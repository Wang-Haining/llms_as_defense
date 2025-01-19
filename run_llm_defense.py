"""
LLM-based defense against authorship attribution attacks. Implements a framework to
evaluate different LLMs' effectiveness in defending against authorship attribution
attacks. Supports multiple research questions (RQ1-RQ3) and various LLM backends.

Directory structure for sample models:
llm_outputs/
├── rj/
│   └── RQ1/
│       └── RQ1.1/
│           ├── gemma-2-9b-it/           # from google/gemma-2-9b-it
│           │   ├── experiment_config.json
│           │   ├── seed_93187.json
│           │   └── seed_95617.json
│           └── llama-3.1-8b-instruct/    # from meta-llama/Llama-3.1-8B-Instruct
│               ├── experiment_config.json
│               ├── seed_93187.json
│               └── seed_95617.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
import openai
from pydantic import BaseModel
from vllm import LLM, SamplingParams

from utils import load_corpus

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FIXED_SEEDS = [
    93187, 95617, 98473, 101089, 103387,
    105673, 108061, 110431, 112757, 115327
]
REWRITE_START_TAG = "[REWRITE]"
REWRITE_END_TAG = "[/REWRITE]"
MIN_WORDS = 50
MAX_RETRIES = 10
BASE_RETRY_DELAY = 1  # seconds


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class ModelInitError(LLMError):
    """Raised when model initialization fails."""
    pass


class APIError(LLMError):
    """Raised when API requests fail."""
    pass


class GenerationError(LLMError):
    """Raised when text generation fails."""
    pass


@dataclass
class ExperimentConfig:
    """
    Combined configuration for experiment and model setup.

    Args:
        corpus: Target corpus for evaluation
        research_question: Main research question category
        sub_question: Specific research question
        output_dir: Directory for saving results
        model_name: Name of the LLM to use
        provider: Model provider (local/anthropic/openai)
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
        num_seeds: Number of seeds (also used for concurrency)
    """
    corpus: str
    research_question: str
    sub_question: str
    model_name: str
    provider: str
    output_dir: str = "llm_outputs"
    temperature: float = 0.7
    max_tokens: int = 4096
    num_seeds: int = 5
    debug: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            corpus=args.corpus,
            research_question=args.rq.split('.')[0],
            sub_question=args.rq,
            output_dir=args.output_dir,
            model_name=args.model,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            num_seeds=args.num_seeds,
            debug=args.debug
        )


class ProviderConfig(BaseModel):
    """Schema for provider-specific prompt configuration."""
    system: str
    user: str
    use_examples: bool = False
    use_cot: bool = False


class PromptTemplate(BaseModel):
    """Schema for model-agnostic prompt templates."""
    description: str
    instruction: Dict[str, str]  # system and user instructions
    use_examples: bool = False
    use_cot: bool = False

    def get_instruction(self) -> Dict[str, str]:
        """Get core instruction configuration."""
        return self.instruction


class PromptManager:
    """Manages prompt formatting and validation for different providers."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def format_prompt(self, template: PromptTemplate, text: str) -> Union[
        str, List[Dict[str, str]]]:
        instruction = template.get_instruction()
        user_prompt = instruction["user"].replace("{{text}}", text)

        if self.config.provider == "anthropic":
            return [
                {"role": "system", "content": instruction["system"]},
                {"role": "user", "content": user_prompt}
            ]

        elif self.config.provider == "openai":
            return [
                {"role": "system", "content": instruction["system"]},
                {"role": "user", "content": user_prompt}
            ]

        else:  # local models
            lower_name = self.config.model_name.lower()
            if "llama" in lower_name or "mistral" in lower_name:
                return f"<s>[INST] {instruction['system']}\n\n{user_prompt} [/INST]"
            elif "olmo" in lower_name or "gemma" in lower_name:
                return f"Human: {instruction['system']}\n\n{user_prompt}\n\nAssistant:"
            else:
                raise ValueError(f"Unsupported local model: {self.config.model_name}")


class ModelManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._api_key = self._get_api_key()
        self.model = self._initialize_model()
        self.prompt_manager = PromptManager(config)

    async def generate_with_validation(
            self,
            prompt: PromptTemplate,
            text: str,
            base_seed: int
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Generate response with validation and retries.
        If it fails the first time, increment the seed by 1 on each retry.
        """
        formatted_prompt = self.prompt_manager.format_prompt(prompt, text)

        for attempt in range(MAX_RETRIES):
            try:
                # Shift the seed by attempt
                used_seed = base_seed + attempt
                random.seed(used_seed)

                if self.config.debug:
                    logger.info(f"Raw input to model:\n{formatted_prompt}")
                    logger.info(f"Using seed={used_seed} for attempt={attempt}")

                response = await self._generate_with_provider(formatted_prompt)

                if response:
                    rewrite = self._validate_and_extract(response)
                    if rewrite:
                        # Return the actual seed used, i.e. base_seed + attempt
                        return response, rewrite, used_seed

            except APIError as e:
                logger.warning(
                    f"API error on attempt {attempt + 1} with seed={used_seed}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                break

        raise GenerationError("Failed to generate valid response after max retries")

    def _get_api_key(self) -> Optional[str]:
        if self.config.provider == "local":
            return None

        key_map = {
            "anthropic": "ANTHROPIC_API_KEY_MALENIA",
            "openai": "OPENAI_API_KEY_MALENIA"
        }
        if self.config.provider not in key_map:
            raise ModelInitError(f"Unsupported provider: {self.config.provider}")

        key = os.environ.get(key_map[self.config.provider])
        if not key:
            raise ModelInitError(
                f"Missing API key for {self.config.provider}. "
                f"Set {key_map[self.config.provider]} environment variable."
            )
        return key

    def _get_quantization_config(self) -> Dict[str, Any]:
        """
        Example: if model size >= 30B => use AWQ + tensor parallel, etc.
        """
        model_name = self.config.model_name.lower()
        size_match = re.search(r'(\d+)b', model_name)
        if size_match:
            model_size = int(size_match.group(1))
            if model_size >= 30:
                return {
                    "quantization": "awq",
                    "max_parallel_loading_workers": 2,
                    "tensor_parallel_size": 2,
                    "gpu_memory_utilization": 0.85
                }
            elif model_size >= 13:
                return {
                    "quantization": "awq",
                    "max_parallel_loading_workers": 1,
                    "gpu_memory_utilization": 0.85
                }
        return {
            "max_parallel_loading_workers": 1,
            "gpu_memory_utilization": 0.85
        }

    def _initialize_model(self) -> Optional[LLM]:
        if self.config.provider != "local":
            return None
        try:
            model_kwargs = {
                "model": self.config.model_name,
                "trust_remote_code": True,
                "dtype": "float16"
            }
            model_kwargs.update(self._get_quantization_config())

            logger.info(f"Initializing local model with config: {model_kwargs}")
            return LLM(**model_kwargs)

        except Exception as e:
            raise ModelInitError(
                f"Failed to initialize local model {self.config.model_name}: {str(e)}"
            )

    async def _generate_with_openai(self, formatted_prompt: list) -> str:
        openai.api_key = self._api_key
        try:
            resp = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=formatted_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return resp.choices[0].message.content
        except openai.error.OpenAIError as e:
            raise APIError(f"OpenAI API error: {str(e)}")

    async def _generate_with_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """
        Use the Anthropic Messages API (new style) with a list of role-content messages.
        """
        client = anthropic.Client(api_key=self._api_key)
        try:
            resp = client.messages.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            return resp.messages[-1]["content"]
        except anthropic.APIStatusError as e:
            raise APIError(f"Anthropic API error: {str(e)}")

    async def _generate_with_provider(
            self,
            formatted_prompt: Union[str, List[Dict[str, str]]]
    ) -> Optional[str]:

        if self.config.debug:
            logger.info(f"Raw input to model:\n{formatted_prompt}")

        if self.config.provider == "anthropic":
            response = await self._generate_with_anthropic(
                formatted_prompt  # type: List[Dict[str, str]]
            )

        elif self.config.provider == "openai":
            response = await self._generate_with_openai(
                formatted_prompt  # type: List[Dict[str, str]]
            )
        else:
            # local model inference
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            response = self.model.generate(formatted_prompt, sampling_params)
            response = response[0].outputs[0].text

        if response and self.config.debug:
            logger.info(f"Raw model output:\n{response}")

        return response

    def _validate_and_extract(self, response: str) -> Optional[str]:
        if not response:
            return None

        start_idx = response.find(REWRITE_START_TAG)
        end_idx = response.find(REWRITE_END_TAG)
        if start_idx == -1 or end_idx == -1:
            logger.warning("Missing rewrite tags in response")
            return None

        rewrite = response[start_idx + len(REWRITE_START_TAG):end_idx].strip()
        if len(rewrite.split()) < MIN_WORDS:
            logger.warning(f"Rewrite too short: {len(rewrite.split())} words")
            return None

        if self.config.debug:
            logger.info(f"Extracted rewrite:\n{rewrite}")

        return rewrite


class ExperimentManager:
    """Manages generation of adversarial examples using LLMs."""

    def __init__(self, config: ExperimentConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.output_dir = self._setup_output_dirs()

    def _setup_output_dirs(self) -> Path:
        model_name = self.config.model_name.split('/')[-1].lower()
        base_dir = (
            Path(self.config.output_dir)
            / self.config.corpus
            / self.config.research_question
            / self.config.sub_question
            / model_name
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    async def _process_single_text(
            self,
            text: str,
            prompt_template: PromptTemplate,
            seed: int
    ) -> Optional[Dict]:
        """Process a single text input."""
        try:
            # Just pass `seed` to generate_with_validation
            raw_output, rewrite, used_seed = await self.model_manager.generate_with_validation(
                prompt_template,
                text,
                base_seed=seed  # <-- new parameter
            )
            if rewrite:
                return {
                    "original": text,
                    "raw_input": {
                        "description": prompt_template.description,
                        "provider": self.model_manager.config.provider
                    },
                    "raw_output": raw_output,
                    "transformed": rewrite,
                    "initial_seed": seed,
                    "actual_seed": used_seed
                }
            return None

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return None

    async def generate_rewrites(
        self,
        texts: List[str],
        prompt_template: PromptTemplate
    ) -> Dict:
        """
        Generate rewritten versions of input texts using LLM transformations.
        We run concurrency = num_seeds, each concurrency job uses a unique seed.
        """

        # pick the seeds
        selected_seeds = FIXED_SEEDS[: self.config.num_seeds]

        async def process_seed(s_idx: int) -> Dict:
            """
            Each seed => one concurrency job
            """
            seed = selected_seeds[s_idx]
            random.seed(seed)

            tasks = [
                self._process_single_text(text, prompt_template, seed)
                for text in texts
            ]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r]

            if valid_results:
                self._save_seed_results(seed, valid_results)

            return {
                "seed": seed,
                "transformations": valid_results
            }

        # build concurrency tasks
        tasks = [process_seed(i) for i in range(self.config.num_seeds)]
        # run them all in parallel
        all_results = await asyncio.gather(*tasks)

        # filter out any that have no transformations
        return {"all_runs": [r for r in all_results if r["transformations"]]}

    def _save_seed_results(self, seed: int, results: List[Dict]) -> None:
        output_file = self.output_dir / f"seed_{seed}.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def save_experiment_config(self) -> None:
        config = {
            "corpus": self.config.corpus,
            "research_question": self.config.research_question,
            "sub_question": self.config.sub_question,
            "num_seeds": self.config.num_seeds,
            "model": {
                "name": self.config.model_name,
                "provider": self.config.provider,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            },
            "generation_constants": {
                "min_words": MIN_WORDS,
                "max_retries": MAX_RETRIES,
                "rewrite_tags": [REWRITE_START_TAG, REWRITE_END_TAG]
            }
        }
        config_file = self.output_dir / "experiment_config.json"
        config_file.write_text(
            json.dumps(config, indent=2),
            encoding='utf-8'
        )


async def main():
    parser = argparse.ArgumentParser(
        description='LLM-based defense against authorship attribution attacks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--corpus',
        required=True,
        choices=['rj', 'ebg', 'lcmc'],
        help='Target corpus for evaluation'
    )
    parser.add_argument(
        '--rq',
        required=True,
        help='Research question identifier (e.g., RQ1.1)'
    )
    parser.add_argument(
        '--output_dir',
        default='llm_outputs',
        help='Base directory for saving results'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Name of the LLM to use'
    )
    parser.add_argument(
        '--provider',
        required=True,
        choices=['local', 'anthropic', 'openai'],
        help='Model provider'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature (0.0-1.0)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help='Maximum tokens for generation'
    )
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=5,
        help='Number of seeds to use (also concurrency). Max 10.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for model inputs/outputs'
    )
    args = parser.parse_args()

    try:
        # setup config
        config = ExperimentConfig.from_args(args)
        logger.info(f"Initialized experiment config for {config.corpus}-{config.sub_question}")

        # load prompt
        prompt_path = Path('prompts') / f"{args.rq}.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt configuration not found: {prompt_path}")

        prompt_config = json.loads(prompt_path.read_text(encoding='utf-8'))
        prompt_template = PromptTemplate(**prompt_config)

        # init managers
        model_manager = ModelManager(config)
        experiment_manager = ExperimentManager(config, model_manager)

        # save experiment config
        experiment_manager.save_experiment_config()
        logger.info(f"Saved experiment config to {experiment_manager.output_dir}")

        # load corpus data
        _, _, test_texts, _ = load_corpus(corpus=config.corpus, task="no_protection")
        logger.info(f"Loaded {len(test_texts)} test samples from {config.corpus}")

        # if debug, truncate to first 3
        if config.debug:
            test_texts = test_texts[:3]
            logger.info("Debug mode ON: using only first 3 samples.")

        # run rewrite generation
        results = await experiment_manager.generate_rewrites(test_texts, prompt_template)

        successful_runs = sum(len(run["transformations"]) for run in results["all_runs"])
        logger.info(
            f"Completed experiment with {successful_runs} successful generations. "
            f"Results saved to: {experiment_manager.output_dir}"
        )

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

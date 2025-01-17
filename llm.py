"""
LLM-based defense against authorship attribution attacks. Implements a framework to
evaluate different LLMs' effectiveness in defending against authorship attribution
attacks. Supports multiple research questions (RQ1-RQ3) and various LLM backends.
"""

import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
import openai
from pydantic import BaseModel
from vllm import LLM, SamplingParams

from utils import load_corpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module level constants
FIXED_SEEDS = [93187, 95617, 98473, 101089, 103387,
               105673, 108061, 110431, 112757, 115327]
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
        batch_size: Number of parallel generations
        output_dir: Directory for saving results
        model_name: Name of the LLM to use
        provider: Model provider (local/anthropic/openai)
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
    """
    corpus: str
    research_question: str
    sub_question: str
    model_name: str
    provider: str
    batch_size: int = 5
    output_dir: str = "llm_outputs"
    temperature: float = 0.7
    max_tokens: int = 4096

    @classmethod
    def from_args(cls, args):
        return cls(
            corpus=args.corpus,
            research_question=args.rq.split('.')[0],
            sub_question=args.rq,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            model_name=args.model,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )


class PromptTemplate(BaseModel):
    """Schema for prompt templates loaded from JSON."""
    system: Optional[str]
    user: str
    examples: Optional[List[Dict[str, str]]] = None
    cot_template: Optional[str] = None
    use_examples: bool = False
    use_cot: bool = False
    format: Optional[Dict[str, Any]] = None


class ModelManager:
    """
    Manages interactions with different LLM backends.

    Handles model initialization, API communication, and generation with retries.

    Attributes:
        config: Configuration settings for the model
        model: Initialized model instance for local providers
        retry_seeds: Seeds used for retry attempts
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = self._initialize_model()
        self.retry_seeds = list(range(1, MAX_RETRIES + 1))
        self._api_key = self._get_api_key()

    def _get_base_model_kwargs(self) -> Dict[str, Any]:
        """Get base kwargs for local model initialization."""
        return {
            "model": self.config.model_name,
            "trust_remote_code": True,
            "dtype": "float16",
            "gpu_memory_utilization": 0.85
        }

    def _get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration for large models."""
        if "70B" in self.config.model_name:
            return {
                "tensor_parallel_size": 2,
                "quantization": "awq",  # Advanced Weight Quantization
                "max_parallel_loading_workers": 2
            }
        return {}

    def _initialize_model(self) -> Optional[Any]:
        """Initialize model based on provider type."""
        try:
            if self.config.provider == "local":
                model_kwargs = self._get_base_model_kwargs()
                model_kwargs.update(self._get_quantization_config())
                return LLM(**model_kwargs)
            return None  # API-based models don't need initialization

        except Exception as e:
            raise ModelInitError(f"Failed to initialize model: {str(e)}")

    def _get_api_key(self) -> Optional[str]:
        """Get appropriate API key from environment."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY_MALENIA",
            "openai": "OPENAI_API_KEY_MALENIA"
        }

        if self.config.provider in key_map:
            key = os.environ.get(key_map[self.config.provider])
            if not key:
                raise APIError(f"Missing API key for {self.config.provider}")
            return key
        return None

    def _format_base_content(self, prompt: PromptTemplate) -> str:
        """Format base content common to all providers."""
        if prompt.system:
            return f"{prompt.system}\n\n{prompt.user}"
        return prompt.user

    def _format_prompt(self, prompt: PromptTemplate) -> Union[str, List[Dict[str, str]]]:
        """Format prompt based on provider requirements."""
        base_content = self._format_base_content(prompt)

        if self.config.provider == "anthropic":
            return f"{anthropic.HUMAN_PROMPT} {base_content}{anthropic.AI_PROMPT}"

        elif self.config.provider == "openai":
            messages = []
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.user})
            return messages

        elif "gemma" in self.config.model_name.lower():
            return f"Human: {base_content}\n\nAssistant:"

        else:  # llama, olmo, mistral
            return f"<s>[INST] {base_content} [/INST]"

    async def _generate_with_provider(
        self,
        formatted_prompt: Union[str, List[Dict[str, str]]]
    ) -> Optional[str]:
        """Generate text using appropriate provider."""
        try:
            if self.config.provider == "anthropic":
                client = anthropic.Client(api_key=self._api_key)
                response = client.messages.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.content[0].text

            elif self.config.provider == "openai":
                client = openai.OpenAI(api_key=self._api_key)
                response = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=formatted_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content

            else:  # local models
                sampling_params = SamplingParams(
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                response = self.model.generate(formatted_prompt, sampling_params)
                return response[0].outputs[0].text

        except Exception as e:
            raise APIError(f"Generation failed: {str(e)}")

    def _validate_and_extract(self, response: str) -> Optional[str]:
        """
        Validate response and extract rewritten content.

        Args:
            response: Raw model response

        Returns:
            Extracted rewrite if valid, None otherwise
        """
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

        return rewrite

    async def generate_with_validation(
        self,
        prompt: PromptTemplate
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Generate response with validation and exponential backoff retries.

        Args:
            prompt: Prompt template to use

        Returns:
            Tuple of (raw_response, validated_rewrite, used_seed)
        """
        formatted_prompt = self._format_prompt(prompt)

        for attempt in range(MAX_RETRIES):
            try:
                # exponential backoff with jitter
                if attempt > 0:
                    delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 1)
                    await asyncio.sleep(delay + jitter)

                # set seed for reproducibility
                used_seed = self.retry_seeds[attempt]
                random.seed(used_seed)

                response = await self._generate_with_provider(formatted_prompt)
                if response:
                    rewrite = self._validate_and_extract(response)
                    if rewrite:
                        return response, rewrite, used_seed

            except APIError as e:
                logger.warning(f"API error on attempt {attempt + 1}: {str(e)}")
                continue

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                break

        raise GenerationError("Failed to generate valid response after max retries")


class ExperimentManager:
    """Manages generation of adversarial examples using LLMs."""

    def __init__(self, config: ExperimentConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.output_dir = self._setup_output_dirs()

    def _setup_output_dirs(self) -> Path:
        """Setup output directory structure."""
        base_dir = Path(self.config.output_dir) / self.config.corpus / \
                   self.config.research_question / self.config.sub_question
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
            raw_output, rewrite, used_seed = await self.model_manager.generate_with_validation(
                prompt_template
            )

            if rewrite:
                return {
                    "original": text,
                    "raw_input": prompt_template.dict(),
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

        Args:
            texts: List of original texts
            prompt_template: Template for generation

        Returns:
            Dict containing results from all generation runs
        """
        async def process_batch(batch_idx: int) -> Dict:
            seed = FIXED_SEEDS[batch_idx % len(FIXED_SEEDS)]
            random.seed(seed)

            tasks = [
                self._process_single_text(text, prompt_template, seed)
                for text in texts
            ]
            results = await asyncio.gather(*tasks)

            valid_results = [r for r in results if r]
            if valid_results:
                self._save_batch_results(seed, valid_results)

            return {
                "seed": seed,
                "transformations": valid_results
            }

        # process batches in parallel
        batch_tasks = [
            process_batch(i)
            for i in range(self.config.batch_size)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        return {"all_runs": [r for r in batch_results if r["transformations"]]}

    def _save_batch_results(self, seed: int, results: List[Dict]) -> None:
        """Save generation results for a specific seed."""
        output_file = self.output_dir / f"seed_{seed}.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def save_experiment_config(self) -> None:
        """Save experiment configuration."""
        config = {
            "corpus": self.config.corpus,
            "research_question": self.config.research_question,
            "sub_question": self.config.sub_question,
            "batch_size": self.config.batch_size,
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
    """
    Main entry point for the LLM defense experiment runner.

    Sets up configuration, initializes components, and runs the rewrite generation
    experiment. Handles command line arguments and manages the overall experiment flow.
    """
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
        '--batch_size',
        type=int,
        default=4,
        help='Number of parallel generation runs'
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

    args = parser.parse_args()

    try:
        # setup experiment configuration
        config = ExperimentConfig.from_args(args)
        logger.info(f"Initialized experiment config for {config.corpus}-{config.sub_question}")

        # load research question specific prompt config
        prompt_path = Path('prompts') / f"{args.rq}.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt configuration not found: {prompt_path}")

        prompt_config = json.loads(prompt_path.read_text(encoding='utf-8'))
        prompt_template = PromptTemplate(
            **prompt_config['prompts'][args.provider.lower()]
        )

        # init managers
        model_manager = ModelManager(config)
        experiment_manager = ExperimentManager(config, model_manager)

        # save experiment configuration
        experiment_manager.save_experiment_config()
        logger.info(f"Saved experiment config to {experiment_manager.output_dir}")

        # load corpus data
        _, _, test_texts, _ = load_corpus(
            corpus=config.corpus,
            task="no_protection"
        )
        logger.info(f"Loaded {len(test_texts)} test samples from {config.corpus}")

        # run rewrite generation
        results = await experiment_manager.generate_rewrites(
            texts=test_texts,
            prompt_template=prompt_template
        )

        successful_runs = sum(
            len(run["transformations"])
            for run in results["all_runs"]
        )
        logger.info(
            f"Completed experiment with {successful_runs} successful generations. "
            f"Results saved to: {experiment_manager.output_dir}"
        )

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

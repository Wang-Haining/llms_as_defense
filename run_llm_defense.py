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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import FIXED_SEEDS, load_corpus

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REWRITE_START_TAG = "[REWRITE]"
REWRITE_END_TAG = "[/REWRITE]"
MIN_WORDS = 50
MAX_RETRIES = 50
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


class RateLimiter:
    """Rate limiter for API requests"""
    def __init__(self, rpm: int = 3):  # openai Tier 1 limit
        self.min_interval = 60.0 / rpm  # minimum seconds between requests
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        """Wait if needed to maintain rate limit."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                logger.info(f"Rate limit: waiting {wait_time:.1f}s before next request")
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()


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

    def get_prompt_path(self) -> Path:
        """Get the path to prompt file, handling special cases for RQ3.x scenarios"""
        base_prompt_dir = Path('prompts')

        # special handling for different RQ3.x scenarios
        if self.sub_question == "rq3.1_obfuscation_w_persona":
            persona_dir = base_prompt_dir / self.sub_question
            if not persona_dir.exists():
                raise FileNotFoundError(f"Persona directory not found: {persona_dir}")

            # list all persona files (excluding non-json files)
            persona_files = [f for f in persona_dir.glob("persona_*.json")]
            if not persona_files:
                raise FileNotFoundError(f"No persona files found in {persona_dir}")

            # randomly select one persona file
            return random.choice(persona_files)

        # handle RQ3.2 imitation with different word counts
        elif self.sub_question.startswith("rq3.2_imitation_w_"):
            imitation_dir = base_prompt_dir / self.sub_question
            if not imitation_dir.exists():
                raise FileNotFoundError(
                    f"Imitation directory not found: {imitation_dir}")

            # list all json files in the directory
            imitation_files = [f for f in imitation_dir.glob("*.json")]
            if not imitation_files:
                raise FileNotFoundError(f"No imitation files found in {imitation_dir}")

            # randomly select one imitation file
            return random.choice(imitation_files)

        # handle RQ3.3 simplification with exemplar
        elif self.sub_question == "rq3.3_simplification_w_exemplar":
            exemplar_dir = base_prompt_dir / self.sub_question
            if not exemplar_dir.exists():
                raise FileNotFoundError(f"Exemplar directory not found: {exemplar_dir}")

            # list all json files in the directory
            exemplar_files = [f for f in exemplar_dir.glob("*.json")]
            if not exemplar_files:
                raise FileNotFoundError(f"No exemplar files found in {exemplar_dir}")

            # randomly select one exemplar file
            return random.choice(exemplar_files)

        # handle RQ3.3 simplification with vocabulary and exemplar
        elif self.sub_question == "rq3.3_simplification_w_vocabulary_and_exemplar":
            vocab_exemplar_dir = base_prompt_dir / self.sub_question
            if not vocab_exemplar_dir.exists():
                raise FileNotFoundError(
                    f"Vocabulary and exemplar directory not found: {vocab_exemplar_dir}")

            # list all json files in the directory
            vocab_exemplar_files = [f for f in vocab_exemplar_dir.glob("*.json")]
            if not vocab_exemplar_files:
                raise FileNotFoundError(
                    f"No vocabulary and exemplar files found in {vocab_exemplar_dir}")

            # randomly select one file
            return random.choice(vocab_exemplar_files)

        # handle RQ3.3 simplification with vocabulary only (single file)
        elif self.sub_question == "rq3.3_simplification_w_vocabulary":
            # direct json file for vocabulary-based simplification
            return base_prompt_dir / f"{self.sub_question}.json"

        # default case: direct json file
        return base_prompt_dir / f"{self.sub_question}.json"


class PromptManager:
    """Manages prompt formatting and validation for different providers."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self._supports_system_role = None

        if self.config.provider == "local":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                raise ModelInitError(
                    f"Failed to load tokenizer for {self.config.model_name}: {str(e)}"
                )

    def format_prompt(self, instructions: Dict[str, str], text: str) -> Union[
        str, List[Dict[str, str]]]:
        """
        Format prompt based on model provider and architecture.
        For local models that explicitly don't support system messages, prepends system to user message.

        Args:
            instructions: Dict containing system and user instructions
            text: Input text to format

        Returns:
            Formatted prompt string or message list depending on provider
        """
        user_prompt = instructions["user"].replace("{{text}}", text)
        system_prompt = instructions["system"].strip()

        if self.config.provider == "anthropic":
            return {
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            }

        elif self.config.provider == "openai":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        else:  # local models
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                error_str = str(e)
                # only handle the specific "System role not supported" error
                if "System role not supported" in error_str:
                    logger.info(
                        f"Model {self.config.model_name} doesn't support system role, combining with user prompt")
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                    user_messages = [{"role": "user", "content": combined_prompt}]
                    return self.tokenizer.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # let any other template errors propagate
                    raise ModelInitError(
                        f"Failed to apply chat template for {self.config.model_name}: {error_str}"
                    )


class ModelManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._api_key = self._get_api_key()
        self.tokenizer = None
        self.model = None
        self.rate_limiter = None
        if self._is_gpt4o_model():
            self.rate_limiter = RateLimiter(rpm=20)  # for tier 2 users

        if self.config.provider == "local":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                self.model = self._initialize_model()
            except Exception as e:
                raise ModelInitError(
                    f"Failed to initialize local model/tokenizer {self.config.model_name}: {str(e)}"
                )

        self.prompt_manager = PromptManager(config)

    def _is_gpt4o_model(self) -> bool:
        """Check if current model is GPT-4o."""
        model_name = self.config.model_name.lower()
        return self.config.provider == "openai" and ("gpt-4o" in model_name or "gpt4o" in model_name)

    async def _generate_with_provider(
            self,
            formatted_prompt: Union[str, List[Dict[str, str]]]
    ) -> Optional[str]:
        """Generate text using specified provider."""

        if self.config.debug:
            logger.info("*" * 90)
            logger.info(f"Raw input to model:\n{formatted_prompt}")
            logger.info("*" * 90)

        if self.config.provider == "anthropic":
            response = await self._generate_with_anthropic(formatted_prompt)

        elif self.config.provider == "openai":
            response = await self._generate_with_openai(formatted_prompt)

        else:  # local model inference with vLLM
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            outputs = self.model.generate(formatted_prompt, sampling_params)
            response = outputs[0].outputs[0].text

        if response and self.config.debug:
            logger.info("*" * 90)
            logger.info(f"Raw model output:\n{response}")
            logger.info("*" * 90)

        return response

    async def generate_with_validation(
            self,
            instructions: Dict[str, str],
            text: str,
            base_seed: int
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        generate response with validation and retries
        if it fails the first time, increment the seed by 1 on each retry
        """
        formatted_prompt = self.prompt_manager.format_prompt(instructions, text)
        self.last_input_text = text

        for attempt in range(MAX_RETRIES):
            try:
                used_seed = base_seed + attempt
                random.seed(used_seed)

                if self.config.debug:
                    logger.info(f"raw input to model:\n*{formatted_prompt}*")
                    logger.info(f"using seed={used_seed} for attempt={attempt}")

                response = await self._generate_with_provider(formatted_prompt)

                if response:
                    rewrite = self._validate_and_extract(response)
                    if rewrite:
                        return response, rewrite, used_seed

            except APIError as e:
                logger.warning(
                    f"api error on attempt {attempt + 1} with seed={used_seed}: {str(e)}\n"
                    f"full input text:\n*{text}*"
                )
                continue
            except Exception as e:
                logger.error(f"unexpected error: {str(e)}")
                break

        raise GenerationError(
            f"failed to generate valid response after max retries.\n"
            f"last attempted input text:\n*{text}*"
        )

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
                "dtype": "float16",
                "gpu_memory_utilization": 0.85,
                "tensor_parallel_size": 1  # set default tensor parallelism
            }

            # get quantization config
            quant_config = self._get_quantization_config()

            # update with quantization settings while preserving tensor_parallel_size
            if "tensor_parallel_size" in quant_config:
                model_kwargs["tensor_parallel_size"] = quant_config[
                    "tensor_parallel_size"]

            # add other quantization settings
            if "quantization" in quant_config:
                model_kwargs["quantization"] = quant_config["quantization"]
            if "max_parallel_loading_workers" in quant_config:
                model_kwargs["max_parallel_loading_workers"] = quant_config[
                    "max_parallel_loading_workers"]

            # remove gpu_memory_utilization from quant_config as it's already set
            quant_config.pop("gpu_memory_utilization", None)

            logger.info(f"Initializing local model with config: {model_kwargs}")
            return LLM(**model_kwargs)

        except Exception as e:
            raise ModelInitError(
                f"Failed to initialize local model {self.config.model_name}: {str(e)}"
            )

    async def _generate_with_openai(self, formatted_prompt: list) -> str:
        """Generate with proper rate limiting for GPT-4o."""
        if self.rate_limiter:
            await self.rate_limiter.wait()

        client = AsyncOpenAI(api_key=self._api_key)
        try:
            resp = await client.chat.completions.create(
                model=self.config.model_name,
                messages=formatted_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "insufficient_quota" in str(e):
                logger.error("Quota exceeded, waiting 60s before retry...")
                await asyncio.sleep(60)
            raise APIError(f"OpenAI API error: {str(e)}")

    async def _generate_with_anthropic(self, prompt_data: dict) -> str:
        """
        Return a single string, not a list of TextBlock.
        """
        client = anthropic.Client(api_key=self._api_key)
        try:
            resp = client.messages.create(
                model=self.config.model_name,
                system=prompt_data["system"],
                messages=prompt_data["messages"],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False
            )
            # 'resp.content' is a list of TextBlock or similar
            # combine them into a single string:
            if isinstance(resp.content, list):
                # e.g. [TextBlock(text='...'), TextBlock(text='...')]
                joined = "".join(block.text for block in resp.content)
                return joined
            else:
                # return a single string
                return resp.content

        except anthropic.APIStatusError as e:
            raise APIError(f"Anthropic API error: {str(e)}")

    def _validate_and_extract(self, response: str) -> Optional[str]:
        if not response:
            logger.warning("Empty response from model")
            return None

        start_idx = response.find(REWRITE_START_TAG)
        end_idx = response.find(REWRITE_END_TAG)
        if start_idx == -1 or end_idx == -1:
            logger.warning(
                f"Missing rewrite tags in response.\n"
                f"Full model response:\n*{response}*"
            )
            return None

        rewrite = response[start_idx + len(REWRITE_START_TAG):end_idx].strip()
        word_count = len(rewrite.split())
        if word_count < MIN_WORDS:
            logger.warning(
                f"Rewrite too short: {word_count} words (minimum: {MIN_WORDS})\n"
                f"Full input text:\n*{self.last_input_text}*\n"
                f"Full model response:\n*{response}*"
            )
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
        self.is_gpt4o = self._is_gpt4o_model()

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
            instructions: Dict[str, str],
            seed: int
    ) -> Optional[Dict]:
        """process a single text input"""
        try:
            raw_output, rewrite, used_seed = await self.model_manager.generate_with_validation(
                instructions,
                text.strip(),  # strip whitespace from input text
                base_seed=seed
            )
            if rewrite:
                return {
                    "original": text,
                    "raw_input": {
                        "provider": self.model_manager.config.provider
                    },
                    "raw_output": raw_output,
                    "transformed": rewrite,
                    "initial_seed": seed,
                    "actual_seed": used_seed
                }
            return None

        except Exception as e:
            logger.error(f"error processing text: {str(e)}")
            return None

    def _is_gpt4o_model(self) -> bool:
        """Check if current model is GPT-4 and needs special handling."""
        model_name = self.config.model_name.lower()
        return self.config.provider == "openai" and ("gpt-4o" in model_name or "gpt4o" in model_name)

    async def _generate_gpt4o_rewrites(self, texts: List[str],
                                       instructions: Dict[str, str]) -> Dict:
        selected_seeds = FIXED_SEEDS[:5]  # Always use 5 seeds
        all_results = []

        for seed_idx, seed in enumerate(selected_seeds):
            random.seed(seed)
            # create tasks for each text, instead of processing texts sequentially
            tasks = [
                self._process_single_text(text, instructions, seed)
                for text in texts
            ]
            transformations = [r for r in await asyncio.gather(*tasks) if r]

            if transformations:
                self._save_seed_results(seed, transformations)
                all_results.append({
                    "seed": seed,
                    "transformations": transformations
                })
                logger.info(
                    f"Completed seed {seed_idx + 1}/5 with {len(transformations)} successful transformations")
            else:
                logger.info(f"No successful transformations for seed {seed}")
        return {"all_runs": [r for r in all_results if r["transformations"]]}

    async def generate_rewrites(
        self,
        texts: List[str],
        instructions: Dict[str, str]
    ) -> Dict:
        """Generate rewritten versions of input texts using LLM transformations."""
        if self.is_gpt4o:
            logger.info("Using sequential processing for GPT-4 model")
            return await self._generate_gpt4o_rewrites(texts, instructions)
        selected_seeds = FIXED_SEEDS[: self.config.num_seeds]

        async def process_seed(s_idx: int) -> Dict:
            seed = selected_seeds[s_idx]
            random.seed(seed)

            tasks = [
                self._process_single_text(text, instructions, seed)
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

        tasks = [process_seed(i) for i in range(self.config.num_seeds)]
        all_results = await asyncio.gather(*tasks)
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
        config = ExperimentConfig.from_args(args)
        logger.info(
            f"initialized experiment config for {config.corpus}-{config.sub_question}")
        # get prompt path (now handles both regular and persona cases)
        prompt_path = config.get_prompt_path()
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt configuration not found: {prompt_path}")

        # log the selected persona file for RQ3.1
        if config.sub_question == "rq3.1_obfuscation_w_persona":
            logger.info(f"Selected persona file: {prompt_path.name}")

        instructions = json.loads(prompt_path.read_text(encoding='utf-8'))

        # init managers
        model_manager = ModelManager(config)
        experiment_manager = ExperimentManager(config, model_manager)

        # save experiment config
        experiment_manager.save_experiment_config()
        logger.info(f"saved experiment config to {experiment_manager.output_dir}")

        # load and process corpus data
        _, _, test_texts, _ = load_corpus(corpus=config.corpus, task="no_protection")
        logger.info(f"loaded {len(test_texts)} test samples from {config.corpus}")

        if config.debug:
            test_texts = test_texts[:3]
            logger.info("debug mode ON: using only first 3 samples")

        results = await experiment_manager.generate_rewrites(test_texts, instructions)

        successful_runs = sum(
            len(run["transformations"]) for run in results["all_runs"])
        logger.info(
            f"completed experiment with {successful_runs} successful generations. "
            f"results saved to: {experiment_manager.output_dir}"
        )

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

"""
Prepare exemplar-based prompts for RQ3.2 using Blog-1K corpus.

This script:
1. Loads the Blog-1K corpus from resources/blog1000.csv.gz
2. Randomly selects 50 authors from the training split
3. For each author:
   a. Collects samples to meet fixed word thresholds (500/1000/2500) [default]
   b. Generates a sample with random length between 100-2500 words [with --variable_length]
4. Generates corresponding prompt JSON files in:
   - prompts/rq3.2_imitation_w_500words/ [default]
   - prompts/rq3.2_imitation_w_1000words/ [default]
   - prompts/rq3.2_imitation_w_2500words/ [default]
   - prompts/rq3.2_imitation_variable_length/ [with --variable_length]

Usage:
    # generate all prompts (fixed thresholds)
    python prepare_prompts_for_rq3.2_imitation.py

    # generate only variable length prompts
    python prepare_prompts_for_rq3.2_imitation.py --variable_length

Source:
    BLOG1K: https://zenodo.org/records/7455623#.Y5-v9uxAphG
"""

import json
import logging
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sacremoses import MosesTokenizer

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# constants
NUM_AUTHORS = 50
WORD_THRESHOLDS = [500, 1000, 2500]
VARIABLE_LENGTH_MIN = 100
VARIABLE_LENGTH_MAX = 2500
RANDOM_SEED = 42
OUTPUT_BASE_DIR = Path('prompts')


def load_blog1k(file_path: str = 'resources/blog1000.csv.gz') -> pd.DataFrame:
    """Load BLOG1K corpus from compressed csv"""
    try:
        df = pd.read_csv(file_path, compression='infer')
        logger.info(f"loaded {len(df)} entries from {file_path}")
        return df
    except Exception as e:
        logger.error(f"failed to load corpus: {e}")
        raise


def get_word_count(text: str, tokenizer: MosesTokenizer) -> int:
    """Get word count using moses tokenizer for consistency"""
    return len(tokenizer.tokenize(text, escape=False))


def select_authors(df: pd.DataFrame, num_authors: int = NUM_AUTHORS) -> List[str]:
    """Randomly select authors from training split"""
    train_authors = df[df.split == 'train']['id'].unique()
    random.seed(RANDOM_SEED)
    selected = random.sample(list(train_authors), num_authors)
    logger.info(f"selected {len(selected)} authors")
    return selected


def collect_author_samples(
    df: pd.DataFrame,
    author_id: str,
    word_thresholds: List[int],
    tokenizer: MosesTokenizer
) -> Dict[int, str]:
    """
    Collect samples for an author to meet different word count thresholds
    returns dict mapping threshold -> concatenated text

    Only uses texts from training split to avoid data leakage
    """
    # explicitly verify we're only getting training texts
    author_df = df[df.id == author_id]
    train_df = author_df[author_df.split == 'train']

    if len(train_df) == 0:
        logger.warning(f"author {author_id} has no training samples")
        return {}

    logger.debug(
        f"author {author_id} has {len(train_df)} training samples "
        f"(out of {len(author_df)} total)"
    )

    # get training texts and shuffle
    author_texts = train_df['text'].tolist()
    random.shuffle(author_texts)

    results = {}
    for threshold in word_thresholds:
        combined = []
        word_count = 0

        for text in author_texts:
            combined.append(text)
            word_count = get_word_count(' '.join(combined), tokenizer)

            if word_count >= threshold:
                break

        if word_count >= threshold:
            results[threshold] = ' '.join(combined)
        else:
            logger.warning(
                f"author {author_id} has insufficient words: {word_count}/{threshold}"
            )

    return results


def collect_variable_length_sample(
    df: pd.DataFrame,
    author_id: str,
    min_words: int,
    max_words: int,
    tokenizer: MosesTokenizer
) -> Tuple[str, int, int]:
    """
    Collect a sample with random target length between min_words and max_words
    returns tuple of (concatenated_text, actual_word_count, target_length)
    """
    # get training texts
    author_df = df[df.id == author_id]
    train_df = author_df[author_df.split == 'train']

    if len(train_df) == 0:
        logger.warning(f"author {author_id} has no training samples")
        return "", 0, 0

    # generate random target length
    target_length = random.randint(min_words, max_words)
    logger.info(f"Target length for author {author_id}: {target_length} words")

    # get training texts and shuffle
    author_texts = train_df['text'].tolist()
    random.shuffle(author_texts)

    combined = []
    word_count = 0

    for text in author_texts:
        # calculate word count of current text
        text_word_count = get_word_count(text, tokenizer)
        new_total = word_count + text_word_count

        # if adding this text would exceed target by too much and we already have min_words, stop
        if new_total > target_length + 200 and word_count >= min_words:
            break

        # otherwise add it
        combined.append(text)
        word_count = new_total

        # if we've reached target or exceeded it slightly, stop
        if word_count >= target_length:
            break

    if combined and word_count >= min_words:
        combined_text = ' '.join(combined)
        final_word_count = get_word_count(combined_text, tokenizer)
        return combined_text, final_word_count, target_length

    logger.warning(f"author {author_id} has insufficient words for variable length sample")
    return "", 0, target_length


def create_prompt_json(
    author_id: str,
    exemplar_text: str,
    word_count: int,
    output_dir: Path,
    index: int,
    target_length: Optional[int] = None
) -> None:
    """
    create prompt json file with exemplar text and metadata

    Args:
        author_id: unique identifier of the author from Blog-1K
        exemplar_text: example text to be mimicked
        word_count: actual word count of the exemplar
        output_dir: directory to save the prompt
        index: prompt index (0-49)
        target_length: for variable length samples, the target word count
    """
    # ensure all values are native Python types, not numpy types
    prompt_data = {
        "system": "You are a helpful writing assistant. Your task is to paraphrase text while preserving its meaning. Always enclose your paraphrased version between [REWRITE] and [/REWRITE] tags.",
        "user": (
            f"Here is an example of the writing style you are expected to mimic:"
            f"\n\n{exemplar_text}\n\n"
            f"Please rewrite the following text to match this writing style while "
            f"maintaining its core meaning.\n\nText to be modified:\n\n{{{{text}}}}\n\n"
            f"Provide your rewrite between [REWRITE] and [/REWRITE] tags."
        ),
        "metadata": {
            "author_id": str(author_id),  # ensure string
            "word_count": int(word_count),  # ensure int
            "prompt_index": int(index)  # ensure int
        }
    }

    # add target_length to metadata if provided (for variable length samples)
    if target_length is not None:
        prompt_data["metadata"]["target_length"] = int(target_length)

    output_path = output_dir / f"prompt{index:02d}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)


def process_fixed_thresholds(df: pd.DataFrame, authors_with_indices: List, tokenizer: MosesTokenizer) -> None:
    """Process fixed threshold samples (500/1000/2500 words)"""
    for threshold in WORD_THRESHOLDS:
        # create output directory
        output_dir = OUTPUT_BASE_DIR / f"rq3.2_imitation_w_{threshold}words"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {threshold} word samples")

        # collect samples and create prompts for each author-index pair
        for i, (author_id, index) in enumerate(authors_with_indices):
            logger.info(f"Processing sample {i+1}/{len(authors_with_indices)}: author {author_id}")
            samples = collect_author_samples(df, author_id, [threshold], tokenizer)

            if threshold in samples:
                create_prompt_json(
                    author_id,
                    samples[threshold],
                    threshold,
                    output_dir,
                    index
                )
                logger.info(f"Created prompt{index:03d}.json with {threshold} words")
            else:
                logger.warning(f"Skipping author {author_id} for {threshold} words")

        prompt_count = len(list(output_dir.glob("prompt*.json")))
        logger.info(f"Created {prompt_count} prompts in {output_dir}")


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Prepare exemplar-based prompts for RQ3.2')
    parser.add_argument('--variable_length', action='store_true',
                        help='Generate only variable length prompts (100-2500 words)')
    args = parser.parse_args()

    # initialize tokenizer
    tokenizer = MosesTokenizer(lang='en')

    # load corpus
    df = load_blog1k()

    # verify we have training data
    train_df = df[df.split == 'train']
    logger.info(
        f"corpus has {len(train_df)} training samples "
        f"from {len(train_df.id.unique())} authors"
    )

    if args.variable_length:
        # only process variable length samples - fixed at 500 prompts
        num_prompts = 500
        logger.info(f"Processing {num_prompts} variable length samples")
        process_variable_length(df, num_prompts, tokenizer)
    else:
        # process only fixed thresholds (original functionality)
        logger.info("Processing fixed threshold samples (500/1000/2500 words)")

        # use original author selection (50 authors)
        authors = select_authors(df, NUM_AUTHORS)
        process_fixed_thresholds(df, authors, tokenizer)


def process_variable_length(df: pd.DataFrame, num_prompts: int,
                            tokenizer: MosesTokenizer) -> None:
    """Process variable length samples (100-2500 words)"""
    # create directory for variable length samples
    variable_length_dir = OUTPUT_BASE_DIR / "rq3.2_imitation_variable_length"
    variable_length_dir.mkdir(parents=True, exist_ok=True)

    # dictionary to store variable length metadata
    variable_length_metadata = {}

    # Select authors with replacement if needed
    train_authors = df[df.split == 'train']['id'].unique()
    random.seed(RANDOM_SEED)

    # Generate author-index pairs for each prompt
    author_selections = []
    for i in range(num_prompts):
        author_id = random.choice(train_authors)
        author_selections.append((author_id, i))

    logger.info(
        f"Selected {len(author_selections)} author-index pairs for variable length prompts")

    # process each author-index pair
    for i, (author_id, index) in enumerate(author_selections):
        logger.info(f"Processing prompt {i + 1}/{num_prompts}: author {author_id}")

        # generate sample with random target length
        variable_text, actual_count, target_length = collect_variable_length_sample(
            df, author_id, VARIABLE_LENGTH_MIN, VARIABLE_LENGTH_MAX, tokenizer
        )

        if variable_text and actual_count >= VARIABLE_LENGTH_MIN:
            # Use 3 digits for indices to accommodate up to 1000 prompts
            prompt_filename = f"prompt{index:03d}.json"

            # Create the prompt
            prompt_data = {
                "system": "You are a helpful writing assistant. Your task is to paraphrase text while preserving its meaning. Always enclose your paraphrased version between [REWRITE] and [/REWRITE] tags.",
                "user": (
                    f"Here is an example of the writing style you are expected to mimic:"
                    f"\n\n{variable_text}\n\n"
                    f"Please rewrite the following text to match this writing style while "
                    f"maintaining its core meaning.\n\nText to be modified:\n\n{{{{text}}}}\n\n"
                    f"Provide your rewrite between [REWRITE] and [/REWRITE] tags."
                ),
                "metadata": {
                    "author_id": str(author_id),  # ensure string
                    "word_count": int(actual_count),  # ensure int
                    "prompt_index": int(index),  # ensure int
                    "target_length": int(target_length)  # target word count
                }
            }

            # save the prompt
            output_path = variable_length_dir / prompt_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)

            # store metadata for analysis
            prompt_key = f"prompt{index:03d}"
            variable_length_metadata[prompt_key] = {
                "author_id": str(author_id),
                "target_length": int(target_length),
                "actual_count": int(actual_count),
                "prompt_index": int(index)
            }

            logger.info(
                f"Created {prompt_filename} with {actual_count} words (target: {target_length})")
        else:
            logger.warning(f"Skipping author {author_id}: insufficient word count")
            # try another author as a replacement
            retry_count = 0
            while retry_count < 5:  # limit retries to avoid infinite loops
                retry_count += 1
                replacement_author = random.choice(train_authors)
                logger.info(f"Trying replacement author {replacement_author}")

                variable_text, actual_count, target_length = collect_variable_length_sample(
                    df, replacement_author, VARIABLE_LENGTH_MIN, VARIABLE_LENGTH_MAX,
                    tokenizer
                )

                if variable_text and actual_count >= VARIABLE_LENGTH_MIN:
                    # use the replacement author
                    prompt_filename = f"prompt{index:03d}.json"

                    # create the prompt
                    prompt_data = {
                        "system": "You are a helpful writing assistant. Your task is to paraphrase text while preserving its meaning. Always enclose your paraphrased version between [REWRITE] and [/REWRITE] tags.",
                        "user": (
                            f"Here is an example of the writing style you are expected to mimic:"
                            f"\n\n{variable_text}\n\n"
                            f"Please rewrite the following text to match this writing style while "
                            f"maintaining its core meaning.\n\nText to be modified:\n\n{{{{text}}}}\n\n"
                            f"Provide your rewrite between [REWRITE] and [/REWRITE] tags."
                        ),
                        "metadata": {
                            "author_id": str(replacement_author),  # ensure string
                            "word_count": int(actual_count),  # ensure int
                            "prompt_index": int(index),  # ensure int
                            "target_length": int(target_length)  # target word count
                        }
                    }

                    # save the prompt
                    output_path = variable_length_dir / prompt_filename
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(prompt_data, f, indent=2, ensure_ascii=False)

                    # store metadata for analysis
                    prompt_key = f"prompt{index:03d}"
                    variable_length_metadata[prompt_key] = {
                        "author_id": str(replacement_author),
                        "target_length": int(target_length),
                        "actual_count": int(actual_count),
                        "prompt_index": int(index)
                    }

                    logger.info(
                        f"Created {prompt_filename} with replacement author {replacement_author}: "
                        f"{actual_count} words (target: {target_length})")
                    break

            if retry_count >= 5:
                logger.error(
                    f"Failed to find suitable replacement for prompt{index:03d}")

    # save metadata for variable length samples
    metadata_path = variable_length_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(variable_length_metadata, f, indent=2, ensure_ascii=False)

    prompt_count = len(list(variable_length_dir.glob("prompt*.json")))
    logger.info(
        f"Created {prompt_count} variable length prompts in {variable_length_dir}")
    logger.info(f"Variable length metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
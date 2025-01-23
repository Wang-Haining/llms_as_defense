"""
Prepare exemplar-based prompts for RQ3.2 using Blog-1K corpus.

This script:
1. Loads the Blog-1K corpus from resources/blog1000.csv.gz
2. Randomly selects 50 authors from the training split
3. For each author, collects enough samples to meet word thresholds (500/1000/2500)
4. Generates corresponding prompt JSON files in prompts/rq3.2_imitation_w_*

Directory structure for generated prompts:
prompts/
├── rq3.2_imitation_w_500words/
│   ├── prompt00.json
│   ├── prompt01.json
│   └── ... (up to prompt49.json)
├── rq3.2_imitation_w_1000words/
│   └── ... (same structure)
└── rq3.2_imitation_w_2500words/
    └── ... (same structure)

Each prompt JSON file contains:
{
    "system": str,         # standard system message for writing assistant
    "user": str,          # instruction with example text and placeholder
    "metadata": {
        "author_id": str,     # unique identifier from Blog-1K
        "word_count": int,    # target word count (500/1000/2500)
        "prompt_index": int   # index of prompt (0-49)
    }
}

The exemplars are extracted only from the training split of Blog-1K corpus
to avoid any data leakage.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

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
RANDOM_SEED = 42
OUTPUT_BASE_DIR = Path('prompts')


def load_blog1k(file_path: str = 'resources/blog1000.csv.gz') -> pd.DataFrame:
    """load blog1k corpus from compressed csv"""
    try:
        df = pd.read_csv(file_path, compression='infer')
        logger.info(f"loaded {len(df)} entries from {file_path}")
        return df
    except Exception as e:
        logger.error(f"failed to load corpus: {e}")
        raise


def get_word_count(text: str, tokenizer: MosesTokenizer) -> int:
    """get word count using moses tokenizer for consistency"""
    return len(tokenizer.tokenize(text, escape=False))


def select_authors(df: pd.DataFrame, num_authors: int = NUM_AUTHORS) -> List[str]:
    """randomly select authors from training split"""
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
    collect samples for an author to meet different word count thresholds
    returns dict mapping threshold -> concatenated text

    only uses texts from training split to avoid data leakage
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


def create_prompt_json(
    author_id: str,
    exemplar_text: str,
    word_count: int,
    output_dir: Path,
    index: int
) -> None:
    """
    create prompt json file with exemplar text and metadata

    Args:
        author_id: unique identifier of the author from Blog-1K
        exemplar_text: example text to be mimicked
        word_count: target word count (500/1000/2500)
        output_dir: directory to save the prompt
        index: prompt index (0-49)
    """
    # ensure all values are native Python types, not numpy types
    prompt_data = {
        "system": "You are a helpful writing assistant. Your task is to paraphrase text while preserving its meaning. Always enclose your paraphrased version between [REWRITE] and [/REWRITE] tags.",
        "user": (
            f"Here is an example of the writing style you are expected to mimic "
            f"({word_count} words):\n\n'''\n{exemplar_text}\n'''\n\n"
            f"Please rewrite the following text to match this writing style while "
            f"maintaining its core meaning:\n\n{{text}}\n\n"
            f"Remember to enclose your rewrite between [REWRITE] and [/REWRITE] tags."
        ),
        "metadata": {
            "author_id": str(author_id),  # ensure string
            "word_count": int(word_count),  # ensure int
            "prompt_index": int(index)  # ensure int
        }
    }

    output_path = output_dir / f"prompt{index:02d}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)

    output_path = output_dir / f"prompt{index:02d}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)


def main():
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

    # select authors
    authors = select_authors(df)

    # verify selected authors have sufficient training data
    authors_data = df[df.id.isin(authors) & (df.split == 'train')]
    author_counts = authors_data.groupby('id').size()
    logger.info(
        f"selected authors have {author_counts.mean():.1f} training samples on average "
        f"(min: {author_counts.min()}, max: {author_counts.max()})"
    )

    # process each word threshold
    for word_count in WORD_THRESHOLDS:
        # create output directory
        output_dir = OUTPUT_BASE_DIR / f"rq3.2_imitation_w_{word_count}words"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"processing {word_count} word samples")

        # collect samples and create prompts for each author
        for idx, author_id in enumerate(authors):
            samples = collect_author_samples(df, author_id, [word_count], tokenizer)

            if word_count in samples:
                create_prompt_json(
                    author_id,
                    samples[word_count],
                    word_count,
                    output_dir,
                    idx
                )
            else:
                logger.warning(f"skipping author {author_id} for {word_count} words")

        logger.info(f"created prompts in {output_dir}")


if __name__ == "__main__":
    main()

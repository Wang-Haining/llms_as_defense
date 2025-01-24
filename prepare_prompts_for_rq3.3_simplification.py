"""
Create all RQ3.3 prompt variants:
1. With VOA vocabulary only
2. With VOA exemplars only
3. With both vocabulary and exemplars combined

This script generates all three sets of prompts with consistent formatting and structure.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def read_voa_words(filepath: str) -> list[str]:
    """read VOA Special English word list from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def read_voa_articles(filepath: str) -> List[Dict]:
    """read VOA articles from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles


def format_word_list(words: list[str]) -> str:
    """format word list into a compact string with single spaces between words."""
    return f"\n\nHere is the complete VOA Special English word list:\n{' '.join(words)}"


def create_vocabulary_prompt(voa_words: List[str]) -> Dict:
    """create a prompt with VOA vocabulary only."""
    return {
        "system": "You are a helpful writing assistant. Do this task as if you are "
                 "asked as part of a college application. Always enclose your "
                 "rewritten version between [REWRITE] and [/REWRITE] tags.",
        "user": ("Please rewrite this text using Special English, a simplified version "
                "of English. There are two basic rules for writing in Special English:\n"
                "1. Write using words in the Special English word list. (Using an "
                "occasional proper noun which is not on the list is acceptable.) "
                "2. Write using short, simple sentences. Try to use the simplest "
                "sentences that make sense.\n\n"
                "Your writing should not raise any eyebrows when a human reads over "
                "it and the point should still be clearly conveyed.\n\n"
                "This is the VOA Special English word list you are permitted to use:"
                f"{format_word_list(voa_words)}"
                "\n\nText to be modified:\n\n{{text}}\n\n"
                "Provide your rewrite between [REWRITE] and [/REWRITE] tags.")
    }


def create_exemplar_prompt(article: Dict) -> Dict:
    """create a prompt with a VOA article as exemplar."""
    return {
        "system": "You are a helpful writing assistant. Do this task as if you are "
                 "asked as part of a college application. Always enclose your "
                 "rewritten version between [REWRITE] and [/REWRITE] tags.",
        "user": ("Please rewrite this text using Special English, a simplified version "
                "of English. There are two basic rules for writing in Special English:\n"
                "1. Write using words in the Special English word list. (Using an "
                "occasional proper noun which is not on the list is acceptable.) "
                "2. Write using short, simple sentences. Try to use the simplest "
                "sentences that make sense.\n\n"
                "Your writing should not raise any eyebrows when a human reads over "
                "it and the point should still be clearly conveyed.\n\n"
                f"Here is an example of Special English writing from VOA:"
                f"\n\n{article['content']}"
                "\n\nText to be modified:\n\n{{text}}\n\n"
                "Provide your rewrite between [REWRITE] and [/REWRITE] tags.")
    }


def create_combined_prompt(article: Dict, voa_words: List[str]) -> Dict:
    """create a prompt with both VOA vocabulary and article exemplar."""
    return {
        "system": "You are a helpful writing assistant. Do this task as if you are "
                 "asked as part of a college application. Always enclose your "
                 "rewritten version between [REWRITE] and [/REWRITE] tags.",
        "user": ("Please rewrite this text using Special English, a simplified version "
                "of English. There are two basic rules for writing in Special English:\n"
                "1. Write using words in the Special English word list. (Using an "
                "occasional proper noun which is not on the list is acceptable.) "
                "2. Write using short, simple sentences. Try to use the simplest "
                "sentences that make sense.\n\n"
                "Your writing should not raise any eyebrows when a human reads over "
                "it and the point should still be clearly conveyed."
                "\nThis is the VOA Special English word list you are permitted to use:"
                f"{format_word_list(voa_words)}\n\n"
                f"Here is an example of Special English writing from VOA:"
                f"\n\n{article['content']}"
                "\n\nText to be modified:\n\n{{text}}\n\n"
                "Provide your rewrite between [REWRITE] and [/REWRITE] tags.")
    }


def save_prompt(prompt: Dict, filepath: Path) -> None:
    """save a single prompt to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt, f, indent=4, ensure_ascii=False)


def save_prompts(prompts: List[Dict], output_dir: Path) -> None:
    """save multiple prompts to numbered JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts, 1):
        output_file = output_dir / f"prompt_{i:02d}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompt, f, indent=4, ensure_ascii=False)


def main():
    """main function to create and save all prompt variants."""
    # setup paths
    voa_words_path = Path("resources/voa.txt")
    voa_articles_path = Path("resources/manythings_voa.json")
    prompts_dir = Path("prompts")

    # create output paths
    vocab_file = prompts_dir / "rq3.3_simplification_w_vocabulary.json"
    exemplar_dir = prompts_dir / "rq3.3_simplification_w_exemplar"
    combined_dir = prompts_dir / "rq3.3_simplification_w_vocabulary_and_exemplar"

    # read input data
    voa_words = read_voa_words(voa_words_path)
    articles = read_voa_articles(voa_articles_path)

    # create and save vocabulary-only prompt
    vocab_prompt = create_vocabulary_prompt(voa_words)
    save_prompt(vocab_prompt, vocab_file)
    print(f"Created vocabulary prompt at {vocab_file}")

    # randomly sample 50 articles
    sampled_articles = random.sample(articles, 50)

    # create and save exemplar-only prompts
    exemplar_prompts = [create_exemplar_prompt(article) for article in sampled_articles]
    save_prompts(exemplar_prompts, exemplar_dir)
    print(f"Created {len(exemplar_prompts)} exemplar prompts in {exemplar_dir}")

    # create and save combined prompts
    combined_prompts = [create_combined_prompt(article, voa_words)
                       for article in sampled_articles]
    save_prompts(combined_prompts, combined_dir)
    print(f"Created {len(combined_prompts)} combined prompts in {combined_dir}")


if __name__ == "__main__":
    # set random seed for reproducibility
    random.seed(42)
    main()

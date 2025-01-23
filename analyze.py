"""
Analyze evaluation results for LLM-based defenses against authorship attribution threat
models.

This script:
1. Reads `.npz` evaluation results from a specified directory structure.
2. Analyzes performance changes and calculates 95% confidence intervals for:
   - Attribution metrics (e.g., MRR change).
   - Text quality metrics (e.g., BLEU, METEOR scores).
3. Handles multiple seeds to ensure statistical robustness.

Usage:
    python analyze.py --corpus rj --rq rq1.1_basic_paraphrase --model meta-llama/Llama-3.1-8B-Instruct
    python analyze.py --corpus ebg --rq rq1.1_basic_paraphrase  # analyze all models in a corpus
"""

import argparse
import numpy as np
import logging
from pathlib import Path
from scipy.stats import t
from typing import Dict
from utils import LLMS, CORPORA, RQS

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_npz(file_path: Path) -> Dict:
    """load an .npz file and return its results."""
    if not file_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    return data["results"].item()


def analyze_results(consolidated_results: Dict, seed_dir: Path, num_seeds: int) -> Dict:
    """
    Analyze performance changes and calculate 95% confidence intervals.

    Args:
        consolidated_results: dict containing aggregated metrics from `evaluation.npz`.
        seed_dir: Path to directory containing individual `seed_{seed}.npz` files.
        num_seeds: Number of seeds expected for evaluation.

    Returns:
        Dict containing mean and 95% confidence intervals for attribution changes and quality scores.
    """
    attribution_changes = []
    quality_scores = []

    # process seed-level `.npz` files if they exist
    for seed_file in seed_dir.glob("seed_*.npz"):
        seed_data = np.load(seed_file, allow_pickle=True)["results"].item()
        for model_type, metrics in seed_data.items():
            effectiveness = metrics["attribution"]["effectiveness"]
            quality = metrics["quality"]

            # append metrics for this seed
            attribution_changes.append(effectiveness["mrr_change"])
            quality_scores.append(quality["bleu"]["bleu"])  # example for BLEU

    if len(attribution_changes) < num_seeds:
        logger.warning(f"Found results for only {len(attribution_changes)} seeds (expected {num_seeds}).")

    if not attribution_changes:
        raise ValueError("No valid seeds found for analysis.")

    # calculate mean and 95% CI for attribution changes and quality scores
    attribution_mean = np.mean(attribution_changes)
    attribution_std = np.std(attribution_changes, ddof=1)
    attribution_ci = t.interval(
        0.95, len(attribution_changes) - 1, loc=attribution_mean,
        scale=attribution_std / np.sqrt(len(attribution_changes)),
    )

    quality_mean = np.mean(quality_scores)
    quality_std = np.std(quality_scores, ddof=1)
    quality_ci = t.interval(
        0.95, len(quality_scores) - 1, loc=quality_mean,
        scale=quality_std / np.sqrt(len(quality_scores)),
    )

    return {
        "attribution_mean": attribution_mean,
        "attribution_ci": attribution_ci,
        "quality_mean": quality_mean,
        "quality_ci": quality_ci,
    }


def analyze_corpus(corpus: str, rq: str, model: str = None, num_seeds: int = 5):
    """
    Analyze evaluation results for a specific corpus, research question, and model.

    Args:
        corpus: Corpus to analyze (e.g., 'rj').
        rq: Research question identifier (e.g., 'rq1.1_basic_paraphrase').
        model: Specific model to analyze (use full name). If not provided, analyze all models.
        num_seeds: Number of seeds expected for evaluation.
    """
    base_dir = Path("defense_evaluation") / corpus / "rq1" / rq
    if model:
        # analyze a specific model
        model_dir = base_dir / model.split('/')[-1].lower()
        evaluation_file = model_dir / "evaluation.npz"
        seed_dir = model_dir

        if not evaluation_file.exists():
            raise FileNotFoundError(f"No evaluation results found for model: {model}")

        logger.info(f"Analyzing {model} for corpus {corpus} and RQ {rq}.")
        consolidated_results = load_npz(evaluation_file)
        analysis = analyze_results(consolidated_results, seed_dir, num_seeds)

        logger.info(f"Results for {model}:")
        logger.info(f"Attribution change: mean={analysis['attribution_mean']:.4f}, CI={analysis['attribution_ci']}")
        logger.info(f"Quality score: mean={analysis['quality_mean']:.4f}, CI={analysis['quality_ci']}")

    else:
        # analyze all models for the corpus and RQ
        model_dirs = list(base_dir.glob("*/"))
        if not model_dirs:
            raise FileNotFoundError(f"No evaluation results found for corpus: {corpus} and RQ: {rq}")

        all_results = []
        for model_dir in model_dirs:
            evaluation_file = model_dir / "evaluation.npz"
            if not evaluation_file.exists():
                logger.warning(f"Missing consolidated file in {model_dir}. Skipping.")
                continue

            logger.info(f"Analyzing results for model in {model_dir.stem}.")
            consolidated_results = load_npz(evaluation_file)
            seed_dir = model_dir
            analysis = analyze_results(consolidated_results, seed_dir, num_seeds)
            all_results.append(analysis)

        # aggregate results across all models
        combined_attribution_changes = [
            res["attribution_mean"] for res in all_results
        ]
        combined_quality_scores = [
            res["quality_mean"] for res in all_results
        ]

        overall_attribution_mean = np.mean(combined_attribution_changes)
        overall_attribution_std = np.std(combined_attribution_changes, ddof=1)
        overall_attribution_ci = t.interval(
            0.95, len(combined_attribution_changes) - 1, loc=overall_attribution_mean,
            scale=overall_attribution_std / np.sqrt(len(combined_attribution_changes)),
        )

        overall_quality_mean = np.mean(combined_quality_scores)
        overall_quality_std = np.std(combined_quality_scores, ddof=1)
        overall_quality_ci = t.interval(
            0.95, len(combined_quality_scores) - 1, loc=overall_quality_mean,
            scale=overall_quality_std / np.sqrt(len(combined_quality_scores)),
        )

        logger.info(f"Overall results for corpus {corpus} and RQ {rq}:")
        logger.info(
            f"Attribution change: mean={overall_attribution_mean:.4f}, CI={overall_attribution_ci}")
        logger.info(
            f"Quality score: mean={overall_quality_mean:.4f}, CI={overall_quality_ci}")


def main():
    parser = argparse.ArgumentParser(
        description="analyze evaluation results for llm-based defenses."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=CORPORA,
        help="corpus to analyze.",
    )
    parser.add_argument(
        "--rq",
        required=True,
        choices=RQS,
        help="research question to analyze (e.g., rq1.1_basic_paraphrase).",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="specific model to analyze (use full name). if not set, analyze all models.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="number of seeds expected for evaluation (default: 5).",
    )

    args = parser.parse_args()

    try:
        analyze_corpus(args.corpus, args.rq, model=args.model, num_seeds=args.num_seeds)
    except Exception as e:
        logger.error(f"error during analysis: {str(e)}")


if __name__ == "__main__":
    main()

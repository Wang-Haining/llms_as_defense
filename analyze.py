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
    python analyze.py --corpus rj --rq rq1.1 --model meta-llama/Llama-3.1-8B-Instruct
    python analyze.py --corpus ebg --rq rq1  # analyze all models in a corpus
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


def analyze_results(results: Dict, num_seeds: int) -> Dict:
    """analyze performance changes and calculate 95% confidence intervals."""
    attribution_changes = []
    quality_scores = []

    for seed, metrics in results.items():
        if len(metrics) != num_seeds:
            logger.warning(
                f"seed {seed} does not have enough results. found {len(metrics)}, expected {num_seeds}."
            )
            continue

        for model_type, data in metrics.items():
            # get attribution and quality metrics
            effectiveness = data["attribution"]["effectiveness"]
            quality = data["quality"]

            attribution_changes.append(effectiveness["mrr_change"])
            quality_scores.append(quality["bleu"]["bleu"])  # example for BLEU

    if not attribution_changes:
        raise ValueError("no valid seeds found for analysis.")

    # calculate mean and 95% CI for attribution changes and quality scores
    attribution_mean = np.mean(attribution_changes)
    attribution_std = np.std(attribution_changes, ddof=1)
    attribution_ci = t.interval(
        0.95, len(attribution_changes) - 1, loc=attribution_mean,
        scale=attribution_std / np.sqrt(len(attribution_changes))
    )

    quality_mean = np.mean(quality_scores)
    quality_std = np.std(quality_scores, ddof=1)
    quality_ci = t.interval(
        0.95, len(quality_scores) - 1, loc=quality_mean,
        scale=quality_std / np.sqrt(len(quality_scores))
    )

    return {
        "attribution_mean": attribution_mean,
        "attribution_ci": attribution_ci,
        "quality_mean": quality_mean,
        "quality_ci": quality_ci,
    }


def analyze_corpus(corpus: str, rq: str, model: str = None, num_seeds: int = 5):
    """analyze evaluation results for a specific corpus, research question, and model."""
    base_dir = Path("defense_evaluation") / corpus / rq
    if model:
        # analyze a specific model
        model_dir = base_dir.glob(f"**/{model.split('/')[-1].lower()}/evaluation.npz")
        model_files = list(model_dir)
        if not model_files:
            raise FileNotFoundError(f"no evaluation results found for model: {model}")

        logger.info(f"analyzing {model} for corpus {corpus} and RQ {rq}.")
        results = load_npz(model_files[0])  # load evaluation results
        analysis = analyze_results(results, num_seeds)

        logger.info(f"results for {model}:")
        logger.info(
            f"attribution change: mean={analysis['attribution_mean']:.4f}, CI={analysis['attribution_ci']}")
        logger.info(
            f"quality score: mean={analysis['quality_mean']:.4f}, CI={analysis['quality_ci']}")

    else:
        # analyze all models for the corpus and RQ
        model_dirs = list(base_dir.glob(f"**/*/evaluation.npz"))
        if not model_dirs:
            raise FileNotFoundError(f"no evaluation results found for corpus: {corpus} and RQ: {rq}")

        models_used = set([str(p.parent.stem) for p in model_dirs])
        logger.info(
            f"found evaluation results for {len(models_used)} models in corpus {corpus} and RQ {rq}.")

        missing_models = set(LLMS) - models_used
        if missing_models:
            logger.warning(
                f"not all llms are evaluated for corpus {corpus} and RQ {rq}. missing: {len(missing_models)} models."
            )

        all_results = []
        for model_file in model_dirs:
            results = load_npz(model_file)
            all_results.append(analyze_results(results, num_seeds))

        # aggregate results across all models
        combined_attribution_changes = [
            res["attribution_mean"] for res in all_results
        ]
        combined_quality_scores = [res["quality_mean"] for res in all_results]

        overall_attribution_mean = np.mean(combined_attribution_changes)
        overall_attribution_std = np.std(combined_attribution_changes, ddof=1)
        overall_attribution_ci = t.interval(
            0.95,
            len(combined_attribution_changes) - 1,
            loc=overall_attribution_mean,
            scale=overall_attribution_std / np.sqrt(len(combined_attribution_changes)),
        )

        overall_quality_mean = np.mean(combined_quality_scores)
        overall_quality_std = np.std(combined_quality_scores, ddof=1)
        overall_quality_ci = t.interval(
            0.95,
            len(combined_quality_scores) - 1,
            loc=overall_quality_mean,
            scale=overall_quality_std / np.sqrt(len(combined_quality_scores)),
        )

        logger.info(f"overall results for corpus {corpus} and RQ {rq}:")
        logger.info(
            f"attribution change: mean={overall_attribution_mean:.4f}, CI={overall_attribution_ci}")
        logger.info(
            f"quality score: mean={overall_quality_mean:.4f}, CI={overall_quality_ci}")


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
        help="research question to analyze (e.g., rq1, rq1.1).",
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

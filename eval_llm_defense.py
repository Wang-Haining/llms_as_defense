"""
Evaluate effectiveness of LLM-based defense against authorship attribution models.

This script:
1. Loads original test samples and LLM-transformed texts
2. Evaluates both using trained attribution models
3. Computes comprehensive metrics including:
   - Attribution performance (accuracy, MRR, MAP, etc.)
   - Defense effectiveness (ranking changes, entropy increases)
   - Text quality (BLEU, METEOR, BERTScore)
4. Saves detailed results for analysis in `.npz` format

Directory Structure:
defense_evaluation/                          # or specified output_dir
├── {corpus}/                               # rj, ebg, or lcmc
│   ├── RQ{N}/                             # main research question (e.g., RQ1)
│   │   ├── RQ{N}.{M}/                     # sub-question (e.g., RQ1.1)
│   │   │   ├── {model_name}/              # e.g., gemma-2b-it
│   │   │   │   └── evaluation.npz         # consolidated results for all seeds
│   │   │   │       ├── seed_{seed}.npz    # per-seed results
│   │   │   └── {another_model}/
│   │   └── RQ{N}.{M+1}/
│   └── RQ{N+1}/
└── {another_corpus}/

File Formats:
1. `seed_{seed}.npz`: Contains results for a single seed, including:
   - Attribution metrics: accuracy, MRR, MAP, entropy, confidence gaps
   - Defense effectiveness measures: MRR change, entropy increase
   - Text quality metrics: PINC, BLEU, METEOR, BERTScore
   Saved in a NumPy `.npz` file, with `allow_pickle=True` required for loading.

2. `evaluation.npz`: Consolidated results for all seeds in the experiment,
   following the same structure as individual per-seed files.

Usage:
    python eval_llm_defense.py --rq rq1.1_basic_paraphrase --model "google/gemma-2-9b-it"

"""


import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score

from eval_text_quality import evaluate_quality
from roberta import RobertaPredictor
from utils import (CORPORA, LLMS, RQS, LogisticRegressionPredictor,
                   SVMPredictor, load_corpus)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def defense_effectiveness(pre_metrics, post_metrics):
    """
    calculate effectiveness of the defense by comparing pre and post metrics.
    focuses on ranking-based and distribution metrics that handle different
    candidate pool sizes appropriately.

    Args:
        pre_metrics: dict with metrics before defense
        post_metrics: dict with metrics after defense

    Returns:
        dict: effectiveness metrics with interpretation guidelines
    """
    effectiveness = {}

    # ranking metric changes
    for k in [1, 3, 5]:
        metric = f'top_{k}_acc'
        if metric in pre_metrics and metric in post_metrics:
            effectiveness[f'top_{k}_change'] = (
                post_metrics[metric] - pre_metrics[metric]
            )

    # mrr change
    if 'mrr' in pre_metrics and 'mrr' in post_metrics:
        rel_change = (post_metrics['mrr'] - pre_metrics['mrr']) / pre_metrics['mrr']
        effectiveness['mrr_change'] = rel_change
        # average rank improvement
        effectiveness['avg_rank_improvement'] = (
            1 / post_metrics['mrr'] - 1 / pre_metrics['mrr']
        )

    # distribution changes
    if 'entropy_normalized' in pre_metrics and 'entropy_normalized' in post_metrics:
        effectiveness['entropy_change'] = (
            post_metrics['entropy_normalized'] - pre_metrics['entropy_normalized']
        )

    if 'conf_gap_normalized' in pre_metrics and 'conf_gap_normalized' in post_metrics:
        effectiveness['conf_gap_change'] = (
            post_metrics['conf_gap_normalized'] - pre_metrics['conf_gap_normalized']
        )

    # ndcg changes for each k
    for k in [1, 3, 5, None]:
        key = f'ndcg@{k if k else "all"}'
        if key in pre_metrics and key in post_metrics:
            effectiveness[f'{key}_change'] = (
                post_metrics[key] - pre_metrics[key]
            )

    # weighted rank score change
    if 'weighted_rank_score' in pre_metrics and 'weighted_rank_score' in post_metrics:
        effectiveness['weighted_rank_change'] = (
            post_metrics['weighted_rank_score'] - pre_metrics['weighted_rank_score']
        )

    return effectiveness


def gini_coefficient(array):
    """
    Calculate the Gini coefficient of an array.
        - Value of 0 expresses perfect equality (all predictions equally likely)
        - Value of 1 expresses maximal inequality (one prediction has all probability)
    """
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 1e-10  # avoid division by zero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def ndcg_score(y_true, y_pred_probs, k=None):
    """
    Calculate Normalized Discounted Cumulative Gain.
        - Measures ranking quality considering position importance
        - Normalized version allows comparison across different numbers of authors

    Args:
        y_true: true author index
        y_pred_probs: predicted probabilities for each author
        k: number of positions to consider (None for all)
    """

    def dcg(y_true, y_pred_probs, k):
        # sort predictions in descending order and get indices
        sorted_indices = np.argsort(y_pred_probs)[::-1]
        if k:
            sorted_indices = sorted_indices[:k]

        # calculate DCG
        gains = [1.0 if idx == y_true else 0.0 for idx in sorted_indices]
        discounts = [1.0 / np.log2(i + 2) for i in range(len(gains))]
        return np.sum(gains * np.array(discounts))

    # calculate actual DCG
    actual_dcg = dcg(y_true, y_pred_probs, k)

    # calculate ideal DCG (true author ranked first)
    ideal_probs = np.zeros_like(y_pred_probs)
    ideal_probs[y_true] = 1.0
    ideal_dcg = dcg(y_true, ideal_probs, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_metrics(y_true, y_pred_probs):
    """calculate attribution metrics focusing on ranking performance."""
    metrics = {}
    n_candidates = y_pred_probs.shape[1]

    # 1. ranking metrics (naturally handle different candidate pool sizes)
    ranks = []
    for true_label, probs in zip(y_true, y_pred_probs):
        rank = len(probs) - np.where(np.argsort(probs) == true_label)[0][0]
        ranks.append(1 / rank)  # MRR naturally scales
    metrics['mrr'] = np.mean(ranks)
    metrics['mrr_std'] = np.std(ranks)

    # 2. top-k accuracy (standard in attribution literature)
    for k in [1, 3, 5]:
        if k > n_candidates:
            continue  # skip if k larger than candidate pool
        top_k_correct = 0
        for true_label, probs in zip(y_true, y_pred_probs):
            top_k_indices = np.argsort(probs)[-k:]
            if true_label in top_k_indices:
                top_k_correct += 1
        metrics[f'top_{k}_acc'] = top_k_correct / len(y_true)

    # 3. entropy normalized by maximum possible entropy for the candidate pool
    max_entropy = np.log2(n_candidates)  # maximum entropy for uniform distribution
    entropies = -np.sum(y_pred_probs * np.log2(y_pred_probs + 1e-10), axis=1)
    normalized_entropies = entropies / max_entropy  # now bounded [0,1]
    metrics['entropy_normalized'] = np.mean(normalized_entropies)
    metrics['entropy_normalized_std'] = np.std(normalized_entropies)

    # 4. confidence gap relative to uniform baseline
    uniform_gap = 1 / n_candidates  # gap if predictions were uniform
    confidence_gaps = []
    for probs in y_pred_probs:
        sorted_probs = np.sort(probs)
        gap = sorted_probs[-1] - sorted_probs[-2]
        confidence_gaps.append(gap / uniform_gap)  # normalize by baseline
    metrics['conf_gap_normalized'] = np.mean(confidence_gaps)
    metrics['conf_gap_normalized_std'] = np.std(confidence_gaps)

    # 5. NDCG at different k values (measures ranking quality with position importance)
    for k in [1, 3, 5, None]:
        ndcg_scores = [
            ndcg_score(true_label, probs, k)
            for true_label, probs in zip(y_true, y_pred_probs)
        ]
        metrics[f'ndcg@{k if k else "all"}'] = np.mean(ndcg_scores)
        metrics[f'ndcg@{k if k else "all"}_std'] = np.std(ndcg_scores)

    # 6. combined ranking metric (weight different k values by candidate pool size)
    rank_weights = np.array(
        [1 / k for k in [1, min(3, n_candidates), min(5, n_candidates)]])
    rank_weights = rank_weights / rank_weights.sum()  # normalize weights

    weighted_rank_score = 0
    for i, k in enumerate([1, 3, 5]):
        if f'top_{k}_acc' in metrics:
            weighted_rank_score += rank_weights[i] * metrics[f'top_{k}_acc']
    metrics['weighted_rank_score'] = weighted_rank_score

    return metrics


class DefenseEvaluator:
    """Evaluates LLM defense effectiveness against attribution models."""

    def __init__(
        self,
        results_dir: Path,
        llm_outputs_dir: Path,
        output_dir: Path
    ):
        """Initialize evaluator with paths to required data.

        Args:
            results_dir: directory containing trained models
            llm_outputs_dir: directory containing LLM transformations
            output_dir: directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.llm_outputs_dir = Path(llm_outputs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # map of model types to predictor classes
        self.predictor_classes = {
            'logreg': LogisticRegressionPredictor,
            'svm': SVMPredictor,
            'roberta': RobertaPredictor
        }

    def _load_predictor(
        self,
        corpus: str,
        model_type: str
    ) -> object:
        """Load trained model predictor for given corpus and type."""
        model_dir = (
            self.results_dir / corpus / "no_protection" /
            model_type / "model"
        )
        return self.predictor_classes[model_type](model_dir)

    def _get_experiment_paths(self, corpus: str, rq: str, model_name: str) -> Path:
        """Get experiment directory based on RQ identifier and model name.

        Args:
            corpus: corpus name (rj/ebg/lcmc)
            rq: research question identifier (e.g. rq1.1_basic_paraphrase)
            model_name: full model name (e.g. google/gemma-2b-it)

        Returns:
            Path to experiment directory
        """
        rq_base = rq.split('_')[0]  # e.g., 'rq1.1'
        rq_main = rq_base.split('.')[0]  # e.g., 'rq1'

        # model dir name: take last part of model path
        model_dir = model_name.split('/')[-1].lower()

        expected_path = self.llm_outputs_dir / corpus / rq_main / rq / model_dir
        logger.info(f"Constructed path: {expected_path}")
        return expected_path

    def _load_llm_outputs(
        self,
        corpus: str,
        rq: str,
        model_name: str
    ) -> List[Dict]:
        """Load LLM-generated transformations for an experiment."""
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)
        logger.info(f"Looking for transformations in: {exp_dir}")

        transformations = []
        seed_files = list(exp_dir.glob('seed_*.json'))
        logger.info(f"Found {len(seed_files)} seed files")

        for seed_file in seed_files:
            logger.info(f"Loading transformations from: {seed_file}")
            with open(seed_file) as f:
                results = json.load(f)
                logger.info(f"Loaded {len(results)} results from seed file")
                if isinstance(results, list):
                    transformations.extend(results)
                else:
                    # handle nested structures
                    if 'all_runs' in results:
                        for run in results['all_runs']:
                            if 'transformations' in run:
                                transformations.extend(run['transformations'])
                                logger.info(
                                    f"Added {len(run['transformations'])} transformations from run"
                                )

        logger.info(f"Total transformations loaded: {len(transformations)}")

        if not transformations:
            logger.warning(f"No transformations found in {exp_dir}")

        return transformations

    def evaluate_experiment(self, corpus: str, rq: str, model_name: str) -> Dict[
        str, Dict]:
        """
        Evaluate LLM-based defense against attribution models for a specific experiment.

        Returns:
            Dict mapping seeds to their evaluation results including attribution and quality metrics.
        """
        logger.info(f"Evaluating {corpus}-{rq} using {model_name}")

        # load original test data
        _, _, test_texts, test_labels = load_corpus(corpus=corpus, task="no_protection")
        logger.info(f"Loaded {len(test_texts)} original test texts")

        # load and process transformations by seed
        transformations_by_seed = {}
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)

        # process each seed file
        for seed_file in exp_dir.glob("seed_*.json"):
            seed_id = int(seed_file.stem.split("_")[1])
            logger.info(f"Loading transformations from: {seed_file}")

            with open(seed_file) as f:
                seed_data = json.load(f)
                # extract transformed texts from the JSON structure
                transformed_texts = []
                # handle nested structure in JSON
                if "transformations" in seed_data:
                    transformed_texts = [t["transformed"] for t in
                                         seed_data["transformations"]]
                else:
                    for entry in seed_data:
                        if "transformed" in entry:
                            transformed_texts.append(entry["transformed"])

                if len(transformed_texts) == len(test_texts):
                    transformations_by_seed[seed_id] = transformed_texts
                else:
                    logger.warning(
                        f"Skipping seed {seed_id}: expected {len(test_texts)} transformations, got {len(transformed_texts)}")

        logger.info(f"Loaded transformations for {len(transformations_by_seed)} seeds")

        # prepare output directory
        output_base = (
                self.output_dir / corpus / rq.split('_')[0] / rq /
                model_name.split('/')[-1].lower()
        )
        output_base.mkdir(parents=True, exist_ok=True)

        # evaluate each seed
        all_seed_results = {}
        for seed_id, transformed_texts in transformations_by_seed.items():
            seed_results = {}
            example_metrics = []

            for model_type in ['logreg', 'svm', 'roberta']:
                logger.info(f"Evaluating seed {seed_id} against {model_type}")

                predictor = self._load_predictor(corpus, model_type)
                orig_preds = predictor.predict_proba(test_texts)
                trans_preds = predictor.predict_proba(transformed_texts)

                original_metrics = calculate_metrics(test_labels, orig_preds)
                transformed_metrics = calculate_metrics(test_labels, trans_preds)
                effectiveness = defense_effectiveness(original_metrics,
                                                      transformed_metrics)

                # collect example-level metrics
                for idx, (true_label, orig_prob, trans_prob) in enumerate(
                        zip(test_labels, orig_preds, trans_preds)
                ):
                    orig_ranks = np.argsort(orig_prob)[::-1]
                    trans_ranks = np.argsort(trans_prob)[::-1]

                    example_metrics.append({
                        "example_id": idx,
                        "true_label": true_label,
                        "orig_probs": orig_prob.tolist(),
                        "trans_probs": trans_prob.tolist(),
                        "original_rank": np.where(orig_ranks == true_label)[0][0],
                        "transformed_rank": np.where(trans_ranks == true_label)[0][0],
                        "mrr_change": (1 / (
                                    np.where(trans_ranks == true_label)[0][0] + 1) -
                                       1 / (np.where(orig_ranks == true_label)[0][
                                                0] + 1))
                    })

                # calculate text quality metrics
                quality_metrics = evaluate_quality(
                    candidate_texts=transformed_texts,
                    reference_texts=test_texts,
                    metrics=['pinc', 'bleu', 'meteor', 'bertscore']
                )

                seed_results[model_type] = {
                    'attribution': {
                        'original_metrics': original_metrics,
                        'transformed_metrics': transformed_metrics,
                        'effectiveness': effectiveness
                    },
                    'quality': quality_metrics
                }

            # save per-seed results with example-level metrics
            output_file = output_base / f"seed_{seed_id}.npz"
            np.savez_compressed(
                output_file,
                results=seed_results,
                example_metrics=example_metrics
            )
            logger.info(f"Saved evaluation results with raw metrics to {output_file}")

            all_seed_results[seed_id] = seed_results

        return all_seed_results

    def save_results(
        self,
        results: Dict,
        corpus: str,
        rq: str,
        model_name: str
    ) -> None:
        """
        Save evaluation results following consistent structure. Also logs
        a quick summary of the first-level metrics (e.g., accuracy, MRR).
        """
        # get experiment directory structure
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)
        save_dir = self.output_dir / exp_dir.relative_to(self.llm_outputs_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # save the entire results dict to a single .npz file
        output_file = save_dir / "evaluation.npz"
        np.savez_compressed(output_file, results=results)
        logger.info(f"Saved final consolidated results to {output_file}")

        # log summary metrics
        logger.info(f"\nResults for {corpus}-{rq} using {model_name}:")
        for seed, seed_dict in results.items():
            logger.info(f"\n--- Seed: {seed} ---")
            for model_type, metrics in seed_dict.items():
                orig = metrics['attribution']['original_metrics']
                transformed = metrics['attribution']['transformed_metrics']

                logger.info(f"\n{model_type.upper()} Results:")

                # Primary metrics (MRR and Top-k)
                logger.info("Ranking Performance:")
                logger.info(f"  MRR: {orig['mrr']:.4f} → {transformed['mrr']:.4f}")
                for k in [1, 3, 5]:
                    if f'top_{k}_acc' in orig:  # only show if k applicable
                        logger.info(
                            f"  Top-{k}: {orig[f'top_{k}_acc']:.4f} → {transformed[f'top_{k}_acc']:.4f}")

                # normalized distribution metrics
                logger.info("\nPrediction Distribution (Normalized):")
                logger.info(
                    f"  Entropy: {orig['entropy_normalized']:.4f} → {transformed['entropy_normalized']:.4f}")
                logger.info(
                    f"  Conf Gap: {orig['conf_gap_normalized']:.4f} → {transformed['conf_gap_normalized']:.4f}")

                # combined score
                logger.info(
                    f"\nWeighted Rank Score: {orig['weighted_rank_score']:.4f} → {transformed['weighted_rank_score']:.4f}")

                # text quality (independent of candidate pool)
                qual = metrics['quality']
                logger.info("\nText Quality:")
                logger.info(f"  BLEU: {qual['bleu']['bleu']:.4f}")
                logger.info(f"  METEOR: {qual['meteor']['meteor_avg']:.4f}")
                logger.info(f"  BERTScore: {qual['bertscore']['bertscore_f1_avg']:.4f}")

    def main_loop(self, args):
        """
        Handles the overall loop of corpora, RQs, and models.
        Separated from if __name__ == "__main__" for clarity/testing.
        """
        corpora = [args.corpus] if args.corpus else CORPORA
        rqs = [args.rq] if args.rq else RQS
        models = [args.model] if args.model else LLMS

        for corpus in corpora:
            for rq in rqs:
                for model in models:
                    try:
                        results = self.evaluate_experiment(corpus, rq, model)
                        self.save_results(results, corpus, rq, model)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating {corpus}-{rq} with {model}: {str(e)}"
                        )
                        continue


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM defense effectiveness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results'),
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--llm_outputs',
        type=Path,
        default=Path('llm_outputs'),
        help='Directory containing LLM outputs'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('defense_evaluation'),
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['rj', 'ebg', 'lcmc'],
        help='Specific corpus to evaluate (default: all)'
    )
    parser.add_argument(
        '--rq',
        type=str,
        help='Research question identifier (e.g. rq1.1_basic_paraphrase)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Full model name (e.g. google/gemma-2b-it)'
    )

    args = parser.parse_args()

    evaluator = DefenseEvaluator(
        results_dir=args.results_dir,
        llm_outputs_dir=args.llm_outputs,
        output_dir=args.output_dir
    )
    evaluator.main_loop(args)


if __name__ == "__main__":
    main()

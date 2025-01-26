"""
Evaluate effectiveness of LLM-based defense against authorship attribution models.

This script implements a comprehensive evaluation framework for assessing the effectiveness
of LLM-based defenses against authorship attribution attacks. It measures both the absolute
improvement and progress toward ideal defensive conditions.

Key Components:
1. Attribution Performance Metrics:
   - Accuracy@1/5 and F1@1/5 scores
   - True class confidence
   - Distribution metrics (entropy, Gini coefficient, TVD)
   - Ranking metrics (MRR, Wasserstein distance)

2. Defense Effectiveness Measures:
   - Absolute changes in all metrics
   - Progress toward ideal defensive conditions:
     * Random guessing accuracy (1/n_classes)
     * Uniform distribution (max entropy)
     * Equal class probabilities (min Gini)
     * Random ranking (MRR = 1/n_classes)
     * Maximum redistribution (Wasserstein)

3. Text Quality Metrics:
   - PINC (Paraphrase In N-gram Changes)
   - BLEU (Bilingual Evaluation Understudy)
   - METEOR (considering synonyms)
   - BERTScore (contextual similarity)

Directory Structure:
defense_evaluation/                        # or specified output_dir
├── {corpus}/                              # rj, ebg, or lcmc
│   ├── RQ{N}/                             # main research question (e.g., RQ1)
│   │   ├── RQ{N}.{M}/                     # sub-question (e.g., RQ1.1)
│   │   │   ├── {model_name}/              # e.g., gemma-2b-it
│   │   │   │   ├── evaluation.json        # consolidated results
│   │   │   │   └── seed_{seed}.json       # per-seed results
│   │   │   └── {another_model}/
│   │   └── RQ{N}.{M+1}/
│   └── RQ{N+1}/
└── {another_corpus}/

File Formats:
1. seed_{seed}.json: Contains per-seed results including:
   - Metadata (corpus, RQ, model details, dimensions)
   - Raw Data:
     * Original and transformed input texts
     * True labels
     * Original and transformed prediction probabilities
   - Attribution Metrics:
     * Original and transformed metrics
     * Progress toward ideal defensive conditions
   - Text Quality Metrics:
     * PINC, BLEU, METEOR, BERTScore
   - Example-level Metrics:
     * Individual sample performance
     * Rank changes
     * Distribution changes

2. evaluation.json: Consolidated results across all seeds, following
   the same structure as individual seed files.

Usage:
    python eval_llm_defense.py --model "google/gemma-2b-it"  # for all rqs
    python eval_llm_defense.py --rq rq1.1_basic_paraphrase --model "google/gemma-2b-it"
"""


import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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


def compute_accuracy_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray,
                          k: int) -> float:
    """Compute top-k accuracy.

    Args:
        y_true: array of true labels
        y_pred_probs: array of prediction probabilities
        k: k for top-k accuracy

    Returns:
        Accuracy@k score
    """
    n_samples = len(y_true)
    top_k_correct = 0
    for true_label, probs in zip(y_true, y_pred_probs):
        top_k_indices = np.argsort(probs)[-k:]
        if true_label in top_k_indices:
            top_k_correct += 1
    return float(top_k_correct / n_samples)


def compute_f1_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """Compute macro-averaged f1 score for top-k predictions.

    Args:
        y_true: array of true labels
        y_pred_probs: array of prediction probabilities
        k: k for top-k f1 score

    Returns:
        f1@k score
    """
    n_classes = y_pred_probs.shape[1]

    # convert to top-k predictions
    y_pred_top_k = []
    for probs in y_pred_probs:
        top_k_indices = np.argsort(probs)[-k:]
        y_pred_top_k.append(top_k_indices)

    # convert to multi-label format
    mlb = MultiLabelBinarizer(classes=range(n_classes))
    y_true_bin = mlb.fit_transform([[label] for label in y_true])
    y_pred_bin = mlb.transform(y_pred_top_k)

    return float(f1_score(y_true_bin, y_pred_bin, average='macro'))


def compute_true_class_confidence(y_true: np.ndarray,
                                  y_pred_probs: np.ndarray) -> float:
    """Compute average confidence for true class.

    Args:
        y_true: array of true labels
        y_pred_probs: array of prediction probabilities

    Returns:
        Mean confidence for true class
    """
    true_class_probs = np.array(
        [probs[true_label] for true_label, probs in zip(y_true, y_pred_probs)])
    return float(np.mean(true_class_probs))


def compute_entropy(y_pred_probs: np.ndarray) -> tuple[float, float]:
    """Compute normalized entropy of predictions.

    Args:
        y_pred_probs: array of prediction probabilities

    Returns:
        Mean normalized entropy & std of normalized entropy
    """
    n_classes = y_pred_probs.shape[1]
    max_entropy = np.log2(n_classes)
    entropies = -np.sum(y_pred_probs * np.log2(y_pred_probs + 1e-10), axis=1)
    normalized_entropies = entropies / max_entropy
    return float(np.mean(normalized_entropies)), float(np.std(normalized_entropies))


def compute_gini(y_pred_probs: np.ndarray) -> tuple[float, float]:
    """compute gini coefficient of predictions.

    Args:
        y_pred_probs: array of prediction probabilities

    Returns:
        Mean gini coefficient & std of gini coefficient
    """
    gini_scores = []
    for probs in y_pred_probs:
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        index = np.arange(1, n + 1)
        gini = ((np.sum((2 * index - n - 1) * sorted_probs)) /
                (n * np.sum(sorted_probs)))
        gini_scores.append(gini)
    return float(np.mean(gini_scores)), float(np.std(gini_scores))


def compute_tvd(y_pred_probs: np.ndarray) -> tuple[float, float]:
    """Compute total variation distance from uniform distribution.

    Args:
        y_pred_probs: array of prediction probabilities

    Returns:
        Mean TVD & std of TVD
    """
    n_classes = y_pred_probs.shape[1]
    uniform_dist = np.ones(n_classes) / n_classes
    tvd_scores = []
    for probs in y_pred_probs:
        tvd = 0.5 * np.sum(np.abs(probs - uniform_dist))
        tvd_scores.append(tvd)
    return float(np.mean(tvd_scores)), float(np.std(tvd_scores))


def compute_mrr(y_true: np.ndarray, y_pred_probs: np.ndarray) -> tuple[float, float]:
    """Compute mean reciprocal rank.

    Args:
        y_true: array of true labels
        y_pred_probs: array of prediction probabilities

    Returns:
        MRR & std of reciprocal ranks
    """
    mrr_scores = []
    for true_label, probs in zip(y_true, y_pred_probs):
        rank = len(probs) - np.where(np.argsort(probs) == true_label)[0][0]
        mrr_scores.append(1 / rank)
    return float(np.mean(mrr_scores)), float(np.std(mrr_scores))


def compute_wasserstein(y_pred_probs: np.ndarray) -> tuple[float, float]:
    """Compute wasserstein distance from uniform distribution.

    Args:
        y_pred_probs: array of prediction probabilities

    Returns:
        Mean wasserstein distance & std of wasserstein distance
    """
    n_classes = y_pred_probs.shape[1]
    uniform_dist = np.ones(n_classes) / n_classes
    wasserstein_scores = []
    for probs in y_pred_probs:
        # compute empirical CDF distance from uniform CDF
        sorted_probs = np.sort(probs)
        wasserstein = np.mean(np.abs(np.cumsum(sorted_probs) - np.cumsum(uniform_dist)))
        wasserstein_scores.append(wasserstein)
    return float(np.mean(wasserstein_scores)), float(np.std(wasserstein_scores))


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """Calculate comprehensive attribution metrics using individual functions.

    Args:
        y_true: array of true labels
        y_pred_probs: array of prediction probabilities

    Returns:
        All metrics with their values
    """
    metrics = {}

    # overall performance metrics
    for k in [1, 5]:
        metrics[f'accuracy@{k}'] = compute_accuracy_at_k(y_true, y_pred_probs, k)
        metrics[f'f1@{k}'] = compute_f1_at_k(y_true, y_pred_probs, k)

    # confidence metrics
    metrics['true_class_confidence'] = compute_true_class_confidence(y_true,
                                                                     y_pred_probs)

    # distribution metrics
    metrics['entropy'], metrics['entropy_std'] = compute_entropy(y_pred_probs)
    metrics['gini'], metrics['gini_std'] = compute_gini(y_pred_probs)
    metrics['tvd'], metrics['tvd_std'] = compute_tvd(y_pred_probs)

    # ranking metrics
    metrics['mrr'], metrics['mrr_std'] = compute_mrr(y_true, y_pred_probs)
    metrics['wasserstein'], metrics['wasserstein_std'] = compute_wasserstein(
        y_pred_probs)

    return metrics


def defense_effectiveness(pre_metrics: dict, post_metrics: dict) -> dict:
    """
    Calculate defense effectiveness metrics by comparing pre/post metrics and measuring
    progress toward ideal conditions.

    Args:
        pre_metrics: metrics before defense
        post_metrics: metrics after defense

    Returns:
        Effectiveness metrics including both absolute changes and progress toward ideals
    """
    effectiveness = {}
    n_classes = len(
        pre_metrics.get('true_class_probs', [1]))  # fallback to 1 if not available

    # Ideal conditions for each metric
    ideals = {
        'accuracy': 1 / n_classes,  # random guessing
        'entropy': 1.0,  # fully uniform (normalized)
        'gini': 0.0,  # perfectly equal distribution
        'tvd': 0.0,  # no deviation from uniform
        'mrr': 1 / n_classes,  # random ranking
        'wasserstein': 1.0,  # maximum redistribution
    }

    # 1. accuracy and F1 changes
    for k in [1, 5]:
        # absolute changes
        effectiveness[f'accuracy@{k}_abs_change'] = float(
            post_metrics[f'accuracy@{k}'] - pre_metrics[f'accuracy@{k}']
        )
        effectiveness[f'f1@{k}_abs_change'] = float(
            post_metrics[f'f1@{k}'] - pre_metrics[f'f1@{k}']
        )

        # progress toward random guessing (ideal)
        pre_acc_gap = abs(pre_metrics[f'accuracy@{k}'] - ideals['accuracy'])
        post_acc_gap = abs(post_metrics[f'accuracy@{k}'] - ideals['accuracy'])
        effectiveness[f'accuracy@{k}_ideal_progress'] = float(
            (pre_acc_gap - post_acc_gap) / pre_acc_gap if pre_acc_gap > 0 else 0.0
        )

    # 2. confidence changes
    effectiveness['confidence_abs_drop'] = float(
        pre_metrics['true_class_confidence'] - post_metrics['true_class_confidence']
    )

    # 3. distribution metrics - progress toward ideal conditions

    # Entropy (toward 1.0 - perfect uniformity)
    pre_entropy_gap = abs(pre_metrics['entropy'] - ideals['entropy'])
    post_entropy_gap = abs(post_metrics['entropy'] - ideals['entropy'])
    effectiveness['entropy_abs_increase'] = float(
        post_metrics['entropy'] - pre_metrics['entropy']
    )
    effectiveness['entropy_ideal_progress'] = float(
        (
                    pre_entropy_gap - post_entropy_gap) / pre_entropy_gap if pre_entropy_gap > 0 else 0.0
    )

    # Gini (toward 0.0 - perfect equality)
    pre_gini_gap = abs(pre_metrics['gini'] - ideals['gini'])
    post_gini_gap = abs(post_metrics['gini'] - ideals['gini'])
    effectiveness['gini_abs_reduction'] = float(
        pre_metrics['gini'] - post_metrics['gini']
    )
    effectiveness['gini_ideal_progress'] = float(
        (pre_gini_gap - post_gini_gap) / pre_gini_gap if pre_gini_gap > 0 else 0.0
    )

    # TVD (toward 0.0 - no deviation from uniform)
    pre_tvd_gap = abs(pre_metrics['tvd'] - ideals['tvd'])
    post_tvd_gap = abs(post_metrics['tvd'] - ideals['tvd'])
    effectiveness['tvd_abs_reduction'] = float(
        pre_metrics['tvd'] - post_metrics['tvd']
    )
    effectiveness['tvd_ideal_progress'] = float(
        (pre_tvd_gap - post_tvd_gap) / pre_tvd_gap if pre_tvd_gap > 0 else 0.0
    )

    # 4. ranking metrics
    # MRR (toward random ranking)
    pre_mrr_gap = abs(pre_metrics['mrr'] - ideals['mrr'])
    post_mrr_gap = abs(post_metrics['mrr'] - ideals['mrr'])
    effectiveness['mrr_abs_reduction'] = float(
        pre_metrics['mrr'] - post_metrics['mrr']
    )
    effectiveness['mrr_ideal_progress'] = float(
        (pre_mrr_gap - post_mrr_gap) / pre_mrr_gap if pre_mrr_gap > 0 else 0.0
    )

    # Wasserstein (toward max redistribution)
    pre_wass_gap = abs(pre_metrics['wasserstein'] - ideals['wasserstein'])
    post_wass_gap = abs(post_metrics['wasserstein'] - ideals['wasserstein'])
    effectiveness['wasserstein_abs_increase'] = float(
        post_metrics['wasserstein'] - pre_metrics['wasserstein']
    )
    effectiveness['wasserstein_ideal_progress'] = float(
        (pre_wass_gap - post_wass_gap) / pre_wass_gap if pre_wass_gap > 0 else 0.0
    )

    return effectiveness


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

    def _get_output_path(self, corpus: str, rq: str, model_name: str) -> Path:
        """Get standardized output path for both experiment and save results.

        Args:
            corpus: corpus name (rj/ebg/lcmc)
            rq: full research question identifier (e.g. rq1.1_basic_paraphrase)
            model_name: full model name

        Returns:
            Path: Standardized path for output files
        """
        rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"  # e.g., 'rq1'
        model_dir = model_name.split('/')[-1].lower()

        return self.output_dir / corpus / rq_main / rq / model_dir

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
        """Evaluate LLM-based defense against attribution models."""
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
        rq_base = rq.split('_')[0]  # e.g., 'rq1.1'
        rq_main = f"rq{rq_base.split('.')[0].lstrip('rq')}"  # e.g., 'rq1'
        model_dir = model_name.split('/')[-1].lower()

        output_base = self._get_output_path(corpus, rq, model_name)
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
                        "example_id": int(idx),  # convert from numpy int
                        "true_label": int(true_label),  # convert from numpy int
                        "orig_probs": orig_prob.tolist(),
                        "trans_probs": trans_prob.tolist(),
                        "original_rank": int(np.where(orig_ranks == true_label)[0][0]),
                        # convert numpy int
                        "transformed_rank": int(
                            np.where(trans_ranks == true_label)[0][0]),
                        # convert numpy int
                        "mrr_change": float(1 / (np.where(trans_ranks == true_label)[0][
                                                     0] + 1) -  # convert numpy float
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
                    'raw': {
                        'inputs': {
                            'original_texts': test_texts,
                            'transformed_texts': transformed_texts,
                            'true_labels': [int(x) for x in test_labels]
                            # convert numpy ints
                        },
                        'predictions': {
                            'original_probs': orig_preds.tolist(),
                            'transformed_probs': trans_preds.tolist()
                        }
                    },
                    'attribution': {
                        'original_metrics': original_metrics,
                        'transformed_metrics': transformed_metrics,
                        'effectiveness': effectiveness
                    },
                    'quality': quality_metrics
                }

                # save detailed per-seed results
                output_file = output_base / f"seed_{seed_id}.json"
                detailed_results = {
                    'metadata': {
                        'corpus': corpus,
                        'research_question': rq,
                        'model_name': model_name,
                        'seed': int(seed_id),  # convert if numpy int
                        'n_examples': int(len(test_texts)),
                        'n_classes': int(len(np.unique(test_labels))) # convert numpy int
                    },
                    'results': seed_results,
                    'example_metrics': example_metrics
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved detailed evaluation results to {output_file}")

            all_seed_results[seed_id] = seed_results

        return all_seed_results

    def _get_experiment_paths(self, corpus: str, rq: str, model_name: str) -> Path:
        """Get experiment directory based on RQ identifier and model name.

        Args:
            corpus: corpus name (rj/ebg/lcmc)
            rq: research question identifier (e.g. rq1.1_basic_paraphrase)
            model_name: full model name (e.g. google/gemma-2b-it)

        Returns:
            Path to experiment directory
        """
        # extract RQ numbers for parent directory
        rq_base = rq.split('_')[0]  # e.g., 'rq1.1'
        rq_main = f"rq{rq_base.split('.')[0].lstrip('rq')}"  # e.g., 'rq1'

        # model dir name: take last part of model path
        model_dir = model_name.split('/')[-1].lower()

        expected_path = (
                self.llm_outputs_dir / corpus / rq_main / rq / model_dir
        )
        logger.info(f"Looking for files in: {expected_path}")
        return expected_path

    def save_results(self, results: Dict, corpus: str, rq: str,
                     model_name: str) -> None:
        """Save evaluation results following the specified directory structure.

        Args:
            results: evaluation results dictionary
            corpus: corpus name (rj/ebg/lcmc)
            rq: research question identifier
            model_name: model name
        """
        # get output directory using common path construction
        save_dir = self._get_output_path(corpus, rq, model_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file = save_dir / "evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved final consolidated results to {output_file}")

        # log summary metrics with new measures
        logger.info(f"\nResults for {corpus}-{rq} using {model_name}:")
        for seed, seed_dict in results.items():
            logger.info(f"\n--- Seed: {seed} ---")
            for model_type, metrics in seed_dict.items():
                orig = metrics['attribution']['original_metrics']
                transformed = metrics['attribution']['transformed_metrics']
                effect = metrics['attribution']['effectiveness']

                logger.info(f"\n{model_type.upper()} Results:")

                # accuracy and F1
                for k in [1, 5]:
                    logger.info(
                        f"Accuracy@{k}: {orig[f'accuracy@{k}']:.4f} → {transformed[f'accuracy@{k}']:.4f}")
                    logger.info(
                        f"  Ideal Progress: {effect[f'accuracy@{k}_ideal_progress']:.4f}")
                    logger.info(
                        f"F1@{k}: {orig[f'f1@{k}']:.4f} → {transformed[f'f1@{k}']:.4f}")

                # distribution metrics
                logger.info("\nDistribution Metrics:")
                logger.info(
                    f"  Entropy: {orig['entropy']:.4f} → {transformed['entropy']:.4f}")
                logger.info(
                    f"    Progress to Uniform: {effect['entropy_ideal_progress']:.4f}")
                logger.info(f"  Gini: {orig['gini']:.4f} → {transformed['gini']:.4f}")
                logger.info(
                    f"    Progress to Equal: {effect['gini_ideal_progress']:.4f}")
                logger.info(f"  TVD: {orig['tvd']:.4f} → {transformed['tvd']:.4f}")
                logger.info(
                    f"    Progress to Uniform: {effect['tvd_ideal_progress']:.4f}")

                # ranking metrics
                logger.info("\nRanking Metrics:")
                logger.info(f"  MRR: {orig['mrr']:.4f} → {transformed['mrr']:.4f}")
                logger.info(
                    f"    Progress to Random: {effect['mrr_ideal_progress']:.4f}")
                logger.info(
                    f"  Wasserstein: {orig['wasserstein']:.4f} → {transformed['wasserstein']:.4f}")
                logger.info(
                    f"    Progress to Max: {effect['wasserstein_ideal_progress']:.4f}")

                # text quality
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

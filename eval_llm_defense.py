"""
Evaluate effectiveness of LLM-based defense against authorship attribution models.

This script implements an evaluation framework for assessing the effectiveness
of LLM-based defenses against authorship attribution attacks. For effectiveness,
only the following attribution measures are computed:
    - Accuracy@1 and Accuracy@5
    - True class confidence
    - Prediction entropy

The output for each experimental run (seed) is saved in a structured directory
under the output directory (default: defense_evaluation/).
The file structure is illustrated below:

defense_evaluation/
└── {corpus}/
    ├── rq{main}/
    │   ├── rq{sub}/
    │   │   └── {model_name}/
    │   │       ├── evaluation.json         # Consolidated results across seeds
    │   │       └── seed_{seed}.json        # Detailed per-seed results containing:
    │   │             ├── metadata          # Corpus, research question, model details, etc.
    │   │             ├── results             # Contains:
    │   │             │      attribution:      # Attribution-related data with three keys:
    │   │             │          pre:           # Aggregated metrics for original predictions:
    │   │             │                  - accuracy@1
    │   │             │                  - accuracy@5
    │   │             │                  - true_class_confidence
    │   │             │                  - entropy
    │   │             │          post:          # Aggregated metrics for transformed predictions:
    │   │             │                  - accuracy@1
    │   │             │                  - accuracy@5
    │   │             │                  - true_class_confidence
    │   │             │                  - entropy
    │   │             │          raw_predictions:  # Raw prediction probability arrays, stored as a dictionary with:
    │   │             │                  - original: raw predictions for original texts
    │   │             │                  - transformed: raw predictions for transformed texts
    │   │             └── quality:          # Text quality metrics (PINC, BLEU, METEOR, BERTScore, SBERT)
    │   │             └── example_metrics   # Example-level performance metrics (one-to-one mapping per test sample)

Usage:
    python eval_llm_defense.py --model "google/gemma-2-9b-it"  # for all research questions
    python eval_llm_defense.py --rq rq1.1_basic_paraphrase --model "google/gemma-2-9b-it"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

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


def compute_accuracy_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """Compute top-k accuracy."""
    n_samples = len(y_true)
    top_k_correct = 0
    for true_label, probs in zip(y_true, y_pred_probs):
        top_k_indices = np.argsort(probs)[-k:]
        if true_label in top_k_indices:
            top_k_correct += 1
    return float(top_k_correct / n_samples)


def compute_true_class_confidence(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Compute average confidence for the true class."""
    true_class_probs = np.array(
        [probs[true_label] for true_label, probs in zip(y_true, y_pred_probs)]
    )
    return float(np.mean(true_class_probs))


def compute_entropy(y_pred_probs: np.ndarray) -> float:
    """Compute mean entropy of predictions (no standard deviation)."""
    entropies = -np.sum(y_pred_probs * np.log2(y_pred_probs + 1e-10), axis=1)
    return float(np.mean(entropies))


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """Calculate the chosen attribution metrics."""
    metrics = {}
    for k in [1, 5]:
        metrics[f'accuracy@{k}'] = compute_accuracy_at_k(y_true, y_pred_probs, k)
    metrics['true_class_confidence'] = compute_true_class_confidence(y_true, y_pred_probs)
    metrics['entropy'] = compute_entropy(y_pred_probs)
    return metrics


class DefenseEvaluator:
    """Evaluates LLM defense effectiveness against attribution models."""

    def __init__(self, threat_models_dir: Path, llm_outputs_dir: Path, output_dir: Path):
        """
        Initialize evaluator with paths to required data.

        Args:
            threat_models_dir: Directory containing trained models.
            llm_outputs_dir: Directory containing LLM outputs.
            output_dir: Directory to save evaluation results.
        """
        self.threat_models_dir = Path(threat_models_dir)
        self.llm_outputs_dir = Path(llm_outputs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # map model types to predictor classes.
        self.predictor_classes = {
            'logreg': LogisticRegressionPredictor,
            'svm': SVMPredictor,
            'roberta': RobertaPredictor
        }

    def _get_output_path(self, corpus: str, rq: str, model_name: str) -> Path:
        """Determine standardized output path for saving results."""
        rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"
        model_dir = model_name.split('/')[-1].lower()
        return self.output_dir / corpus / rq_main / rq / model_dir

    def _load_predictor(self, corpus: str, model_type: str) -> object:
        """Load the predictor for a given corpus and model type."""
        model_dir = self.threat_models_dir / corpus / "no_protection" / model_type / "model"
        return self.predictor_classes[model_type](model_dir)

    def _get_experiment_paths(self, corpus: str, rq: str, model_name: str) -> Path:
        """Locate the experiment directory based on research question and model name."""
        rq_base = rq.split('_')[0]
        rq_main = f"rq{rq_base.split('.')[0].lstrip('rq')}"
        model_dir = model_name.split('/')[-1].lower()
        expected_path = self.llm_outputs_dir / corpus / rq_main / rq / model_dir
        logger.info(f"Looking for files in: {expected_path}")
        return expected_path

    def evaluate_experiment(self, corpus: str, rq: str, model_name: str) -> Dict[str, Dict]:
        """Evaluate the LLM-based defense against attribution models."""
        logger.info(f"Evaluating {corpus}-{rq} using {model_name}")
        _, _, test_texts, test_labels = load_corpus(corpus=corpus, task="no_protection")
        logger.info(f"Loaded {len(test_texts)} original test texts")

        # load and process transformations by seed
        transformations_by_seed = {}
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)

        for seed_file in exp_dir.glob("seed_*.json"):
            seed_id = int(seed_file.stem.split("_")[1])
            logger.info(f"Loading transformations from: {seed_file}")

            with open(seed_file) as f:
                seed_data = json.load(f)
                transformed_texts = []
                if "transformations" in seed_data:
                    transformed_texts = [t["transformed"] for t in seed_data["transformations"]]
                else:
                    for entry in seed_data:
                        if "transformed" in entry:
                            transformed_texts.append(entry["transformed"])

                if len(transformed_texts) == len(test_texts):
                    transformations_by_seed[seed_id] = transformed_texts
                else:
                    logger.warning(
                        f"Skipping seed {seed_id}: expected {len(test_texts)} transformations, got {len(transformed_texts)}"
                    )

        logger.info(f"Loaded transformations for {len(transformations_by_seed)} seeds")
        output_base = self._get_output_path(corpus, rq, model_name)
        output_base.mkdir(parents=True, exist_ok=True)

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

                # create one-to-one mapping of pre and post metrics
                pre_attribution = {
                    "accuracy@1": original_metrics["accuracy@1"],
                    "accuracy@5": original_metrics["accuracy@5"],
                    "true_class_confidence": original_metrics["true_class_confidence"],
                    "entropy": original_metrics["entropy"]
                }
                post_attribution = {
                    "accuracy@1": transformed_metrics["accuracy@1"],
                    "accuracy@5": transformed_metrics["accuracy@5"],
                    "true_class_confidence": transformed_metrics["true_class_confidence"],
                    "entropy": transformed_metrics["entropy"]
                }

                # collect example-level metrics (raw predictions are one-to-one with test samples)
                for idx, (true_label, orig_prob, trans_prob) in enumerate(
                        zip(test_labels, orig_preds, trans_preds)
                ):
                    orig_ranks = np.argsort(orig_prob)[::-1]
                    trans_ranks = np.argsort(trans_prob)[::-1]
                    example_metrics.append({
                        "example_id": int(idx),
                        "true_label": int(true_label),
                        "orig_probs": orig_prob.tolist(),
                        "trans_probs": trans_prob.tolist(),
                        "original_rank": int(np.where(orig_ranks == true_label)[0][0]),
                        "transformed_rank": int(np.where(trans_ranks == true_label)[0][0]),
                        "mrr_change": float(1 / (np.where(trans_ranks == true_label)[0][0] + 1) -
                                              1 / (np.where(orig_ranks == true_label)[0][0] + 1))
                    })

                quality_metrics = evaluate_quality(
                    candidate_texts=transformed_texts,
                    reference_texts=test_texts,
                    metrics=['pinc', 'bleu', 'meteor', 'bertscore', 'sbert']
                )

                seed_results[model_type] = {
                    "attribution": {
                        "pre": pre_attribution,
                        "post": post_attribution,
                        "raw_predictions": {
                            "original": orig_preds.tolist(),
                            "transformed": trans_preds.tolist()
                        }
                    },
                    "quality": quality_metrics
                }

                output_file = output_base / f"seed_{seed_id}.json"
                detailed_results = {
                    'metadata': {
                        'corpus': corpus,
                        'research_question': rq,
                        'model_name': model_name,
                        'seed': int(seed_id),
                        'n_examples': int(len(test_texts)),
                        'n_classes': int(len(np.unique(test_labels)))
                    },
                    'results': seed_results,
                    'example_metrics': example_metrics
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved detailed evaluation results to {output_file}")

            all_seed_results[seed_id] = seed_results

        return all_seed_results

    def save_results(self, results: Dict, corpus: str, rq: str, model_name: str) -> None:
        """Save the consolidated evaluation results."""
        save_dir = self._get_output_path(corpus, rq, model_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file = save_dir / "evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved final consolidated results to {output_file}")

        logger.info(f"\nResults for {corpus}-{rq} using {model_name}:")
        for seed, seed_dict in results.items():
            logger.info(f"\n--- Seed: {seed} ---")
            for model_type, metrics in seed_dict.items():
                pre_attr = metrics['attribution']['pre']
                post_attr = metrics['attribution']['post']
                logger.info(f"\n{model_type.upper()} Attribution Metrics:")
                for key in ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'entropy']:
                    logger.info(f"{key}: {pre_attr[key]:.4f} → {post_attr[key]:.4f}")
                qual = metrics['quality']
                logger.info("\nText Quality:")
                logger.info(f"  BLEU: {qual['bleu']['bleu']:.4f}")
                logger.info(f"  METEOR: {qual['meteor']['meteor_avg']:.4f}")
                logger.info(f"  BERTScore: {qual['bertscore']['bertscore_f1_avg']:.4f}")
                logger.info(f"  SBERT: {qual['sbert']['sbert_similarity_avg']:.4f}")

    def main_loop(self, args):
        """Main loop iterating over corpora, research questions, and models."""
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
                        logger.error(f"Error evaluating {corpus}-{rq} with {model}: {str(e)}")
                        continue


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM defense effectiveness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--threat_models_dir',
        type=Path,
        default=Path('threat_models'),
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
        choices=CORPORA,
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
        threat_models_dir=args.threat_models_dir,
        llm_outputs_dir=args.llm_outputs,
        output_dir=args.output_dir
    )
    evaluator.main_loop(args)


if __name__ == "__main__":
    main()

"""
Calculate human performance metrics for various strategies (imitation, obfuscation, etc.)
across different corpora and threat models.

This script evaluates the effectiveness of manual intervention strategies by:
1. Loading the threat models (LogisticRegression, SVM, and RoBERTa) trained on baseline data
2. Making predictions on manually transformed test samples
3. Calculating performance metrics: accuracy@1, accuracy@5, true class confidence, entropy

Key differences between corpora:
- EBG corpus has the same 45 authors across all strategies (fixed cohort)
- RJ corpus has different cohorts in different strategies:
  * no_protection: 21 authors
  * imitation: 17 authors
  * obfuscation: 27 authors
  * special_english (simplification): 18 authors

This script handles these differences by:
1. For EBG: Direct evaluation using models trained on no_protection
2. For RJ: Label validation and filtering to ensure correct metrics calculation
3. For RoBERTa on RJ: Using strategy-specific models trained separately

Usage:
    python eval_manual_defense.py --corpus rj  # only on rj and using svm/logreg
    python eval_manual_defense.py --include_roberta  # evaluate all corpora with RoBERTa

Output:
    Printed metrics for each corpus, task, and threat model combination
"""

import argparse
import logging
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from utils import (CORPUS_TASK_MAP, LogisticRegressionPredictor, SVMPredictor,
                   calculate_metrics, load_corpus)

# Import RoBERTa evaluation function
try:
    from evaluate_roberta_models import evaluate_roberta
    ROBERTA_AVAILABLE = True
except ImportError:
    ROBERTA_AVAILABLE = False

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_human_performance(
        corpus: str,
        task: str,
        models_dir: str = "threat_models"
) -> Dict:
    """
    Calculate performance metrics for human transformations.

    Args:
        corpus: The corpus to evaluate ('rj' or 'ebg')
        task: The task/strategy to evaluate (e.g., 'imitation', 'obfuscation')
        models_dir: Directory containing saved models

    Returns:
        Dictionary containing metrics for both LogisticRegression and SVM models
    """
    # load data for the task being evaluated
    _, _, test_text, test_labels = load_corpus(corpus, task)

    # also load the no_protection data to ensure label encoding consistency
    train_text, train_labels, _, _ = load_corpus(corpus, "no_protection")

    # get base models trained on 'no_protection' data
    logreg_model_path = os.path.join(models_dir, corpus, "no_protection", "logreg", "model")
    svm_model_path = os.path.join(models_dir, corpus, "no_protection", "svm", "model")

    results = {"logreg": {}, "svm": {}}

    # evaluate with logistic regression model
    try:
        logreg = LogisticRegressionPredictor(logreg_model_path)
        logreg_probs = logreg.predict_proba(test_text)
        # ensure labels are within bounds (in case different tasks have different authors)
        if max(test_labels) >= logreg_probs.shape[1]:
            logger.warning(f"Label mismatch in {corpus}-{task} for LogisticRegression")
            # create a version of test_labels that only uses authors present in training
            valid_labels = np.array([l if l < logreg_probs.shape[1] else -1 for l in test_labels])
            # filter out invalid labels (-1)
            mask = valid_labels >= 0
            if np.any(mask):
                results["logreg"] = calculate_metrics(valid_labels[mask], logreg_probs[mask])
            else:
                logger.error(f"No valid labels for {corpus}-{task} with LogisticRegression")
                results["logreg"] = None
        else:
            results["logreg"] = calculate_metrics(test_labels, logreg_probs)
    except Exception as e:
        logger.error(f"Error evaluating LogisticRegression on {corpus}-{task}: {str(e)}")
        results["logreg"] = None

    # evaluate with SVM model
    try:
        svm = SVMPredictor(svm_model_path)
        svm_probs = svm.predict_proba(test_text)
        # ensure labels are within bounds (in case different tasks have different authors)
        if max(test_labels) >= svm_probs.shape[1]:
            logger.warning(f"Label mismatch in {corpus}-{task} for SVM")
            # create a version of test_labels that only uses authors present in training
            valid_labels = np.array([l if l < svm_probs.shape[1] else -1 for l in test_labels])
            # filter out invalid labels (-1)
            mask = valid_labels >= 0
            if np.any(mask):
                results["svm"] = calculate_metrics(valid_labels[mask], svm_probs[mask])
            else:
                logger.error(f"No valid labels for {corpus}-{task} with SVM")
                results["svm"] = None
        else:
            results["svm"] = calculate_metrics(test_labels, svm_probs)
    except Exception as e:
        logger.error(f"Error evaluating SVM on {corpus}-{task}: {str(e)}")
        results["svm"] = None

    return results


def calculate_baseline_performance(
        corpus: str,
        models_dir: str = "threat_models"
) -> Dict:
    """
    Calculate baseline performance metrics for 'no_protection' scenario.

    Args:
        corpus: The corpus to evaluate ('rj' or 'ebg')
        models_dir: Directory containing saved models

    Returns:
        Dictionary containing metrics for both LogisticRegression and SVM models
    """
    # load test data for no_protection
    _, _, test_text, test_labels = load_corpus(corpus, "no_protection")

    # get base models trained on 'no_protection' data
    logreg_model_path = os.path.join(models_dir, corpus, "no_protection", "logreg", "model")
    svm_model_path = os.path.join(models_dir, corpus, "no_protection", "svm", "model")

    results = {"logreg": {}, "svm": {}}

    # evaluate with logistic regression model
    try:
        logreg = LogisticRegressionPredictor(logreg_model_path)
        logreg_probs = logreg.predict_proba(test_text)
        results["logreg"] = calculate_metrics(test_labels, logreg_probs)
    except Exception as e:
        logger.error(f"Error evaluating baseline LogisticRegression on {corpus}: {str(e)}")
        results["logreg"] = None

    # evaluate with SVM model
    try:
        svm = SVMPredictor(svm_model_path)
        svm_probs = svm.predict_proba(test_text)
        results["svm"] = calculate_metrics(test_labels, svm_probs)
    except Exception as e:
        logger.error(f"Error evaluating baseline SVM on {corpus}: {str(e)}")
        results["svm"] = None

    return results


def load_roberta_evaluation(
        corpus: str,
        task: str,
        evaluation_dir: str = "roberta_evaluations"
) -> Dict:
    """
    Load pre-computed RoBERTa evaluation results.

    Args:
        corpus: Corpus name ('rj' or 'ebg')
        task: Task name ('no_protection', 'imitation', etc.)
        evaluation_dir: Directory containing RoBERTa evaluation results

    Returns:
        Dictionary with RoBERTa metrics or None if not found
    """
    try:
        eval_path = os.path.join(evaluation_dir, f"{corpus}_{task}.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            y_true = data['y_true']
            y_pred_probs = data['y_pred_probs']
            return calculate_metrics(y_true, y_pred_probs)
        else:
            logger.info(f"No saved evaluation found for {corpus}-{task}, trying direct evaluation")

            # If we have the evaluate_roberta module, use it for direct evaluation
            if ROBERTA_AVAILABLE:
                return evaluate_roberta(corpus, task)
            else:
                logger.warning(f"RoBERTa evaluation module not available")
                return None
    except Exception as e:
        logger.error(f"Error loading RoBERTa evaluation for {corpus}-{task}: {str(e)}")
        return None


def format_metrics(metrics: Dict) -> str:
    """format metrics for display, rounding to 3 decimal places"""
    if metrics is None:
        return "N/A"

    return (f"Acc@1: {metrics['accuracy@1']:.3f}, "
            f"Acc@5: {metrics['accuracy@5']:.3f}, "
            f"Conf: {metrics['true_class_confidence']:.3f}, "
            f"Entropy: {metrics['entropy']:.3f}")


def main(args):
    # determine what to evaluate
    if args.corpus:
        if args.corpus not in CORPUS_TASK_MAP:
            raise ValueError(f"Invalid corpus: {args.corpus}")
        corpora = [args.corpus]
    else:
        corpora = list(CORPUS_TASK_MAP.keys())

    # Check for CUDA availability if using RoBERTa
    if args.include_roberta:
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # print header
    print("\n" + "="*100)
    print(f"{'CORPUS':<10} {'TASK':<20} {'MODEL':<10} {'METRICS'}")
    print("="*100)

    # first calculate baselines for comparison
    for corpus in corpora:
        baseline_results = calculate_baseline_performance(
            corpus=corpus,
            models_dir=args.models_dir
        )

        print(f"{corpus:<10} {'no_protection':<20} {'LogReg':<10} {format_metrics(baseline_results['logreg'])}")
        print(f"{corpus:<10} {'no_protection':<20} {'SVM':<10} {format_metrics(baseline_results['svm'])}")

        # Add RoBERTa results if available
        if args.include_roberta:
            roberta_results = load_roberta_evaluation(corpus, "no_protection")
            print(f"{corpus:<10} {'no_protection':<20} {'RoBERTa':<10} {format_metrics(roberta_results)}")

        print("-"*100)

    # evaluate human performance for each strategy
    for corpus in corpora:
        # get all tasks except 'no_protection' which we already calculated
        tasks = [t for t in CORPUS_TASK_MAP[corpus] if t != 'no_protection']

        for task in tasks:
            try:
                metrics = calculate_human_performance(
                    corpus=corpus,
                    task=task,
                    models_dir=args.models_dir
                )

                print(f"{corpus:<10} {task:<20} {'LogReg':<10} {format_metrics(metrics['logreg'])}")
                print(f"{corpus:<10} {task:<20} {'SVM':<10} {format_metrics(metrics['svm'])}")

                # Add RoBERTa results if available
                if args.include_roberta:
                    roberta_results = load_roberta_evaluation(corpus, task)
                    print(f"{corpus:<10} {task:<20} {'RoBERTa':<10} {format_metrics(roberta_results)}")

                print("-"*100)

            except Exception as e:
                logger.error(f"Error processing {corpus}-{task}: {str(e)}")
                print(f"{corpus:<10} {task:<20} {'ERROR':<10} {str(e)}")
                print("-"*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate human performance metrics across different corpora and tasks"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=['rj', 'ebg'],
        help="Specific corpus to evaluate (default: all corpora)"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="threat_models",
        help="Base directory containing saved models"
    )
    parser.add_argument(
        "--include_roberta",
        action="store_true",
        help="Include RoBERTa model evaluation results if available"
    )

    args = parser.parse_args()
    main(args)

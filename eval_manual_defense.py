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
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from utils import (CORPUS_TASK_MAP, LogisticRegressionPredictor, SVMPredictor,
                   calculate_metrics, load_corpus)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobertaPredictor:
    """Utility class for making predictions with saved RoBERTa models"""

    def __init__(self, model_path: str, device: str = None):
        """Initialize predictor with a saved model"""
        self.model_path = Path(model_path)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model and tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)

        # Get the number of labels
        self.num_labels = self.model.config.num_labels

        logger.info(f"Loaded model from {model_path} with {self.num_labels} labels on {self.device}")

    def predict_proba(self, texts, batch_size=32):
        """
        Get probability predictions for texts.

        Args:
            texts: List of raw text strings to predict on
            batch_size: Batch size for prediction

        Returns:
            Numpy array of prediction probabilities (n_samples, n_classes)
        """
        all_probs = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        # Concatenate all batch results
        return np.vstack(all_probs)


def evaluate_roberta(corpus: str, task: str, models_dir: str = "threat_models"):
    """
    Directly evaluate a RoBERTa model on a specific task.

    Args:
        corpus: The corpus to evaluate ('rj' or 'ebg')
        task: The task/strategy to evaluate
        models_dir: Directory containing saved models

    Returns:
        Dictionary with metrics or None if model not found
    """
    # For RJ, each strategy has its own model
    if corpus == "rj":
        model_path = os.path.join(models_dir, corpus, task, "roberta", "model")
    else:
        # For EBG, there's a single model in no_protection
        model_path = os.path.join(models_dir, corpus, "no_protection", "roberta", "model")

    if not os.path.exists(model_path):
        logger.warning(f"No RoBERTa model found at {model_path}")
        return None

    try:
        # Load data for the task being evaluated
        from utils import load_corpus
        _, _, test_text, test_labels = load_corpus(corpus, task)

        # Load model and get predictions
        roberta = RobertaPredictor(model_path)
        probs = roberta.predict_proba(test_text)

        # Calculate metrics
        metrics = calculate_metrics(test_labels, probs)

        # Optionally save predictions for future use
        save_dir = Path("roberta_evaluations")
        save_dir.mkdir(exist_ok=True)

        np.savez(
            save_dir / f"{corpus}_{task}.npz",
            y_true=test_labels,
            y_pred_probs=probs
        )

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating RoBERTa on {corpus}-{task}: {str(e)}")
        return None


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


def load_roberta_evaluation(corpus: str, task: str, evaluation_dir: str = "roberta_evaluations") -> Dict:
    try:
        eval_path = os.path.join(evaluation_dir, f"{corpus}_{task}.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            y_true = data['y_true']
            y_pred_probs = data['y_pred_probs']
            return calculate_metrics(y_true, y_pred_probs)
        else:
            logger.info(f"No saved evaluation found for {corpus}-{task}, trying direct evaluation")
            # Use direct evaluation since all the code is now in this file
            return evaluate_roberta(corpus, task)
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

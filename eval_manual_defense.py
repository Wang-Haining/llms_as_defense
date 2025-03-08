#!/usr/bin/env python3
"""
Calculate human performance metrics for various strategies (imitation, obfuscation, etc.)
across different corpora and threat models.

This script evaluates the effectiveness of manual intervention strategies by:
1. Loading the threat models (LogisticRegression and SVM) trained on baseline data
2. Making predictions on manually transformed test samples
3. Calculating performance metrics: accuracy@1, accuracy@5, true class confidence, entropy

Usage:
    python calculate_human_performance.py

Output:
    Printed metrics for each corpus, task, and threat model combination

Note:
    This script does not save any files and is meant for analysis purposes.
"""

import argparse
import logging
import os
from typing import Dict

from utils import (CORPUS_TASK_MAP, LogisticRegressionPredictor, SVMPredictor,
                   calculate_metrics, load_corpus)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_human_performance(
        corpus: str,
        task: str,
        models_dir: str = "results",
        save_path: str = "baselines"
) -> Dict:
    """
    Calculate performance metrics for human transformations.

    Args:
        corpus: The corpus to evaluate ('rj' or 'ebg')
        task: The task/strategy to evaluate (e.g., 'imitation', 'obfuscation')
        models_dir: Directory containing saved models
        save_path: Subdirectory under models_dir where models are saved

    Returns:
        Dictionary containing metrics for both LogisticRegression and SVM models
    """
    # load test data for the specified task
    _, _, test_text, test_labels = load_corpus(corpus, task)

    # get base models trained on 'no_protection' data
    logreg_model_path = os.path.join(models_dir, save_path, corpus, "no_protection", "logreg", "model")
    svm_model_path = os.path.join(models_dir, save_path, corpus, "no_protection", "svm", "model")

    results = {"logreg": {}, "svm": {}}

    # evaluate with logistic regression model
    try:
        logreg = LogisticRegressionPredictor(logreg_model_path)
        logreg_probs = logreg.predict_proba(test_text)
        results["logreg"] = calculate_metrics(test_labels, logreg_probs)
    except Exception as e:
        logger.error(f"Error evaluating LogisticRegression on {corpus}-{task}: {str(e)}")
        results["logreg"] = None

    # evaluate with SVM model
    try:
        svm = SVMPredictor(svm_model_path)
        svm_probs = svm.predict_proba(test_text)
        results["svm"] = calculate_metrics(test_labels, svm_probs)
    except Exception as e:
        logger.error(f"Error evaluating SVM on {corpus}-{task}: {str(e)}")
        results["svm"] = None

    return results

def calculate_baseline_performance(
        corpus: str,
        models_dir: str = "results",
        save_path: str = "baselines"
) -> Dict:
    """
    Calculate baseline performance metrics for 'no_protection' scenario.

    Args:
        corpus: The corpus to evaluate ('rj' or 'ebg')
        models_dir: Directory containing saved models
        save_path: Subdirectory under models_dir where models are saved

    Returns:
        Dictionary containing metrics for both LogisticRegression and SVM models
    """
    # load test data for no_protection
    _, _, test_text, test_labels = load_corpus(corpus, "no_protection")

    # get base models trained on 'no_protection' data
    logreg_model_path = os.path.join(models_dir, save_path, corpus, "no_protection", "logreg", "model")
    svm_model_path = os.path.join(models_dir, save_path, corpus, "no_protection", "svm", "model")

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
    corpora = list(CORPUS_TASK_MAP.keys()) if args.all else ["rj"]

    # print header
    print("\n" + "="*100)
    print(f"{'CORPUS':<10} {'TASK':<20} {'MODEL':<10} {'METRICS'}")
    print("="*100)

    # first calculate baselines for comparison
    for corpus in corpora:
        baseline_results = calculate_baseline_performance(
            corpus=corpus,
            models_dir=args.models_dir,
            save_path=args.save_path
        )

        print(f"{corpus:<10} {'no_protection':<20} {'LogReg':<10} {format_metrics(baseline_results['logreg'])}")
        print(f"{corpus:<10} {'no_protection':<20} {'SVM':<10} {format_metrics(baseline_results['svm'])}")
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
                    models_dir=args.models_dir,
                    save_path=args.save_path
                )

                print(f"{corpus:<10} {task:<20} {'LogReg':<10} {format_metrics(metrics['logreg'])}")
                print(f"{corpus:<10} {task:<20} {'SVM':<10} {format_metrics(metrics['svm'])}")
                print("-"*100)

            except Exception as e:
                logger.error(f"Error processing {corpus}-{task}: {str(e)}")
                print(f"{corpus:<10} {task:<20} {'ERROR':<10} {str(e)}")
                print("-"*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate human performance metrics across different corpora and tasks"
    )

    # path arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="baselines",
        help="Subdirectory under results for saved models"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="results",
        help="Base directory containing saved models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all corpora (default: only RJ corpus)"
    )

    args = parser.parse_args()
    main(args)
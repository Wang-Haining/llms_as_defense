"""
This module implements baseline authorship attribution models and evaluates their
performance on different corpora.

The implementation includes three attribution models:
1. Logistic Regression with Koppel's 512 function word features
2. SVM with Writeprints-static features
3. RoBERTa with 10-fold cross-validation

Directory Structure:
results/baselines/
├── {corpus}/                    # RJ, EBG, or LCMC
│   ├── {task}/                 # e.g., control, imitation, obfuscation
│   │   ├── logreg/
│   │   │   └── predictions.npz # contains y_true, y_pred_probs, feature_type
│   │   ├── svm/
│   │   │   └── predictions.npz # contains y_true, y_pred_probs, feature_type
│   │   └── roberta/
│   │       ├── fold_0/
│   │       │   ├── ckpts/     # contains saved model weights
│   │       │   └── predictions.npz # contains fold-specific predictions
│   │       ├── fold_1/
│   │       │   ├── ckpts/
│   │       │   └── predictions.npz
│   │       ...
│   │       ├── fold_9/
│   │       │   ├── ckpts/
│   │       │   └── predictions.npz
│   │       └── ensemble_predictions.npz # contains y_pred_probs (mean_probs), std_probs, true_labels, feature_type
│   └── {another_task}/
└── {another_corpus}/

Usage:
    # For traditional models (logreg, svm)
    python eval.py --corpus RJ --task control --model logreg
    python eval.py --corpus EBG --task no_protection --model svm

    # For RoBERTa with 10-fold CV
    python eval.py --corpus RJ --task control --model roberta

Notes:
    - Roberta relevant scripts refer to roberta_cv.py
    - Traditional models (logreg, svm) provide point estimates
    - RoBERTa provides both point estimates (ensemble mean) and uncertainty (std)
- All predictions are saved as probability distributions over authors
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

from utils import (load_ebg, load_lcmc, load_rj, vectorize_koppel512,
                       vectorize_writeprints_static)

# setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/baselines"


def evaluate_logistic_regression(
    train_text: List[str],
    train_labels: List[int],
    test_text: List[str],
    test_labels: List[int],
) -> Dict:
    """Evaluate logistic regression with Koppel512 features"""
    X_train = vectorize_koppel512(train_text)
    X_test = vectorize_koppel512(test_text)

    model = Pipeline(
        [
            ("normalizer", Normalizer(norm="l1")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=5000, multi_class="multinomial"
                ),
            ),
        ]
    )

    model.fit(X_train, train_labels)
    y_pred_probs = model.predict_proba(X_test)

    return {
        "y_true": np.array(test_labels),
        "y_pred_probs": y_pred_probs,
        "feature_type": "koppel512",
    }


def evaluate_svm(
    train_text: List[str],
    train_labels: List[int],
    test_text: List[str],
    test_labels: List[int],
) -> Dict:
    """Evaluate SVM with Writeprints-static features"""
    X_train = vectorize_writeprints_static(train_text)
    X_test = vectorize_writeprints_static(test_text)

    model = Pipeline(
        [
            ("normalizer", Normalizer(norm="l1")),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=0.01,
                    kernel="poly",
                    degree=3,
                    gamma="scale",
                    coef0=100,
                    max_iter=-1,
                    probability=True,
                ),
            ),
        ]
    )

    model.fit(X_train, train_labels)
    y_pred_probs = model.predict_proba(X_test)

    return {
        "y_true": np.array(test_labels),
        "y_pred_probs": y_pred_probs,
        "feature_type": "writeprints_static",
    }


def evaluate_traditional_model(
    train_text: List[str],
    train_labels: List[int],
    test_text: List[str],
    test_labels: List[int],
    model_type: str,
) -> Dict:
    """Evaluate traditional ML models"""
    if model_type == "logreg":
        results = evaluate_logistic_regression(
            train_text, train_labels, test_text, test_labels
        )
    else:
        results = evaluate_svm(train_text, train_labels, test_text, test_labels)

    return results


def save_prediction_results(
    results: Dict, corpus: str, task: str, model: str, output_dir: Path
) -> None:
    """Save prediction results consistently"""
    # create model-specific directory
    model_dir = output_dir / corpus / task / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # save predictions
    np.savez(
        model_dir / "predictions.npz",
        y_true=results["y_true"],
        y_pred_probs=results["y_pred_probs"],
        feature_type=results.get("feature_type", "roberta"),
    )

    # save additional metrics if available
    if "metrics" in results:
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(results["metrics"], f)

    logger.info(f"Saved results to {model_dir}")


def main(args):
    # import RoBERTa-related modules only if needed
    if args.model == "roberta":
        import torch
        import wandb
        from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                                Trainer, TrainingArguments, EarlyStoppingCallback)
        from roberta_cv import RobertaCV, CommonDataset
    # load data
    loaders = {"rj": load_rj, "ebg": load_ebg, "lcmc": load_lcmc}
    train_text, train_labels, test_text, test_labels = loaders[args.corpus](args.task)

    # create results directory
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ["logreg", "svm"]:
        results = evaluate_traditional_model(
            train_text, train_labels, test_text, test_labels, args.model
        )
        save_prediction_results(results, args.corpus, args.task, args.model, output_dir)

    else:  # roberta
        roberta = RobertaCV()
        results = roberta.train_and_evaluate(
            train_text, train_labels, test_text, test_labels, args.corpus, args.task
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate authorship attribution models"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        choices=["rj", "ebg", "lcmc"],
        help="Corpus to evaluate",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task/condition to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["roberta", "logreg", "svm"],
        help="Model to evaluate",
    )

    args = parser.parse_args()
    main(args)

"""
Evaluate LogisticRegression and SVM models' performance as authorship attribution models
across corpora and tasks.

The implementation includes two attribution models:
1. Logistic Regression with Koppel's 512 function word features
2. SVM with Writeprints-static features

Directory Structure:
results/{save_path}/              # or other supplied subdirectory
├── {corpus}/                     # rj, ebg, or lcmc
│   ├── {task}/                  # e.g., no_protection, imitation, obfuscation
│   │   ├── logreg/
│   │   │   ├── predictions.npz  # contains y_true, y_pred_probs
│   │   │   └── model/
│   │   │       ├── model.pkl    # saved sklearn pipeline
│   │   │       └── metadata.json # contains model_type and n_labels
│   │   └── svm/
│   │       ├── predictions.npz
│   │       └── model/
│   │           ├── model.pkl
│   │           └── metadata.json
│   └── {another_task}/
└── {another_corpus}/

Usage:
    # run specific corpus and task
    python train_ml.py --corpus rj --task no_protection --model logreg

    # run all tasks for a specific corpus
    python train_ml.py --corpus rj --model svm

    # run everything (default)
    python train_ml.py --model logreg

Notes:
    - Both models provide point estimates as probability distributions over authors
    - LogisticRegression uses Koppel512 function word features
    - SVM uses Writeprints-static features
    - Models are saved as complete sklearn pipelines including preprocessing steps
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._sgd_fast import Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, Normalizer,
                                   StandardScaler)
from sklearn.svm import SVC

from utils import (load_corpus, vectorize_koppel512,
                   vectorize_writeprints_static)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# define scenarios
CORPUS_TASK_MAP = {
    'rj': ['no_protection', 'imitation', 'obfuscation', 'special_english'],
    'ebg': ['no_protection', 'imitation', 'obfuscation'],
    'lcmc': ['no_protection']
}


class LogisticRegressionPredictor:
    """Utility class for making predictions with saved LogisticRegression models"""

    def __init__(self, model_path: str):
        """Initialize predictor with a saved model"""
        self.model_path = Path(model_path)

        # load metadata
        with open(self.model_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # load model
        with open(self.model_path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict_proba(self, texts):
        """Get probability predictions for texts"""
        # extract features
        features = vectorize_koppel512(texts)
        return self.model.predict_proba(features)


class SVMPredictor:
    """Utility class for making predictions with saved SVM models"""

    def __init__(self, model_path: str):
        """Initialize predictor with a saved model"""
        self.model_path = Path(model_path)

        # load metadata
        with open(self.model_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # load model
        with open(self.model_path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict_proba(self, texts):
        """Get probability predictions for texts"""
        # extract features
        features = vectorize_writeprints_static(texts)
        return self.model.predict_proba(features)


class MLModel:
    def __init__(
            self,
            output_dir: str,
            save_path: str,
            model_type: str = "logreg",
            corpus: str = None
    ):
        self.output_dir = Path(output_dir)
        self.save_path = save_path
        self.model_type = model_type
        self.corpus = corpus
        if corpus is not None:
            self._initialize_pipeline(corpus)

    def _initialize_pipeline(self, corpus: str):
        """Initialize the pipeline with appropriate parameters for the given corpus"""
        if self.model_type == "logreg":
            # use corpus-specific parameters based on grid search results
            if corpus == 'rj':
                logreg_params = {
                    'C': 100.0,
                    'solver': 'lbfgs',
                    'max_iter': 5000,
                }
            elif corpus == 'ebg':
                logreg_params = {
                    'C': 1.0,
                    'solver': 'lbfgs',
                    'max_iter': 5000,
                }
            elif corpus == 'lcmc':
                logreg_params = {
                    'C': 0.01,
                    'solver': 'lbfgs',
                    'max_iter': 5000,
                }
            else:
                raise ValueError(f"unrecognized corpus type: {corpus}")

            self.pipeline = Pipeline([
                ("features", FunctionTransformer(vectorize_koppel512)),
                ("normalizer", Normalizer(norm="l1")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**logreg_params))
            ])
        else:  # svm
            # use corpus-specific parameters based on grid search results
            if corpus in ['rj', 'ebg']:
                svm_params = {
                    'C': 0.0001,
                    'kernel': 'poly',
                    'degree': 3,
                    'gamma': 'scale',
                    'coef0': 100.0,
                    'max_iter': -1,
                    'probability': True
                }
            elif corpus == 'lcmc':
                svm_params = {
                    'C': 0.1,
                    'kernel': 'poly',
                    'degree': 2,
                    'gamma': 'scale',
                    'coef0': 10.0,
                    'max_iter': -1,
                    'probability': True
                }
            else:
                raise ValueError(f"unrecognized corpus type: {corpus}")

            self.pipeline = Pipeline([
                ("features", FunctionTransformer(vectorize_writeprints_static)),
                ("normalizer", Normalizer(norm="l1")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(**svm_params))
            ])

    def set_corpus(self, corpus: str):
        """Update pipeline with corpus-specific parameters"""
        if corpus not in ['rj', 'ebg', 'lcmc']:
            raise ValueError(f"unrecognized corpus type: {corpus}")
        self.corpus = corpus
        self._initialize_pipeline(corpus)

    def train_and_evaluate(
            self,
            train_text: List[str],
            train_labels: np.ndarray,
            test_text: List[str],
            test_labels: np.ndarray,
            corpus: str,
            task: str,
    ) -> Dict:
        """Train model and evaluate on test set"""
        exp_dir = self.output_dir / corpus / task / self.model_type
        exp_dir.mkdir(parents=True, exist_ok=True)

        # train model
        logger.info(f"Training {self.model_type} model...")
        self.pipeline.fit(train_text, train_labels)

        # get predictions
        test_probs = self.pipeline.predict_proba(test_text)

        # save model
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(self.pipeline, f)

        # save metadata
        metadata = {
            "model_type": self.model_type,
            "n_labels": len(np.unique(train_labels))
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # save predictions
        np.savez(
            exp_dir / "predictions.npz",
            y_true=test_labels,
            y_pred_probs=test_probs
        )

        # calculate accuracy for logging
        accuracy = np.mean(
            np.argmax(test_probs, axis=1) == test_labels
        )

        logger.info(
            f"Results for {corpus}-{task} using {self.model_type}: "
            f"accuracy={accuracy:.4f}"
        )

        return {
            "model_path": str(model_dir),
            "accuracy": accuracy
        }


def evaluate_corpus_task(
        model: MLModel,
        corpus: str,
        task: str,
        logger: logging.Logger
) -> None:
    """Evaluate ML model on a specific corpus and task."""
    logger.info(f"Evaluating {corpus}-{task}")

    # load data
    train_text, train_labels, test_text, test_labels = load_corpus(corpus, task)

    # train and evaluate
    results = model.train_and_evaluate(
        train_text=train_text,
        train_labels=train_labels,
        test_text=test_text,
        test_labels=test_labels,
        corpus=corpus,
        task=task
    )

    logger.info(
        f"Model saved at: {results['model_path']}, "
        f"accuracy: {results['accuracy']:.4f}"
    )


def main(args):
    # initialize model with default parameters
    model = MLModel(
        output_dir=args.output_dir,
        save_path=args.save_path,
        model_type=args.model
    )

    # determine what to evaluate
    if args.corpus:
        if args.corpus not in CORPUS_TASK_MAP:
            raise ValueError(f"Invalid corpus: {args.corpus}")
        corpora = [args.corpus]
    else:
        corpora = list(CORPUS_TASK_MAP.keys())

    # evaluate specified scenarios
    for corpus in corpora:
        logger.info(f"Processing corpus: {corpus}")
        # update model parameters for current corpus
        model.set_corpus(corpus)

        if args.task:
            if args.task not in CORPUS_TASK_MAP[corpus]:
                logger.warning(
                    f"Task {args.task} not available for corpus {corpus}, skipping")
                continue
            tasks = [args.task]
        else:
            tasks = CORPUS_TASK_MAP[corpus]

        for task in tasks:
            try:
                evaluate_corpus_task(model, corpus, task, logger)
            except Exception as e:
                logger.error(f"Error processing {corpus}-{task}: {str(e)}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate traditional ML models across different corpora and tasks"
    )
    # corpus and task selection
    parser.add_argument(
        "--corpus",
        type=str,
        choices=['rj', 'ebg', 'lcmc'],
        help="Specific corpus to evaluate (default: all)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Specific task to evaluate (default: all tasks for chosen corpus)"
    )

    # model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['logreg', 'svm'],
        help="ML model to evaluate"
    )

    # path arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="baselines",
        help="Subdirectory under results for saving"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory for results"
    )

    args = parser.parse_args()

    # validate task if provided
    if args.task and args.corpus and args.task not in CORPUS_TASK_MAP[args.corpus]:
        parser.error(
            f"Task '{args.task}' is not valid for corpus '{args.corpus}'. "
            f"Valid tasks are: {CORPUS_TASK_MAP[args.corpus]}"
        )

    main(args)

# ==================================================
# Logistic Regression
# ==================================================
# 2025-01-15 21:38:56,993 - INFO - Processing corpus: rj
# 2025-01-15 21:38:56,993 - INFO - Evaluating rj-no_protection
# 2025-01-15 21:38:57,308 - INFO - Training logreg model...
# 2025-01-15 21:39:09,275 - INFO - Results for rj-no_protection using logreg: accuracy=0.2857
# 2025-01-15 21:39:09,275 - INFO - Model saved at: results/rj/no_protection/logreg/model, accuracy: 0.2857
# 2025-01-15 21:39:09,276 - INFO - Evaluating rj-imitation
# 2025-01-15 21:39:09,602 - INFO - Training logreg model...
# 2025-01-15 21:39:17,540 - INFO - Results for rj-imitation using logreg: accuracy=0.1176
# 2025-01-15 21:39:17,540 - INFO - Model saved at: results/rj/imitation/logreg/model, accuracy: 0.1176
# 2025-01-15 21:39:17,540 - INFO - Evaluating rj-obfuscation
# 2025-01-15 21:39:18,126 - INFO - Training logreg model...
# 2025-01-15 21:39:34,574 - INFO - Results for rj-obfuscation using logreg: accuracy=0.0741
# 2025-01-15 21:39:34,575 - INFO - Model saved at: results/rj/obfuscation/logreg/model, accuracy: 0.0741
# 2025-01-15 21:39:34,575 - INFO - Evaluating rj-special_english
# 2025-01-15 21:39:34,923 - INFO - Training logreg model...
# 2025-01-15 21:39:43,552 - INFO - Results for rj-special_english using logreg: accuracy=0.1667
# 2025-01-15 21:39:43,552 - INFO - Model saved at: results/rj/special_english/logreg/model, accuracy: 0.1667
# 2025-01-15 21:39:43,552 - INFO - Processing corpus: ebg
# 2025-01-15 21:39:43,552 - INFO - Evaluating ebg-no_protection
# 2025-01-15 21:39:48,668 - INFO - Training logreg model...
# 2025-01-15 21:40:21,793 - INFO - Results for ebg-no_protection using logreg: accuracy=0.6667
# 2025-01-15 21:40:21,793 - INFO - Model saved at: results/ebg/no_protection/logreg/model, accuracy: 0.6667
# 2025-01-15 21:40:21,794 - INFO - Evaluating ebg-imitation
# 2025-01-15 21:40:26,566 - INFO - Training logreg model...
# 2025-01-15 21:41:00,678 - INFO - Results for ebg-imitation using logreg: accuracy=0.0889
# 2025-01-15 21:41:00,679 - INFO - Model saved at: results/ebg/imitation/logreg/model, accuracy: 0.0889
# 2025-01-15 21:41:00,679 - INFO - Evaluating ebg-obfuscation
# 2025-01-15 21:41:05,377 - INFO - Training logreg model...
# 2025-01-15 21:41:39,734 - INFO - Results for ebg-obfuscation using logreg: accuracy=0.0222
# 2025-01-15 21:41:39,734 - INFO - Model saved at: results/ebg/obfuscation/logreg/model, accuracy: 0.0222
# 2025-01-15 21:41:39,735 - INFO - Processing corpus: lcmc
# 2025-01-15 21:41:39,737 - INFO - Evaluating lcmc-no_protection
# Loading saved LCMC no_protection.
# 2025-01-15 21:41:39,743 - INFO - Training logreg model...
# 2025-01-15 21:41:56,229 - INFO - Results for lcmc-no_protection using logreg: accuracy=0.2381
# 2025-01-15 21:41:56,230 - INFO - Model saved at: results/lcmc/no_protection/logreg/model, accuracy: 0.2381

# ==================================================
# SVM
# ==================================================
# 2025-01-15 21:43:01,941 - INFO - Processing corpus: rj
# 2025-01-15 21:43:01,941 - INFO - Evaluating rj-no_protection
# 2025-01-15 21:43:02,123 - INFO - Training svm model...
# 2025-01-15 21:43:22,474 - INFO - Results for rj-no_protection using svm: accuracy=0.2857
# 2025-01-15 21:43:22,474 - INFO - Model saved at: results/rj/no_protection/svm/model, accuracy: 0.2857
# 2025-01-15 21:43:22,474 - INFO - Evaluating rj-imitation
# 2025-01-15 21:43:22,671 - INFO - Training svm model...
# 2025-01-15 21:43:40,356 - INFO - Results for rj-imitation using svm: accuracy=0.1176
# 2025-01-15 21:43:40,356 - INFO - Model saved at: results/rj/imitation/svm/model, accuracy: 0.1176
# 2025-01-15 21:43:40,356 - INFO - Evaluating rj-obfuscation
# 2025-01-15 21:43:40,651 - INFO - Training svm model...
# 2025-01-15 21:44:08,752 - INFO - Results for rj-obfuscation using svm: accuracy=0.0000
# 2025-01-15 21:44:08,752 - INFO - Model saved at: results/rj/obfuscation/svm/model, accuracy: 0.0000
# 2025-01-15 21:44:08,753 - INFO - Evaluating rj-special_english
# 2025-01-15 21:44:08,927 - INFO - Training svm model...
# 2025-01-15 21:44:27,530 - INFO - Results for rj-special_english using svm: accuracy=0.2778
# 2025-01-15 21:44:27,530 - INFO - Model saved at: results/rj/special_english/svm/model, accuracy: 0.2778
# 2025-01-15 21:44:27,530 - INFO - Processing corpus: ebg
# 2025-01-15 21:44:27,530 - INFO - Evaluating ebg-no_protection
# 2025-01-15 21:44:32,686 - INFO - Training svm model...
# 2025-01-15 21:45:20,419 - INFO - Results for ebg-no_protection using svm: accuracy=0.7778
# 2025-01-15 21:45:20,419 - INFO - Model saved at: results/ebg/no_protection/svm/model, accuracy: 0.7778
# 2025-01-15 21:45:20,419 - INFO - Evaluating ebg-imitation
# 2025-01-15 21:45:25,211 - INFO - Training svm model...
# 2025-01-15 21:46:15,686 - INFO - Results for ebg-imitation using svm: accuracy=0.0000
# 2025-01-15 21:46:15,686 - INFO - Model saved at: results/ebg/imitation/svm/model, accuracy: 0.0000
# 2025-01-15 21:46:15,686 - INFO - Evaluating ebg-obfuscation
# 2025-01-15 21:46:20,409 - INFO - Training svm model...
# 2025-01-15 21:47:09,869 - INFO - Results for ebg-obfuscation using svm: accuracy=0.1333
# 2025-01-15 21:47:09,869 - INFO - Model saved at: results/ebg/obfuscation/svm/model, accuracy: 0.1333
# 2025-01-15 21:47:09,869 - INFO - Processing corpus: lcmc
# 2025-01-15 21:47:09,869 - INFO - Evaluating lcmc-no_protection
# Loading saved LCMC no_protection.
# 2025-01-15 21:47:09,875 - INFO - Training svm model...
# 2025-01-15 21:47:31,809 - INFO - Results for lcmc-no_protection using svm: accuracy=0.1905
# 2025-01-15 21:47:31,809 - INFO - Model saved at: results/lcmc/no_protection/svm/model, accuracy: 0.1905

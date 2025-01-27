"""
Evaluate LogisticRegression and SVM models' performance as authorship attribution models
across corpora and tasks.

The implementation includes two attribution models:
1. Logistic Regression with Koppel's 512 function word features
2. SVM with Writeprints-static features

Directory Structure:
threat_models/
├── {corpus}/                    # rj and ebg
│   ├── {task}/                 # e.g., no_protection, imitation, etc
│   │   ├── logreg/
│   │   │   ├── model/          # training data differs by corpus:
│   │   │   │   ├── model.pkl   # - RJ: uses task-specific training data
│   │   │   │   └── metadata.json# - EBG: uses common no_protection training data
│   │   │   └── predictions.json # predictions on task's test set
│   │   └── svm/
│   │       ├── model/
│   │       └── predictions.json
│   └── {another_task}/
└── {another_corpus}/

Usage:
    python train_ml.py  --model svm
    python train_ml.py  --model logreg

Notes:
    - RJ: models trained with *task-specific* training data
    - EBG: models trained with common no_protection training data
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, Normalizer,
                                   StandardScaler)
from sklearn.svm import SVC

from utils import (load_corpus, vectorize_koppel512,
                   vectorize_writeprints_static, CORPORA, CORPUS_TASK_MAP)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            svm_params = {
                'C': 0.0001,
                'kernel': 'poly',
                'degree': 3,
                'gamma': 'scale',
                'coef0': 100.0,
                'max_iter': -1,
                'probability': True
            }

            self.pipeline = Pipeline([
                ("features", FunctionTransformer(vectorize_writeprints_static)),
                ("normalizer", Normalizer(norm="l1")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(**svm_params))
            ])

    def set_corpus(self, corpus: str):
        """Update pipeline with corpus-specific parameters"""
        if corpus not in CORPORA:
            raise ValueError(f"unrecognized corpus type: {corpus}")
        self.corpus = corpus
        self._initialize_pipeline(corpus)

    def train_and_evaluate(
            self,
            train_text: List[str],
            train_labels: np.ndarray,
            corpus: str,
            task: str,
    ) -> Dict:
        """
        Train model and evaluate. Handles corpus-specific training data:
        - RJ: Uses task-specific training data
        - EBG: Uses common no_protection training data
        Both use same directory structure and evaluation approach.

        Args:
            train_text: training texts (task-specific for RJ, common for EBG)
            train_labels: training labels
            corpus: corpus name (e.g., 'rj', 'ebg')
            task: task name (e.g., 'no_protection', 'imitation')

        Returns:
            Dict with model path and accuracy
        """
        # use consistent directory structure for both corpora
        exp_dir = self.output_dir / corpus / task / self.model_type
        exp_dir.mkdir(parents=True, exist_ok=True)

        # train model
        logger.info(f"Training {self.model_type} model for {corpus}-{task}...")
        self.pipeline.fit(train_text, train_labels)

        # save model and metadata
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(self.pipeline, f)

        metadata = {
            "model_type": self.model_type,
            "n_labels": int(len(np.unique(train_labels))),
            "corpus": corpus,
            "task": task,
            "training_data": "task_specific" if corpus == "rj" else "common_no_protection"
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # load test data and make predictions
        _, _, test_text, test_labels = load_corpus(corpus, task)
        test_probs = self.pipeline.predict_proba(test_text)

        # save predictions
        predictions = {
            "y_true": [int(x) for x in test_labels],
            "y_pred_probs": [[float(p) for p in row] for row in test_probs]
        }
        with open(exp_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # calculate accuracy
        accuracy = float(np.mean(np.argmax(test_probs, axis=1) == test_labels))
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
    """
    Evaluate ML model on corpus/task.

    Uses appropriate training data based on corpus:
    - RJ: Uses task-specific training data
    - EBG: Uses common no_protection training data for all tasks
    """
    logger.info(f"Evaluating {corpus}-{task}")

    # load appropriate training data
    if corpus == 'rj':
        # RJ: use task-specific training data
        train_text, train_labels, _, _ = load_corpus(corpus, task)
    else:  # EBG
        # EBG: use common training data from no_protection
        train_text, train_labels, _, _ = load_corpus(corpus, "no_protection")

    results = model.train_and_evaluate(
        train_text=train_text,
        train_labels=train_labels,
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
        choices=CORPORA,
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
        default="threat_models",
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
# 2025-01-27 10:08:35,462 - INFO - Processing corpus: rj
# 2025-01-27 10:08:35,462 - INFO - Evaluating rj-no_protection
# 2025-01-27 10:08:35,723 - INFO - Training logreg model for rj-no_protection...
# 2025-01-27 10:08:46,614 - INFO - Results for rj-no_protection using logreg: accuracy=0.2857
# 2025-01-27 10:08:46,614 - INFO - Model saved at: threat_models/rj/no_protection/logreg/model, accuracy: 0.2857
# 2025-01-27 10:08:46,614 - INFO - Evaluating rj-imitation
# 2025-01-27 10:08:46,835 - INFO - Training logreg model for rj-imitation...
# 2025-01-27 10:08:55,022 - INFO - Results for rj-imitation using logreg: accuracy=0.1176
# 2025-01-27 10:08:55,022 - INFO - Model saved at: threat_models/rj/imitation/logreg/model, accuracy: 0.1176
# 2025-01-27 10:08:55,022 - INFO - Evaluating rj-obfuscation
# 2025-01-27 10:08:55,429 - INFO - Training logreg model for rj-obfuscation...
# 2025-01-27 10:09:10,625 - INFO - Results for rj-obfuscation using logreg: accuracy=0.0741
# 2025-01-27 10:09:10,625 - INFO - Model saved at: threat_models/rj/obfuscation/logreg/model, accuracy: 0.0741
# 2025-01-27 10:09:10,625 - INFO - Evaluating rj-special_english
# 2025-01-27 10:09:10,941 - INFO - Training logreg model for rj-special_english...
# 2025-01-27 10:09:19,732 - INFO - Results for rj-special_english using logreg: accuracy=0.1667
# 2025-01-27 10:09:19,733 - INFO - Model saved at: threat_models/rj/special_english/logreg/model, accuracy: 0.1667
# 2025-01-27 10:09:19,733 - INFO - Processing corpus: ebg
# 2025-01-27 10:09:19,733 - INFO - Evaluating ebg-no_protection
# 2025-01-27 10:09:24,644 - INFO - Training logreg model for ebg-no_protection...
# 2025-01-27 10:09:58,859 - INFO - Results for ebg-no_protection using logreg: accuracy=0.6667
# 2025-01-27 10:09:58,859 - INFO - Model saved at: threat_models/ebg/no_protection/logreg/model, accuracy: 0.6667
# 2025-01-27 10:09:58,859 - INFO - Evaluating ebg-imitation
# 2025-01-27 10:10:03,314 - INFO - Training logreg model for ebg-imitation...
# 2025-01-27 10:10:37,435 - INFO - Results for ebg-imitation using logreg: accuracy=0.0889
# 2025-01-27 10:10:37,435 - INFO - Model saved at: threat_models/ebg/imitation/logreg/model, accuracy: 0.0889
# 2025-01-27 10:10:37,436 - INFO - Evaluating ebg-obfuscation
# 2025-01-27 10:10:41,858 - INFO - Training logreg model for ebg-obfuscation...
# 2025-01-27 10:11:18,218 - INFO - Results for ebg-obfuscation using logreg: accuracy=0.0444
# 2025-01-27 10:11:18,218 - INFO - Model saved at: threat_models/ebg/obfuscation/logreg/model, accuracy: 0.0444

# ==================================================
# SVM
# ==================================================
# 2025-01-27 10:11:32,850 - INFO - Processing corpus: rj
# 2025-01-27 10:11:32,850 - INFO - Evaluating rj-no_protection
# 2025-01-27 10:11:33,069 - INFO - Training svm model for rj-no_protection...
# 2025-01-27 10:11:58,140 - INFO - Results for rj-no_protection using svm: accuracy=0.2857
# 2025-01-27 10:11:58,140 - INFO - Model saved at: threat_models/rj/no_protection/svm/model, accuracy: 0.2857
# 2025-01-27 10:11:58,140 - INFO - Evaluating rj-imitation
# 2025-01-27 10:11:58,352 - INFO - Training svm model for rj-imitation...
# 2025-01-27 10:12:20,659 - INFO - Results for rj-imitation using svm: accuracy=0.1176
# 2025-01-27 10:12:20,659 - INFO - Model saved at: threat_models/rj/imitation/svm/model, accuracy: 0.1176
# 2025-01-27 10:12:20,659 - INFO - Evaluating rj-obfuscation
# 2025-01-27 10:12:20,981 - INFO - Training svm model for rj-obfuscation...
# 2025-01-27 10:12:56,698 - INFO - Results for rj-obfuscation using svm: accuracy=0.0370
# 2025-01-27 10:12:56,698 - INFO - Model saved at: threat_models/rj/obfuscation/svm/model, accuracy: 0.0370
# 2025-01-27 10:12:56,698 - INFO - Evaluating rj-special_english
# 2025-01-27 10:12:56,910 - INFO - Training svm model for rj-special_english...
# 2025-01-27 10:13:20,259 - INFO - Results for rj-special_english using svm: accuracy=0.3333
# 2025-01-27 10:13:20,259 - INFO - Model saved at: threat_models/rj/special_english/svm/model, accuracy: 0.3333
# 2025-01-27 10:13:20,259 - INFO - Processing corpus: ebg
# 2025-01-27 10:13:20,259 - INFO - Evaluating ebg-no_protection
# 2025-01-27 10:13:25,145 - INFO - Training svm model for ebg-no_protection...
# 2025-01-27 10:14:29,302 - INFO - Results for ebg-no_protection using svm: accuracy=0.7333
# 2025-01-27 10:14:29,303 - INFO - Model saved at: threat_models/ebg/no_protection/svm/model, accuracy: 0.7333
# 2025-01-27 10:14:29,303 - INFO - Evaluating ebg-imitation
# 2025-01-27 10:14:33,766 - INFO - Training svm model for ebg-imitation...
# 2025-01-27 10:15:38,055 - INFO - Results for ebg-imitation using svm: accuracy=0.0000
# 2025-01-27 10:15:38,055 - INFO - Model saved at: threat_models/ebg/imitation/svm/model, accuracy: 0.0000
# 2025-01-27 10:15:38,056 - INFO - Evaluating ebg-obfuscation
# 2025-01-27 10:15:42,497 - INFO - Training svm model for ebg-obfuscation...
# 2025-01-27 10:16:46,400 - INFO - Results for ebg-obfuscation using svm: accuracy=0.0889
# 2025-01-27 10:16:46,400 - INFO - Model saved at: threat_models/ebg/obfuscation/svm/model, accuracy: 0.0889

"""
Training script for LogisticRegression and SVM authorship attribution models focused on RJ manual strategies.

This script trains separate models for each RJ manual strategy (imitation, obfuscation, simplification)
using the same hyperparameters as the main train_ml.py script. This is necessary because
RJ has different cohorts in different scenarios, unlike EBG.

Usage:
    # Train both model types on all RJ manual strategies
    python train_ml_rj_strategies.py

    # Train a specific model type on all strategies
    python train_ml_rj_strategies.py --model logreg

    # Train a specific model on a specific strategy
    python train_ml_rj_strategies.py --model svm --strategy imitation
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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

# RJ manual strategies
RJ_STRATEGIES = ["imitation", "obfuscation", "special_english"]


class MLStrategyModel:
    def __init__(
            self,
            output_dir: str,
            model_type: str = "logreg"
    ):
        self.output_dir = Path(output_dir)
        self.model_type = model_type

        # initialize the pipeline with RJ-specific parameters based on grid search results
        if model_type == "logreg":
            logreg_params = {
                'C': 100.0,
                'solver': 'lbfgs',
                'max_iter': 5000,
            }
            self.pipeline = Pipeline([
                ("features", FunctionTransformer(vectorize_koppel512)),
                ("normalizer", Normalizer(norm="l1")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**logreg_params))
            ])
        else:  # svm
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

    def train_and_evaluate(self, strategy):
        """
        Train a model specifically for the given RJ strategy and evaluate.

        Args:
            strategy: The RJ strategy to train on ('imitation', 'obfuscation', 'special_english')

        Returns:
            Dict with model path and accuracy
        """
        corpus = "rj"
        exp_dir = self.output_dir / corpus / strategy / self.model_type
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Load the task-specific training data
        train_text, train_labels, test_text, test_labels = load_corpus(corpus, strategy)

        # Train the model
        logger.info(f"Training {self.model_type} model for {corpus}-{strategy}...")
        self.pipeline.fit(train_text, train_labels)

        # Save model and metadata
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(self.pipeline, f)

        metadata = {
            "model_type": self.model_type,
            "n_labels": int(len(np.unique(train_labels))),
            "corpus": corpus,
            "task": strategy,
            "training_data": "task_specific"
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Make predictions on the test set
        test_probs = self.pipeline.predict_proba(test_text)

        # Save predictions
        predictions = {
            "y_true": [int(x) for x in test_labels],
            "y_pred_probs": [[float(p) for p in row] for row in test_probs]
        }
        with open(exp_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Calculate accuracy
        accuracy = float(np.mean(np.argmax(test_probs, axis=1) == test_labels))
        logger.info(
            f"Results for {corpus}-{strategy} using {self.model_type}: "
            f"accuracy={accuracy:.4f}"
        )

        return {
            "model_path": str(model_dir),
            "accuracy": accuracy
        }


def main(args):
    # Determine which model types to train
    if args.model:
        model_types = [args.model]
    else:
        model_types = ["logreg", "svm"]

    # Determine which strategies to train on
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = RJ_STRATEGIES

    # Train each model type on each strategy
    for model_type in model_types:
        logger.info(f"Processing model type: {model_type}")
        model = MLStrategyModel(
            output_dir=args.output_dir,
            model_type=model_type
        )

        for strategy in strategies:
            try:
                results = model.train_and_evaluate(strategy)
                logger.info(
                    f"Model saved at: {results['model_path']}, "
                    f"accuracy: {results['accuracy']:.4f}"
                )
                print("#" * 80)
                print(f"Completed {model_type} training for RJ {strategy}")
                print(f"Model saved at: {results['model_path']}")
                print(f"Test accuracy: {results['accuracy']:.4f}")
                print("#" * 80)
            except Exception as e:
                logger.error(f"Error processing {model_type} for strategy {strategy}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models specifically for RJ manual strategies"
    )
    # strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        choices=RJ_STRATEGIES,
        help="Specific RJ strategy to train on (default: all)"
    )
    # model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "svm"],
        help="ML model to train (default: both)"
    )
    # path arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="threat_models",
        help="Base directory for saving models"
    )

    args = parser.parse_args()
    main(args)

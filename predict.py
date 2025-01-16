# predict.py
"""
Make predictions using saved authorship attribution models.

Usage:
    python predict.py --model_dir path/to/model/dir --texts path/to/texts.txt

The script will automatically detect the model type from metadata.json and use the
appropriate predictor class.

Models can be:
- LogisticRegression (uses Koppel512 features)
- SVM (uses Writeprints-static features)
- RoBERTa (uses raw text)
"""

__author__ = 'hw56@indiana.edu'
__license__ = 'OBSD'

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from utils import LogisticRegressionPredictor, SVMPredictor
from roberta import RobertaPredictor

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_texts(file_path: str) -> list:
    """Load texts from file, one per line"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_predictor(model_dir: str):
    """Load appropriate predictor based on model type in metadata"""
    model_dir = Path(model_dir)

    # Read metadata to determine model type
    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type",
                              "roberta")  # default to roberta if not specified

    # Initialize appropriate predictor
    if model_type == "logreg":
        return LogisticRegressionPredictor(model_dir)
    elif model_type == "svm":
        return SVMPredictor(model_dir)
    else:  # roberta
        return RobertaPredictor(model_dir)


def main(args):
    # Load texts
    logger.info(f"Loading texts from {args.texts}")
    texts = load_texts(args.texts)

    # Load appropriate predictor
    logger.info(f"Loading model from {args.model_dir}")
    predictor = get_predictor(args.model_dir)

    # Get predictions
    logger.info("Making predictions...")
    predictions = predictor.predict_proba(texts)

    # Save predictions
    output_path = Path(args.output_file or "predictions.npz")
    np.savez(
        output_path,
        texts=texts,
        predictions=predictions
    )
    logger.info(f"Saved predictions to {output_path}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions using saved authorship attribution models"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to directory containing saved model"
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=True,
        help="Path to file containing texts (one per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save predictions (default: predictions.npz)"
    )

    args = parser.parse_args()
    main(args)
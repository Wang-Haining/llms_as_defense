"""
Function to directly evaluate RoBERTa models for RJ strategies.

This module can be imported into eval_manual_defense.py to add direct RoBERTa evaluation
without requiring pre-saved evaluation files.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from utils import calculate_metrics

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

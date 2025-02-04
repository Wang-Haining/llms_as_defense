"""
Evaluate RoBERTa-based authorship attribution models.

This script loads the best saved RoBERTa models (saved as Hugging Face checkpoints)
for each specified corpus and computes the validation and test accuracy and loss.
It tokenizes input texts, performs inference on the validation and test sets,
and prints out the resulting metrics.

Usage Examples:
    # Evaluate models for a specific corpus (e.g., rj)
    python evaluate_roberta.py --corpus rj

    # Evaluate models for all corpora
    python evaluate_roberta.py
"""
import argparse
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import load_corpus, CORPORA

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_dir):
    """
    Load the saved model and tokenizer from the specified directory.

    Args:
        model_dir (str): Path to the saved model directory.

    Returns:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def create_val_split(train_text, train_labels):
    """
    Create a validation set by taking the first sample from each class.

    Args:
        train_text (list): List of training texts.
        train_labels (list or np.ndarray): Corresponding training labels.

    Returns:
        new_train_text, new_train_labels, val_text, val_labels
    """
    label_to_idx = {}
    for i, label in enumerate(train_labels):
        if label not in label_to_idx:
            label_to_idx[label] = i
    val_indices = list(label_to_idx.values())
    train_indices = [i for i in range(len(train_labels)) if i not in val_indices]

    val_text = [train_text[i] for i in val_indices]
    val_labels = np.array([train_labels[i] for i in val_indices])
    new_train_text = [train_text[i] for i in train_indices]
    new_train_labels = np.array([train_labels[i] for i in train_indices])
    return new_train_text, new_train_labels, val_text, val_labels


def tokenize_texts(tokenizer, texts, max_length=512):
    """
    Tokenize texts using the provided tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer.
        texts (list): List of text strings.
        max_length (int): Maximum token length.

    Returns:
        Dictionary with tokenized inputs.
    """
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoding


def evaluate_model(model, tokenizer, texts, true_labels, device):
    """
    Make predictions with the model and compute accuracy and cross-entropy loss.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        texts (list): List of text samples.
        true_labels (array-like): True labels.
        device: Device on which to run inference (cpu or cuda).

    Returns:
        accuracy, loss: Tuple containing the accuracy and cross-entropy loss.
    """
    model.eval()
    encoding = tokenize_texts(tokenizer, texts)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    pred_labels = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    loss = log_loss(true_labels, probabilities)
    return accuracy, loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpora = [args.corpus] if args.corpus else CORPORA

    for corpus in corpora:
        logger.info(f"Evaluating model for corpus: {corpus}")

        # expected directory structure: threat_models/{corpus}/no_protection/roberta/model
        model_dir = os.path.join("threat_models", corpus, "no_protection", "roberta", "model")
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist. Skipping {corpus}.")
            continue

        model, tokenizer = load_model(model_dir)
        model.to(device)
        logger.info(f"Loaded model from {model_dir}")

        # load data (using the no_protection dataset)
        train_text, train_labels, test_text, test_labels = load_corpus(corpus, "no_protection")

        # create validation split (first sample from each class)
        _, _, val_text, val_labels = create_val_split(train_text, train_labels)

        # evaluate on validation set
        val_accuracy, val_loss = evaluate_model(model, tokenizer, val_text, val_labels, device)
        # evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, tokenizer, test_text, test_labels, device)

        # Print out the results
        print(f"Corpus: {corpus}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
        logger.info(f"Corpus: {corpus}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa-based authorship attribution models")
    parser.add_argument("--corpus", type=str, choices=CORPORA, help="Specify a corpus to evaluate (default: all corpora)")
    args = parser.parse_args()
    main(args)

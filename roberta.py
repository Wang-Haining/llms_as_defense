"""Model and predictor classes for RoBERTa-based authorship attribution.

The module implements RoBERTa-based authorship attribution using the no_protection
dataset for training and evaluation. The trained models serve as threat models that try
to identify the author of a given text.

Directory Structure:
threat_models/
├── {corpus}/                    # rj or ebg
│   └── no_protection/          # uses no_protection data for training/testing
│       └── roberta/            # model type
│           ├── model/          # saved model, tokenizer and metadata
│           │   ├── config.json
│           │   ├── metadata.json  # includes corpus, task info
│           │   ├── pytorch_model.bin
│           │   └── tokenizer files...
│           └── predictions.json  # predictions on no_protection test set
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import wandb
from transformers import (
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

__author__ = 'hw56@indiana.edu'
__license__ = 'OBSD'

PROJECT_NAME = 'LLM as Defense'


import logging
logger = logging.getLogger(__name__)


class CommonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long).view(1)
        return item

    def __len__(self):
        return len(self.labels)


class RobertaBest:
    def __init__(
            self,
            model_name: str = "roberta-base",
            training_args: Dict = None
    ):
        """Initialize RoBERTa model and tokenizer."""
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.training_args = training_args or {}

    def get_training_args(self, output_dir: str, run_name: str) -> TrainingArguments:
        """Get training arguments with wandb integration."""
        return TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            learning_rate=self.training_args.get('learning_rate', 3e-5),
            per_device_train_batch_size=self.training_args.get('batch_size', 32),
            per_device_eval_batch_size=self.training_args.get('batch_size', 32),
            num_train_epochs=self.training_args.get('num_epochs', 100),
            warmup_steps=self.training_args.get('warmup_steps', 20),
            load_best_model_at_end=True,        # load best at end of training
            save_total_limit=1,              # only keep the best checkpoint
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            bf16=True,
            report_to="wandb",
            ddp_find_unused_parameters=False,
            no_cuda=self.device == "cpu",
        )

    def train_and_evaluate(
            self,
            train_text: List[str],
            train_labels: np.ndarray,
            val_text: List[str],
            val_labels: np.ndarray,
            test_text: List[str],
            test_labels: np.ndarray,
            corpus: str,
            seed: int,
    ) -> Dict[str, Any]:
        """Train model using no_protection training data and evaluate on test set.

        Args:
            train_text: training texts from no_protection dataset
            train_labels: training labels
            val_text: validation texts
            val_labels: validation labels
            test_text: test texts from no_protection dataset
            test_labels: test labels
            corpus: corpus name (e.g., 'rj', 'ebg')
            seed: random seed for reproducibility

        Returns:
            Dict with model path and metrics
        """
        # setup directory structure
        exp_dir = Path("threat_models") / corpus / "no_protection" / "roberta"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # set all random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # tokenize all datasets
        train_encodings = self.tokenizer(
            train_text, truncation=True, padding=True, max_length=512
        )
        val_encodings = self.tokenizer(
            val_text, truncation=True, padding=True, max_length=512
        )
        test_encodings = self.tokenizer(
            test_text, truncation=True, padding=True, max_length=512
        )

        # create datasets
        train_dataset = CommonDataset(train_encodings, train_labels)
        val_dataset = CommonDataset(val_encodings, val_labels)
        test_dataset = CommonDataset(test_encodings, test_labels)

        # initialize model
        model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(np.unique(train_labels))
        ).to(self.device)

        trainer = Trainer(
            model=model,
            args=self.get_training_args(
                output_dir=str(exp_dir / "checkpoints"),
                run_name=f"{corpus}_roberta_no_protection_seed_{seed}",
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
        )

        # train model
        trainer.train()

        # get validation and test metrics
        val_output = trainer.predict(val_dataset)
        test_output = trainer.predict(test_dataset)

        # compute validation metrics
        val_metrics = {
            "loss": float(val_output.metrics["test_loss"]),
            "accuracy": float(np.mean(
                np.argmax(val_output.predictions, axis=1) == val_labels
            )),
        }

        # get test probabilities and metrics
        test_probs = torch.nn.functional.softmax(
            torch.from_numpy(test_output.predictions), dim=-1
        ).numpy()
        test_metrics = {
            "loss": float(test_output.metrics["test_loss"]),
            "accuracy": float(np.mean(
                np.argmax(test_output.predictions, axis=1) == test_labels
            )),
        }

        # log final metrics to wandb and print
        final_metrics = {
            "final/val_loss": val_metrics['loss'],
            "final/val_accuracy": val_metrics['accuracy'],
            "final/test_loss": test_metrics['loss'],
            "final/test_accuracy": test_metrics['accuracy']
        }
        wandb.log(final_metrics)

        print(f"\nFinal metrics for {corpus}-no_protection:")
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test loss: {test_metrics['loss']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

        # save model and metadata
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        metadata = {
            "model_type": "roberta",
            "model_name": self.model_name,
            "n_labels": int(len(np.unique(train_labels))),
            "corpus": corpus,
            "task": "no_protection",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # save predictions
        predictions = {
            "y_true": [int(x) for x in test_labels],
            "y_pred_probs": [[float(p) for p in row] for row in test_probs]
        }
        with open(exp_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # save the final best model
        final_model = trainer.state.best_model_checkpoint
        if final_model:
            logger.info(f"Loading best model from {final_model}")
            model = RobertaForSequenceClassification.from_pretrained(
                final_model,
                num_labels=len(np.unique(train_labels))
            ).to(self.device)

        # cleanup checkpoints directory
        checkpoint_dir = exp_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)

        return {
            "model_path": str(model_dir),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }


class RobertaPredictor:
    """Utility class for making predictions with saved RoBERTa models."""

    def __init__(self, model_path: str):
        """Initialize predictor with a saved model."""
        self.model_path = Path(model_path)

        # load metadata
        with open(self.model_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # load model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.metadata["n_labels"]
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get probability predictions for texts."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()
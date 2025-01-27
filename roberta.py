"""Model and predictor classes for RoBERTa-based authorship attribution.

The module implements RoBERTa-based authorship attribution using fixed splits for
training, validation and testing. The trained models serve as threat models that try
to identify the author of a given text.

The module expects input data to already be split into training and test sets. It
automatically creates a validation set by taking the first sample from each author in
the training set.

Model outputs are saved under threat_models/{corpus}/roberta/ with the following structure:
    threat_models/
    └── {corpus}/
        └── roberta/
            ├── checkpoints/      # training checkpoints
            ├── model/           # saved model and tokenizer
            │   ├── config.json
            │   ├── metadata.json
            │   ├── pytorch_model.bin
            │   └── tokenizer_config.json
            └── predictions.npz  # test set predictions
"""

import json
from pathlib import Path
from typing import Dict, List

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
            load_best_model_at_end=True,
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
            save_total_limit=1,
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
    ) -> Dict:
        """Train model on fixed splits and save best model."""
        output_dir = Path("threat_models") / corpus / "roberta"
        output_dir.mkdir(parents=True, exist_ok=True)

        # start wandb run
        wandb.init(
            project=PROJECT_NAME,
            name=f"roberta_{corpus}",
            tags=[corpus, "roberta", self.model_name],
            config={
                "model_type": "roberta",
                "model_name": self.model_name,
                "corpus": corpus,
                "n_authors": len(np.unique(train_labels)),
                "learning_rate": self.training_args.get("learning_rate", 3e-5),
                "batch_size": self.training_args.get("batch_size", 32),
                "num_epochs": self.training_args.get("num_epochs", 100),
                "warmup_steps": self.training_args.get("warmup_steps", 20),
            },
        )

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
                output_dir=str(output_dir / "checkpoints"),
                run_name=f"{corpus}_roberta",
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
            "loss": val_output.metrics["test_loss"],
            "accuracy": np.mean(
                np.argmax(val_output.predictions, axis=1) == val_labels
            ),
        }

        # compute test metrics
        test_probs = torch.nn.functional.softmax(
            torch.from_numpy(test_output.predictions), dim=-1
        ).numpy()
        test_metrics = {
            "loss": test_output.metrics["test_loss"],
            "accuracy": np.mean(
                np.argmax(test_output.predictions, axis=1) == test_labels
            ),
        }

        # log final metrics to wandb and print
        final_metrics = {
            "final/val_loss": val_metrics['loss'],
            "final/val_accuracy": val_metrics['accuracy'],
            "final/test_loss": test_metrics['loss'],
            "final/test_accuracy": test_metrics['accuracy']
        }
        wandb.log(final_metrics)

        print(f"\nFinal metrics for {corpus}:")
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test loss: {test_metrics['loss']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

        # save model and metadata
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        metadata = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_name": self.model_name,
            "n_labels": len(np.unique(train_labels)),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # save predictions
        np.savez(
            output_dir / "predictions.npz",
            y_true=test_labels,
            y_pred_probs=test_probs,
        )

        # cleanup non-optimal checkpoints
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)

        wandb.finish()

        return {
            "model_path": str(model_dir),
            "val_metrics": val_metrics,
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

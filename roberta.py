import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold
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
        # ensure labels are returned as a tensor with proper shape
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long).view(1)
        return item

    def __len__(self):
        return len(self.labels)


class RobertaBest:
    def __init__(
            self,
            output_dir: str,
            save_path: str,
            seeds: Tuple[int, ...] = (42, 2025, 20250115),
            n_splits: int = 5,
            model_name: str = "roberta-base",
            training_args: Dict = None
    ):
        self.output_dir = Path(output_dir)
        self.save_path = save_path
        self.seeds = seeds
        self.n_splits = n_splits
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.training_args = training_args or {}

    def get_training_args(
            self,
            output_dir: str,
            run_name: str
    ) -> TrainingArguments:
        """Get training arguments with wandb integration"""
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
            test_text: List[str],
            test_labels: np.ndarray,
            corpus: str,
            task: str,
    ) -> Dict:
        """Train models with multiple seeds and CV, save best model."""
        exp_dir = self.output_dir / corpus / task / "roberta"
        exp_dir.mkdir(parents=True, exist_ok=True)

        best_overall_metrics = {"loss": float("inf"), "accuracy": 0}
        best_overall_model = None
        best_overall_probs = None
        best_seed = None
        best_fold = None

        experiment_group = f"{corpus}_{task}_roberta"

        for seed in self.seeds:
            # start a new run for each seed
            wandb.init(
                project=PROJECT_NAME,
                group=experiment_group,
                name=f"roberta_{corpus}_{task}_seed_{seed}",
                tags=[corpus, task, "roberta", self.model_name],
                config={
                    "model_type": "roberta",
                    "model_name": self.model_name,
                    "corpus": corpus,
                    "task": task,
                    "seed": seed,
                    "n_splits": self.n_splits,
                    "n_authors": len(np.unique(train_labels)),
                    "learning_rate": self.training_args.get("learning_rate", 3e-5),
                    "batch_size": self.training_args.get("batch_size", 32),
                    "num_epochs": self.training_args.get("num_epochs", 100),
                    "warmup_steps": self.training_args.get("warmup_steps", 20),
                },
            )

            # set all random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

            for fold, (train_idx, val_idx) in enumerate(kf.split(train_text)):
                # create a descriptive fold name
                fold_name = f"fold_{fold}"

                # prepare fold data
                fold_train_text = [train_text[i] for i in train_idx]
                fold_train_labels = train_labels[train_idx]
                fold_val_text = [train_text[i] for i in val_idx]
                fold_val_labels = train_labels[val_idx]

                # tokenize
                train_encodings = self.tokenizer(
                    fold_train_text, truncation=True, padding=True, max_length=512
                )
                val_encodings = self.tokenizer(
                    fold_val_text, truncation=True, padding=True, max_length=512
                )
                test_encodings = self.tokenizer(
                    test_text, truncation=True, padding=True, max_length=512
                )

                # create datasets
                train_dataset = CommonDataset(train_encodings, fold_train_labels)
                val_dataset = CommonDataset(val_encodings, fold_val_labels)
                test_dataset = CommonDataset(test_encodings, test_labels)

                # init model
                model = RobertaForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(np.unique(train_labels))
                ).to(self.device)

                trainer = Trainer(
                    model=model,
                    args=self.get_training_args(
                        output_dir=str(exp_dir / f"seed_{seed}_{fold_name}"),
                        run_name=f"{corpus}_{task}_seed_{seed}_{fold_name}",
                    ),
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
                )

                trainer.train()

                val_output = trainer.predict(val_dataset)
                test_output = trainer.predict(test_dataset)

                val_metrics = {
                    "loss": val_output.metrics["test_loss"],
                    "accuracy": np.mean(
                        np.argmax(val_output.predictions, axis=1) == fold_val_labels
                    ),
                }

                wandb.log(
                    {
                        "fold_name": fold_name,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "epoch": trainer.state.epoch,
                        "global_step": trainer.state.global_step,
                    }
                )

                # get test probabilities
                test_probs = torch.nn.functional.softmax(
                    torch.from_numpy(test_output.predictions), dim=-1
                ).numpy()

                # update best overall model if better
                if (
                        val_metrics["loss"] < best_overall_metrics["loss"]
                        and val_metrics["accuracy"] > best_overall_metrics["accuracy"]
                ):
                    best_overall_metrics = val_metrics
                    best_overall_model = model
                    best_overall_probs = test_probs
                    best_seed = seed
                    best_fold = fold

            # move on to the next seed
            wandb.finish()

        # save best overall model and metadata
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        best_overall_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        metadata = {
            "seed": best_seed,
            "fold": best_fold,
            "val_metrics": best_overall_metrics,
            "model_name": self.model_name,
            "n_labels": len(np.unique(train_labels)),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # save predictions
        np.savez(
            exp_dir / "predictions.npz",
            y_true=test_labels,
            y_pred_probs=best_overall_probs,
        )

        return {
            "model_path": str(model_dir),
            "best_seed": best_seed,
            "best_fold": best_fold,
            "best_val_metrics": best_overall_metrics,
        }


class RobertaPredictor:
    """Utility class for making predictions with saved RoBERTa models"""

    def __init__(self, model_path: str):
        """Initialize predictor with a saved model"""
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

    def predict_proba(self, texts):
        """Get probability predictions for texts"""
        # tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

import json
from pathlib import Path
from typing import Dict, List

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


class CommonDataset(torch.utils.data.Dataset):
    """Required by HuggingFace Trainer API"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class RobertaCV:
    def __init__(
        self,
        per_device_train_batch_size: int = 16,
        n_splits: int = 10,
        model_name: str = "roberta-base",
        base_dir: str = "results/baselines",
        seed: int = 42,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.n_splits = n_splits
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_training_args(
        self, fold: int, output_dir: str, run_name: str
    ) -> TrainingArguments:
        """Get training arguments with wandb integration"""
        return TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            seed=self.seed,
            do_eval=True,
            learning_rate=3e-5,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            logging_steps=self.per_device_train_batch_size,
            save_steps=self.per_device_train_batch_size,
            eval_steps=self.per_device_train_batch_size,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            bf16=True,
            overwrite_output_dir=True,
            num_train_epochs=200,
            save_total_limit=3,
            report_to="wandb",
        )

    def train_and_evaluate(
        self,
        train_text: List[str],
        train_labels: List[int],
        test_text: List[str],
        test_labels: List[int],
        corpus: str,
        task: str,
    ) -> Dict:
        """Run 10-fold CV with wandb logging"""

        # create experiment directory
        exp_dir = self.base_dir / corpus / task / "roberta"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # initialize wandb for ensemble
        wandb.init(
            project="LLM as Defense",
            group=f"{corpus}_{task}",
            name=f"{corpus}_{task}_roberta",
            config={
                "model": "roberta-base",
                "n_splits": self.n_splits,
                "seed": self.seed,
                "n_authors": len(set(train_labels)),
            },
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        results = {
            "fold_predictions": [],
            "test_predictions": [],
            "model_paths": [],
            "fold_metrics": [],
        }

        # run CV
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_text)):
            fold_dir = exp_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            # initialize fold-specific wandb run
            run_name = f"{corpus}_{task}_roberta_fold_{fold}"
            wandb.init(
                project="LLM as Defense",
                group=f"{corpus}_{task}",
                name=run_name,
                config={"fold": fold},
                reinit=True,
            )

            # prepare fold data
            fold_train_text = [train_text[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_text = [train_text[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]

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

            # initialize model
            model = RobertaForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(set(train_labels))
            ).to(self.device)

            # training args with wandb run name
            training_args = self.get_training_args(
                fold=fold, output_dir=str(fold_dir / "ckpts"), run_name=run_name
            )

            # initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],
            )

            # train
            trainer.train()

            # save best model
            ckpt_path = fold_dir / "ckpts"
            trainer.save_model(ckpt_path)
            results["model_paths"].append(str(ckpt_path))

            # get predictions
            with torch.no_grad():
                val_output = trainer.predict(val_dataset)
                val_probs = torch.nn.functional.softmax(
                    torch.from_numpy(val_output.predictions), dim=-1
                ).numpy()

                test_output = trainer.predict(test_dataset)
                test_probs = torch.nn.functional.softmax(
                    torch.from_numpy(test_output.predictions), dim=-1
                ).numpy()

            # log metrics
            fold_metrics = {
                "val_loss": val_output.metrics["test_loss"],
                "val_accuracy": val_output.metrics["test_accuracy"],
            }
            wandb.log({**fold_metrics, "fold": fold})

            # save fold results
            fold_result = {
                "fold": fold,
                "val_indices": val_idx.tolist(),
                "val_true": fold_val_labels,
                "val_pred_probs": val_probs,
                "test_true": test_labels,
                "test_pred_probs": test_probs,
                "metrics": fold_metrics,
            }

            np.savez(
                fold_dir / "predictions.npz",
                **{k: v for k, v in fold_result.items() if k != "metrics"},
            )
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(fold_metrics, f)

            # update results
            results["fold_predictions"].append(fold_result)
            results["test_predictions"].append(test_probs)
            results["fold_metrics"].append(fold_metrics)

            wandb.finish()

            # cleanup
            del model, trainer
            torch.cuda.empty_cache()

        # calculate ensemble predictions
        test_predictions_array = np.array(results["test_predictions"])
        test_ensemble_probs = np.mean(test_predictions_array, axis=0)
        test_ensemble_std = np.std(test_predictions_array, axis=0)

        # save ensemble results
        np.savez(
            exp_dir / "ensemble_predictions.npz",
            y_pred_probs=test_ensemble_probs,  # mean_probs
            std_probs=test_ensemble_std,
            true_labels=test_labels,
            feature_type="roberta",
        )

        # calculate ensemble metrics
        ensemble_accuracy = np.mean(
            [m["val_accuracy"] for m in results["fold_metrics"]]
        )
        ensemble_std = np.std([m["val_accuracy"] for m in results["fold_metrics"]])

        # log final ensemble metrics
        wandb.log(
            {
                "ensemble_accuracy": ensemble_accuracy,
                "ensemble_accuracy_std": ensemble_std,
                "completed_folds": self.n_splits,
            }
        )

        wandb.finish()

        return {
            "experiment_dir": str(exp_dir),
            "fold_metrics": results["fold_metrics"],
            "model_paths": results["model_paths"],
            "ensemble_metrics": {
                "mean_accuracy": ensemble_accuracy,
                "std_accuracy": ensemble_std,
            },
        }

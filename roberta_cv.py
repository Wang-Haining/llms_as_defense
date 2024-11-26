import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)


class CommonDataset(torch.utils.data.Dataset):
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
            output_dir: str,
            per_device_train_batch_size: int = 64,
            n_splits: int = 10,
            model_name: str = "FacebookAI/roberta-base",
            seed: int = 42,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.n_splits = n_splits
        self.model_name = model_name
        self.model = model_name.split('/')[-1].lower()
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # initialize label encoder for consistent label mapping
        self.label_encoder = LabelEncoder()

    def get_training_args(
            self, output_dir: str, run_name: str, fold: int
    ) -> TrainingArguments:
        """Get training arguments with wandb integration"""
        # avoid collision
        fold_seed = (self.seed * 31337 + fold) % (2 ** 32)

        return TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            seed=fold_seed,
            do_eval=True,
            learning_rate=3e-5,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_train_batch_size,
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
            save_total_limit=1,
            report_to="wandb",
        )

    def save_fold_predictions(
            self,
            fold_dir: Path,
            fold_result: Dict,
    ) -> None:
        """Save fold predictions matching traditional ML format"""
        np.savez(
            fold_dir / "predictions.npz",
            y_true=fold_result["test_true"],
            y_pred_probs=fold_result["test_pred_probs"],
            feature_type=self.model,
        )

        # save metrics separately
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result["metrics"], f)

    def train_and_evaluate(
            self,
            train_text: List[str],
            train_labels: List[int],
            test_text: List[str],
            test_labels: List[int],
            corpus: str,
            task: str,
    ) -> Dict:
        """Run 10-fold CV with wandb logging and consistent result saving"""
        # fit label encoder on all labels
        all_labels = list(train_labels) + list(test_labels)
        self.label_encoder.fit(all_labels)

        # transform all labels to numerical indices
        train_labels_num = self.label_encoder.transform(train_labels)
        test_labels_num = self.label_encoder.transform(test_labels)

        # create experiment directory
        exp_dir = self.output_dir / corpus / task / self.model
        exp_dir.mkdir(parents=True, exist_ok=True)

        # initialize wandb for ensemble
        wandb.init(
            project="LLM as Defense",
            group=f"{corpus}_{task}",
            name=f"{corpus}_{task}_{self.model}",
            config={
                "model": self.model,
                "n_splits": self.n_splits,
                "seed": self.seed,
                "n_authors": len(set(train_labels)),
                "device": self.device,
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
            run_name = f"{corpus}_{task}_{self.model}_fold_{fold}"
            wandb.init(
                project="LLM as Defense",
                group=f"{corpus}_{task}",
                name=run_name,
                config={"fold": fold},
                reinit=True,
            )

            # prepare fold data with numerical labels
            fold_train_text = [train_text[i] for i in train_idx]
            fold_train_labels = train_labels_num[train_idx]
            fold_val_text = [train_text[i] for i in val_idx]
            fold_val_labels = train_labels_num[val_idx]

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
            test_dataset = CommonDataset(test_encodings, test_labels_num)

            # initialize model
            model = RobertaForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(set(train_labels))
            ).to(self.device)

            # training args with wandb run name and fold-specific seed
            training_args = self.get_training_args(
                output_dir=str(fold_dir / "ckpts"),
                run_name=run_name,
                fold=fold
            )

            # initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            )

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
                val_preds = np.argmax(val_probs, axis=1)
                val_accuracy = np.mean(val_preds == fold_val_labels)

                test_output = trainer.predict(test_dataset)
                test_probs = torch.nn.functional.softmax(
                    torch.from_numpy(test_output.predictions), dim=-1
                ).numpy()
                test_preds = np.argmax(test_probs, axis=1)
                test_accuracy = np.mean(test_preds == test_labels_num)

            # log metrics
            fold_metrics = {
                "val_loss": val_output.metrics["test_loss"],
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
            }
            wandb.log({**fold_metrics, "fold": fold})

            # prepare fold results (only test predictions like traditional ML)
            fold_result = {
                "fold": fold,
                "test_true": test_labels,
                "test_pred_probs": test_probs,
                "metrics": fold_metrics,
            }

            # save fold predictions
            self.save_fold_predictions(fold_dir, fold_result)

            # update results
            results["fold_predictions"].append(fold_result)
            results["test_predictions"].append(test_probs)
            results["fold_metrics"].append(fold_metrics)

            wandb.finish()

            # tear down
            del model, trainer
            torch.cuda.empty_cache()

        # calculate and save ensemble predictions
        test_predictions_array = np.array(results["test_predictions"])
        test_ensemble_probs = np.mean(test_predictions_array, axis=0)
        test_ensemble_std = np.std(test_predictions_array, axis=0)

        # save ensemble results matching traditional ML format
        np.savez(
            exp_dir / "ensemble_predictions.npz",
            y_true=test_labels,
            y_pred_probs=test_ensemble_probs,
            std_probs=test_ensemble_std,
            feature_type=self.model
        )

        # calculate ensemble metrics
        ensemble_val_accuracy = np.mean(
            [m["val_accuracy"] for m in results["fold_metrics"]]
        )
        ensemble_val_std = np.std([m["val_accuracy"] for m in results["fold_metrics"]])

        ensemble_test_accuracy = np.mean(
            [m["test_accuracy"] for m in results["fold_metrics"]]
        )
        ensemble_test_std = np.std(
            [m["test_accuracy"] for m in results["fold_metrics"]])

        # log final ensemble metrics
        wandb.log(
            {
                "ensemble_val_accuracy": ensemble_val_accuracy,
                "ensemble_val_accuracy_std": ensemble_val_std,
                "ensemble_test_accuracy": ensemble_test_accuracy,
                "ensemble_test_accuracy_std": ensemble_test_std,
                "completed_folds": self.n_splits,
            }
        )

        wandb.finish()

        return {
            "experiment_dir": str(exp_dir),
            "fold_metrics": results["fold_metrics"],
            "model_paths": results["model_paths"],
            "ensemble_metrics": {
                "mean_val_accuracy": ensemble_val_accuracy,
                "std_val_accuracy": ensemble_val_std,
                "mean_test_accuracy": ensemble_test_accuracy,
                "std_test_accuracy": ensemble_test_std,
            },
        }

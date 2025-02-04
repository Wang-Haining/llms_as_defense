"""
Training script for RoBERTa-based authorship attribution models.

This script trains RoBERTa models for authorship attribution on different text corpora.
It trains models with multiple seeds on each corpus using the 'no_protection' dataset.
A validation set is automatically created by taking the first sample from each author
in the training set.

The script supports training on:
- RJ (Riddell-Juola): Cross-topic authorship attribution
- EBG (Extended Brennan-Greenstadt): Topic-overlap authorship attribution

Usage:
    # Train on a specific corpus
    python train_roberta.py --corpus rj --learning_rate 3e-5 --batch_size 32 --num_seeds 3

    # Train on all corpora with default parameters
    python train_roberta.py
"""

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb
from utils import load_corpus, CORPORA, FIXED_SEEDS
from roberta import RobertaBest

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "LLM as Defense"


def create_val_split(train_text, train_labels):
    """Create validation set using first sample from each class."""
    # group samples by label
    label_to_idx = defaultdict(list)
    for i, label in enumerate(train_labels):
        label_to_idx[label].append(i)

    # get validation indices (first sample of each class)
    val_indices = [indices[0] for indices in label_to_idx.values()]
    train_indices = [i for i in range(len(train_labels)) if i not in val_indices]

    # split data
    val_text = [train_text[i] for i in val_indices]
    val_labels = train_labels[val_indices]
    new_train_text = [train_text[i] for i in train_indices]
    new_train_labels = train_labels[train_indices]

    return new_train_text, new_train_labels, val_text, val_labels


def train_corpus(
        model: RobertaBest,
        corpus: str,
        num_seeds: int,
        training_args: dict,
        logger: logging.Logger
) -> None:
    """
    Train RoBERTa with multiple seeds on a specific corpus and reserve only the best checkpoint.

    After all seeds have been run, the checkpoint with the lowest validation loss is copied to
    the canonical folder: threat_models/{corpus}/no_protection/roberta/model.
    """
    logger.info(f"Training on corpus: {corpus}")

    # load data (using no_protection for both training and testing)
    train_text, train_labels, test_text, test_labels = load_corpus(
        corpus, "no_protection"
    )

    # create validation split
    train_text, train_labels, val_text, val_labels = create_val_split(
        train_text, train_labels
    )

    # track results across seeds; each entry is a tuple: (seed, results_dict)
    results_list = []

    # train with multiple seeds
    seeds = FIXED_SEEDS[:num_seeds]
    for seed in seeds:
        # To avoid overwriting checkpoints, append seed to the corpus name for the run.
        corpus_for_run = f"{corpus}_seed_{seed}"

        # initialize wandb for this run
        run = wandb.init(
            project=PROJECT_NAME,
            group=f"{corpus}_roberta",
            name=f"roberta_{corpus}_seed_{seed}",
            config={
                "model_type": "roberta",
                "model_name": "roberta-base",
                "corpus": corpus,
                "seed": seed,
                "n_authors": len(np.unique(train_labels)),
                **training_args
            },
            reinit=True  # allow multiple runs in same process
        )

        logger.info(f"Training with seed: {seed}")

        try:
            results = model.train_and_evaluate(
                train_text=train_text,
                train_labels=train_labels,
                val_text=val_text,
                val_labels=val_labels,
                test_text=test_text,
                test_labels=test_labels,
                corpus=corpus_for_run,
                seed=seed
            )

            # Log final metrics to wandb for this seed
            wandb.log({
                "final_val_accuracy": results['val_metrics']['accuracy'],
                "final_test_accuracy": results['test_metrics']['accuracy'],
                "final_val_loss": results['val_metrics']['loss'],
                "final_test_loss": results['test_metrics']['loss'],
            })

            logger.info(
                f"Completed training for {corpus} seed {seed}: "
                f"val_loss={results['val_metrics']['loss']:.4f}, "
                f"val_acc={results['val_metrics']['accuracy']:.4f}, "
                f"test_loss={results['test_metrics']['loss']:.4f}, "
                f"test_acc={results['test_metrics']['accuracy']:.4f}"
            )

            # Save the results for later comparison
            results_list.append((seed, results))
        except Exception as e:
            logger.error(f"Error training seed {seed}: {str(e)}")
        finally:
            run.finish()

    # pick the overall best checkpoint across seeds (lowest validation loss)
    if results_list:
        best_seed, best_result = min(results_list, key=lambda x: x[1]['val_metrics']['loss'])
        logger.info(
            f"Best checkpoint for corpus {corpus} is from seed {best_seed} with "
            f"val_loss {best_result['val_metrics']['loss']:.4f}, "
            f"val_acc {best_result['val_metrics']['accuracy']:.4f}, "
            f"test_loss {best_result['test_metrics']['loss']:.4f}, "
            f"test_acc {best_result['test_metrics']['accuracy']:.4f}. "
            f"Model saved at: {best_result['model_path']}"
        )
        wandb.run.summary.update({
            "best_seed": best_seed,
            "best_val_loss": best_result['val_metrics']['loss'],
            "best_val_accuracy": best_result['val_metrics']['accuracy'],
            "best_test_loss": best_result['test_metrics']['loss'],
            "best_test_accuracy": best_result['test_metrics']['accuracy']
        })

        # copy best checkpoint to canonical folder
        canonical_model_dir = Path("threat_models") / corpus / "no_protection" / "roberta" / "model"
        source_model_dir = Path(best_result['model_path'])
        logger.info(f"Copying best model from {source_model_dir} to canonical location {canonical_model_dir}")
        # remove canonical folder if it exists to avoid copytree errors.
        if canonical_model_dir.exists():
            shutil.rmtree(canonical_model_dir)
        shutil.copytree(source_model_dir, canonical_model_dir)
        logger.info(f"Best model is now available at the canonical location: {canonical_model_dir}")
        wandb.run.summary.update({"canonical_model_path": str(canonical_model_dir)})

    else:
        logger.warning(f"No successful training runs for corpus {corpus}.")


def main(args):
    # initialize model with training params
    training_args = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'warmup_steps': args.warmup_steps
    }

    model = RobertaBest(training_args=training_args)

    # determine corpora to process
    corpora = [args.corpus] if args.corpus else CORPORA

    # train on each corpus
    for corpus in corpora:
        try:
            train_corpus(
                model=model,
                corpus=corpus,
                num_seeds=args.num_seeds,
                training_args=training_args,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error processing {corpus}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RoBERTa as an authorship attribution model"
    )
    # corpus selection
    parser.add_argument(
        "--corpus",
        type=str,
        choices=CORPORA,
        help="Specific corpus to train on (default: all)"
    )

    # training arguments
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to use (max 10)"
    )

    args = parser.parse_args()

    # validate num_seeds
    if args.num_seeds > 10:
        parser.error("Maximum number of seeds is 10")

    main(args)

"""
Training script for RoBERTa-based authorship attribution models focused on RJ manual strategies.

This script trains RoBERTa models specifically for authorship attribution on RJ's manual
strategies (imitation, obfuscation, simplification). It trains models with multiple seeds
for each strategy, saving the best model for each scenario. This is necessary because
RJ has different cohorts in different scenarios, unlike EBG.

Usage:
    # Train on all RJ manual strategies
    python train_roberta_rj_strategies.py

    # Train on a specific strategy
    python train_roberta_rj_strategies.py --strategy imitation

    # Train with custom parameters
    python train_roberta_rj_strategies.py --strategy obfuscation --learning_rate 3e-5 --batch_size 32 --num_seeds 3
"""

import argparse
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb

from roberta import RobertaBest
from utils import FIXED_SEEDS, load_corpus

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "LLM as Defense"

# RJ manual strategies
RJ_STRATEGIES = ["imitation", "obfuscation", "special_english"]


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


def save_test_predictions(model_dir, y_true, y_pred_probs):
    """Save test predictions for later evaluation."""
    eval_dir = Path("roberta_evaluations")
    eval_dir.mkdir(exist_ok=True)

    # Extract the corpus and task from the model directory path
    path_parts = Path(model_dir).parts
    corpus = path_parts[-4]  # Should be 'rj'
    task = path_parts[-3]    # Should be the strategy name

    # Save the predictions
    output_path = eval_dir / f"{corpus}_{task}.npz"
    np.savez(
        output_path,
        y_true=y_true,
        y_pred_probs=y_pred_probs
    )
    logger.info(f"Saved test predictions to {output_path}")


def train_strategy(
        model: RobertaBest,
        strategy: str,
        num_seeds: int,
        training_args: dict,
        logger: logging.Logger
) -> None:
    """
    Train RoBERTa with multiple seeds on the specified RJ strategy and save the best checkpoint.

    After all seeds have been run, the checkpoint with the lowest validation loss is copied to
    the canonical folder: threat_models/rj/{strategy}/roberta/model.
    """
    logger.info(f"Training on RJ strategy: {strategy}")

    # load data
    train_text, train_labels, test_text, test_labels = load_corpus(
        "rj", strategy
    )

    # create validation split
    train_text, train_labels, val_text, val_labels = create_val_split(
        train_text, train_labels
    )

    # track results across seeds; each entry is a tuple: (seed, results_dict)
    results_list = []

    # train with multiple seeds
    seeds = FIXED_SEEDS[:num_seeds]

    # initialize wandb run for the overall training process
    with wandb.init(
            project=PROJECT_NAME,
            group=f"rj_{strategy}_roberta",
            name=f"roberta_rj_{strategy}_overall",
            config={
                "model_type": "roberta",
                "model_name": "roberta-base",
                "corpus": "rj",
                "strategy": strategy,
                "seeds": seeds,
                "n_authors": len(np.unique(train_labels)),
                **training_args
            }
    ) as overall_run:
        for seed in seeds:
            # to avoid overwriting checkpoints, append seed to the strategy name for the run
            strategy_for_run = f"rj_{strategy}_seed_{seed}"
            logger.info(f"Training with seed: {seed}")

            try:
                results = model.train_and_evaluate(
                    train_text=train_text,
                    train_labels=train_labels,
                    val_text=val_text,
                    val_labels=val_labels,
                    test_text=test_text,
                    test_labels=test_labels,
                    corpus=strategy_for_run,
                    seed=seed
                )

                # log metrics for this seed
                overall_run.log({
                    f"seed_{seed}/val_accuracy": results['val_metrics']['accuracy'],
                    f"seed_{seed}/test_accuracy": results['test_metrics']['accuracy'],
                    f"seed_{seed}/val_loss": results['val_metrics']['loss'],
                    f"seed_{seed}/test_loss": results['test_metrics']['loss'],
                })

                logger.info(
                    f"Completed training for RJ {strategy} seed {seed}: "
                    f"val_loss={results['val_metrics']['loss']:.4f}, "
                    f"val_acc={results['val_metrics']['accuracy']:.4f}, "
                    f"test_loss={results['test_metrics']['loss']:.4f}, "
                    f"test_acc={results['test_metrics']['accuracy']:.4f}"
                )

                # save the results for later comparison
                results_list.append((seed, results))

            except Exception as e:
                logger.error(f"Error training seed {seed}: {str(e)}")

        # pick the overall best checkpoint across seeds (lowest validation loss)
        if results_list:
            best_seed, best_result = min(results_list,
                                         key=lambda x: x[1]['val_metrics']['loss'])
            logger.info(
                f"Best checkpoint for RJ {strategy} is from seed {best_seed} with "
                f"val_loss {best_result['val_metrics']['loss']:.4f}, "
                f"val_acc {best_result['val_metrics']['accuracy']:.4f}, "
                f"test_loss {best_result['test_metrics']['loss']:.4f}, "
                f"test_acc {best_result['test_metrics']['accuracy']:.4f}"
            )

            # log best results to wandb
            overall_run.summary.update({
                "best_seed": best_seed,
                "best_val_loss": best_result['val_metrics']['loss'],
                "best_val_accuracy": best_result['val_metrics']['accuracy'],
                "best_test_loss": best_result['test_metrics']['loss'],
                "best_test_accuracy": best_result['test_metrics']['accuracy']
            })

            # copy the best checkpoint to canonical folder
            canonical_model_dir = Path(
                "threat_models") / "rj" / strategy / "roberta" / "model"
            source_model_dir = Path(best_result['model_path'])
            logger.info(
                f"Copying best model from {source_model_dir} to canonical location {canonical_model_dir}")

            # remove canonical folder if it exists to avoid copytree errors
            if canonical_model_dir.exists():
                shutil.rmtree(canonical_model_dir)

            # make parent directories if they don't exist
            canonical_model_dir.parent.mkdir(parents=True, exist_ok=True)

            shutil.copytree(source_model_dir, canonical_model_dir)

            logger.info(
                f"Best model is now available at the canonical location: {canonical_model_dir}")
            overall_run.summary.update(
                {"canonical_model_path": str(canonical_model_dir)})

            # Save test predictions for later evaluation
            if 'test_predictions' in best_result:
                save_test_predictions(
                    canonical_model_dir,
                    best_result['test_labels'],
                    best_result['test_predictions']
                )

            # print out the best checkpoint info wrapped with #####
            print("#" * 80)
            print(
                f"Best checkpoint is from seed {best_seed} and saved at {canonical_model_dir}")
            print("#" * 80)

        else:
            logger.warning(f"No successful training runs for RJ {strategy}.")


def main(args):
    # initialize model with training params
    training_args = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'warmup_steps': args.warmup_steps
    }

    model = RobertaBest(training_args=training_args)

    # determine strategies to process
    strategies = [args.strategy] if args.strategy else RJ_STRATEGIES

    # train on each strategy
    for strategy in strategies:
        train_strategy(
            model=model,
            strategy=strategy,
            num_seeds=args.num_seeds,
            training_args=training_args,
            logger=logger
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RoBERTa specifically for RJ manual strategies"
    )
    # strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        choices=RJ_STRATEGIES,
        help="Specific RJ strategy to train on (default: all)"
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
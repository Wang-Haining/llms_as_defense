"""
Evaluate RoBERTa-base's performance as an authorship attribution model across corpora
and tasks.

Usage:
    # Run specific corpus and task
    python eval_roberta.py --corpus rj --task no_protection

    # Run all tasks for a specific corpus
    python eval_roberta.py --corpus rj

    # Run everything (default)
    python eval_roberta.py
"""

import argparse
import logging

import wandb
from utils import load_corpus

from roberta import RobertaBest

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# define valid scenarios
CORPUS_TASK_MAP = {
    'rj': ['no_protection', 'imitation', 'obfuscation', 'special_english'],
    'ebg': ['no_protection', 'imitation', 'obfuscation'],
    'lcmc': ['no_protection']
}


def evaluate_corpus_task(
        model: RobertaBest,
        corpus: str,
        task: str,
        logger: logging.Logger
) -> None:
    """Evaluate RoBERTa on a specific corpus and task."""
    logger.info(f"Evaluating {corpus}-{task}")

    # load data
    train_text, train_labels, test_text, test_labels = load_corpus(corpus, task)

    # train and evaluate
    results = model.train_and_evaluate(
        train_text=train_text,
        train_labels=train_labels,
        test_text=test_text,
        test_labels=test_labels,
        corpus=corpus,
        task=task
    )

    logger.info(
        f"Best model for {corpus}-{task}: "
        f"seed={results['best_seed']}, "
        f"fold={results['best_fold']}, "
        f"val_acc={results['best_val_metrics']['accuracy']:.4f}"
    )


def main(args):
    # initialize wandb
    wandb.init(project="LLM as Defense")

    # initialize model with training params
    model = RobertaBest(
        output_dir=args.output_dir,
        save_path=args.save_path,
        training_args={
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'warmup_steps': args.warmup_steps
        }
    )

    # determine what to evaluate
    if args.corpus:
        if args.corpus not in CORPUS_TASK_MAP:
            raise ValueError(f"Invalid corpus: {args.corpus}")
        corpora = [args.corpus]
    else:
        corpora = list(CORPUS_TASK_MAP.keys())

    # evaluate specified scenarios
    for corpus in corpora:
        logger.info(f"Processing corpus: {corpus}")

        if args.task:
            if args.task not in CORPUS_TASK_MAP[corpus]:
                logger.warning(f"Task {args.task} not available for corpus {corpus}, skipping")
                continue
            tasks = [args.task]
        else:
            tasks = CORPUS_TASK_MAP[corpus]

        for task in tasks:
            try:
                evaluate_corpus_task(model, corpus, task, logger)
            except Exception as e:
                logger.error(f"Error processing {corpus}-{task}: {str(e)}")
                continue

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa across different corpora and scenarios"
    )
    # corpus and task selection
    parser.add_argument(
        "--corpus",
        type=str,
        choices=['rj', 'ebg', 'lcmc'],
        help="Specific corpus to evaluate (default: all)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Specific task to evaluate (default: all tasks for chosen corpus)"
    )

    # path arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="baselines",
        help="Subdirectory under results for saving"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory for results"
    )

    # training arguments
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=50)

    args = parser.parse_args()

    # validate task if provided
    if args.task and args.corpus and args.task not in CORPUS_TASK_MAP[args.corpus]:
        parser.error(f"Task '{args.task}' is not valid for corpus '{args.corpus}'. "
                    f"Valid tasks are: {CORPUS_TASK_MAP[args.corpus]}")

    main(args)

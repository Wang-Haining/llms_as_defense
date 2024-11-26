import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from utils import load_ebg, load_lcmc, load_rj, vectorize_writeprints_static

# setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_svm_pipeline() -> Pipeline:
    """Create SVM pipeline with Writeprints-static features"""
    return Pipeline(
        [
            ("normalizer", Normalizer(norm="l1")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="poly", gamma="scale", probability=True, max_iter=-1)),
        ]
    )


def grid_search_svm(
        X_train: np.ndarray, y_train: np.ndarray, n_jobs: int = -1
) -> Tuple[Dict, float]:
    """
    Perform grid search for SVM parameters.

    Args:
        X_train: Training features
        y_train: Training labels
        n_jobs: Number of parallel jobs

    Returns:
        best_params: Dictionary of best parameters
        best_score: Best CV score
    """
    pipeline = create_svm_pipeline()

    # define param grid
    param_grid = {
        "svm__C": np.logspace(-3, 3, num=7),  # 1e-3 ~ 1e3
        "svm__degree": np.arange(2, 9),  # 2 ~ 8
        "svm__coef0": [0, 1, 10, 100, 1000],
    }

    # setup grid search with stratified k-fold
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=2,
    )

    # perform grid search
    grid_search.fit(X_train, y_train)

    # convert NumPy types to native Python types in best_params
    best_params = {
        key: int(value) if isinstance(value, np.integer) else
        float(value) if isinstance(value, np.floating) else value
        for key, value in grid_search.best_params_.items()
    }

    return best_params, float(grid_search.best_score_)


def optimize_svm_for_corpus(
        corpus_name: str, data_loader, output_dir: Path, n_jobs: int = -1
) -> Dict:
    """
    Optimize SVM parameters for a specific corpus.

    Args:
        corpus_name: Name of the corpus
        data_loader: Function to load corpus data
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Optimizing SVM for {corpus_name} corpus")

    # load data (use control/no_protection)
    task = "control" if corpus_name == "rj" else "no_protection"
    train_text, train_labels, test_text, test_labels = data_loader(task)

    # convert labels to native Python types
    train_labels = [int(label) if isinstance(label, np.integer) else label
                    for label in train_labels]
    test_labels = [int(label) if isinstance(label, np.integer) else label
                   for label in test_labels]

    # extract Writeprints-static features
    X_train = vectorize_writeprints_static(train_text)
    X_test = vectorize_writeprints_static(test_text)

    # perform grid search
    best_params, cv_score = grid_search_svm(X_train, train_labels, n_jobs)

    # train final model with best parameters
    pipeline = create_svm_pipeline()
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, train_labels)
    test_score = pipeline.score(X_test, test_labels)

    # save results
    results = {
        "corpus": corpus_name,
        "best_parameters": best_params,
        "cv_accuracy": float(cv_score),
        "test_accuracy": float(test_score),
        "n_authors": len(set(train_labels)),
        "n_train_samples": len(train_labels),
        "n_test_samples": len(test_labels),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{corpus_name.lower()}_svm_optimization.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results for {corpus_name}:")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"CV accuracy: {cv_score:.3f}")
    logger.info(f"Test accuracy: {test_score:.3f}")

    return results


def print_summary(all_results: Dict):
    """Print a summary of optimization results for all corpora."""
    logger.info("\n" + "=" * 50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 50)

    for corpus_name, results in all_results.items():
        logger.info(f"\n{corpus_name.upper()} Corpus:")
        logger.info(f"Best Parameters:")
        for param, value in results["best_parameters"].items():
            logger.info(f"  {param}: {value}")
        logger.info(f"CV Accuracy: {results['cv_accuracy']:.3f}")
        logger.info(f"Test Accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"Number of authors: {results['n_authors']}")
        logger.info(f"Training samples: {results['n_train_samples']}")
        logger.info(f"Test samples: {results['n_test_samples']}")

    logger.info("\n" + "=" * 50)


def main():
    # setup output directory
    output_dir = Path("results/optimization/svm")

    # define corpora
    corpora = {"rj": load_rj, "ebg": load_ebg, "lcmc": load_lcmc}

    # optimize for each corpus
    all_results = {}
    for corpus_name, data_loader in tqdm(corpora.items(), desc="Optimizing corpora"):
        try:
            results = optimize_svm_for_corpus(
                corpus_name=corpus_name, data_loader=data_loader, output_dir=output_dir
            )
            all_results[corpus_name] = results
        except Exception as e:
            logger.error(f"Error optimizing {corpus_name}: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    # save combined results
    with open(output_dir / "all_optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary of results
    print_summary(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize poly SVM parameters for each corpus"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs for grid search"
    )
    args = parser.parse_args()

    main()

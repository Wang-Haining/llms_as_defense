import json
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

from utils import load_ebg, load_rj, vectorize_writeprints_static

# setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_svm_pipeline() -> Pipeline:
    """Create SVM pipeline with writeprints-static features"""
    return Pipeline(
        [
            ("normalizer", Normalizer(norm="l1")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="poly", gamma="scale", probability=True, max_iter=-1)),
        ]
    )


def grid_search_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int = -1
) -> Tuple[Dict, float]:
    """
    Perform grid search for SVM parameters.

    Args:
        X_train: training features
        y_train: training labels
        n_jobs: number of parallel jobs

    Returns:
        best_params: dictionary of best parameters
        best_score: best CV score
    """
    pipeline = create_svm_pipeline()

    # more granular param grid
    param_grid = {
        "svm__C": np.logspace(-4, 4, num=9),  # 1e-4 ~ 1e4
        "svm__degree": [2, 3, 4],  # focus on lower degrees for efficiency
        "svm__coef0": [0.0, 0.1, 1.0, 10.0, 100.0],
    }

    # setup stratified k-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # setup grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )

    # perform grid search
    grid_search.fit(X_train, y_train)

    # log detailed results
    cv_results = grid_search.cv_results_
    for mean_score, std_score, params in zip(
            cv_results['mean_test_score'],
            cv_results['std_test_score'],
            cv_results['params']
    ):
        logger.info(
            f"CV accuracy: {mean_score:.3f} (+/- {std_score * 2:.3f}) for {params}")

    # convert NumPy types to native Python types
    best_params = {
        key: int(value) if isinstance(value, np.integer) else
        float(value) if isinstance(value, np.floating) else value
        for key, value in grid_search.best_params_.items()
    }

    return best_params, float(grid_search.best_score_)


def optimize_svm_for_corpus(
        corpus_name: str,
        data_loader,
        output_dir: Path,
        n_jobs: int = -1
) -> Dict:
    """
    Optimize SVM parameters for a specific corpus.

    Args:
        corpus_name: name of the corpus
        data_loader: function to load corpus data
        output_dir: directory to save results
        n_jobs: number of parallel jobs

    Returns:
        dictionary with optimization results
    """
    logger.info(f"optimizing SVM for {corpus_name} corpus")

    # load data (use control/no_protection baseline)
    task = "no_protection"
    train_text, train_labels, test_text, test_labels = data_loader(task)

    # verify label consistency
    train_unique = set(train_labels)
    test_unique = set(test_labels)
    if train_unique != test_unique:
        logger.warning(
            f"label mismatch! Train labels: {train_unique}, Test labels: {test_unique}")

    logger.info(
        f"data loaded - Train: {len(train_text)} samples, Test: {len(test_text)} samples")
    logger.info(f"number of authors: {len(train_unique)}")

    # extract writeprints-static features
    X_train = vectorize_writeprints_static(train_text)
    X_test = vectorize_writeprints_static(test_text)

    logger.info(f"features extracted - Train: {X_train.shape}, Test: {X_test.shape}")

    # check for zero features
    zero_features_train = np.sum(X_train.sum(axis=0) == 0)
    zero_features_test = np.sum(X_test.sum(axis=0) == 0)
    logger.info(
        f"zero features - Train: {zero_features_train}, Test: {zero_features_test}")

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
        "feature_dims": X_train.shape[1],
        "zero_features_train": int(zero_features_train),
        "zero_features_test": int(zero_features_test),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{corpus_name.lower()}_svm_optimization.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"results for {corpus_name}:")
    logger.info(f"best parameters: {best_params}")
    logger.info(f"CV accuracy: {cv_score:.3f}")
    logger.info(f"test accuracy: {test_score:.3f}")

    return results


def main():
    # setup output directory
    output_dir = Path("threat_models/optimization/svm")

    # define corpora
    corpora = {
        "rj": load_rj,
        "ebg": load_ebg,
        # "lcmc": load_lcmc
    }

    # optimize for each corpus
    all_results = {}
    for corpus_name, data_loader in tqdm(corpora.items(), desc="optimizing corpora"):
        try:
            results = optimize_svm_for_corpus(
                corpus_name=corpus_name,
                data_loader=data_loader,
                output_dir=output_dir,
                n_jobs=-1  # use all cores
            )
            all_results[corpus_name] = results
        except Exception as e:
            logger.error(f"Error optimizing {corpus_name}: {str(e)}")
            raise  # re-raise to see full traceback

    # save combined results
    with open(output_dir / "all_optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary
    logger.info("\n" + "=" * 50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 50)

    for corpus_name, results in all_results.items():
        logger.info(f"\n{corpus_name.upper()} corpus:")
        logger.info(f"best parameters:")
        for param, value in results["best_parameters"].items():
            logger.info(f"  {param}: {value}")
        logger.info(f"CV accuracy: {results['cv_accuracy']:.3f}")
        logger.info(f"test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"number of authors: {results['n_authors']}")
        logger.info(f"training samples: {results['n_train_samples']}")
        logger.info(f"test samples: {results['n_test_samples']}")
        logger.info(f"feature dimensions: {results['feature_dims']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize SVM parameters for each corpus"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for grid search"
    )
    args = parser.parse_args()

    main()

# ==================================================
# 2025-01-15 21:02:06,525 - INFO - OPTIMIZATION SUMMARY
# ==================================================
# 2025-01-15 21:02:06,525 - INFO -
# RJ corpus:
# 2025-01-15 21:02:06,525 - INFO - best parameters:
# 2025-01-15 21:02:06,525 - INFO -   svm__C: 0.0001
# 2025-01-15 21:02:06,525 - INFO -   svm__coef0: 100.0
# 2025-01-15 21:02:06,525 - INFO -   svm__degree: 3
# 2025-01-15 21:02:06,525 - INFO - CV accuracy: 0.858
# 2025-01-15 21:02:06,525 - INFO - test accuracy: 0.286
# 2025-01-15 21:02:06,525 - INFO - number of authors: 21
# 2025-01-15 21:02:06,525 - INFO - training samples: 254
# 2025-01-15 21:02:06,525 - INFO - test samples: 21
# 2025-01-15 21:02:06,525 - INFO - feature dimensions: 552
# 2025-01-15 21:02:06,525 - INFO -
# EBG corpus:
# 2025-01-15 21:02:06,525 - INFO - best parameters:
# 2025-01-15 21:02:06,525 - INFO -   svm__C: 0.0001
# 2025-01-15 21:02:06,525 - INFO -   svm__coef0: 100.0
# 2025-01-15 21:02:06,525 - INFO -   svm__degree: 3
# 2025-01-15 21:02:06,525 - INFO - CV accuracy: 0.797
# 2025-01-15 21:02:06,525 - INFO - test accuracy: 0.711
# 2025-01-15 21:02:06,525 - INFO - number of authors: 45
# 2025-01-15 21:02:06,525 - INFO - training samples: 654
# 2025-01-15 21:02:06,525 - INFO - test samples: 45
# 2025-01-15 21:02:06,525 - INFO - feature dimensions: 552
# 2025-01-15 21:02:06,525 - INFO -
# LCMC corpus:
# 2025-01-15 21:02:06,525 - INFO - best parameters:
# 2025-01-15 21:02:06,525 - INFO -   svm__C: 0.1
# 2025-01-15 21:02:06,525 - INFO -   svm__coef0: 10.0
# 2025-01-15 21:02:06,525 - INFO -   svm__degree: 2
# 2025-01-15 21:02:06,525 - INFO - CV accuracy: 0.633
# 2025-01-15 21:02:06,525 - INFO - test accuracy: 0.143
# 2025-01-15 21:02:06,525 - INFO - number of authors: 21
# 2025-01-15 21:02:06,525 - INFO - training samples: 378
# 2025-01-15 21:02:06,525 - INFO - test samples: 21
# 2025-01-15 21:02:06,525 - INFO - feature dimensions: 552
# ==================================================

import numpy as np
from collections import Counter
from utils import load_corpus

from sklearn.metrics import confusion_matrix
import logging


def debug_pipeline(train_text, train_labels, test_text, test_labels, pipeline, task):
    """Debug ML pipeline for authorship attribution."""
    logging.info(f"\nDebugging pipeline for task: {task}")

    # 1. Check data shapes and distributions
    logging.info(f"Train samples: {len(train_text)}, Test samples: {len(test_text)}")
    logging.info(f"Unique train labels: {np.unique(train_labels)}")
    logging.info(f"Unique test labels: {np.unique(test_labels)}")
    logging.info(f"Label distribution in train: {np.bincount(train_labels)}")
    logging.info(f"Label distribution in test: {np.bincount(test_labels)}")

    # 2. Feature extraction check
    X_train = pipeline.named_steps['features'].transform(train_text)
    X_test = pipeline.named_steps['features'].transform(test_text)
    logging.info(f"Feature shapes - Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Check for zero/sparse features
    logging.info(f"Zero features in train: {np.sum(X_train.sum(axis=0) == 0)}")
    logging.info(f"Zero features in test: {np.sum(X_test.sum(axis=0) == 0)}")

    # 4. Train model and get predictions
    pipeline.fit(train_text, train_labels)
    test_probs = pipeline.predict_proba(test_text)
    predictions = np.argmax(test_probs, axis=1)

    # 5. Detailed prediction analysis
    cm = confusion_matrix(test_labels, predictions)
    logging.info(f"Confusion matrix:\n{cm}")

    # 6. Check prediction confidence
    mean_confidence = np.mean(np.max(test_probs, axis=1))
    logging.info(f"Mean prediction confidence: {mean_confidence:.3f}")

    return {
        "features_shape": X_train.shape,
        "zero_features": np.sum(X_train.sum(axis=0) == 0),
        "mean_confidence": mean_confidence,
        "confusion_matrix": cm
    }


# Add this to your evaluate_corpus_task function:
def evaluate_corpus_task_with_debug(model, corpus, task, logger):
    """Evaluate ML model with debugging information."""
    logger.info(f"Evaluating {corpus}-{task}")

    # Load data
    train_text, train_labels, test_text, test_labels = load_corpus(corpus, task)

    # Run debugging
    debug_info = debug_pipeline(
        train_text, train_labels,
        test_text, test_labels,
        model.pipeline, task
    )

    # Train and evaluate as before
    results = model.train_and_evaluate(
        train_text=train_text,
        train_labels=train_labels,
        test_text=test_text,
        test_labels=test_labels,
        corpus=corpus,
        task=task
    )

    return results, debug_info

def debug_label_mapping(train_text, train_labels, test_text, test_labels, task):
    """Debug label mapping between train and test sets."""
    print(f"\nDebugging labels for task: {task}")

    # Basic counts
    print(f"Training samples: {len(train_text)}")
    print(f"Test samples: {len(test_text)}")

    # Label distributions
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    print("\nTraining label distribution:")
    print(train_dist)
    print("\nTest label distribution:")
    print(test_dist)

    # Check label overlap
    train_unique = set(train_labels)
    test_unique = set(test_labels)
    print(f"\nUnique train labels: {len(train_unique)}")
    print(f"Unique test labels: {len(test_unique)}")
    print(f"Labels in test but not train: {test_unique - train_unique}")
    print(f"Labels in train but not test: {train_unique - test_unique}")

    # Samples per author
    print("\nAvg samples per author in train:", len(train_labels) / len(train_unique))

    # Sample verification
    print("\nFirst few training samples for each label:")
    for label in sorted(train_unique)[:3]:
        idx = train_labels.index(label)
        print(f"\nLabel {label}:")
        print(train_text[idx][:100] + "...")

    return {
        "train_samples": len(train_text),
        "test_samples": len(test_text),
        "train_labels": len(train_unique),
        "test_labels": len(test_unique),
        "train_dist": dict(train_dist),
        "test_dist": dict(test_dist)
    }


# Use this to check each corpus and task
for corpus in ['rj', 'ebg', 'lcmc']:
    if corpus == 'rj':
        tasks = ['no_protection', 'imitation', 'obfuscation', 'special_english']
    elif corpus == 'ebg':
        tasks = ['no_protection', 'imitation', 'obfuscation']
    else:
        tasks = ['no_protection']

    for task in tasks:
        train_text, train_labels_raw, test_text, test_labels_raw = load_corpus(corpus,
                                                                               task)
        debug_info = debug_label_mapping(train_text, train_labels_raw,
                                         test_text, test_labels_raw, f"{corpus}-{task}")
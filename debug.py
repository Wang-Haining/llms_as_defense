import numpy as np
from collections import Counter
from utils import load_corpus


# def debug_label_mapping(train_text, train_labels, test_text, test_labels, task):
#     """Debug label mapping between train and test sets."""
#     print(f"\nDebugging labels for task: {task}")
#
#     # Basic counts
#     print(f"Training samples: {len(train_text)}")
#     print(f"Test samples: {len(test_text)}")
#
#     # Label distributions
#     train_dist = Counter(train_labels)
#     test_dist = Counter(test_labels)
#     print("\nTraining label distribution:")
#     print(train_dist)
#     print("\nTest label distribution:")
#     print(test_dist)
#
#     # Check label overlap
#     train_unique = np.unique(train_labels)
#     test_unique = np.unique(test_labels)
#     print(f"\nUnique train labels: {len(train_unique)}")
#     print(f"Unique test labels: {len(test_unique)}")
#     print(f"Labels in test but not train: {set(test_unique) - set(train_unique)}")
#     print(f"Labels in train but not test: {set(train_unique) - set(test_unique)}")
#
#     # Samples per author
#     print("\nAvg samples per author in train:", len(train_labels) / len(train_unique))
#
#     # Sample verification
#     print("\nFirst few training samples for each label:")
#     for label in sorted(train_unique)[:3]:
#         idx = np.where(train_labels == label)[0][0]  # Get first occurrence of label
#         print(f"\nLabel {label}:")
#         print(f"Sample length: {len(train_text[idx])}")
#         print(train_text[idx][:200] + "...")
#
#     # Text length statistics
#     train_lengths = [len(text) for text in train_text]
#     test_lengths = [len(text) for text in test_text]
#     print("\nText length statistics:")
#     print(
#         f"Train - Mean: {np.mean(train_lengths):.1f}, Std: {np.std(train_lengths):.1f}")
#     print(f"Test - Mean: {np.mean(test_lengths):.1f}, Std: {np.std(test_lengths):.1f}")
#
#     return {
#         "train_samples": len(train_text),
#         "test_samples": len(test_text),
#         "train_labels": len(train_unique),
#         "test_labels": len(test_unique),
#         "train_dist": dict(train_dist),
#         "test_dist": dict(test_dist)
#     }
#
#
# # Use this to check each corpus and task
# print("Starting debug analysis...")
# for corpus in ['rj', 'ebg', 'lcmc']:
#     print(f"\n{'=' * 50}\nAnalyzing corpus: {corpus}")
#     if corpus == 'rj':
#         tasks = ['no_protection', 'imitation', 'obfuscation', 'special_english']
#     elif corpus == 'ebg':
#         tasks = ['no_protection', 'imitation', 'obfuscation']
#     else:
#         tasks = ['no_protection']
#
#     for task in tasks:
#         print(f"\n{'-' * 30}\nTask: {task}")
#         train_text, train_labels_raw, test_text, test_labels_raw = load_corpus(corpus,
#                                                                                task)
#         debug_info = debug_label_mapping(train_text, train_labels_raw,
#                                          test_text, test_labels_raw,
#                                          f"{corpus}-{task}")

import numpy as np
from sklearn.preprocessing import StandardScaler
from writeprints_static import WriteprintsStatic
from utils import load_corpus


def analyze_feature_shifts(corpus, normal_task='no_protection',
                           modified_task='obfuscation'):
    """Analyze how features shift between normal and modified writing."""

    # Load normal and modified texts
    train_normal, _, test_normal, _ = load_corpus(corpus, normal_task)
    train_mod, _, test_mod, _ = load_corpus(corpus, modified_task)

    # Extract features
    vec = WriteprintsStatic()
    features_normal = vec.transform(test_normal).toarray()
    features_mod = vec.transform(test_mod).toarray()

    # Standardize features
    scaler = StandardScaler()
    scaler.fit(features_normal)
    features_normal_scaled = scaler.transform(features_normal)
    features_mod_scaled = scaler.transform(features_mod)

    # Compare distributions
    feature_shifts = np.mean(np.abs(features_normal_scaled - features_mod_scaled),
                             axis=0)
    print(f"\nFeature shift analysis for {corpus}:")
    print(f"Average absolute shift: {np.mean(feature_shifts):.3f}")
    print(f"Max shift: {np.max(feature_shifts):.3f}")
    print(
        f"Feature similarity: {np.corrcoef(features_normal.flatten(), features_mod.flatten())[0, 1]:.3f}")


# Run analysis
for corpus in ['rj', 'ebg']:
    analyze_feature_shifts(corpus)
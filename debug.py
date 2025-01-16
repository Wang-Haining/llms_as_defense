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
    """Analyze how features shift between normal and modified writing styles."""
    print(f"\nAnalyzing feature shifts for {corpus}")

    # Load data
    train_normal, labels_normal, test_normal, test_labels_normal = load_corpus(corpus,
                                                                               normal_task)
    train_mod, labels_mod, test_mod, test_labels_mod = load_corpus(corpus,
                                                                   modified_task)

    # Extract features
    vec = WriteprintsStatic()
    features_normal = vec.transform(test_normal).toarray()
    features_mod = vec.transform(test_mod).toarray()

    # Standardize features
    scaler = StandardScaler()
    scaler.fit(np.vstack([features_normal, features_mod]))
    features_normal_scaled = scaler.transform(features_normal)
    features_mod_scaled = scaler.transform(features_mod)

    # Get unique authors that appear in both sets
    normal_authors = set(np.unique(test_labels_normal))
    mod_authors = set(np.unique(test_labels_mod))
    common_authors = normal_authors.intersection(mod_authors)

    print(f"Authors in normal: {len(normal_authors)}")
    print(f"Authors in modified: {len(mod_authors)}")
    print(f"Common authors: {len(common_authors)}")

    # Compare distributions per author
    all_shifts = []
    similarities = []

    for author in common_authors:
        # Find this author's samples using numpy where
        normal_idx = np.where(test_labels_normal == author)[0]
        mod_idx = np.where(test_labels_mod == author)[0]

        if len(normal_idx) > 0 and len(mod_idx) > 0:
            # Get features for this author
            author_normal = features_normal_scaled[normal_idx[0]]
            author_mod = features_mod_scaled[mod_idx[0]]

            # Calculate shift
            shift = np.abs(author_normal - author_mod)
            all_shifts.append(shift)

            # Calculate similarity
            sim = np.corrcoef(author_normal, author_mod)[0, 1]
            similarities.append(sim)

    all_shifts = np.array(all_shifts)

    print(f"\nAnalysis Results:")
    print(f"Number of authors analyzed: {len(all_shifts)}")
    print(f"Average absolute shift: {np.mean(all_shifts):.3f}")
    print(f"Max shift per author (mean): {np.mean(np.max(all_shifts, axis=1)):.3f}")

    # Calculate top shifted features
    mean_shifts = np.mean(all_shifts, axis=0)
    top_indices = np.argsort(mean_shifts)[-5:]  # Top 5 most shifted features
    print("\nTop shifted features:")
    for idx in reversed(top_indices):
        print(f"Feature {idx}: {mean_shifts[idx]:.3f}")

    print(f"\nFeature Similarities:")
    print(f"Mean similarity: {np.mean(similarities):.3f}")
    print(f"Std similarity: {np.std(similarities):.3f}")

    return {
        "mean_shift": float(np.mean(all_shifts)),
        "max_shift": float(np.mean(np.max(all_shifts, axis=1))),
        "feature_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "n_authors": len(all_shifts)
    }


print("Starting analysis...")
results = {}
for corpus in ['rj', 'ebg']:
    print(f"\n{'=' * 50}")
    print(f"Analyzing corpus: {corpus}")

    # Compare no_protection vs obfuscation
    results[f"{corpus}_obfuscation"] = analyze_feature_shifts(
        corpus, 'no_protection', 'obfuscation'
    )

    # Compare no_protection vs imitation
    results[f"{corpus}_imitation"] = analyze_feature_shifts(
        corpus, 'no_protection', 'imitation'
    )

print("\nFinal Summary:")
for task, result in results.items():
    print(f"\n{task}:")
    for metric, value in result.items():
        print(f"  {metric}: {value}")
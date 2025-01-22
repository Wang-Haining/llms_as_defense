"""
Text quality evaluation metrics for assessing paraphrase quality and semantic retention.

Key metrics:
- PINC (Paraphrase In N-gram Changes): Measures n-gram novelty
- SacreBLEU: Language-agnostic BLEU implementation
- METEOR: Metric incorporating synonyms and paraphrases
- BERTScore: Contextual semantic similarity using BERT

"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import nltk
import numpy as np
import torch
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import BLEU

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


def get_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    """
    Extract n-grams from text.

    Args:
        text: Input text
        n: n-gram size

    Returns:
        Set of n-gram tuples
    """
    tokens = word_tokenize(text.lower())
    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.add(ngram)
    return ngrams


def compute_pinc(
        source_texts: List[str],
        candidate_texts: List[str],
        max_n: int = 4
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute PINC scores measuring n-gram novelty between source and candidate texts.

    PINC measures percentage of n-grams in candidate that don't appear in source.
    Formula: PINC(s,c) = 1/N ∑[1 - |n-grams ∩ n-gramc| / |n-gramc|]

    Args:
        source_texts: Original texts
        candidate_texts: Generated/paraphrased texts
        max_n: Maximum n-gram size to consider

    Returns:
        Dictionary with mean/std scores per n-gram size
    """
    if len(source_texts) != len(candidate_texts):
        raise ValueError("Number of source and candidate texts must match")

    # store scores for each n-gram size
    pinc_scores = defaultdict(list)

    for source, candidate in zip(source_texts, candidate_texts):
        # calculate PINC for each n-gram size
        for n in range(1, max_n + 1):
            source_ngrams = get_ngrams(source, n)
            candidate_ngrams = get_ngrams(candidate, n)

            if not candidate_ngrams:
                continue

            # compute n-gram overlap ratio
            overlap = len(source_ngrams & candidate_ngrams)
            ratio = 1 - (overlap / len(candidate_ngrams))
            pinc_scores[f'pinc_{n}'].append(ratio)

    # compute statistics
    results = {}
    for n in range(1, max_n + 1):
        key = f'pinc_{n}'
        scores = pinc_scores[key]
        if scores:
            results[f'{key}_avg'] = float(np.mean(scores))
            results[f'{key}_std'] = float(np.std(scores))
            results[f'{key}_scores'] = scores

    # compute overall PINC as average across n-gram sizes
    all_scores = []
    for n in range(1, max_n + 1):
        if pinc_scores[f'pinc_{n}']:
            all_scores.append(np.mean(pinc_scores[f'pinc_{n}']))

    if all_scores:
        results['pinc_overall_avg'] = float(np.mean(all_scores))
        results['pinc_overall_std'] = float(np.std(all_scores))

    return results


def compute_bleu(
        candidate_texts: List[str],
        reference_texts: List[str],
        smooth_method: str = 'exp'
) -> Dict[str, float]:
    """
    Compute SacreBLEU score between generated texts and references.

    Args:
        candidate_texts: Generated/paraphrased texts
        reference_texts: Original reference texts
        smooth_method: BLEU smoothing method

    Returns:
        Dictionary with BLEU score and n-gram precisions
    """
    # sacreBLEU expects list of references for each candidate
    refs = [[ref] for ref in reference_texts]

    bleu = BLEU(smooth_method=smooth_method)
    scores = bleu.corpus_score(candidate_texts, refs)

    return {
        'bleu': scores.score / 100,  # normalize to [0,1]
        'precisions': [p / 100 for p in scores.precisions]
    }


def compute_meteor(
        candidate_texts: List[str],
        reference_texts: List[str]
) -> Dict[str, float]:
    """
    Compute METEOR scores incorporating synonyms and paraphrases.

    Args:
        candidate_texts: Generated/paraphrased texts
        reference_texts: Original reference texts

    Returns:
        Dictionary with METEOR statistics
    """
    scores = []
    for hyp, ref in zip(candidate_texts, reference_texts):
        hyp_tokens = word_tokenize(hyp)
        ref_tokens = word_tokenize(ref)
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)

    return {
        'meteor_avg': float(np.mean(scores)),
        'meteor_std': float(np.std(scores)),
        'meteor_scores': scores
    }


def compute_bertscore(
        candidate_texts: List[str],
        reference_texts: List[str],
        model_type: str = "bert-large-uncased",
        num_layers: int = 18,
        batch_size: int = 32,
        device: Optional[str] = None
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute BERTScore measuring semantic similarity using contextual embeddings.
    Uses BERT-large-uncased layer 18 by default as in the original paper.
    Automatically selects first available GPU or falls back to CPU.

    Args:
        candidate_texts: Generated/paraphrased texts
        reference_texts: Original reference texts
        model_type: BERT model variant (default: bert-large-uncased)
        num_layers: Which layer to use for embeddings (default: 18)
        batch_size: Inference batch size
        device: Computation device (default: auto-detect first GPU or fall back to CPU)

    Returns:
        Dictionary with precision, recall and F1 statistics
    """
    # auto-detect device if not specified
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            logger.warning("No GPU detected, falling back to CPU for BERTScore computation")
        else:
            logger.info(f"Using {device} for BERTScore computation")

    precision_scores, recall_scores, f1_scores = bert_score(
        candidate_texts,
        reference_texts,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device
    )

    # convert tensors to numpy
    precision_scores = precision_scores.cpu().numpy()
    recall_scores = recall_scores.cpu().numpy()
    f1_scores = f1_scores.cpu().numpy()

    return {
        'bertscore_precision_avg': float(np.mean(precision_scores)),
        'bertscore_precision_std': float(np.std(precision_scores)),
        'bertscore_recall_avg': float(np.mean(recall_scores)),
        'bertscore_recall_std': float(np.std(recall_scores)),
        'bertscore_f1_avg': float(np.mean(f1_scores)),
        'bertscore_f1_std': float(np.std(f1_scores)),
        'bertscore_individual': [
            {
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1)
            }
            for prec, rec, f1 in zip(precision_scores, recall_scores, f1_scores)
        ]
    }


def evaluate_quality(
        candidate_texts: List[str],
        reference_texts: List[str],
        source_texts: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute comprehensive text quality metrics.

    Args:
        candidate_texts: Generated/paraphrased texts
        reference_texts: Original reference texts
        source_texts: Source texts for PINC (if different from references)
        metrics: List of metrics to compute. Options:
            - 'pinc': N-gram novelty
            - 'bleu': BLEU score
            - 'meteor': METEOR score
            - 'bertscore': BERTScore

    Returns:
        Dictionary with results for each requested metric
    """
    if metrics is None:
        metrics = ['pinc', 'bleu', 'meteor', 'bertscore']

    if not source_texts:
        source_texts = reference_texts

    results = {}

    if 'pinc' in metrics:
        results['pinc'] = compute_pinc(source_texts, candidate_texts)

    if 'bleu' in metrics:
        results['bleu'] = compute_bleu(candidate_texts, reference_texts)

    if 'meteor' in metrics:
        results['meteor'] = compute_meteor(candidate_texts, reference_texts)

    if 'bertscore' in metrics:
        results['bertscore'] = compute_bertscore(candidate_texts, reference_texts)

    return results


# if __name__ == "__main__":
#     # example usage
#     sources = [
#         "The quick brown fox jumps over the lazy dog.",
#         "The cat sat on the mat."
#     ]
#
#     candidates = [
#         "A swift brown fox leaps across a lazy dog.",
#         "A feline rests on the carpet."
#     ]
#
#     # evaluate with all metrics
#     results = evaluate_quality(candidates, sources)
#
#     # print results
#     for metric, scores in results.items():
#         print(f"\n{metric.upper()} Scores:")
#         for k, v in scores.items():
#             if isinstance(v, (float, int)):
#                 print(f"{k}: {v:.4f}")
#             elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (float, int)):
#                 print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

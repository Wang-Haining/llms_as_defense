"""
Evaluate LLM-generated adversarial examples using saved authorship attribution models.

This script:
1. Loads LLM-generated adversarial examples from llm.py output directories
2. Uses saved attribution models to evaluate their effectiveness
3. Computes comprehensive metrics including attribution performance deltas and text quality

Directory Structure:

INPUT:
llm_outputs/{corpus}/{research_question}/{sub_question}/
├── experiment_config.json
└── seed_{seed}.json  # contains original and transformed texts

MODEL:
results/{corpus}/no_protection/
├── logreg/
│   ├── predictions.npz
│   └── model/
│       ├── model.pkl
│       └── metadata.json
├── svm/
└── roberta/

OUTPUT:
defense_results/{corpus}/{research_question}/{sub_question}/
└── seed_{seed}.json   # individual seed results containing:
    ├── metrics/
    │   ├── attribution/           # attribution metrics
    │   │   ├── original_*/        # metrics on original texts:
    │   │   │   ├── accuracy      # basic classification
    │   │   │   ├── mrr          # mean reciprocal rank
    │   │   │   ├── map          # mean average precision
    │   │   │   ├── entropy      # prediction uncertainty
    │   │   │   ├── entropy_std
    │   │   │   ├── conf_gap     # confidence gap between top predictions
    │   │   │   ├── conf_gap_std
    │   │   │   ├── top_k_acc    # top-k accuracy (k=1,3,5)
    │   │   │   ├── ndcg         # NDCG scores (k=1,3,5,all)
    │   │   │   ├── gini         # distribution inequality
    │   │   │   └── gini_std
    │   │   ├── transformed_*/    # same metrics for transformed texts
    │   │   └── effectiveness/    # defense effectiveness metrics:
    │   │       ├── mrr_change
    │   │       ├── map_change
    │   │       ├── conf_gap_change
    │   │       ├── entropy_increase
    │   │       ├── avg_rank_improvement
    │   │       ├── ndcg_change
    │   │       └── gini_change
    │   └── quality/              # text quality metrics
    │       ├── pinc/             # n-gram novelty
    │       │   ├── pinc_overall_avg
    │       │   ├── pinc_overall_std
    │       │   ├── pinc_1_avg    # metrics for unigrams
    │       │   ├── pinc_1_std
    │       │   ├── pinc_1_scores
    │       │   └── ... # similar for n=2,3,4
    │       ├── bleu/             # semantic retention
    │       │   ├── bleu         # overall score
    │       │   └── precisions   # n-gram precisions
    │       ├── meteor/           # semantic similarity with synonyms
    │       │   ├── meteor_avg
    │       │   ├── meteor_std
    │       │   └── meteor_scores
    │       └── bertscore/        # contextual semantic similarity
    │           ├── bertscore_precision_avg
    │           ├── bertscore_precision_std
    │           ├── bertscore_recall_avg
    │           ├── bertscore_recall_std
    │           ├── bertscore_f1_avg
    │           ├── bertscore_f1_std
    │           └── bertscore_individual  # per-example scores
    └── texts/                    # evaluated texts
        ├── original
        └── transformed

Usage:
    python eval_llm_defense.py --llm_outputs path/to/llm/outputs --model_type [logreg|svm|roberta]
"""


import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from tqdm import tqdm

from utils import (
    LogisticRegressionPredictor, 
    SVMPredictor,
    evaluate_attribution_defense
)
from roberta import RobertaPredictor
from eval_text_quality import evaluate_quality

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of different types of attribution models."""

    PREDICTOR_MAP = {
        'logreg': LogisticRegressionPredictor,
        'svm': SVMPredictor,
        'roberta': RobertaPredictor
    }

    @classmethod
    def load_predictor(cls, corpus: str, model_type: str) -> Any:
        """
        Load appropriate predictor based on model type.

        Args:
            corpus: Corpus name (rj, ebg, or lcmc)
            model_type: Type of model (logreg, svm, or roberta)

        Returns:
            Initialized predictor instance
        """
        if model_type not in cls.PREDICTOR_MAP:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_dir = Path("results") / corpus / "no_protection" / model_type / "model"

        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        predictor_cls = cls.PREDICTOR_MAP[model_type]
        return predictor_cls(model_dir)


def load_llm_outputs(output_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load LLM-generated outputs from directory structure.

    Args:
        output_dir: Base directory containing LLM outputs

    Returns:
        Dictionary mapping research questions to lists of transformations per seed
    """
    results = {}

    # find all seed_*.json files recursively
    json_files = glob.glob(str(output_dir / "**" / "seed_*.json"), recursive=True)

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # extract path components
        path = Path(json_file)
        corpus = next(p for p in path.parts if p in ['rj', 'ebg', 'lcmc'])
        rq = next(p for p in path.parts if p.startswith('RQ'))
        sub_q = path.parts[path.parts.index(rq) + 1]
        seed = path.stem.split('_')[1]

        # organize by research question and sub-question
        key = f"{corpus}/{rq}/{sub_q}"
        if key not in results:
            results[key] = {}

        # store transformations by seed
        results[key][seed] = [
            {
                'original': item['original'],
                'transformed': item['transformed'],
                'metadata': {
                    'seed': item['initial_seed'],
                    'actual_seed': item['actual_seed']
                }
            }
            for item in data if 'original' in item and 'transformed' in item
        ]

    return results


def evaluate_transformations(
    predictor: Any,
    original_texts: List[str],
    transformed_texts: List[str],
    true_labels: np.ndarray
) -> Dict:
    """
    Evaluate original and transformed texts using both attribution and quality metrics.

    Args:
        predictor: Model predictor instance
        original_texts: Original texts
        transformed_texts: Transformed texts
        true_labels: True author labels

    Returns:
        Dictionary with combined metrics following the documented structure
    """
    # get attribution predictions
    logger.info("Getting predictions for original texts...")
    original_preds = predictor.predict_proba(original_texts)

    logger.info("Getting predictions for transformed texts...")
    transformed_preds = predictor.predict_proba(transformed_texts)

    # get comprehensive attribution metrics
    attribution_metrics = evaluate_attribution_defense(
        true_labels,
        original_preds,
        transformed_preds
    )

    # calculate text quality metrics
    logger.info("Computing text quality metrics...")
    quality_metrics = evaluate_quality(
        candidate_texts=transformed_texts,
        reference_texts=original_texts,
        metrics=['pinc', 'bleu', 'meteor', 'bertscore']
    )

    # organize results according to documented structure
    results = {
        'metrics': {
            'attribution': {
                'original': attribution_metrics['original_metrics'],
                'transformed': attribution_metrics['transformed_metrics'],
                'effectiveness': attribution_metrics['effectiveness']
            },
            'quality': quality_metrics
        }
    }

    return results


def format_metrics_for_logging(metrics: Dict) -> str:
    """Format metrics dictionary into a readable string."""
    lines = []

    # attribution metrics
    attribution = metrics['metrics']['attribution']
    lines.append("\nAttribution Performance:")
    
    # original metrics
    orig = attribution['original']
    lines.append("Original:")
    lines.append(f"  Accuracy: {orig['accuracy']:.4f}")
    lines.append(f"  MRR: {orig['mrr']:.4f}")
    lines.append(f"  MAP: {orig['map']:.4f}")
    
    # transformed metrics
    trans = attribution['transformed']
    lines.append("\nTransformed:")
    lines.append(f"  Accuracy: {trans['accuracy']:.4f}")
    lines.append(f"  MRR: {trans['mrr']:.4f}")
    lines.append(f"  MAP: {trans['map']:.4f}")
    
    # effectiveness metrics
    eff = attribution['effectiveness']
    lines.append("\nEffectiveness:")
    lines.append(f"  MRR Change: {eff['mrr_change']:.4f}")
    lines.append(f"  MAP Change: {eff['map_change']:.4f}")
    lines.append(f"  Entropy Increase: {eff['entropy_increase']:.4f}")

    # quality metrics
    quality = metrics['metrics']['quality']
    lines.append("\nText Quality:")
    lines.append(f"  BLEU: {quality['bleu']['bleu']:.4f}")
    lines.append(f"  METEOR: {quality['meteor']['meteor_avg']:.4f}")
    lines.append(f"  BERTScore F1: {quality['bertscore']['bertscore_f1_avg']:.4f}")
    lines.append(f"  PINC: {quality['pinc']['pinc_overall_avg']:.4f}")

    return "\n".join(lines)


def save_results(
    metrics: Dict,
    original_texts: List[str],
    transformed_texts: List[str],
    output_dir: Path,
    experiment_path: str,
    seed: str
):
    """
    Save evaluation results following documented structure.

    Args:
        metrics: Evaluation metrics
        original_texts: Original texts
        transformed_texts: Transformed texts
        output_dir: Base output directory
        experiment_path: Path components (corpus/RQ/sub_q)
        seed: Seed used for generation
    """
    save_dir = output_dir / experiment_path
    save_dir.mkdir(parents=True, exist_ok=True)

    # structure the output data
    output_data = {
        'metrics': {
            'attribution': metrics['metrics']['attribution'],
            'quality': {
                'pinc': metrics['metrics']['quality']['pinc'],
                'bleu': metrics['metrics']['quality']['bleu'],
                'meteor': metrics['metrics']['quality']['meteor'],
                'bertscore': metrics['metrics']['quality']['bertscore']
            }
        },
        'texts': {
            'original': original_texts,
            'transformed': transformed_texts
        }
    }

    # save results
    with open(save_dir / f"seed_{seed}.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults for {experiment_path} (seed {seed}):")
    logger.info(format_metrics_for_logging(metrics))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated adversarial examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--llm_outputs",
        type=Path,
        required=True,
        help="Directory containing LLM-generated outputs"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['logreg', 'svm', 'roberta'],
        help="Type of attribution model to use"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("defense_results"),
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # load LLM outputs
    logger.info(f"Loading LLM outputs from {args.llm_outputs}")
    outputs_by_experiment = load_llm_outputs(args.llm_outputs)

    # create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # evaluate each experiment
    for exp_path, seed_outputs in outputs_by_experiment.items():
        logger.info(f"\nEvaluating experiment: {exp_path}")
        
        # get corpus from path
        corpus = exp_path.split('/')[0]
        
        # load model once per corpus
        predictor = ModelLoader.load_predictor(corpus, args.model_type)

        # evaluate each seed's outputs
        for seed, transformations in seed_outputs.items():
            logger.info(f"Processing seed {seed}")
            
            # prepare texts and labels
            originals = [t['original'] for t in transformations]
            transformed = [t['transformed'] for t in transformations]
            true_labels = np.arange(len(originals))  # indices as labels
            
            # run evaluation
            metrics = evaluate_transformations(
                predictor,
                originals,
                transformed,
                true_labels
            )
            
            # save results
            save_results(
                metrics,
                originals,
                transformed,
                args.output_dir,
                exp_path,
                seed
            )


if __name__ == "__main__":
    main()

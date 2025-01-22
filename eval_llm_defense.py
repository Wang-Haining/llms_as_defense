"""
Evaluate effectiveness of LLM-based defense against authorship attribution models.

This script:
1. Loads original test samples and LLM-transformed texts
2. Evaluates both using trained attribution models
3. Computes comprehensive metrics including:
   - Attribution performance (accuracy, MRR, MAP, etc.)
   - Defense effectiveness (ranking changes, entropy increases)
   - Text quality (BLEU, METEOR, BERTScore)
4. Saves detailed results for analysis

Directory Structure:
defense_evaluation/                          # or specified output_dir
├── {corpus}/                               # rj, ebg, or lcmc
│   ├── RQ{N}/                             # main research question (e.g., RQ1)
│   │   ├── RQ{N}.{M}/                     # sub-question (e.g., RQ1.1)
│   │   │   ├── {model_name}/              # e.g., gemma-2b-it
│   │   │   │   └── evaluation.json        # evaluation results containing:
│   │   │   │       ├── logreg/           # results for LogReg model
│   │   │   │       │   ├── attribution/  # attribution metrics
│   │   │   │       │   │   ├── original/ # metrics on original texts
│   │   │   │       │   │   ├── transformed/ # metrics on transformed texts
│   │   │   │       │   │   └── effectiveness/ # defense effectiveness
│   │   │   │       │   └── quality/      # text quality metrics
│   │   │   │       ├── svm/             # similar structure for SVM
│   │   │   │       └── roberta/         # similar structure for RoBERTa
│   │   │   └── {another_model}/
│   │   └── RQ{N}.{M+1}/
│   └── RQ{N+1}/
└── {another_corpus}/

Usage:
    python eval_llm_defense.py # run all the experiments
    python eval_llm_defense.py --corpus rj  --rq rq1.1 --model "google/gemma-2-9b-it"

The evaluation.json files contain:
1. Attribution Metrics:
   - Accuracy, MRR, MAP for both original and transformed texts
   - Entropy, confidence gaps, and ranking metrics
   - Defense effectiveness measures (e.g., MRR change, entropy increase)
2. Text Quality Metrics:
   - PINC: n-gram novelty scores
   - BLEU: overlap with original texts
   - METEOR: semantic similarity with synonyms
   - BERTScore: contextual semantic similarity
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle

from eval_text_quality import evaluate_quality
from roberta import RobertaPredictor
from utils import (CORPORA, LLMS, RQS, LogisticRegressionPredictor,
                   SVMPredictor, _calculate_metrics, load_corpus, defense_effectiveness)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefenseEvaluator:
    """evaluates LLM defense effectiveness against attribution models."""

    def __init__(
        self,
        results_dir: Path,
        llm_outputs_dir: Path,
        output_dir: Path
    ):
        """initialize evaluator with paths to required data.

        Args:
            results_dir: directory containing trained models
            llm_outputs_dir: directory containing LLM transformations
            output_dir: directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.llm_outputs_dir = Path(llm_outputs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # map of model types to predictor classes
        self.predictor_classes = {
            'logreg': LogisticRegressionPredictor,
            'svm': SVMPredictor,
            'roberta': RobertaPredictor
        }

    def _load_predictor(
        self,
        corpus: str,
        model_type: str
    ) -> object:
        """load trained model predictor for given corpus and type."""
        model_dir = (
            self.results_dir / corpus / "no_protection" /
            model_type / "model"
        )
        return self.predictor_classes[model_type](model_dir)

    def _get_experiment_paths(self, corpus: str, rq: str, model_name: str) -> Path:
        """get experiment directory based on RQ identifier and model name.

        Args:
            corpus: corpus name (rj/ebg/lcmc)
            rq: research question identifier (e.g. rq1.1_basic_paraphrase)
            model_name: full model name (e.g. google/gemma-2b-it)

        Returns:
            Path to experiment directory
        """
        # extract main RQ category (e.g. rq1)
        rq_base = rq.split('_')[0]  # get rq1.1 part
        rq_main = rq_base.split('.')[0]  # get rq1

        # get model directory name (last part of model path)
        model_dir = model_name.split('/')[-1].lower()

        expected_path = self.llm_outputs_dir / corpus / rq_main / rq / model_dir
        logger.info(f"Constructed path: {expected_path}")

        return expected_path

    def _load_llm_outputs(
            self,
            corpus: str,
            rq: str,
            model_name: str
    ) -> List[Dict]:
        """load LLM-generated transformations for an experiment."""
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)
        logger.info(f"Looking for transformations in: {exp_dir}")

        transformations = []
        seed_files = list(exp_dir.glob('seed_*.json'))
        logger.info(f"Found {len(seed_files)} seed files")

        for seed_file in seed_files:
            logger.info(f"Loading transformations from: {seed_file}")
            with open(seed_file) as f:
                results = json.load(f)
                logger.info(f"Loaded {len(results)} results from seed file")
                if isinstance(results, list):
                    transformations.extend(results)
                else:
                    # handle case where results might be nested differently
                    if 'all_runs' in results:
                        for run in results['all_runs']:
                            if 'transformations' in run:
                                transformations.extend(run['transformations'])
                                logger.info(
                                    f"Added {len(run['transformations'])} transformations from run")

        logger.info(f"Total transformations loaded: {len(transformations)}")

        if not transformations:
            logger.warning(f"No transformations found in {exp_dir}")

        return transformations

    def evaluate_experiment(
            self,
            corpus: str,
            rq: str,
            model_name: str
    ) -> Dict[str, Dict]:
        """
        Evaluate LLM-based defense against attribution models for a specific experiment.

        Returns:
            A dictionary of the form:
                {
                    seed_1: {
                        "logreg": { ... },
                        "svm": { ... },
                        "roberta": { ... }
                    },
                    seed_2: {
                        ...
                    },
                    ...
                }
            where each attribution model includes attribution + quality metrics.
        """
        logger.info(f"Evaluating {corpus}-{rq} using {model_name}")

        # 1. load original test data
        _, _, test_texts, test_labels = load_corpus(
            corpus=corpus,
            task="no_protection"
        )
        logger.info(f"Loaded {len(test_texts)} original test texts")

        # 2. load LLM transformations
        llm_outputs = self._load_llm_outputs(corpus, rq, model_name)

        # 3. group transformations by seed
        transformations_by_seed = {}
        for output in llm_outputs:
            if isinstance(output, dict) and 'transformed' in output:
                seed = output.get('actual_seed', output.get('initial_seed'))
                if seed not in transformations_by_seed:
                    transformations_by_seed[seed] = []
                transformations_by_seed[seed].append(output['transformed'])

        logger.info(f"Loaded {len(transformations_by_seed)} seeds with transformations")

        # 4. prepare output directory (mirroring input structure)
        output_base = (
                self.output_dir
                / corpus
                / rq.split('_')[0]
                / rq
                / model_name.split('/')[-1].lower()
        )
        output_base.mkdir(parents=True, exist_ok=True)

        # collect all seeds' results in a single dict
        all_seed_results = {}

        # 5. evaluate each seed
        for seed, transformed_texts in transformations_by_seed.items():
            if len(transformed_texts) != len(test_texts):
                logger.warning(
                    f"Seed {seed} has {len(transformed_texts)} transformations "
                    f"but expected {len(test_texts)}. Skipping."
                )
                continue

            seed_results = {}  # per-model results for this seed

            for model_type in ['logreg', 'svm', 'roberta']:
                logger.info(f"Evaluating seed {seed} against {model_type}")

                # 5a. load the trained predictor (LogReg, SVM, or RoBERTa)
                predictor = self._load_predictor(corpus, model_type)

                # 5b. get predictions
                orig_preds = predictor.predict_proba(test_texts)
                trans_preds = predictor.predict_proba(transformed_texts)

                # 5c. calculate attribution metrics
                original_metrics = _calculate_metrics(test_labels, orig_preds)
                transformed_metrics = _calculate_metrics(test_labels, trans_preds)

                # 5d. measure effectiveness
                effectiveness = defense_effectiveness(
                    original_metrics,
                    transformed_metrics
                )

                # 5e. evaluate text quality
                quality_metrics = evaluate_quality(
                    candidate_texts=transformed_texts,
                    reference_texts=test_texts,
                    metrics=['pinc', 'bleu', 'meteor', 'bertscore']
                )

                # 5f. store results
                seed_results[model_type] = {
                    'attribution': {
                        'original_metrics': original_metrics,
                        'transformed_metrics': transformed_metrics,
                        'effectiveness': effectiveness
                    },
                    'quality': quality_metrics
                }

            # 6. save *this seed's* results to a pickle file
            output_file = output_base / f"evaluation_seed_{seed}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(seed_results, f)

            logger.info(f"Saved evaluation results to {output_file}")

            # 7. add to our big dictionary
            all_seed_results[seed] = seed_results

        # 8. return all seeds’ results so that save_results (or other code) can use them
        return all_seed_results

    def save_results(
        self,
        results: Dict,
        corpus: str,
        rq: str,
        model_name: str
    ) -> None:
        """save evaluation results following consistent structure."""
        # get experiment directory structure
        exp_dir = self._get_experiment_paths(corpus, rq, model_name)
        save_dir = self.output_dir / exp_dir.relative_to(self.llm_outputs_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file = save_dir / "evaluation.npz"
        # storing one dictionary in npz requires allow_pickle=True on load
        np.savez_compressed(output_file, results=results)

        # log summary metrics
        logger.info(f"\nResults for {corpus}-{rq} using {model_name}:")
        for model_type, metrics in results.items():
            logger.info(f"\n{model_type.upper()} Attribution Results:")

            # original performance
            orig = metrics['attribution']['original_metrics']
            logger.info(f"Original Accuracy: {orig['accuracy']:.4f}")
            logger.info(f"Original MRR: {orig['mrr']:.4f}")

            # defense effectiveness
            eff = metrics['attribution']['effectiveness']
            logger.info(f"MRR Change: {eff['mrr_change']:.4f}")
            logger.info(f"Entropy Increase: {eff['entropy_increase']:.4f}")

            # text quality
            qual = metrics['quality']
            logger.info(f"BLEU Score: {qual['bleu']['bleu']:.4f}")
            logger.info(f"METEOR Score: {qual['meteor']['meteor_avg']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM defense effectiveness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results'),
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--llm_outputs',
        type=Path,
        default=Path('llm_outputs'),
        help='Directory containing LLM outputs'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('defense_evaluation'),
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['rj', 'ebg', 'lcmc'],
        help='Specific corpus to evaluate (default: all)'
    )
    parser.add_argument(
        '--rq',
        type=str,
        help='Research question identifier (e.g. rq1.1_basic_paraphrase)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Full model name (e.g. google/gemma-2b-it)'
    )

    args = parser.parse_args()

    # initialize evaluator
    evaluator = DefenseEvaluator(
        results_dir=args.results_dir,
        llm_outputs_dir=args.llm_outputs,
        output_dir=args.output_dir
    )

    # determine what to evaluate
    corpora = [args.corpus] if args.corpus else CORPORA
    rqs = [args.rq] if args.rq else RQS
    models = [args.model] if args.model else LLMS

    # run evaluation
    for corpus in corpora:
        for rq in rqs:
            for model in models:
                try:
                    results = evaluator.evaluate_experiment(
                        corpus=corpus,
                        rq=rq,
                        model_name=model
                    )
                    evaluator.save_results(
                        results=results,
                        corpus=corpus,
                        rq=rq,
                        model_name=model
                        )
                except Exception as e:
                    logger.error(
                        f"Error evaluating {corpus}-{rq} "
                        f"with {model}: {str(e)}"
                    )
                    continue


if __name__ == "__main__":
    main()

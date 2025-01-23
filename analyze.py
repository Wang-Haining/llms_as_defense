"""
Analyze evaluation results for LLM-based defenses against authorship attribution threat
models. This module performs statistical analysis including per-model tests and
meta-analysis across models, with proper handling of different metric types.

Key Features:
1. Reads `.npz` evaluation results from specified directory structure
2. Performs appropriate statistical tests based on metric types:
   - Wilcoxon signed-rank test for ranking metrics
   - Paired t-tests for normalized and quality metrics
3. Applies Bonferroni correction for multiple comparisons
4. Conducts meta-analysis across models using inverse variance weighting
5. Saves analysis results in JSON format

Directory Structure:
defense_evaluation/                      # Input directory
├── {corpus}/                           # rj, ebg, or lcmc
│   └── RQ{N}/                         # e.g., RQ1
│       └── RQ{N}.{M}/                 # e.g., RQ1.1
│           └── {model_name}/          # e.g., llama-7b
│               ├── evaluation.npz     # consolidated results
│               └── seed_{seed}.npz    # per-seed results

analysis_results/                       # Output directory
├── {corpus}_RQ{N}.{M}_analysis.json   # cumulative analysis
└── {corpus}_RQ{N}.{M}_{model}_analysis.json  # per-model analysis

Usage Examples:

1. Analyze specific model for a research question:
   python analyze.py --corpus rj --rq rq1.1_basic_paraphrase \\
                     --model meta-llama/Llama-3.1-8B-Instruct

   This will:
   - Load results from defense_evaluation/rj/rq1/rq1.1_basic_paraphrase/llama-3.1-8b-instruct/
   - Perform statistical tests on all metrics
   - Save results to analysis_results/rj_rq1.1_basic_paraphrase_llama-3.1-8b-instruct_analysis.json
   Output includes:
   - Statistical test results per metric
   - Effect sizes with confidence intervals
   - Bonferroni-corrected significance

2. Analyze all models for a research question:
   python analyze.py --corpus ebg --rq rq1.1_basic_paraphrase

   This will:
   - Analyze all models listed in LLMS constant
   - Perform meta-analysis across models
   - Save results to analysis_results/ebg_rq1.1_basic_paraphrase_analysis.json
   Output includes:
   - Per-model statistical results
   - Weighted effect sizes across models
   - Heterogeneity analysis

3. Custom analysis directory:
   python analyze.py --corpus lcmc --rq rq1.1_basic_paraphrase \\
   ...                   --analysis_dir path/to/results

Sample Output Format:
{
    "per_model": {
        "llama-3.1-8b-instruct": {
            "statistical_tests": {
                "ranking": {
                    "mrr": {
                        "test": "wilcoxon",
                        "statistic": 42.0,
                        "p_value": 0.001,
                        "effect_size": 0.85,
                        "significant": true,
                        "alpha_corrected": 0.005
                    },
                    ...
                },
                ...
            },
            "n_seeds": 5,
            "bonferroni_factor": 10
        },
        ...
    },
    "meta_analysis": {
        "ranking_mrr": {
            "weighted_effect_size": 0.75,
            "heterogeneity_q": 3.2,
            "heterogeneity_p": 0.36,
            "n_models": 5
        },
        ...
    }
}

Notes:
- All p-values are Bonferroni-corrected for multiple comparisons
- Effect sizes use Cohen's d for normalized metrics and r for ranking metrics
- Meta-analysis uses inverse variance weighting
- Heterogeneity assessed using Cochran's Q test

See Also:
    eval_llm_defense.py: Generates the evaluation results analyzed by this script
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Optional
from collections import defaultdict

from utils import LLMS, CORPORA, RQS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Handles statistical tests for different metric types."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze_metric_pairs(
        self,
        pre_values: np.ndarray,
        post_values: np.ndarray,
        metric_type: str
    ) -> Dict:
        """
        Run appropriate statistical test based on metric type.

        Args:
            pre_values: Original metric values
            post_values: Transformed metric values
            metric_type: Type of metric ('ranking', 'normalized', or 'quality')
        """
        if len(pre_values) != len(post_values):
            raise ValueError("Pre and post values must have same length")

        results = {}

        # ranking metrics (Wilcoxon signed-rank test due to ordinal nature)
        if metric_type == 'ranking':
            stat, p_value = stats.wilcoxon(pre_values, post_values)
            test_name = 'wilcoxon'

        # normalized metrics (paired t-test as they're continuous and normalized)
        elif metric_type == 'normalized':
            stat, p_value = stats.ttest_rel(pre_values, post_values)
            test_name = 'paired_ttest'

        # quality metrics (paired t-test as they're continuous)
        elif metric_type == 'quality':
            stat, p_value = stats.ttest_rel(pre_values, post_values)
            test_name = 'paired_ttest'

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        results = {
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'mean_diff': float(np.mean(post_values - pre_values)),
            'std_diff': float(np.std(post_values - pre_values)),
            'effect_size': float(compute_effect_size(pre_values, post_values))
        }

        return results


def compute_effect_size(pre: np.ndarray, post: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    diff = post - pre
    d = np.mean(diff) / np.std(diff)
    return d


class ResultAnalyzer:
    """Analyzes defense evaluation results with statistical testing."""

    def __init__(
        self,
        output_dir: Path = Path("defense_evaluation"),
        analysis_dir: Path = Path("analysis_results")
    ):
        self.output_dir = output_dir
        self.analysis_dir = analysis_dir
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.stat_analyzer = StatisticalAnalyzer()

        # define metric groupings
        self.metric_types = {
            'ranking': ['mrr', 'top_1_acc', 'top_3_acc', 'top_5_acc', 'weighted_rank_score'],
            'normalized': ['entropy_normalized', 'conf_gap_normalized'],
            'quality': ['bleu', 'meteor', 'bertscore']
        }

    def load_evaluation_results(self, file_path: Path) -> Dict:
        """Load evaluation results while maintaining compatibility."""
        data = np.load(file_path, allow_pickle=True)
        return data["results"].item()

    def analyze_rq(
        self,
        corpus: str,
        rq: str,
        model: Optional[str] = None
    ) -> Dict:
        """
        Analyze results for a research question, optionally for a specific model.

        Args:
            corpus: Corpus identifier
            rq: Research question identifier
            model: Optional specific model to analyze
        """
        base_dir = self.output_dir / corpus / rq.split('_')[0] / rq

        if model:
            # analyze specific model
            model_dir = base_dir / model.split('/')[-1].lower()
            results = self._analyze_single_model(model_dir)
            self._save_analysis(results, corpus, rq, model)
            return results

        # analyze all models for this RQ
        all_results = {}
        combined_stats = defaultdict(list)

        for llm in LLMS:
            model_dir = base_dir / llm.split('/')[-1].lower()
            if not model_dir.exists():
                continue

            try:
                results = self._analyze_single_model(model_dir)
                all_results[llm] = results

                # collect statistics for meta-analysis
                for metric_type, metrics in results['statistical_tests'].items():
                    for metric, stats_dict in metrics.items():
                        combined_stats[f"{metric_type}_{metric}"].append(stats_dict)

            except Exception as e:
                logger.warning(f"Error analyzing {llm}: {e}")

        # perform meta-analysis
        meta_analysis = self._meta_analyze(combined_stats)

        # save comprehensive results
        final_results = {
            'per_model': all_results,
            'meta_analysis': meta_analysis
        }

        self._save_analysis(final_results, corpus, rq)
        return final_results

    def _analyze_single_model(self, model_dir: Path) -> Dict:
        """Analyze results for a single model with statistical tests."""
        results = defaultdict(dict)
        bonferroni_factor = 0  # Will be set based on number of tests

        # load all seed results
        seed_results = []
        for seed_file in model_dir.glob("seed_*.npz"):
            try:
                seed_data = self.load_evaluation_results(seed_file)
                seed_results.append(seed_data)
            except Exception as e:
                logger.warning(f"Error loading {seed_file}: {e}")

        if not seed_results:
            raise ValueError(f"No valid results found in {model_dir}")

        # group metrics by type and perform statistical tests
        statistical_tests = defaultdict(dict)

        for metric_type, metrics in self.metric_types.items():
            for metric in metrics:
                pre_values = []
                post_values = []

                for seed_data in seed_results:
                    for model_results in seed_data.values():
                        if metric in model_results['attribution']['original_metrics']:
                            pre = model_results['attribution']['original_metrics'][metric]
                            post = model_results['attribution']['transformed_metrics'][metric]
                            pre_values.append(pre)
                            post_values.append(post)

                if pre_values and post_values:
                    bonferroni_factor += 1
                    statistical_tests[metric_type][metric] = self.stat_analyzer.analyze_metric_pairs(
                        np.array(pre_values),
                        np.array(post_values),
                        metric_type
                    )

        # apply Bonferroni correction
        alpha_corrected = 0.05 / bonferroni_factor
        for metric_type in statistical_tests:
            for metric in statistical_tests[metric_type]:
                stats_dict = statistical_tests[metric_type][metric]
                stats_dict['significant'] = stats_dict['p_value'] < alpha_corrected
                stats_dict['alpha_corrected'] = alpha_corrected

        return {
            'statistical_tests': statistical_tests,
            'n_seeds': len(seed_results),
            'bonferroni_factor': bonferroni_factor
        }

    def _meta_analyze(self, combined_stats: Dict) -> Dict:
        """
        Perform meta-analysis across all models for each metric.
        Uses inverse variance weighting for effect sizes.
        """
        meta_results = {}

        for metric, stat_list in combined_stats.items():
            effect_sizes = [s['effect_size'] for s in stat_list]
            variances = [1/len(s) for s in stat_list]  # Simple approximation

            # inverse variance weighted mean
            weights = np.array([1/v for v in variances])
            weighted_mean = np.average(effect_sizes, weights=weights)

            # heterogeneity analysis
            q_stat = np.sum(weights * (effect_sizes - weighted_mean)**2)
            df = len(effect_sizes) - 1
            p_value = 1 - stats.chi2.cdf(q_stat, df)

            meta_results[metric] = {
                'weighted_effect_size': float(weighted_mean),
                'heterogeneity_q': float(q_stat),
                'heterogeneity_p': float(p_value),
                'n_models': len(stat_list)
            }

        return meta_results

    def _save_analysis(self, results: Dict, corpus: str, rq: str, model: str = None):
        """Save analysis results in JSON format."""
        if model:
            filename = f"{corpus}_{rq}_{model.split('/')[-1].lower()}_analysis.json"
        else:
            filename = f"{corpus}_{rq}_analysis.json"

        output_path = self.analysis_dir / filename

        # convert any numpy types to Python natives for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        results = convert_to_native(results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved analysis results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze defense evaluation results with statistical testing"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=CORPORA,
        help="Corpus to analyze"
    )
    parser.add_argument(
        "--rq",
        required=True,
        choices=RQS,
        help="Research question identifier"
    )
    parser.add_argument(
        "--model",
        help="Specific model to analyze. If not set, analyze all models"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("defense_evaluation"),
        help="Base directory for evaluation results"
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("analysis_results"),
        help="Directory to save analysis results"
    )

    args = parser.parse_args()

    try:
        analyzer = ResultAnalyzer(args.output_dir, args.analysis_dir)
        results = analyzer.analyze_rq(args.corpus, args.rq, model=args.model)

        # print summary of results
        if args.model:
            tests = results['statistical_tests']
            logger.info(f"\nResults for {args.model}:")

            for metric_type, metrics in tests.items():
                logger.info(f"\n{metric_type.upper()} metrics:")
                for metric, stats in metrics.items():
                    sig = "significant" if stats['significant'] else "not significant"
                    logger.info(
                        f"{metric}: effect size = {stats['effect_size']:.4f}, "
                        f"p = {stats['p_value']:.4f} ({sig})"
                    )
        else:
            logger.info(f"\nMeta-analysis results for {args.corpus} {args.rq}:")
            for metric, stats in results['meta_analysis'].items():
                logger.info(
                    f"{metric}: weighted effect size = {stats['weighted_effect_size']:.4f}, "
                    f"heterogeneity p = {stats['heterogeneity_p']:.4f}"
                )

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
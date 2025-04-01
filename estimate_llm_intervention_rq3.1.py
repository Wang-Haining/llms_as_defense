"""
Compare two defense approaches using Bayesian analysis.

This script compares two different defense approaches (e.g., RQ2.1 obfuscation vs. RQ3.1
persona-enhanced obfuscation) to determine if one approach provides measurable
improvements over the other in terms of effectiveness, robustness, or text quality.

Usage:
    python estimate_llm_intervention_rq3.1.py --baseline rq2.1_obfuscation --enhanced rq3.1_obfuscation_w_persona
    python estimate_llm_intervention_rq3.1.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class DifferenceResult(NamedTuple):
    """Results of comparing two defense approaches."""
    baseline_mean: float
    enhanced_mean: float
    difference_mean: float
    difference_std: float
    difference_ci_lower: float
    difference_ci_upper: float
    conclusion: str  # overall conclusion about effectiveness


def load_defense_data(base_dir: str, corpus: str, baseline_rq: str,
                      enhanced_rq: str, threat_model: str) -> Tuple[Dict, Dict]:
    """
    Load evaluation data for baseline and enhanced defense approaches.

    Args:
        base_dir: Base directory containing evaluation results
        corpus: Corpus name (e.g., 'ebg', 'rj')
        baseline_rq: Research question for baseline approach (e.g., 'rq2.1_obfuscation')
        enhanced_rq: Research question for enhanced approach (e.g., 'rq3.1_obfuscation_w_persona')
        threat_model: Threat model to analyze (e.g., 'logreg', 'svm', 'roberta')

    Returns:
        Tuple of (baseline_data, enhanced_data) dictionaries
    """
    baseline_rq_main = f"rq{baseline_rq.split('_')[0].split('.')[0].lstrip('rq')}"
    enhanced_rq_main = f"rq{enhanced_rq.split('_')[0].split('.')[0].lstrip('rq')}"

    baseline_path = Path(base_dir) / corpus / baseline_rq_main / baseline_rq
    enhanced_path = Path(base_dir) / corpus / enhanced_rq_main / enhanced_rq

    baseline_data = {}
    enhanced_data = {}

    # load baseline data
    for model_dir in baseline_path.glob("*"):
        if not model_dir.is_dir():
            continue

        eval_file = model_dir / "evaluation.json"
        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            results = json.load(f)

        model_name = model_dir.name.lower()
        baseline_data[model_name] = {
            seed: data.get(threat_model, {})
            for seed, data in results.items()
        }

    # load enhanced data
    for model_dir in enhanced_path.glob("*"):
        if not model_dir.is_dir():
            continue

        eval_file = model_dir / "evaluation.json"
        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            results = json.load(f)

        model_name = model_dir.name.lower()
        enhanced_data[model_name] = {
            seed: data.get(threat_model, {})
            for seed, data in results.items()
        }

    return baseline_data, enhanced_data


def extract_paired_observations(baseline_data: Dict, enhanced_data: Dict,
                                metric: str, metric_type: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract paired observations for a specified metric across all models.

    Args:
        baseline_data: Data from baseline defense approach
        enhanced_data: Data from enhanced defense approach
        metric: Metric name to extract (e.g., 'accuracy@1', 'entropy', 'pinc')
        metric_type: Type of metric ('attribution' or 'quality')

    Returns:
        Dictionary mapping model names to lists of paired observations (baseline, enhanced)
    """
    paired_observations = {}

    # For each model present in both datasets
    for model_name in set(baseline_data.keys()) & set(enhanced_data.keys()):
        model_pairs = []

        # Handle run-level metrics from attribution data (except entropy)
        if metric_type == 'attribution' and metric != 'entropy':
            for seed in set(baseline_data[model_name].keys()) & set(enhanced_data[model_name].keys()):
                if 'attribution' in baseline_data[model_name][seed] and 'attribution' in enhanced_data[model_name][seed]:
                    baseline_val = baseline_data[model_name][seed]['attribution']['post'].get(metric)
                    enhanced_val = enhanced_data[model_name][seed]['attribution']['post'].get(metric)

                    if baseline_val is not None and enhanced_val is not None:
                        model_pairs.append((baseline_val, enhanced_val))

        # Handle entropy (sample-level metric from attribution data)
        elif metric_type == 'attribution' and metric == 'entropy':
            for seed in set(baseline_data[model_name].keys()) & set(enhanced_data[model_name].keys()):
                if ('attribution' in baseline_data[model_name][seed] and
                        'attribution' in enhanced_data[model_name][seed] and
                        'raw_predictions' in baseline_data[model_name][seed]['attribution'] and
                        'raw_predictions' in enhanced_data[model_name][seed]['attribution']):

                    baseline_preds = baseline_data[model_name][seed]['attribution']['raw_predictions']['transformed']
                    enhanced_preds = enhanced_data[model_name][seed]['attribution']['raw_predictions']['transformed']

                    if len(baseline_preds) == len(enhanced_preds):
                        for b_pred, e_pred in zip(baseline_preds, enhanced_preds):
                            b_pred = np.array(b_pred)
                            e_pred = np.array(e_pred)

                            b_entropy = -np.sum(b_pred * np.log2(b_pred + 1e-10))
                            e_entropy = -np.sum(e_pred * np.log2(e_pred + 1e-10))

                            model_pairs.append((b_entropy, e_entropy))

        # Handle quality metrics
        elif metric_type == 'quality':
            if metric == 'pinc':
                for seed in set(baseline_data[model_name].keys()) & set(enhanced_data[model_name].keys()):
                    if ('quality' in baseline_data[model_name][seed] and
                            'quality' in enhanced_data[model_name][seed] and
                            'pinc' in baseline_data[model_name][seed]['quality'] and
                            'pinc' in enhanced_data[model_name][seed]['quality']):

                        baseline_pinc = baseline_data[model_name][seed]['quality']['pinc']
                        enhanced_pinc = enhanced_data[model_name][seed]['quality']['pinc']

                        n_samples = min(
                            len(baseline_pinc.get("pinc_1_scores", [])),
                            len(enhanced_pinc.get("pinc_1_scores", []))
                        )

                        if n_samples > 0:
                            for i in range(n_samples):
                                baseline_scores = []
                                enhanced_scores = []

                                for k in range(1, 5):
                                    b_scores_key = f"pinc_{k}_scores"
                                    e_scores_key = f"pinc_{k}_scores"

                                    if (b_scores_key in baseline_pinc and
                                            e_scores_key in enhanced_pinc and
                                            i < len(baseline_pinc[b_scores_key]) and
                                            i < len(enhanced_pinc[e_scores_key])):
                                        baseline_scores.append(baseline_pinc[b_scores_key][i])
                                        enhanced_scores.append(enhanced_pinc[e_scores_key][i])

                                if baseline_scores and enhanced_scores:
                                    avg_baseline_pinc = np.mean(baseline_scores)
                                    avg_enhanced_pinc = np.mean(enhanced_scores)
                                    model_pairs.append((avg_baseline_pinc, avg_enhanced_pinc))

            elif metric == 'bertscore':
                for seed in set(baseline_data[model_name].keys()) & set(enhanced_data[model_name].keys()):
                    if ('quality' in baseline_data[model_name][seed] and
                            'quality' in enhanced_data[model_name][seed] and
                            'bertscore' in baseline_data[model_name][seed]['quality'] and
                            'bertscore' in enhanced_data[model_name][seed]['quality'] and
                            'bertscore_individual' in baseline_data[model_name][seed]['quality']['bertscore'] and
                            'bertscore_individual' in enhanced_data[model_name][seed]['quality']['bertscore']):
                        baseline_scores = [item['f1'] for item in
                                           baseline_data[model_name][seed]['quality']['bertscore']['bertscore_individual']]
                        enhanced_scores = [item['f1'] for item in
                                           enhanced_data[model_name][seed]['quality']['bertscore']['bertscore_individual']]

                        if len(baseline_scores) == len(enhanced_scores):
                            for b_score, e_score in zip(baseline_scores, enhanced_scores):
                                model_pairs.append((b_score, e_score))

        if model_pairs:
            paired_observations[model_name] = model_pairs

    return paired_observations


def analyze_difference(paired_samples: List[Tuple[float, float]], metric: str,
                       higher_is_better: bool) -> DifferenceResult:
    """
    Analyze difference between baseline and enhanced approach using Bayesian modeling.

    Args:
        paired_samples: List of paired (baseline, enhanced) observations
        metric: Metric name being analyzed
        higher_is_better: Whether higher values of the metric indicate better performance

    Returns:
        DifferenceResult with analysis statistics
    """
    baseline_values = np.array([pair[0] for pair in paired_samples])
    enhanced_values = np.array([pair[1] for pair in paired_samples])

    if metric == 'entropy':
        max_entropy = max(np.max(baseline_values), np.max(enhanced_values))
        log_base = np.log2(45) if max_entropy > 5.0 else np.log2(21)
        baseline_values /= log_base
        enhanced_values /= log_base

    raw_differences = enhanced_values - baseline_values
    differences = raw_differences if higher_is_better else -raw_differences

    min_diff = min(0, np.min(differences))
    max_diff = max(0, np.max(differences))
    range_diff = max_diff - min_diff

    if range_diff < 1e-6:
        if metric == 'entropy':
            baseline_mean = float(np.mean(baseline_values * log_base))
            enhanced_mean = float(np.mean(enhanced_values * log_base))
            difference_mean = float(np.mean(differences * log_base))
        else:
            baseline_mean = float(np.mean(baseline_values))
            enhanced_mean = float(np.mean(enhanced_values))
            difference_mean = float(np.mean(differences))

        return DifferenceResult(
            baseline_mean=baseline_mean,
            enhanced_mean=enhanced_mean,
            difference_mean=difference_mean,
            difference_std=0.0,
            difference_ci_lower=difference_mean,
            difference_ci_upper=difference_mean,
            conclusion="Practically Equivalent"
        )

    normalized_diffs = (differences - min_diff) / range_diff
    normalized_diffs = np.clip(normalized_diffs, 1e-6, 1 - 1e-6)

    n_obs = len(normalized_diffs)
    if n_obs == 5:
        prior_mean = float(np.median(normalized_diffs))
        alpha_prior = prior_mean * 2.0
        beta_prior = (1 - prior_mean) * 2.0
    else:
        alpha_prior, beta_prior = 1, 1

    with pm.Model() as model:
        mu = pm.Beta("mu", alpha=alpha_prior, beta=beta_prior)
        kappa = pm.HalfNormal("kappa", sigma=10)
        _ = pm.Beta("obs", alpha=mu * kappa, beta=(1 - mu) * kappa,
                    observed=normalized_diffs)
        trace = pm.sample(2000, tune=1000, chains=4, random_seed=42, cores=4,
                          return_inferencedata=True)

    mu_samples = trace.posterior["mu"].values.flatten()
    diff_samples = mu_samples * range_diff + min_diff

    if metric == 'entropy':
        baseline_mean = float(np.mean(baseline_values * log_base))
        enhanced_mean = float(np.mean(enhanced_values * log_base))
        diff_samples *= log_base
    else:
        baseline_mean = float(np.mean(baseline_values))
        enhanced_mean = float(np.mean(enhanced_values))

    diff_samples_original = diff_samples if higher_is_better else -diff_samples

    difference_mean = float(np.mean(diff_samples_original))
    difference_std = float(np.std(diff_samples_original))

    ci_lower, ci_upper = az.hdi(diff_samples_original)
    difference_ci_lower = float(ci_lower)
    difference_ci_upper = float(ci_upper)

    rope_width = 0.1 * difference_std
    if metric == 'entropy':
        rope_width *= log_base

    if -rope_width <= difference_ci_lower and difference_ci_upper <= rope_width:
        conclusion = "Practically Equivalent"
    elif difference_ci_lower > rope_width:
        conclusion = "Significant Increase (Enhanced > Baseline)" if higher_is_better else "Significant Decrease (Enhanced < Baseline)"
    elif difference_ci_upper < -rope_width:
        conclusion = "Significant Decrease (Enhanced < Baseline)" if higher_is_better else "Significant Increase (Enhanced > Baseline)"
    else:
        conclusion = "Inconclusive"

    return DifferenceResult(
        baseline_mean=baseline_mean,
        enhanced_mean=enhanced_mean,
        difference_mean=difference_mean,
        difference_std=difference_std,
        difference_ci_lower=difference_ci_lower,
        difference_ci_upper=difference_ci_upper,
        conclusion=conclusion
    )


def compare_defenses(
        base_dir: str,
        corpus: str,
        baseline_rq: str,
        enhanced_rq: str,
        output_dir: Optional[str] = None,
        threat_models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare two defense approaches and generate a summary report.

    Args:
        base_dir: Base directory containing evaluation results
        corpus: Corpus name (e.g., 'ebg', 'rj')
        baseline_rq: Research question for baseline approach (e.g., 'rq2.1_obfuscation')
        enhanced_rq: Research question for enhanced approach (e.g., 'rq3.1_obfuscation_w_persona')
        output_dir: Directory to save results (optional)
        threat_models: List of threat models to analyze (default: ['logreg', 'svm', 'roberta'])

    Returns:
        DataFrame with comparison results

    Raises:
        ValueError: If fewer than 3 paired observations are found for any model, metric, and threat model combination.
    """
    if threat_models is None:
        threat_models = ['logreg', 'svm', 'roberta']

    attribution_metrics = {
        'accuracy@1': False,
        'accuracy@5': False,
        'true_class_confidence': False,
        'entropy': True,
    }

    quality_metrics = {
        'pinc': True,
        'bertscore': True,
    }

    results = []

    for threat_model in threat_models:
        logging.info(f"Analyzing {corpus} - {threat_model}")

        baseline_data, enhanced_data = load_defense_data(
            base_dir, corpus, baseline_rq, enhanced_rq, threat_model
        )

        if not baseline_data or not enhanced_data:
            raise ValueError(f"No data found for {corpus} - {threat_model}")

        # Process attribution metrics
        for metric, higher_is_better in attribution_metrics.items():
            paired_observations = extract_paired_observations(
                baseline_data, enhanced_data, metric, 'attribution'
            )

            for model_name, pairs in paired_observations.items():
                if len(pairs) < 3:
                    raise ValueError(
                        f"Insufficient paired observations for model '{model_name}', metric '{metric}', threat model '{threat_model}'. "
                        f"Found {len(pairs)} samples; at least 3 are required."
                    )

                diff_result = analyze_difference(pairs, metric, higher_is_better)
                display_name = _get_model_display_name(model_name)

                results.append({
                    'Corpus': corpus.upper(),
                    'Threat Model': threat_model,
                    'Defense Model': display_name,
                    'Metric': metric,
                    'Higher is Better': higher_is_better,
                    'Metric Type': 'Effectiveness',
                    'Baseline Mean': diff_result.baseline_mean,
                    'Enhanced Mean': diff_result.enhanced_mean,
                    'Mean Difference': diff_result.difference_mean,
                    'Difference Std': diff_result.difference_std,
                    '95% HDI Lower': diff_result.difference_ci_lower,
                    '95% HDI Upper': diff_result.difference_ci_upper,
                    'Conclusion': diff_result.conclusion
                })

        # Process quality metrics
        for metric, higher_is_better in quality_metrics.items():
            paired_observations = extract_paired_observations(
                baseline_data, enhanced_data, metric, 'quality'
            )

            for model_name, pairs in paired_observations.items():
                if len(pairs) < 3:
                    raise ValueError(
                        f"Insufficient paired observations for model '{model_name}', metric '{metric}', threat model '{threat_model}' (quality). "
                        f"Found {len(pairs)} samples; at least 3 are required."
                    )

                diff_result = analyze_difference(pairs, metric, higher_is_better)
                display_name = _get_model_display_name(model_name)

                results.append({
                    'Corpus': corpus.upper(),
                    'Threat Model': threat_model,
                    'Defense Model': display_name,
                    'Metric': metric,
                    'Higher is Better': higher_is_better,
                    'Metric Type': 'Quality',
                    'Baseline Mean': diff_result.baseline_mean,
                    'Enhanced Mean': diff_result.enhanced_mean,
                    'Mean Difference': diff_result.difference_mean,
                    'Difference Std': diff_result.difference_std,
                    '95% HDI Lower': diff_result.difference_ci_lower,
                    '95% HDI Upper': diff_result.difference_ci_upper,
                    'Conclusion': diff_result.conclusion
                })

    df = pd.DataFrame(results)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{corpus}_{baseline_rq}_vs_{enhanced_rq}_comparison.csv"
        df.to_csv(output_path / filename, index=False)

        significant_df = df[df['Conclusion'].str.startswith('Significant')]
        if not significant_df.empty:
            significant_df.to_csv(output_path / f"{corpus}_significant_changes.csv", index=False)

    return df


def _get_model_display_name(model_dir_name: str) -> str:
    """Get standardized display name for model."""
    name = model_dir_name.lower()
    if 'llama' in name:
        return 'Llama-3.1'
    elif 'gemma' in name:
        return 'Gemma-2'
    elif 'ministral' in name:
        return 'Ministral'
    elif 'sonnet' in name:
        return 'Claude-3.5'
    elif 'gpt' in name:
        return 'GPT-4o'
    return model_dir_name


def main():
    """Main entry point for script."""
    import argparse
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Compare effectiveness of two defense approaches"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='defense_evaluation',
        help='Base directory containing evaluation results'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['ebg', 'rj', 'all'],
        default='all',
        help='Corpus to analyze'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='rq2.1_obfuscation',
        help='Research question for baseline approach'
    )
    parser.add_argument(
        '--enhanced',
        type=str,
        default='rq3.1_obfuscation_w_persona',
        help='Research question for enhanced approach'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/comparison_results_rq3.1_rq2.1',
        help='Output directory for results'
    )

    args = parser.parse_args()

    corpora = ['ebg', 'rj'] if args.corpus == 'all' else [args.corpus]

    for corpus in corpora:
        logging.info(f"Comparing {args.baseline} vs {args.enhanced} on {corpus}")
        results_df = compare_defenses(
            base_dir=args.base_dir,
            corpus=corpus,
            baseline_rq=args.baseline,
            enhanced_rq=args.enhanced,
            output_dir=args.output
        )

        increases = results_df[results_df['Conclusion'].str.contains('Increase')]
        if not increases.empty:
            logging.info(f"\nSignificant increases ({corpus}):")
            for _, row in increases.iterrows():
                logging.info(f"- {row['Defense Model']} | {row['Threat Model']} | {row['Metric']}: "
                             f"{row['Baseline Mean']:.4f} → {row['Enhanced Mean']:.4f} "
                             f"(diff: {row['Mean Difference']:.4f}, "
                             f"95% HDI: [{row['95% HDI Lower']:.4f}, {row['95% HDI Upper']:.4f}])")

        decreases = results_df[results_df['Conclusion'].str.contains('Decrease')]
        if not decreases.empty:
            logging.info(f"\nSignificant decreases ({corpus}):")
            for _, row in decreases.iterrows():
                logging.info(f"- {row['Defense Model']} | {row['Threat Model']} | {row['Metric']}: "
                             f"{row['Baseline Mean']:.4f} → {row['Enhanced Mean']:.4f} "
                             f"(diff: {row['Mean Difference']:.4f}, "
                             f"95% HDI: [{row['95% HDI Lower']:.4f}, {row['95% HDI Upper']:.4f}])")

        logging.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

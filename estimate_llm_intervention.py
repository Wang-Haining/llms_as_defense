"""
Defense evaluation script that performs Bayesian analysis comparing pre/post defense
performance across different metrics, corpora, and threat models.

This version has been updated to compute and document the standard deviation (std) of
the full posterior distribution for each metric. For accuracy and true class confidence,
run-level (aggregated) values are used, but for entropy and text quality metrics we now
load sample-level values from each seed file for a finer analysis.

Expected directory structure:
defense_evaluation/
├── {corpus}/                              # e.g., rj and ebg
│   ├── rq{N}/                            # main research question (e.g., rq1)
│   │   ├── rq{N}.{M}/                    # sub-question (e.g., rq1.1)
│   │   │   ├── {model_name}/             # e.g., gemma-2b-it
│   │   │   │   ├── evaluation.json       # consolidated results across seeds
│   │   │   │   └── seed_{seed}.json      # detailed per-seed results containing:
│   │   │   │           attribution:
│   │   │   │               pre:  aggregated pre-defense metrics
│   │   │   │               post: aggregated post-defense metrics
│   │   │   │               raw_predictions: raw probability arrays for original and transformed texts
│   │   │   │           quality: text quality metrics (aggregated)
│   │   │   │           example_metrics: example-level attribution data
│   │   │   │           example_quality: (NEW!) example-level quality metrics
│   │   │   └── {another_model}/
│   └── rq{N+1}/
└── {another_corpus}/

Results are saved as CSV files (pre, post, absolute change, and relative change)
in the specified output folder.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from tqdm import tqdm


# Updated BayesResult with standard deviation (std) included
class BayesResult(NamedTuple):
    pre_value: float            # baseline value (pre-intervention)
    post_mean: float            # posterior mean of the parameter
    std: float                  # standard deviation (spread) of the posterior
    ci_lower: float             # lower bound of the 95% credible interval
    ci_upper: float             # upper bound of the 95% credible interval


def estimate_beta_metric(post_values: List[float]) -> dict:
    """
    Estimate beta distribution parameters with 95% HDI intervals for metrics naturally bounded in [0,1],
    and compute the estimated standard deviation (std) of the full posterior of mu.
    """
    epsilon = 1e-6
    post_values = np.clip(np.array(post_values), epsilon, 1 - epsilon)

    with pm.Model() as model:
        mu = pm.Beta("mu", alpha=1, beta=1)
        kappa = pm.HalfNormal("kappa", sigma=10)
        _ = pm.Beta("obs",
                    alpha=mu * kappa,
                    beta=(1 - mu) * kappa,
                    observed=post_values)

        trace = pm.sample(
            2000,
            tune=1000,
            cores=4,
            progressbar=True,
            random_seed=42,
            target_accept=0.95,
            return_inferencedata=True
        )

    mu_samples = trace.posterior["mu"].values.flatten()
    mean_mu = float(np.mean(mu_samples))
    std_mu = float(np.std(mu_samples))
    hdi_bounds = az.hdi(mu_samples, hdi_prob=0.95)
    lower_bound = float(max(0, hdi_bounds[0]))
    upper_bound = float(min(1, hdi_bounds[1]))

    return {
        "mean": mean_mu,
        "std": std_mu,
        "hdi_lower": lower_bound,
        "hdi_upper": upper_bound
    }


class DefenseStats:
    """Container for defense evaluation Bayesian estimates for each metric."""

    def __init__(self, corpus: str, threat_model: str, defense_model: str):
        self.corpus = corpus
        self.threat_model = threat_model
        self.defense_model = defense_model
        self.effectiveness_estimates: Dict[str, BayesResult] = {}  # metric_name -> BayesResult
        self.quality_estimates: Dict[str, BayesResult] = {}        # metric_name -> BayesResult

    def add_estimate(self, metric_name: str, metric_type: str,
                     pre_value: float, post_values: List[float]) -> None:
        """
        Add Bayesian estimate for a metric based on post_values.
        For effectiveness metrics, the baseline is the run-level pre_value;
        for quality metrics, the baseline is 1 for BLEU/METEOR/BERTScore/SBERT and 0 for PINC.

        For the 'entropy' metric, the raw values are first normalized to [0,1]
        by dividing by the scaling factor (np.log2(21) for 'rj' and np.log2(45) for 'ebg'),
        then the beta model is run, and the resulting posterior estimates are scaled back.
        """
        if not post_values:
            return

        baseline = pre_value if metric_type == 'effectiveness' else (
            0.0 if metric_name == 'pinc' else 1.0)

        if metric_name == 'entropy':
            if self.corpus.lower() == 'rj':
                factor = np.log2(21)
            elif self.corpus.lower() == 'ebg':
                factor = np.log2(45)
            else:
                raise RuntimeError('Entropy normalization fails.')
            scaled_post_values = [v / factor for v in post_values]
            result = estimate_beta_metric(scaled_post_values)
            result = {
                "mean": result["mean"] * factor,
                "std": result["std"] * factor,
                "hdi_lower": result["hdi_lower"] * factor,
                "hdi_upper": result["hdi_upper"] * factor,
            }
        else:
            result = estimate_beta_metric(post_values)

        bayes_result = BayesResult(
            pre_value=baseline,
            post_mean=result["mean"],
            std=result["std"],
            ci_lower=result["hdi_lower"],
            ci_upper=result["hdi_upper"],
        )
        if metric_type == 'effectiveness':
            self.effectiveness_estimates[metric_name] = bayes_result
        else:
            self.quality_estimates[metric_name] = bayes_result

    def format_results(self, metric_name: str, metric_type: str) -> str:
        """Format test results for a specific metric."""
        tests = self.effectiveness_estimates if metric_type == 'effectiveness' else self.quality_estimates
        if metric_name not in tests:
            return '-'
        result = tests[metric_name]
        return format_estimate_with_hdi(
            result.post_mean,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper
        )

    def print_summary(self) -> None:
        """Print a summary of all Bayesian estimates."""
        print("\nDefense analysis summary")
        print("=" * 80)
        print(f"Corpus: {self.corpus}")
        print(f"Threat Model: {self.threat_model}")
        print(f"Defense Model: {self.defense_model}\n")

        def print_group(name: str, estimates: Dict[str, BayesResult]) -> None:
            if not estimates:
                return
            print(f"\n{name} metrics:")
            print("-" * 40)
            for metric_name, result in estimates.items():
                print(f"\n{metric_name}:")
                print(f"  Pre-defense:  {result.pre_value:.3f}")
                print(f"  Post-defense: {result.post_mean:.3f}")
                print(f"  Std:          {result.std:.3f}")
                print(f"  95% credible interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

        print_group("Effectiveness", self.effectiveness_estimates)
        print_group("Quality", self.quality_estimates)


def get_scenario_name(rq: str) -> str:
    """Convert RQ identifier to a readable scenario name."""
    parts = rq.split('_')
    if len(parts) > 1:
        return ' '.join(part.capitalize() for part in parts[1:])
    return rq


def format_estimate_with_hdi(
        value: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        as_percent: bool = False
) -> str:
    """Format a value with its 95% HDI interval in a consistent way."""
    if as_percent:
        if ci_lower is not None and ci_upper is not None:
            return f"{value:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"
        return f"{value:.1f}%"
    else:
        if ci_lower is not None and ci_upper is not None:
            return f"{value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        return f"{value:.3f}"


def print_defense_stats(
    stats_dict: Dict[str, DefenseStats],
    corpus: Optional[str] = None,
    threat_model: Optional[str] = None,
    defense_model: Optional[str] = None
) -> None:
    """Print Bayesian summaries with optional filtering."""
    for key, stats in stats_dict.items():
        if ((corpus is None or stats.corpus == corpus) and
            (threat_model is None or stats.threat_model == threat_model) and
            (defense_model is None or stats.defense_model == defense_model)):
            stats.print_summary()


def get_defense_tables(
        base_dir: str = "defense_evaluation",
        rqs: Union[str, List[str]] = "rq1.1_basic_paraphrase",
        corpora: Optional[List[str]] = None,
        threat_models: Optional[Dict[str, str]] = None,
        mode: str = "pre",
) -> pd.DataFrame:
    """
    Generate defense analysis tables for pre-defense values.

    For pre mode, quality measures are set to their ideal (1 for most, 0 for PINC).
    """
    if isinstance(rqs, str):
        rqs = [rqs]

    if corpora is None:
        corpora = ['ebg', 'rj']
    if threat_models is None:
        threat_models = {'logreg': 'LogReg', 'svm': 'SVM', 'roberta': 'RoBERTa'}
    if mode not in ["pre", "post"]:
        raise ValueError('mode must be one of ["pre", "post"]')

    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Class Conf ↓',
        'entropy': 'Entropy ↑'
    }

    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↑'),
        'meteor': ('meteor_avg', 'METEOR ↑'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↑'),
        'sbert': ('sbert_similarity_avg', 'SBERT ↑')
    }

    pre_rows = []
    for corpus in corpora:
        for threat_model_key, threat_model_name in threat_models.items():
            for rq in rqs:
                rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"
                corpus_path = Path(base_dir) / corpus / rq_main / rq
                if not corpus_path.exists():
                    continue
                for model_dir in corpus_path.glob("*"):
                    eval_file = model_dir / "evaluation.json"
                    if not eval_file.exists():
                        continue
                    with open(eval_file) as f:
                        results = json.load(f)
                    if not results:
                        continue
                    seed_keys = list(results.keys())
                    if not seed_keys:
                        continue
                    first_seed = seed_keys[0]
                    if threat_model_key not in results[first_seed]:
                        continue
                    # use aggregated run-level pre values
                    metrics = results[first_seed][threat_model_key]['attribution']['pre']
                    row = {'Corpus': corpus.upper(),
                           'Scenario': 'Combined' if len(rqs) > 1 else 'No protection',
                           'Threat Model': threat_model_name}
                    for metric_key, display_name in metrics_map.items():
                        value = metrics.get(metric_key)
                        row[display_name] = format_estimate_with_hdi(value)
                    for qm_key, (q_key, display_name) in quality_metrics.items():
                        baseline = 0.0 if qm_key == 'pinc' else 1.0
                        row[display_name] = format_estimate_with_hdi(baseline)
                    pre_rows.append(row)
                    break  # Only one model's data per RQ is needed
    columns = ['Corpus', 'Scenario', 'Threat Model'] + list(metrics_map.values()) + [v[1] for v in quality_metrics.values()]
    return pd.DataFrame(pre_rows, columns=columns)


def get_defense_tables_with_stats(
        base_dir: str = "defense_evaluation",
        rqs: Union[str, List[str]] = "rq1.1_basic_paraphrase",
        corpora: Optional[List[str]] = None,
        threat_models: Optional[Dict[str, str]] = None,
        mode: str = "post",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], DefenseStats]]:
    """
    Generate defense analysis tables with Bayesian credible intervals (95% HDI)
    and additional dataframes for absolute and relative changes.

    For post mode, we use run-level values for accuracy and confidence, and
    sample-level values for entropy and quality metrics.
    """
    if isinstance(rqs, str):
        rqs = [rqs]

    if corpora is None:
        corpora = ['ebg', 'rj']
    if threat_models is None:
        threat_models = {'logreg': 'LogReg', 'svm': 'SVM', 'roberta': 'RoBERTa'}
    if mode not in ["pre", "post"]:
        raise ValueError('mode must be one of ["pre", "post"]')

    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Class Conf ↓',
        'entropy': 'Entropy ↑'
    }

    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↑'),
        'meteor': ('meteor_avg', 'METEOR ↑'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↑'),
        'sbert': ('sbert_similarity_avg', 'SBERT ↑')
    }

    # Dictionaries to collect values for later modeling.
    all_metrics = defaultdict(list)  # key: (config_key, metric_key)
    all_quality = defaultdict(list)  # key: (config_key, quality_metric)
    stats_dict = {}  # key: (corpus, threat_model, defense_model) -> DefenseStats

    for rq in rqs:
        rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"
        for corpus in tqdm(corpora, desc=f"Processing corpus for {rq}"):
            for threat_model_key, threat_model_name in threat_models.items():
                corpus_path = Path(base_dir) / corpus / rq_main / rq
                if not corpus_path.exists():
                    continue
                for model_dir in tqdm(list(corpus_path.glob("*")),
                                      desc=f"Processing models for {corpus} {threat_model_name}",
                                      leave=False):
                    if not model_dir.is_dir():
                        continue

                    # Determine a display name for the defense model.
                    model_dir_name = model_dir.name.lower()
                    if 'llama' in model_dir_name:
                        model_name_disp = 'Llama-3.1'
                    elif 'gemma' in model_dir_name:
                        model_name_disp = 'Gemma-2'
                    elif 'ministral' in model_dir_name:
                        model_name_disp = 'Ministral'
                    elif 'sonnet' in model_dir_name:
                        model_name_disp = 'Claude-3.5'
                    elif 'gpt' in model_dir_name:
                        model_name_disp = 'GPT-4o'
                    else:
                        model_name_disp = model_dir.name

                    config_key = (corpus, threat_model_name, model_name_disp)
                    if config_key not in stats_dict:
                        stats_dict[config_key] = DefenseStats(corpus, threat_model_name, model_name_disp)

                    eval_file = model_dir / "evaluation.json"
                    if not eval_file.exists():
                        continue

                    with open(eval_file) as f:
                        results = json.load(f)

                    # For run-level effectiveness metrics (accuracy and confidence),
                    # take values from one seed's aggregated results.
                    for seed_results in results.values():
                        if threat_model_key not in seed_results:
                            continue
                        run_pre = seed_results[threat_model_key]['attribution']['pre']
                        run_post = seed_results[threat_model_key]['attribution']['post']
                        for metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
                            orig_val = run_pre.get(metric)
                            post_val = run_post.get(metric)
                            if orig_val is not None and post_val is not None:
                                all_metrics[(config_key, metric)].append({'pre': float(orig_val), 'post': float(post_val)})
                        # For entropy, take the aggregated pre value…
                        orig_entropy = run_pre.get('entropy')
                        break  # Use one seed’s aggregated run-level values

                    # Now load sample-level data from each seed file for entropy and quality.
                    entropy_samples = []
                    quality_scores = {'bleu': [], 'meteor': [], 'pinc': [], 'bertscore': [], 'sbert': []}
                    for seed_file in model_dir.glob("seed_*.json"):
                        with open(seed_file) as f:
                            seed_data = json.load(f)
                        # From example_metrics, compute sample-level entropy.
                        for ex in seed_data.get("example_metrics", []):
                            if ex.get("model_type", "").lower() == threat_model_key:
                                trans_probs = ex.get("trans_probs")
                                if trans_probs:
                                    entropy_samples.append(
                                        -np.sum(np.array(trans_probs) * np.log2(np.array(trans_probs) + 1e-10))
                                    )
                        # From example_quality, collect quality scores.
                        for ex in seed_data.get("example_quality", []):
                            if ex.get("model_type", "").lower() == threat_model_key:
                                for qm in quality_scores.keys():
                                    if qm in ex:
                                        quality_scores[qm].append(float(ex[qm]))
                    # Add entropy sample-level values if available.
                    if orig_entropy is not None and entropy_samples:
                        all_metrics[(config_key, 'entropy')].append({'pre': float(orig_entropy), 'post_samples': entropy_samples})
                    # For each quality metric, add all sample scores.
                    for qm, scores in quality_scores.items():
                        if scores:
                            all_quality[(config_key, qm)].extend(scores)
                # End loop over model_dir.
            # End loop over threat model.
        # End loop over corpus.

    # Build tables for each configuration.
    post_rows = []
    abs_change_rows = []
    rel_change_rows = []

    for config_key, stats in stats_dict.items():
        corpus_val, threat_model_name_val, model_name_disp_val = config_key
        base_row = {
            'Corpus': corpus_val.upper(),
            'Threat Model': threat_model_name_val,
            'Defense Model': model_name_disp_val,
            'Scenario': 'Combined' if len(rqs) > 1 else get_scenario_name(rqs[0])
        }
        # Process effectiveness metrics.
        for metric_key, display_name in metrics_map.items():
            key = (config_key, metric_key)
            if key in all_metrics:
                values = all_metrics[key]
                pre_values = [v['pre'] for v in values if v.get('pre') is not None]
                pre_value = np.mean(pre_values) if pre_values else None
                if metric_key == 'entropy':
                    post_values = []
                    for v in values:
                        if 'post_samples' in v:
                            post_values.extend(v['post_samples'])
                        elif 'post' in v:
                            post_values.append(v['post'])
                else:
                    post_values = [v['post'] for v in values if 'post' in v]
                if pre_value is not None and post_values:
                    stats.add_estimate(metric_key, 'effectiveness', pre_value, post_values)
                    result = estimate_beta_metric(post_values)
                    if metric_key == 'entropy':
                        if corpus_val.lower() == 'rj':
                            factor = np.log2(21)
                        elif corpus_val.lower() == 'ebg':
                            factor = np.log2(45)
                        else:
                            factor = 1.0
                        result = {
                            "mean": result["mean"] * factor,
                            "std": result["std"] * factor,
                            "hdi_lower": result["hdi_lower"] * factor,
                            "hdi_upper": result["hdi_upper"] * factor
                        }
                    base_row[display_name] = format_estimate_with_hdi(result["mean"], result["hdi_lower"], result["hdi_upper"])
                else:
                    base_row[display_name] = '-'
            else:
                base_row[display_name] = '-'
        # Process quality metrics.
        for qm, (q_key, display_name) in quality_metrics.items():
            quality_key = (config_key, qm)
            if quality_key in all_quality:
                values = all_quality[quality_key]
                baseline = 0.0 if qm == 'pinc' else 1.0
                stats.add_estimate(qm, 'quality', baseline, values)
                result = estimate_beta_metric(values)
                base_row[display_name] = format_estimate_with_hdi(result["mean"], result["hdi_lower"], result["hdi_upper"])
        # Compute absolute and relative changes for effectiveness metrics.
        abs_row = {k: v for k, v in base_row.items() if k in ['Corpus', 'Threat Model', 'Defense Model']}
        rel_row = abs_row.copy()
        for metric_key, display_name in metrics_map.items():
            if metric_key in stats.effectiveness_estimates:
                br = stats.effectiveness_estimates[metric_key]
                abs_change = br.post_mean - br.pre_value
                abs_ci_lower = br.ci_lower - br.pre_value
                abs_ci_upper = br.ci_upper - br.pre_value
                abs_row[display_name] = format_estimate_with_hdi(abs_change, abs_ci_lower, abs_ci_upper)
                if br.pre_value != 0:
                    rel_change = (abs_change / br.pre_value) * 100
                    rel_ci_lower = ((br.ci_lower - br.pre_value) / br.pre_value) * 100
                    rel_ci_upper = ((br.ci_upper - br.pre_value) / br.pre_value) * 100
                    rel_row[display_name] = format_estimate_with_hdi(rel_change, rel_ci_lower, rel_ci_upper, as_percent=True)
                else:
                    rel_row[display_name] = '-'
        for qm, (q_key, display_name) in quality_metrics.items():
            if qm in stats.quality_estimates:
                br = stats.quality_estimates[qm]
                abs_change = br.post_mean - br.pre_value
                abs_ci_lower = br.ci_lower - br.pre_value
                abs_ci_upper = br.ci_upper - br.pre_value
                abs_row[display_name] = format_estimate_with_hdi(abs_change, abs_ci_lower, abs_ci_upper)
                rel_change = abs_change * 100
                rel_ci_lower = abs_ci_lower * 100
                rel_ci_upper = abs_ci_upper * 100
                rel_row[display_name] = format_estimate_with_hdi(rel_change, rel_ci_lower, rel_ci_upper, as_percent=True)
        post_rows.append(base_row.copy())
    columns = ['Corpus', 'Scenario', 'Threat Model', 'Defense Model'] + list(metrics_map.values()) + [v[1] for v in quality_metrics.values()]
    change_cols = ['Corpus', 'Threat Model', 'Defense Model'] + list(metrics_map.values()) + [v[1] for v in quality_metrics.values()]

    post_df = pd.DataFrame(post_rows, columns=columns)
    # For absolute and relative change tables, we reuse the change values from post_rows.
    abs_change_df = pd.DataFrame([{k: v for k, v in row.items() if k in change_cols} for row in post_rows], columns=change_cols)
    rel_change_df = pd.DataFrame([{k: v for k, v in row.items() if k in change_cols} for row in post_rows], columns=change_cols)

    return post_df, abs_change_df, rel_change_df, stats_dict


def serialize_defense_stats(stats: DefenseStats) -> dict:
    """
    Convert a DefenseStats instance to a dictionary.
    """
    return {
        "corpus": stats.corpus,
        "threat_model": stats.threat_model,
        "defense_model": stats.defense_model,
        "effectiveness_estimates": {k: v._asdict() for k, v in stats.effectiveness_estimates.items()},
        "quality_estimates": {k: v._asdict() for k, v in stats.quality_estimates.items()}
    }


def main():
    """Example usage: parse arguments and output CSV tables and aggregated stats JSON in the results folder."""
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Evaluate defense effectiveness with Bayesian analysis"
    )

    parser.add_argument(
        '--base_dir',
        type=str,
        default='defense_evaluation',
        help='Base directory containing evaluation results'
    )
    parser.add_argument(
        '--rqs',
        type=str,
        nargs='+',
        default=['rq1.1_basic_paraphrase'],
        help='Research question identifiers (can specify multiple for combining data)'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['rj', 'ebg'],
        help='Specific corpus to analyze'
    )
    parser.add_argument(
        '--defense_model',
        type=str,
        help='Specific defense model to analyze (not yet implemented for filtering)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output folder to save CSV tables and stats JSON'
    )

    args = parser.parse_args()

    all_post_dfs = []
    all_abs_change_dfs = []
    all_rel_change_dfs = []
    all_stats_dicts = []

    for rq in args.rqs:
        post_df, abs_change_df, rel_change_df, stats_dict = get_defense_tables_with_stats(
            base_dir=args.base_dir,
            rqs=rq,
            corpora=[args.corpus] if args.corpus else None,
            threat_models=None,
            mode="post"
        )
        all_post_dfs.append(post_df)
        all_abs_change_dfs.append(abs_change_df)
        all_rel_change_dfs.append(rel_change_df)
        all_stats_dicts.append(stats_dict)

    post_df = pd.concat(all_post_dfs, ignore_index=True)
    abs_change_df = pd.concat(all_abs_change_dfs, ignore_index=True)
    rel_change_df = pd.concat(all_rel_change_dfs, ignore_index=True)

    print("\nPost-intervention results (Bayesian estimates with 95% HDI):")
    print(post_df)
    print("\nBayesian absolute changes (post_mean - pre_value):")
    print(abs_change_df)
    print("\nBayesian relative changes (in percent):")
    print(rel_change_df)

    pre_df = get_defense_tables(
        base_dir=args.base_dir,
        rqs=args.rqs,
        corpora=[args.corpus] if args.corpus else None,
        threat_models=None,
        mode="pre"
    )

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    if len(args.rqs) == 1:
        rq_part = args.rqs[0]
    else:
        common_prefix = os.path.commonprefix(args.rqs).rstrip('_')
        rq_part = f"{common_prefix}_combined"

    base_filename = f"{rq_part}_{args.corpus}" if args.corpus else rq_part

    post_df.to_csv(output_folder / f"{base_filename}_post.csv", index=False)
    abs_change_df.to_csv(output_folder / f"{base_filename}_abs_change.csv", index=False)
    rel_change_df.to_csv(output_folder / f"{base_filename}_rel_change.csv", index=False)
    pre_df.to_csv(output_folder / f"{base_filename}_pre.csv", index=False)

    merged_stats = {}
    for sd in all_stats_dicts:
        merged_stats.update(sd)

    merged_stats_serializable = {
        str(key): serialize_defense_stats(value) for key, value in merged_stats.items()
    }

    stats_file = output_folder / f"{base_filename}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(merged_stats_serializable, f, indent=2)

    print(f"\nResults saved in folder '{output_folder.absolute()}'")
    print("\nNote: Run-level details are stored in the per-seed JSON files within each experiment directory.")


if __name__ == "__main__":
    main()

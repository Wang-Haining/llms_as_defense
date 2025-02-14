"""
Defense evaluation script that performs bayesian analysis comparing pre/post defense
performance across different metrics, corpora, and threat models

expected directory structure:
defense_evaluation/
├── {corpus}/                              # e.g., rj and ebg
│   ├── rq{N}/                            # main research question (e.g., rq1)
│   │   ├── rq{N}.{M}/                    # sub-question (e.g., rq1.1)
│   │   │   ├── {model_name}/             # e.g., gemma-2b-it
│   │   │   │   ├── evaluation.json       # consolidated results
│   │   │   │   └── seed_{seed}.json      # per-seed results
│   │   │   └── {another_model}/
│   └── rq{N+1}/
└── {another_corpus}/
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from tqdm import tqdm

# define the expected direction for each metric
METRIC_DIRECTIONS = {
    # effectiveness metrics – we want metrics to decrease except for entropy and kl divergence
    'accuracy@1': 'less',
    'accuracy@5': 'less',
    'true_class_confidence': 'less',
    'wrong_entropy': 'greater',
    'mrr': 'less',
    'kl_divergence': 'greater',

    # quality metrics – note: for pre, bleu, meteor, bertscore are 1 and pinc is 0
    'bleu': 'less',
    'meteor': 'less',
    'pinc': 'greater',
    'bertscore': 'less'
}


# for posteriors
class BayesResult(NamedTuple):
    pre_value: float            # baseline value (pre-intervention)
    post_mean: float            # posterior mean of the parameter
    ci_lower: float             # lower bound of the 95% credible interval
    ci_upper: float             # upper bound of the 95% credible interval
    effect_size: float          # (post_mean - baseline) / (approximate uncertainty)
    direction: str              # expected direction ('less' or 'greater')


def estimate_beta_metric(post_values: List[float]) -> dict:
    """Estimate beta distribution parameters with 95% hdi intervals for metrics
    naturally bounded in [0,1]"""
    # ensure values are strictly within (0, 1)
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
    # we ignore standard deviation and only use hdi
    hdi_bounds = az.hdi(mu_samples, hdi_prob=0.95)
    lower_bound = float(max(0, hdi_bounds[0]))
    upper_bound = float(min(1, hdi_bounds[1]))

    return {
        "mean": mean_mu,
        "hdi_lower": lower_bound,
        "hdi_upper": upper_bound
    }


def bayesian_estimate(metric_name: str, metric_type: str,
                      post_values: List[float], baseline: float) -> BayesResult:
    """Use the beta estimation function for all metrics (except for entropy change,
    which should be normalized before calling this function)

    args:
        metric_name: name of the metric
        metric_type: either 'effectiveness' or 'quality'
        post_values: post-intervention values in [0,1]
        baseline: pre-intervention value (or fixed baseline for quality metrics)

    returns:
        a BayesResult object with posterior summaries
    """
    result = estimate_beta_metric(post_values)
    # approximate uncertainty is inferred from the hdi width (for effect size calculation)
    uncertainty = (result["hdi_upper"] - result["hdi_lower"]) / 4  # approx. std estimate for normal
    effect_size = (result["mean"] - baseline) / uncertainty if uncertainty != 0 else np.nan
    direction = METRIC_DIRECTIONS.get(metric_name, "less")

    return BayesResult(
        pre_value=baseline,
        post_mean=result["mean"],
        ci_lower=result["hdi_lower"],
        ci_upper=result["hdi_upper"],
        effect_size=effect_size,
        direction=direction
    )


class DefenseStats:
    """Container for defense evaluation bayesian estimates for each metric"""

    def __init__(self, corpus: str, threat_model: str, defense_model: str):
        self.corpus = corpus
        self.threat_model = threat_model
        self.defense_model = defense_model
        self.effectiveness_estimates: Dict[str, BayesResult] = {}  # metric_name -> BayesResult
        self.quality_estimates: Dict[str, BayesResult] = {}        # metric_name -> BayesResult

    def add_estimate(self, metric_name: str, metric_type: str,
                     pre_value: float, post_values: List[float],
                     display_name: str) -> None:
        """
        Add bayesian estimate for a metric based on post_values
        for effectiveness metrics, baseline is pre_value; for quality metrics,
        baseline is 1 for bleu/meteor/bertscore and 0 for pinc
        """
        if not post_values:
            return

        baseline = pre_value if metric_type == 'effectiveness' else (0.0 if metric_name == 'pinc' else 1.0)
        bayes_result = bayesian_estimate(metric_name, metric_type, post_values, baseline)
        if metric_type == 'effectiveness':
            self.effectiveness_estimates[metric_name] = bayes_result
        else:
            self.quality_estimates[metric_name] = bayes_result

    def format_results(self, metric_name: str, metric_type: str) -> str:
        """format test results for a specific metric"""
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
        """print a summary of all bayesian estimates"""
        print("\ndefense analysis summary")
        print("=" * 80)
        print(f"corpus: {self.corpus}")
        print(f"threat model: {self.threat_model}")
        print(f"defense model: {self.defense_model}\n")

        def print_group(name: str, estimates: Dict[str, BayesResult]) -> None:
            if not estimates:
                return
            print(f"\n{name} metrics:")
            print("-" * 40)
            for metric_name, result in estimates.items():
                print(f"\n{metric_name}:")
                print(f"  pre-defense:  {result.pre_value:.3f}")
                print(f"  post-defense: {result.post_mean:.3f}")
                print(f"  95% credible interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
                print(f"  effect size:  {result.effect_size:.3f}")
                print(f"  direction:    {result.direction}")

        print_group("effectiveness", self.effectiveness_estimates)
        print_group("quality", self.quality_estimates)


def get_scenario_name(rq: str) -> str:
    """Convert rq identifier to a readable scenario name"""
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
    """Format a value with its 95% HDI interval in a consistent way.

    Args:
        value: point estimate
        ci_lower: lower bound of 95% HDI
        ci_upper: upper bound of 95% HDI
        as_percent: whether to format as percentage

    Returns:
        Formatted string with value and HDI
    """
    if as_percent:
        if ci_lower is not None and ci_upper is not None:
            return f"{value:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"
        return f"{value:.1f}%"
    else:
        if ci_lower is not None and ci_upper is not None:
            return f"{value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        return f"{value:.3f}"


def calculate_effect_size(pre_value: float, post_values: List[float]) -> float:
    """Calculate glass's d effect size using post values"""
    post_mean = np.mean(post_values)
    post_std = np.std(post_values)
    if post_std == 0:
        if post_mean > pre_value:
            return float('inf')
        elif post_mean < pre_value:
            return float('-inf')
        return 0.0
    return (post_mean - pre_value) / post_std


def print_defense_stats(
    stats_dict: Dict[str, DefenseStats],
    corpus: Optional[str] = None,
    threat_model: Optional[str] = None,
    defense_model: Optional[str] = None
) -> None:
    """Print bayesian summaries with optional filtering"""
    for key, stats in stats_dict.items():
        if ((corpus is None or stats.corpus == corpus) and
            (threat_model is None or stats.threat_model == threat_model) and
            (defense_model is None or stats.defense_model == defense_model)):
            stats.print_summary()


# def display_copyable(df: pd.DataFrame) -> HTML:
#     """display dataframe in a copyable format with a copy button"""
#     csv_string = df.to_csv(index=False, sep='\t')
#     return HTML(f"""
#     <textarea id="copyable_text" style="width: 100%; height: 200px;">{csv_string}</textarea>
#     <button onclick="copyText()">Copy to Clipboard</button>
#     <script>
#     function copyText() {{
#         var copyText = document.getElementById("copyable_text");
#         copyText.select();
#         document.execCommand("copy");
#     }}
#     </script>
#     """)


def get_defense_tables(
        base_dir: str = "defense_evaluation",
        rq: str = "rq1.1_basic_paraphrase",
        corpora: Optional[List[str]] = None,
        threat_models: Optional[Dict[str, str]] = None,
        mode: str = "post",
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Generate defense analysis tables

    For pre mode, quality measures are set to their ideal (1 for most, 0 for pinc)
    For post mode, the output includes the post values for effectiveness and quality metrics

    Args:
        base_dir: base directory containing evaluation results
        rq: research question identifier (e.g. 'rq1.1_basic_paraphrase')
        corpora: list of corpora to analyze (default: ['ebg', 'rj'])
        threat_models: dict mapping model keys to display names
        mode: either 'pre' or 'post'

    Returns:
        if mode=="pre": dataframe with pre-defense values
        if mode=="post": single dataframe with post values and HDI
    """
    if corpora is None:
        corpora = ['ebg', 'rj']
    if threat_models is None:
        threat_models = {'logreg': 'LogReg', 'svm': 'SVM', 'roberta': 'RoBERTa'}
    if mode not in ["pre", "post"]:
        raise ValueError('mode must be one of ["pre", "post"]')

    # effectiveness metrics (run-level)
    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Class Conf ↓',
        'entropy': 'Entropy ↑'
    }

    # quality metrics (sample-level)
    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↑'),
        'meteor': ('meteor_avg', 'METEOR ↑'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↑'),
        'sbert': ('sbert_similarity_avg', 'SBERT ↑')
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"

    if mode == "pre":
        pre_rows = []
        for corpus in corpora:
            for threat_model_key, threat_model_name in threat_models.items():
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
                    metrics = results[first_seed][threat_model_key]['attribution'][
                        'original_metrics']
                    row = {'Corpus': corpus.upper(), 'Scenario': 'No protection',
                           'Threat Model': threat_model_name}
                    for metric_key, display_name in metrics_map.items():
                        value = metrics.get(metric_key)
                        row[display_name] = format_estimate_with_hdi(value)
                    for _, (_, display_name) in quality_metrics.items():
                        # pre-defense quality is ideal (1 for most, 0 for pinc)
                        baseline = 0.0 if _ == 'pinc' else 1.0
                        row[display_name] = format_estimate_with_hdi(baseline)
                    pre_rows.append(row)
                    break  # only need one model's data
        columns = ['Corpus', 'Scenario', 'Threat Model'] + list(
            metrics_map.values()) + [v[1] for v in quality_metrics.values()]
        return pd.DataFrame(pre_rows, columns=columns)

    else:  # mode == "post"
        post_rows = []
        for corpus in corpora:
            for threat_model_key, threat_model_name in threat_models.items():
                corpus_path = Path(base_dir) / corpus / rq_main / rq
                if not corpus_path.exists():
                    continue
                for model_dir in corpus_path.glob("*"):
                    if not model_dir.is_dir():
                        continue
                    # determine display name for model
                    model_dir_name = model_dir.name.lower()
                    if 'llama' in model_dir_name:
                        model_name = 'Llama-3.1'
                    elif 'gemma' in model_dir_name:
                        model_name = 'Gemma-2'
                    elif 'ministral' in model_dir_name:
                        model_name = 'Ministral'
                    elif 'sonnet' in model_dir_name:
                        model_name = 'Claude-3.5'
                    elif 'gpt' in model_dir_name:
                        model_name = 'GPT-4o'
                    else:
                        model_name = model_dir.name

                    eval_file = model_dir / "evaluation.json"
                    if not eval_file.exists():
                        continue
                    post_metrics = defaultdict(list)
                    quality_scores = defaultdict(list)
                    with open(eval_file) as f:
                        results = json.load(f)
                    for seed_results in results.values():
                        if threat_model_key not in seed_results:
                            continue
                        metrics = seed_results[threat_model_key]['attribution']
                        for metric_key, _ in metrics_map.items():
                            post_val = metrics['transformed_metrics'].get(metric_key)
                            if post_val is not None:
                                post_metrics[metric_key].append(float(post_val))
                        quality = seed_results[threat_model_key].get('quality', {})
                        for qm, (key, _) in quality_metrics.items():
                            score = quality.get(qm, {}).get(key)
                            if score is not None:
                                quality_scores[qm].append(float(score))

                    # Create row with 95% HDI using Bayesian estimation
                    base_row = {
                        'Corpus': corpus.upper(),
                        'Scenario': get_scenario_name(rq),
                        'Threat Model': threat_model_name,
                        'Defense Model': model_name
                    }
                    # For effectiveness metrics
                    for metric_key, display_name in metrics_map.items():
                        post_vals = post_metrics[metric_key]
                        if post_vals:
                            result = estimate_beta_metric(post_vals)
                            base_row[display_name] = format_estimate_with_hdi(
                                result["mean"],
                                result["hdi_lower"],
                                result["hdi_upper"]
                            )
                        else:
                            base_row[display_name] = '-'

                    # For quality metrics
                    for qm, (key, display_name) in quality_metrics.items():
                        values = quality_scores[qm]
                        if values:
                            result = estimate_beta_metric(values)
                            base_row[display_name] = format_estimate_with_hdi(
                                result["mean"],
                                result["hdi_lower"],
                                result["hdi_upper"]
                            )
                        else:
                            base_row[display_name] = '-'

                    post_rows.append(base_row)

        columns = ['Corpus', 'Scenario', 'Threat Model', 'Defense Model'] + \
                  list(metrics_map.values()) + [v[1] for v in quality_metrics.values()]
        return pd.DataFrame(post_rows, columns=columns)


def get_defense_tables_with_stats(
        base_dir: str = "defense_evaluation",
        rq: str = "rq1.1_basic_paraphrase",
        corpora: Optional[List[str]] = None,
        threat_models: Optional[Dict[str, str]] = None,
        mode: str = "post",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, DefenseStats]]:
    """
    Generate defense analysis tables with bayesian credible intervals (95% HDI)
    and additional dataframes for absolute and relative changes.
    Progress bars are added to track processing.

    Args:
        base_dir: base directory containing evaluation results
        rq: research question identifier (e.g. 'rq1.1_basic_paraphrase')
        corpora: list of corpora to analyze (default: ['ebg', 'rj'])
        threat_models: dict mapping model keys to display names
        mode: either 'pre' or 'post'

    Returns:
        Tuple containing:
        - post_df: formatted post-intervention values with 95% HDI
        - abs_change_df: absolute change with 95% HDI
        - rel_change_df: relative change (%) with 95% HDI
        - stats_dict: dict mapping (corpus, threat_model, defense_model) to DefenseStats
    """
    if corpora is None:
        corpora = ['ebg', 'rj']
    if threat_models is None:
        threat_models = {'logreg': 'LogReg', 'svm': 'SVM', 'roberta': 'RoBERTa'}
    if mode not in ["pre", "post"]:
        raise ValueError('mode must be one of ["pre", "post"]')

    # effectiveness metrics (run-level)
    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Class Conf ↓',
        'entropy': 'Entropy ↑'
    }

    # quality metrics (sample-level)
    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↑'),
        'meteor': ('meteor_avg', 'METEOR ↑'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↑'),
        'sbert': ('sbert_similarity_avg', 'SBERT ↑')
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"

    # get the post values with HDI from modified get_defense_tables
    post_df = get_defense_tables(base_dir, rq, corpora, threat_models, mode)

    # prepare dataframes for changes
    abs_change_rows = []
    rel_change_rows = []
    stats_dict = {}

    for corpus in tqdm(corpora, desc="Processing corpora"):
        for threat_model_key, threat_model_name in threat_models.items():
            corpus_path = Path(base_dir) / corpus / rq_main / rq
            if not corpus_path.exists():
                continue

            for model_dir in tqdm(list(corpus_path.glob("*")),
                                desc=f"Processing models for {corpus} {threat_model_name}",
                                leave=False):
                if not model_dir.is_dir():
                    continue

                # determine display name for model
                model_dir_name = model_dir.name.lower()
                if 'llama' in model_dir_name:
                    model_name = 'Llama-3.1'
                elif 'gemma' in model_dir_name:
                    model_name = 'Gemma-2'
                elif 'ministral' in model_dir_name:
                    model_name = 'Ministral'
                elif 'sonnet' in model_dir_name:
                    model_name = 'Claude-3.5'
                elif 'gpt' in model_dir_name:
                    model_name = 'GPT-4o'
                else:
                    model_name = model_dir.name

                eval_file = model_dir / "evaluation.json"
                if not eval_file.exists():
                    continue

                # initialize defense stats object
                stats = DefenseStats(corpus, threat_model_name, model_name)

                # collectors for metrics
                orig_metrics = defaultdict(list)
                post_metrics = defaultdict(list)
                quality_scores = defaultdict(list)

                # load evaluation results
                with open(eval_file) as f:
                    results = json.load(f)

                # collect metrics across seeds
                for seed_results in results.values():
                    if threat_model_key not in seed_results:
                        continue

                    metrics = seed_results[threat_model_key]['attribution']

                    # effectiveness metrics
                    for metric_key, _ in metrics_map.items():
                        orig_val = metrics['original_metrics'].get(metric_key)
                        post_val = metrics['transformed_metrics'].get(metric_key)
                        if orig_val is not None:
                            orig_metrics[metric_key].append(float(orig_val))
                        if post_val is not None:
                            post_metrics[metric_key].append(float(post_val))

                    # quality metrics
                    quality = seed_results[threat_model_key].get('quality', {})
                    for qm, (key, _) in quality_metrics.items():
                        score = quality.get(qm, {}).get(key)
                        if score is not None:
                            quality_scores[qm].append(float(score))

                # process effectiveness metrics using bayesian estimates
                for metric_key, display_name in metrics_map.items():
                    orig_vals = orig_metrics[metric_key]
                    post_vals = post_metrics[metric_key]
                    if orig_vals and post_vals:
                        stats.add_estimate(metric_key, 'effectiveness',
                                        np.mean(orig_vals), post_vals, display_name)

                # process quality metrics using bayesian estimates
                for qm, (key, display_name) in quality_metrics.items():
                    values = quality_scores[qm]
                    if values:
                        stats.add_estimate(qm, 'quality',
                                        1.0 if qm != 'pinc' else 0.0,
                                        values, display_name)

                stats_dict[(corpus, threat_model_name, model_name)] = stats

                # prepare rows for change tables
                base_row_abs = {
                    'Corpus': corpus.upper(),
                    'Threat Model': threat_model_name,
                    'Defense Model': model_name
                }
                base_row_rel = base_row_abs.copy()

                # effectiveness metrics changes
                for metric_key, display_name in metrics_map.items():
                    if metric_key in stats.effectiveness_estimates:
                        br = stats.effectiveness_estimates[metric_key]
                        abs_change = br.post_mean - br.pre_value
                        abs_ci_lower = br.ci_lower - br.pre_value
                        abs_ci_upper = br.ci_upper - br.pre_value
                        base_row_abs[display_name] = format_estimate_with_hdi(
                            abs_change, abs_ci_lower, abs_ci_upper
                        )

                        if br.pre_value != 0:
                            rel_change = (abs_change / br.pre_value) * 100
                            rel_ci_lower = ((br.ci_lower - br.pre_value) / br.pre_value) * 100
                            rel_ci_upper = ((br.ci_upper - br.pre_value) / br.pre_value) * 100
                            base_row_rel[display_name] = format_estimate_with_hdi(
                                rel_change, rel_ci_lower, rel_ci_upper, as_percent=True
                            )
                        else:
                            base_row_rel[display_name] = '-'
                    else:
                        base_row_abs[display_name] = '-'
                        base_row_rel[display_name] = '-'

                # quality metrics changes
                for qm, (key, display_name) in quality_metrics.items():
                    if qm in stats.quality_estimates:
                        br = stats.quality_estimates[qm]
                        base_val = 0.0 if qm == 'pinc' else 1.0
                        abs_change = br.post_mean - base_val
                        abs_ci_lower = br.ci_lower - base_val
                        abs_ci_upper = br.ci_upper - base_val
                        base_row_abs[display_name] = format_estimate_with_hdi(
                            abs_change, abs_ci_lower, abs_ci_upper
                        )

                        # quality metrics relative changes
                        rel_change = (br.post_mean - base_val) * 100
                        rel_ci_lower = (br.ci_lower - base_val) * 100
                        rel_ci_upper = (br.ci_upper - base_val) * 100
                        base_row_rel[display_name] = format_estimate_with_hdi(
                            rel_change, rel_ci_lower, rel_ci_upper, as_percent=True
                        )
                    else:
                        base_row_abs[display_name] = '-'
                        base_row_rel[display_name] = '-'

                abs_change_rows.append(base_row_abs)
                rel_change_rows.append(base_row_rel)

    # create final dataframes
    cols = ['Corpus', 'Threat Model', 'Defense Model'] + \
           list(metrics_map.values()) + \
           [v[1] for v in quality_metrics.values()]

    abs_change_df = pd.DataFrame(abs_change_rows, columns=cols)
    rel_change_df = pd.DataFrame(rel_change_rows, columns=cols)

    return post_df, abs_change_df, rel_change_df, stats_dict


def main():
    """Example usage of defense evaluation; parse args and output csv tables in results
    folder"""
    import argparse
    parser = argparse.ArgumentParser(
        description="evaluate defense effectiveness with bayesian analysis"
    )

    parser.add_argument(
        '--base_dir',
        type=str,
        default='defense_evaluation',
        help='base directory containing evaluation results'
    )
    parser.add_argument(
        '--rq',
        type=str,
        default='rq1.1_basic_paraphrase',
        help='research question identifier'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['rj', 'ebg'],
        help='specific corpus to analyze'
    )
    parser.add_argument(
        '--defense_model',
        type=str,
        help='specific defense model to analyze (not yet implemented for filtering)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='output folder to save csv tables'
    )

    args = parser.parse_args()

    # Note changed return values here
    (post_df, abs_change_df, rel_change_df, stats_dict) = get_defense_tables_with_stats(
        base_dir=args.base_dir,
        rq=args.rq,
        corpora=[args.corpus] if args.corpus else None,
        threat_models=None,  # uses default if not provided
        mode="post"
    )

    print("\npost-intervention results (bayesian estimates with 95% HDI):")
    print(post_df)
    print("\nbayesian absolute changes (post_mean - pre_value):")
    print(abs_change_df)
    print("\nbayesian relative changes (in percent):")
    print(rel_change_df)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    base_filename = args.rq
    post_df.to_csv(output_folder / f"{base_filename}_post.csv", index=False)
    abs_change_df.to_csv(output_folder / f"{base_filename}_abs_change.csv", index=False)
    rel_change_df.to_csv(output_folder / f"{base_filename}_rel_change.csv", index=False)

    print(f"\nresults saved in folder '{output_folder.absolute()}'")


if __name__ == "__main__":
    main()

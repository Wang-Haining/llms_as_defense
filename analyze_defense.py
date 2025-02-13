"""Defense evaluation script that performs Bayesian analysis comparing pre/post defense.

This module provides functionality to analyze and compare performance metrics before and
after applying defense strategies against authorship attribution attacks. It implements
Bayesian estimation using PyMC for robust statistical analysis.

The script expects results in the following directory structure:
    defense_evaluation/
    ├── {corpus}/                # rj and ebg
    │   ├── rq{N}/              # main research question (e.g., rq1)
    │   │   ├── rq{N}.{M}/      # sub-question (e.g., rq1.1)
    │   │   │   ├── {model}/    # e.g., gemma-2b-it
    │   │   │   │   ├── evaluation.json  # consolidated results
    │   │   │   │   └── seed_{seed}.json # per-seed results
    │   │   │   └── {another_model}/
    │   │   └── rq{N}.{M+1}/
    │   └── rq{N+1}/
    └── {another_corpus}/

Typical usage example:
    ```python
    post_df, abs_imp_df, rel_imp_df, stats = get_defense_tables_with_stats(
        rq="rq1.1_basic_paraphrase",
        corpora=["rj", "ebg"]
    )
    print_defense_stats(stats)
    ```
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
from IPython.display import HTML

# suppress pymc warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# define metric directions for evaluation
METRIC_DIRECTIONS = {
    # effectiveness metrics - want metrics to decrease except entropy and kl_divergence
    'accuracy@1': 'less',
    'accuracy@5': 'less',
    'true_class_confidence': 'less',
    'wrong_entropy': 'greater',
    'mrr': 'less',
    'kl_divergence': 'greater',

    # quality metrics - note: pre bleu, meteor, bertscore are 1 and pinc is 0
    'bleu': 'less',
    'meteor': 'less',
    'pinc': 'greater',
    'bertscore': 'less'
}


class BayesResult(NamedTuple):
    """Container for Bayesian analysis results."""
    pre_value: float  # baseline value (pre-intervention)
    post_mean: float  # posterior mean of the parameter
    post_std: float  # posterior std of the parameter
    ci_lower: float  # lower bound of the 95% credible interval
    ci_upper: float  # upper bound of the 95% credible interval
    effect_size: float  # (post_mean - baseline)/post_std
    direction: str  # expected direction ('less' or 'greater')


def estimate_beta_metric(post_values: list) -> dict:
    """Estimates beta distribution parameters with improved numerical stability.

    Args:
        post_values: List of values between 0 and 1

    Returns:
        Dict with posterior mean, std and credible intervals
    """
    post_values = np.array(post_values)

    # clip values to avoid boundary issues (epsilon away from 0 or 1)
    eps = 1e-7
    post_values = np.clip(post_values, eps, 1 - eps)

    # use mean of data for initialization, but keep away from bounds
    init_mu = np.clip(np.mean(post_values), 0.1, 0.9)
    init_kappa = max(10.0, 1 / np.var(post_values)) if np.var(post_values) > 0 else 10.0

    with pm.Model() as model:
        mu = pm.Beta("mu", alpha=1, beta=1)  # noncommittal prior
        kappa = pm.HalfNormal("kappa", sigma=10)  # weakly informative prior

        obs = pm.Beta("obs",
                      alpha=pm.math.clip(mu * kappa, eps, 1e6),
                      beta=pm.math.clip((1 - mu) * kappa, eps, 1e6),
                      observed=post_values)

        initvals = {"mu": init_mu, "kappa": init_kappa}

        trace = pm.sample(2000,
                          tune=1000,
                          cores=1,
                          progressbar=False,
                          random_seed=42,
                          initvals=initvals,
                          target_accept=0.95,
                          return_inferencedata=True)

    mu_samples = trace.posterior["mu"].values.flatten()
    mean_mu = np.mean(mu_samples)
    std_mu = np.std(mu_samples)
    ci_lower, ci_upper = np.percentile(mu_samples, [2.5, 97.5])

    return {
        "mean": mean_mu,
        "std": std_mu,
        "ci_lower": max(0, ci_lower),
        "ci_upper": min(1, ci_upper)
    }


def estimate_gamma_metric(post_values: list) -> dict:
    """Uses gamma model for metrics not bounded in [0,1].

    Args:
        post_values: List of positive real values

    Returns:
        Dict with posterior mean, std and credible intervals
    """
    post_values = np.array(post_values)
    with pm.Model() as model:
        alpha_param = pm.Exponential("alpha", lam=1)
        beta_param = pm.Exponential("beta", lam=1)
        obs = pm.Gamma("obs", alpha=alpha_param, beta=beta_param, observed=post_values)
        initvals = {"alpha": 2.0, "beta": 2.0}
        trace = pm.sample(
            2000,
            tune=1000,
            cores=1,
            progressbar=False,
            random_seed=42,
            initvals=initvals,
            target_accept=0.95,
        )
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()
    mean_samples = alpha_samples / beta_samples
    mean_mean = np.mean(mean_samples)
    std_mean = np.std(mean_samples)
    ci_lower, ci_upper = np.percentile(mean_samples, [2.5, 97.5])

    return {
        "mean": mean_mean,
        "std": std_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


def bayesian_estimate(
        metric_name: str,
        metric_type: str,
        post_values: List[float],
        baseline: float
) -> BayesResult:
    """Estimates posterior distribution parameters for a metric.

    Args:
        metric_name: Name of the metric to estimate
        metric_type: Type of metric ('effectiveness' or 'quality')
        post_values: List of observed values
        baseline: Pre-intervention value or fixed baseline

    Returns:
        BayesResult containing posterior summaries
    """
    # for metrics in [0,1] use beta model; for others use gamma
    if metric_name in ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'mrr',
                       'bleu', 'meteor', 'bertscore', 'pinc']:
        result = estimate_beta_metric(post_values)
    elif metric_name in ['wrong_entropy', 'kl_divergence']:
        result = estimate_gamma_metric(post_values)
    else:
        # fallback to beta if uncertain
        result = estimate_beta_metric(post_values)

    effect_size = (result["mean"] - baseline) / result["std"] if result[
                                                                     "std"] != 0 else np.nan
    direction = METRIC_DIRECTIONS.get(metric_name, "less")

    return BayesResult(
        pre_value=baseline,
        post_mean=result["mean"],
        post_std=result["std"],
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        effect_size=effect_size,
        direction=direction
    )


class DefenseStats:
    """Container for defense evaluation Bayesian estimates."""

    def __init__(self, corpus: str, threat_model: str, defense_model: str):
        """Initializes container for a specific experimental configuration.

        Args:
            corpus: Name of the dataset
            threat_model: Name of the threat model
            defense_model: Name of the defense model
        """
        self.corpus = corpus
        self.threat_model = threat_model
        self.defense_model = defense_model
        self.effectiveness_estimates = {}  # metric_name -> BayesResult
        self.quality_estimates = {}  # metric_name -> BayesResult

    def add_estimate(
            self,
            metric_name: str,
            metric_type: str,
            pre_value: float,
            post_values: List[float],
            display_name: str
    ) -> None:
        """Adds a Bayesian estimate for a metric.

        Args:
            metric_name: Name of the metric
            metric_type: Type of metric ('effectiveness' or 'quality')
            pre_value: Pre-intervention value
            post_values: List of post-intervention values
            display_name: Display name for the metric
        """
        if len(post_values) < 1:
            return

        # determine baseline based on metric type
        if metric_type == 'effectiveness':
            baseline = pre_value
        else:  # quality metrics
            if metric_name == 'pinc':
                baseline = 0.0
            else:
                baseline = 1.0

        bayes_result = bayesian_estimate(metric_name, metric_type, post_values,
                                         baseline)
        if metric_type == 'effectiveness':
            self.effectiveness_estimates[metric_name] = bayes_result
        else:
            self.quality_estimates[metric_name] = bayes_result

    def format_results(self, metric_name: str, metric_type: str) -> str:
        """Formats Bayesian results as a string.

        Args:
            metric_name: Name of the metric
            metric_type: Type of metric

        Returns:
            Formatted string with results
        """
        tests = self.effectiveness_estimates if metric_type == 'effectiveness' else self.quality_estimates
        if metric_name not in tests:
            return '-'

        result = tests[metric_name]
        base_str = f"{result.post_mean:.3f} (±{result.post_std:.3f})"
        # append the 95% credible interval
        base_str += f" [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        # optionally append effect size
        base_str += f" (d={result.effect_size:.3f})"
        return base_str

    def print_summary(self) -> None:
        """Prints a summary of all Bayesian estimates."""
        print("\nDefense Analysis Summary")
        print("=" * 80)
        print(f"Corpus: {self.corpus}")
        print(f"Threat Model: {self.threat_model}")
        print(f"Defense Model: {self.defense_model}\n")

        def print_group(name: str, estimates: Dict[str, BayesResult]) -> None:
            if not estimates:
                return
            print(f"\n{name} Metrics:")
            print("-" * 40)
            for metric_name, result in estimates.items():
                print(f"\n{metric_name}:")
                print(f"  Pre-defense:  {result.pre_value:.3f}")
                print(
                    f"  Post-defense: {result.post_mean:.3f} (±{result.post_std:.3f})")
                print(
                    f"  95% Credible Interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
                print(f"  Effect Size:  {result.effect_size:.3f}")
                print(f"  Direction:    {result.direction}")

        print_group("Effectiveness", self.effectiveness_estimates)
        print_group("Quality", self.quality_estimates)


def get_scenario_name(rq: str) -> str:
    """Converts research question identifier to readable name.

    Args:
        rq: Research question identifier

    Returns:
        Human-readable scenario name
    """
    parts = rq.split('_')
    if len(parts) > 1:
        return ' '.join(part.capitalize() for part in parts[1:])
    return rq


def format_value_with_significance(
        value: float,
        std: Optional[float] = None,
        brevity: bool = False,
        show_std: bool = True,
        metric_name: str = ""
) -> str:
    """Formats value with optional std and significance markers.

    Args:
        value: Numeric value to format
        std: Optional standard deviation
        brevity: Whether to use shortened format
        show_std: Whether to show standard deviation
        metric_name: Name of metric (for special formatting)

    Returns:
        Formatted string
    """
    val_str = f"{value:.3f}"
    if brevity:
        val_str = val_str.lstrip('0')
        if not val_str.startswith('.'):
            val_str = f"1{val_str}"
    base_str = val_str

    if std is not None and show_std:
        std_str = f"{std:.3f}"
        if brevity:
            std_str = std_str.lstrip('0')
            if not std_str.startswith('.'):
                std_str = f"1{std_str}"
        base_str = f"{val_str} (±{std_str})"
    return base_str


def calculate_effect_size(pre_value: float, post_values: List[float]) -> float:
    """Calculates Glass's d effect size.

    Args:
        pre_value: Pre-intervention value
        post_values: List of post-intervention values

    Returns:
        Effect size value
    """
    post_mean = np.mean(post_values)
    post_std = np.std(post_values)
    if post_std == 0:
        if post_mean > pre_value:
            return float('inf')
        elif post_mean < pre_value:
            return float('-inf')
        return 0.0
    return (post_mean - pre_value) / post_std


def calculate_improvement(
        orig_mean: float,
        post_mean: float,
        metric_type: str,
        relative: bool = True
) -> float:
    """Calculates improvement with consistent signs.

    Args:
        orig_mean: Original mean valueorig_mean: Original mean value
        post_mean: Post-intervention mean value
        metric_type: Type of metric (with direction indicator)
        relative: Whether to compute relative improvement

    Returns:
        Improvement value (signed appropriately)
    """
    if '↓' in metric_type:  # metrics we want to decrease
        if relative:
            return -((post_mean - orig_mean) / orig_mean) * 100
        return orig_mean - post_mean
    else:  # metrics we want to increase
        if relative:
            if orig_mean == 1.0:
                return (post_mean - orig_mean) * 100
            return ((post_mean - orig_mean) / (1 - orig_mean)) * 100
        return post_mean - orig_mean


def print_defense_stats(
    stats_dict: Dict[str, DefenseStats],
    corpus: Optional[str] = None,
    threat_model: Optional[str] = None,
    defense_model: Optional[str] = None
) -> None:
    """Prints Bayesian summaries with optional filtering.

    Args:
        stats_dict: Dictionary of DefenseStats objects
        corpus: Optional corpus filter
        threat_model: Optional threat model filter
        defense_model: Optional defense model filter
    """
    for key, stats in stats_dict.items():
        if (corpus is None or stats.corpus == corpus) and \
           (threat_model is None or stats.threat_model == threat_model) and \
           (defense_model is None or stats.defense_model == defense_model):
            stats.print_summary()


def display_copyable(df: pd.DataFrame) -> HTML:
    """Creates copyable display of dataframe with button.

    Args:
        df: Pandas DataFrame to display

    Returns:
        HTML object containing copyable text area and button
    """
    csv_string = df.to_csv(index=False, sep='\t')
    return HTML(f"""
    <textarea id="copyable_text" style="width: 100%; height: 200px;">{csv_string}</textarea>
    <button onclick="copyText()">Copy to Clipboard</button>
    <script>
    function copyText() {{
        var copyText = document.getElementById("copyable_text");
        copyText.select();
        document.execCommand("copy");
    }}
    </script>
    """)


def get_defense_tables(
    base_dir: str = "defense_evaluation",
    rq: str = "rq1.1_basic_paraphrase",
    corpora: Optional[List[str]] = None,
    threat_models: Optional[Dict[str, str]] = None,
    mode: str = "post",
    brevity: bool = False,
    show_std_with_significance: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Generates defense analysis tables.

    Args:
        base_dir: Base directory containing results
        rq: Research question to analyze
        corpora: List of corpora to analyze
        threat_models: Mapping of model types to display names
        mode: "pre" for baseline only, "post" for full analysis
        brevity: If true, remove leading zeros in output
        show_std_with_significance: If true, show std even with extra markers

    Returns:
        Either a single DataFrame (pre mode) or tuple of three DataFrames
        (post mode: post values, absolute improvements, relative improvements)

    Raises:
        ValueError: If mode is not "pre" or "post"
    """
    if corpora is None:
        corpora = ['ebg', 'rj']

    if threat_models is None:
        threat_models = {
            'logreg': 'LogReg',
            'svm': 'SVM',
            'roberta': 'RoBERTa'
        }

    if mode not in ["pre", "post"]:
        raise ValueError('mode must be one of ["pre", "post"]')

    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Label Conf ↓',
        'wrong_entropy': 'Wrong Class Entropy ↑',
        'mrr': 'MRR ↓',
        'kl_divergence': 'KL Divergence ↑'
    }

    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↓'),
        'meteor': ('meteor_avg', 'METEOR ↓'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↓')
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"

    if mode == "pre":
        # handle pre-intervention analysis
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

                    metrics = results[first_seed][threat_model_key]['attribution']['original_metrics']
                    row = {
                        'Corpus': corpus.upper(),
                        'Scenario': 'No protection',
                        'Threat Model': threat_model_name
                    }
                    for metric_key, display_name in metrics_map.items():
                        if metric_key != 'kl_divergence':
                            value = metrics.get(metric_key)
                            row[display_name] = format_value_with_significance(
                                value, None, brevity, show_std=show_std_with_significance
                            ) if value is not None else '-'
                    for _, (_, display_name) in quality_metrics.items():
                        row[display_name] = format_value_with_significance(
                            1.0, None, brevity, show_std=show_std_with_significance
                        )
                    pre_rows.append(row)
                    break  # only need one model's data

        columns = ['Corpus', 'Scenario', 'Threat Model'] + \
                  [v for k, v in metrics_map.items() if k != 'kl_divergence'] + \
                  [v[1] for v in quality_metrics.values()]
        return pd.DataFrame(pre_rows, columns=columns)

    else:  # mode == "post"
        # handle post-intervention analysis
        post_rows = []
        abs_imp_rows = []
        rel_imp_rows = []
        scenario_name = get_scenario_name(rq)

        for corpus in corpora:
            for threat_model_key, threat_model_name in threat_models.items():
                corpus_path = Path(base_dir) / corpus / rq_main / rq
                if not corpus_path.exists():
                    continue

                # iterate through each model directory
                for model_dir in corpus_path.glob("*"):
                    if not model_dir.is_dir():
                        continue

                    # determine display name for the model
                    model_dir_name = model_dir.name.lower()
                    if 'llama' in model_dir_name:
                        model_name = 'Llama 3.1'
                    elif 'gemma' in model_dir_name:
                        model_name = 'Gemma 2'
                    elif 'ministral' in model_dir_name:
                        model_name = 'Ministral'
                    elif 'sonnet' in model_dir_name:
                        model_name = 'Sonnet 3.5'
                    elif 'gpt' in model_dir_name:
                        model_name = 'GPT4o'
                    else:
                        model_name = model_dir.name

                    eval_file = model_dir / "evaluation.json"
                    if not eval_file.exists():
                        continue

                    # collect metrics
                    orig_metrics = defaultdict(list)
                    post_metrics = defaultdict(list)
                    quality_scores = defaultdict(list)

                    with open(eval_file) as f:
                        results = json.load(f)

                    for seed_results in results.values():
                        if threat_model_key not in seed_results:
                            continue
                        metrics = seed_results[threat_model_key]['attribution']

                        # process attribution metrics
                        for metric_key in metrics_map.keys():
                            if metric_key == 'kl_divergence':
                                if 'kl_divergence' in metrics['transformed_metrics']:
                                    post_metrics[metric_key].append(
                                        float(metrics['transformed_metrics']['kl_divergence'])
                                    )
                                orig_metrics[metric_key].append(0.0)
                                continue
                            orig_val = metrics['original_metrics'].get(metric_key)
                            post_val = metrics['transformed_metrics'].get(metric_key)
                            if orig_val is not None:
                                orig_metrics[metric_key].append(float(orig_val))
                            if post_val is not None:
                                post_metrics[metric_key].append(float(post_val))

                        # process quality metrics
                        quality = seed_results[threat_model_key].get('quality', {})
                        for qm, (key, _) in quality_metrics.items():
                            score = quality.get(qm, {}).get(key)
                            if score is not None:
                                quality_scores[qm].append(float(score))

                    # prepare result rows
                    base_row = {
                        'Corpus': corpus.upper(),
                        'Scenario': scenario_name,
                        'Threat Model': threat_model_name,
                        'Defense Model': model_name
                    }
                    post_row = base_row.copy()
                    abs_imp_row = base_row.copy()
                    rel_imp_row = base_row.copy()

                    # process effectiveness metrics
                    for metric_key, display_name in metrics_map.items():
                        if metric_key == 'kl_divergence':
                            if metric_key in post_metrics:
                                post_mean = np.mean(post_metrics[metric_key])
                                post_std = np.std(post_metrics[metric_key])
                                post_row[display_name] = format_value_with_significance(
                                    post_mean, post_std, brevity,
                                    show_std=show_std_with_significance
                                )
                                abs_imp_row[display_name] = '-'
                                rel_imp_row[display_name] = '-'
                            continue

                        orig_vals = orig_metrics[metric_key]
                        post_vals = post_metrics[metric_key]
                        if orig_vals and post_vals:
                            baseline = np.mean(orig_vals)
                            post_mean = np.mean(post_vals)
                            post_std = np.std(post_vals)
                            post_row[display_name] = format_value_with_significance(
                                post_mean, post_std, brevity,
                                show_std=show_std_with_significance
                            )

                            # compute improvements
                            orig_mean = np.mean(orig_vals)
                            abs_improvements = []
                            rel_improvements = []
                            for orig, post in zip(orig_vals, post_vals):
                                abs_imp = calculate_improvement(
                                    orig, post, display_name, False
                                )
                                rel_imp = calculate_improvement(
                                    orig, post, display_name, True
                                )
                                abs_improvements.append(abs_imp)
                                rel_improvements.append(rel_imp)

                            # format improvement values
                            abs_mean = np.mean(abs_improvements)
                            abs_std = np.std(abs_improvements)
                            abs_imp_row[display_name] = format_value_with_significance(
                                abs_mean, abs_std, brevity,
                                show_std=show_std_with_significance,
                                metric_name=display_name
                            )

                            rel_mean = np.mean(rel_improvements)
                            rel_std = np.std(rel_improvements)
                            rel_imp_row[display_name] = format_value_with_significance(
                                rel_mean, rel_std, brevity,
                                show_std=show_std_with_significance,
                                metric_name=display_name
                            ) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'

                    # process quality metrics
                    for qm, (key, display_name) in quality_metrics.items():
                        values = quality_scores[qm]
                        if values:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            post_row[display_name] = format_value_with_significance(
                                mean_val, std_val, brevity,
                                show_std=show_std_with_significance
                            )

                            # compute quality improvements
                            abs_improvements = [v - 1.0 for v in values]
                            rel_improvements = [(v - 1.0) * 100 for v in values]

                            abs_mean = np.mean(abs_improvements)
                            abs_std = np.std(abs_improvements)
                            abs_imp_row[display_name] = format_value_with_significance(
                                abs_mean, abs_std, brevity,
                                show_std=show_std_with_significance
                            )

                            rel_mean = np.mean(rel_improvements)
                            rel_std = np.std(rel_improvements)
                            rel_imp_row[display_name] = format_value_with_significance(
                                rel_mean, rel_std, brevity,
                                show_std=show_std_with_significance
                            ) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'

                    post_rows.append(post_row)
                    abs_imp_rows.append(abs_imp_row)
                    rel_imp_rows.append(rel_imp_row)

        # prepare output dataframes
        columns = ['Corpus', 'Scenario', 'Threat Model', 'Defense Model'] + \
                  list(metrics_map.values()) + \
                  [v[1] for v in quality_metrics.values()]

        post_df = pd.DataFrame(post_rows, columns=columns)
        abs_imp_df = pd.DataFrame(abs_imp_rows, columns=columns)
        rel_imp_df = pd.DataFrame(rel_imp_rows, columns=columns)
        return post_df, abs_imp_df, rel_imp_df


def get_defense_tables_with_stats(
    base_dir: str = "defense_evaluation",
    rq: str = "rq1.1_basic_paraphrase",
    corpora: Optional[List[str]] = None,
    threat_models: Optional[Dict[str, str]] = None,
    mode: str = "post",
    brevity: bool = False,
    show_std_with_significance: bool = True
) -> Union[pd.DataFrame, Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, DefenseStats]]]:
    """Enhanced version of get_defense_tables that includes Bayesian estimation.
    
    Args:
        base_dir: Base directory containing results
        rq: Research question to analyze
        corpora: List of corpora to analyze
        threat_models: Mapping of model types to display names
        mode: "pre" for baseline only, "post" for full analysis
        brevity: If true, remove leading zeros in output
        show_std_with_significance: If true, show std even with extra markers

    Returns:
        In pre mode: single DataFrame
        In post mode: tuple of (post_df, abs_imp_df, rel_imp_df, stats_dict)
    """
    if mode == "pre":
        return get_defense_tables(
            base_dir, rq, corpora, threat_models, mode, brevity,
            show_std_with_significance
        )

    post_df, abs_imp_df, rel_imp_df = get_defense_tables(
        base_dir, rq, corpora, threat_models, mode, brevity,
        show_std_with_significance
    )

    if corpora is None:
        corpora = ['ebg', 'rj']

    if threat_models is None:
        threat_models = {
            'logreg': 'LogReg',
            'svm': 'SVM',
            'roberta': 'RoBERTa'
        }

    stats_dict = {}
    metrics_map = {
        'accuracy@1': 'Acc@1 ↓',
        'accuracy@5': 'Acc@5 ↓',
        'true_class_confidence': 'True Label Conf ↓',
        'wrong_entropy': 'Wrong Class Entropy ↑',
        'mrr': 'MRR ↓',
        'kl_divergence': 'KL Divergence ↑'
    }

    quality_metrics = {
        'bleu': {'key': 'bleu', 'display': 'BLEU ↓', 'baseline': 1.0},
        'meteor': {'key': 'meteor_avg', 'display': 'METEOR ↓', 'baseline': 1.0},
        'pinc': {'key': 'pinc_overall_avg', 'display': 'PINC ↑', 'baseline': 0.0},
        'bertscore': {'key': 'bertscore_f1_avg', 'display': 'BERTScore ↓', 'baseline': 1.0}
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"
    
    # populate stats_dict with bayesian estimates
    for corpus in corpora:
        for threat_model_key, threat_model_name in threat_models.items():
            corpus_path = Path(base_dir) / corpus / rq_main / rq
            if not corpus_path.exists():
                continue
                
            for model_dir in corpus_path.glob("*"):
                if not model_dir.is_dir():
                    continue
                    
                # determine model display name
                model_dir_name = model_dir.name.lower()
                if 'llama' in model_dir_name:
                    model_name = 'Llama 3.1'
                elif 'gemma' in model_dir_name:
                    model_name = 'Gemma 2'
                elif 'ministral' in model_dir_name:
                    model_name = 'Ministral'
                elif 'sonnet' in model_dir_name:
                    model_name = 'Sonnet 3.5'
                elif 'gpt' in model_dir_name:
                    model_name = 'GPT4o'
                else:
                    model_name = model_dir.name

                eval_file = model_dir / "evaluation.json"
                if not eval_file.exists():
                    continue

                # initialize stats container
                stats = DefenseStats(corpus, threat_model_name, model_name)
                orig_metrics = defaultdict(list)
                post_metrics = defaultdict(list)
                quality_scores = defaultdict(list)
                
                # load and process results
                with open(eval_file) as f:
                    results = json.load(f)
                    
                for seed_results in results.values():
                    if threat_model_key not in seed_results:
                        continue
                        
                    metrics = seed_results[threat_model_key]['attribution']
                    
                    # collect effectiveness metrics
                    for metric_key in metrics_map.keys():
                        if metric_key == 'kl_divergence':
                            if 'kl_divergence' in metrics['transformed_metrics']:
                                post_metrics[metric_key].append(
                                    float(metrics['transformed_metrics']['kl_divergence'])
                                )
                            orig_metrics[metric_key].append(0.0)
                            continue
                        orig_val = metrics['original_metrics'].get(metric_key)
                        post_val = metrics['transformed_metrics'].get(metric_key)
                        if orig_val is not None:
                            orig_metrics[metric_key].append(float(orig_val))
                        if post_val is not None:
                            post_metrics[metric_key].append(float(post_val))
                            
                    # collect quality metrics
                    quality = seed_results[threat_model_key].get('quality', {})
                    for qm, metric_info in quality_metrics.items():
                        score = quality.get(qm, {}).get(metric_info['key'])
                        if score is not None:
                            quality_scores[qm].append(float(score))

                # add bayesian estimates for effectiveness metrics
                for metric_key, display_name in metrics_map.items():
                    orig_vals = orig_metrics[metric_key]
                    post_vals = post_metrics[metric_key]
                    if orig_vals and post_vals:
                        baseline = np.mean(orig_vals)
                        stats.add_estimate(
                            metric_key, 'effectiveness', baseline, post_vals,
                            display_name
                        )

                # add bayesian estimates for quality metrics
                for qm, metric_info in quality_metrics.items():
                    values = quality_scores[qm]
                    if values:
                        stats.add_estimate(
                            qm, 'quality', metric_info['baseline'], values,
                            metric_info['display']
                        )

                stats_dict[(corpus, threat_model_name, model_name)] = stats

    return post_df, abs_imp_df, rel_imp_df, stats_dict

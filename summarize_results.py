"""
defense evaluation script that performs bayesian analysis comparing pre/post defense
performance across different metrics, corpora, and threat models.

directory structure expected:
defense_evaluation/
├── {corpus}/                              # rj and ebg
│   ├── rq{N}/                            # main research question (e.g., rq1)
│   │   ├── rq{N}.{M}/                    # sub-question (e.g., rq1.1)
│   │   │   ├── {model_name}/             # e.g., gemma-2b-it
│   │   │   │   ├── evaluation.json       # consolidated results
│   │   │   │   └── seed_{seed}.json      # per-seed results
│   │   │   └── {another_model}/
│   │   └── rq{N}.{M+1}/
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
from IPython.display import HTML

# define the expected direction for each metric
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


# bayesian result namedtuple to store posterior summaries
class BayesResult(NamedTuple):
    pre_value: float            # baseline value (pre-intervention)
    post_mean: float            # posterior mean of the parameter (e.g., mu for beta, mean for gamma)
    post_std: float             # posterior std of the parameter
    ci_lower: float             # lower bound of the 95% credible interval
    ci_upper: float             # upper bound of the 95% credible interval
    effect_size: float          # (post_mean - baseline)/post_std
    direction: str              # expected direction ('less' or 'greater')


import arviz as az
# helper function to estimate a beta-model parameter from post_values
import numpy as np
import pymc as pm


def estimate_beta_metric(post_values: list) -> dict:
    """
    estimate beta distribution parameters with 95% HDI intervals.

    args:
        post_values: list of values between 0 and 1

    returns:
        dict with posterior mean, std and HDI intervals
    """
    post_values = np.array(post_values)

    with pm.Model() as model:
        # priors
        mu = pm.Beta("mu", alpha=1, beta=1)  # uniform prior over (0,1)
        kappa = pm.HalfNormal("kappa", sigma=10)  # weakly informative prior

        # likelihood
        obs = pm.Beta("obs",
                      alpha=mu * kappa,
                      beta=(1 - mu) * kappa,
                      observed=post_values)

        # sample from posterior
        trace = pm.sample(2000,
                          tune=1000,
                          cores=1,  # reduced from 4 for better reproducibility
                          progressbar=False,
                          random_seed=42,
                          target_accept=0.95,
                          return_inferencedata=True)

    # compute posterior statistics
    mu_samples = trace.posterior["mu"].values.flatten()
    mean_mu = float(np.mean(mu_samples))  # ensure python float
    std_mu = float(np.std(mu_samples))

    # compute 95% HDI
    hdi_bounds = az.hdi(trace.posterior["mu"], hdi_prob=0.95)

    return {
        "mean": mean_mu,
        "std": std_mu,
        "hdi_lower": float(max(0, hdi_bounds[0])),  # ensure bounds in [0,1]
        "hdi_upper": float(min(1, hdi_bounds[1]))
    }

# main bayesian estimation function for a metric
def bayesian_estimate(metric_name: str, metric_type: str, post_values: List[float], baseline: float) -> BayesResult:
    """
    ; given a list of post_values for a metric, use an appropriate likelihood
    ; to estimate the posterior mean, std, and 95% credible interval.
    ; baseline is the pre-intervention value (or fixed baseline for quality metrics)
    """
    # ; for metrics naturally in [0,1] use beta model; for others (e.g., wrong_entropy, kl_divergence) use gamma
    if metric_name in ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'mrr',
                       'bleu', 'meteor', 'bertscore', 'pinc']:
        result = estimate_beta_metric(post_values)
    elif metric_name in ['wrong_entropy', 'kl_divergence']:
        result = estimate_entropy_post(post_values)
    else:
        # ; fallback to beta if uncertain (should not occur)
        result = estimate_beta_metric(post_values)

    effect_size = (result["mean"] - baseline) / result["std"] if result["std"] != 0 else np.nan
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


# new defense stats container that holds bayesian estimates (no hypothesis testing)
class DefenseStats:
    """container for defense evaluation bayesian estimates for each metric."""

    def __init__(self, corpus: str, threat_model: str, defense_model: str):
        self.corpus = corpus
        self.threat_model = threat_model
        self.defense_model = defense_model
        self.effectiveness_estimates = {}  # metric_name -> BayesResult
        self.quality_estimates = {}        # metric_name -> BayesResult

    def add_estimate(self, metric_name: str,
                     metric_type: str,
                     pre_value: float,
                     post_values: List[float],
                     display_name: str):
        """add bayesian estimate for a metric based on post_values.
        ; for effectiveness metrics, baseline is pre_value; for quality metrics,
        ; baseline is 1 for bleu/meteor/bertscore and 0 for pinc.
        """
        if len(post_values) < 1:
            return

        # ; determine baseline based on metric type
        if metric_type == 'effectiveness':
            baseline = pre_value
        else:  # quality metrics
            if metric_name == 'pinc':
                baseline = 0.0
            else:
                baseline = 1.0

        bayes_result = bayesian_estimate(metric_name, metric_type, post_values, baseline)
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

        # Format with credible interval instead of std
        return format_value_with_significance(
            result.post_mean,
            std=None,
            brevity=False,
            show_std=False,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper
        )

    def print_summary(self):
        """print a summary of all bayesian estimates."""
        print("\ndefense analysis summary")
        print("=" * 80)
        print(f"corpus: {self.corpus}")
        print(f"threat model: {self.threat_model}")
        print(f"defense model: {self.defense_model}\n")

        def print_group(name: str, estimates: Dict[str, BayesResult]):
            if not estimates:
                return
            print(f"\n{name} metrics:")
            print("-" * 40)
            for metric_name, result in estimates.items():
                print(f"\n{metric_name}:")
                print(f"  pre-defense:  {result.pre_value:.3f}")
                print(f"  post-defense: {result.post_mean:.3f} (±{result.post_std:.3f})")
                print(f"  95% credible interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
                print(f"  effect size:  {result.effect_size:.3f}")
                print(f"  direction:    {result.direction}")

        print_group("effectiveness", self.effectiveness_estimates)
        print_group("quality", self.quality_estimates)


def get_scenario_name(rq: str) -> str:
    """convert rq identifier to readable scenario name."""
    parts = rq.split('_')
    if len(parts) > 1:
        return ' '.join(part.capitalize() for part in parts[1:])
    return rq


def format_value_with_significance(
        value: float,
        std: Optional[float] = None,
        brevity: bool = False,
        show_std: bool = True,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        metric_name: Optional[str] = None  # Add this parameter
) -> str:
    """Format value with optional std deviation or credible intervals.

    Args:
        value: value to format
        std: standard deviation (optional)
        brevity: if True, remove leading zeros
        show_std: if True, show std even when CI is present
        ci_lower: lower bound of credible interval (optional)
        ci_upper: upper bound of credible interval (optional)
        metric_name: name of metric (optional, for special formatting)
    """
    if brevity:
        val_str = f"{value:.3f}".lstrip('0')
        val_str = val_str if val_str.startswith('.') else f"1{val_str}"
    else:
        val_str = f"{value:.3f}"

    # If credible intervals are provided, use those instead of std
    if ci_lower is not None and ci_upper is not None:
        if brevity:
            ci_lower_str = f"{ci_lower:.3f}".lstrip('0')
            ci_upper_str = f"{ci_upper:.3f}".lstrip('0')
            ci_lower_str = ci_lower_str if ci_lower_str.startswith('.') else f"1{ci_lower_str}"
            ci_upper_str = ci_upper_str if ci_upper_str.startswith('.') else f"1{ci_upper_str}"
        else:
            ci_lower_str = f"{ci_lower:.3f}"
            ci_upper_str = f"{ci_upper:.3f}"

        return f"{val_str} [{ci_lower_str}, {ci_upper_str}]"

    # Fall back to std if no CI provided
    if std is not None and show_std:
        if brevity:
            std_str = f"{std:.3f}".lstrip('0')
            std_str = std_str if std_str.startswith('.') else f"1{std_str}"
        else:
            std_str = f"{std:.3f}"
        return f"{val_str} (±{std_str})"

    return val_str

def calculate_effect_size(pre_value: float, post_values: List[float]) -> float:
    """calculate glass's d effect size."""
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
    """calculate improvement with consistent signs."""
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
):
    """print bayesian summaries with optional filtering."""
    for key, stats in stats_dict.items():
        if (corpus is None or stats.corpus == corpus) and \
                (threat_model is None or stats.threat_model == threat_model) and \
                (defense_model is None or stats.defense_model == defense_model):
            stats.print_summary()


def display_copyable(df: pd.DataFrame) -> HTML:
    """display dataframe in a copyable format with copy button."""
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
    """generate defense analysis tables.

    args:
        base_dir: base directory containing results
        rq: research question to analyze
        corpora: list of corpora to analyze
        threat_models: mapping of model types to display names
        mode: "pre" for baseline only, "post" for full analysis
        brevity: if true, remove leading zeros in output
        show_std_with_significance: if true, show std even with extra markers
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

                    # ; determine display name for the model
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

                    orig_metrics = defaultdict(list)
                    post_metrics = defaultdict(list)
                    quality_scores = defaultdict(list)

                    with open(eval_file) as f:
                        results = json.load(f)

                    for seed_results in results.values():
                        if threat_model_key not in seed_results:
                            continue
                        metrics = seed_results[threat_model_key]['attribution']
                        for metric_key in metrics_map.keys():
                            if metric_key == 'kl_divergence':
                                if 'kl_divergence' in metrics['transformed_metrics']:
                                    post_metrics[metric_key].append(
                                        float(metrics['transformed_metrics']['kl_divergence'])
                                    )
                                # ; add a pre-value of 0 for kl_divergence
                                orig_metrics[metric_key].append(0.0)
                                continue
                            orig_val = metrics['original_metrics'].get(metric_key)
                            post_val = metrics['transformed_metrics'].get(metric_key)
                            if orig_val is not None:
                                orig_metrics[metric_key].append(float(orig_val))
                            if post_val is not None:
                                post_metrics[metric_key].append(float(post_val))
                        quality = seed_results[threat_model_key].get('quality', {})
                        for qm, (key, _) in quality_metrics.items():
                            score = quality.get(qm, {}).get(key)
                            if score is not None:
                                quality_scores[qm].append(float(score))

                    base_row = {
                        'Corpus': corpus.upper(),
                        'Scenario': scenario_name,
                        'Threat Model': threat_model_name,
                        'Defense Model': model_name
                    }
                    post_row = base_row.copy()
                    abs_imp_row = base_row.copy()
                    rel_imp_row = base_row.copy()

                    for metric_key, display_name in metrics_map.items():
                        if metric_key == 'kl_divergence':
                            if metric_key in post_metrics:
                                post_mean = np.mean(post_metrics[metric_key])
                                post_std = np.std(post_metrics[metric_key])
                                post_row[display_name] = format_value_with_significance(
                                    post_mean, post_std, brevity, show_std=show_std_with_significance
                                )
                                abs_imp_row[display_name] = '-'
                                rel_imp_row[display_name] = '-'
                            continue

                        orig_vals = orig_metrics[metric_key]
                        post_vals = post_metrics[metric_key]
                        if orig_vals and post_vals:
                            # ; compute bayesian estimate for post-defense values
                            # ; baseline for effectiveness is the mean of original values
                            baseline = np.mean(orig_vals)
                            # ; add bayesian estimate to defense stats later; for now, display summary stats
                            post_mean = np.mean(post_vals)
                            post_std = np.std(post_vals)
                            post_row[display_name] = format_value_with_significance(
                                post_mean, post_std, brevity, show_std=show_std_with_significance
                            )
                            # ; compute improvements for absolute and relative
                            orig_mean = np.mean(orig_vals)
                            abs_improvements = []
                            rel_improvements = []
                            for orig, post in zip(orig_vals, post_vals):
                                abs_imp = calculate_improvement(orig, post, display_name, False)
                                rel_imp = calculate_improvement(orig, post, display_name, True)
                                abs_improvements.append(abs_imp)
                                rel_improvements.append(rel_imp)
                            abs_mean = np.mean(abs_improvements)
                            abs_std = np.std(abs_improvements)
                            abs_imp_row[display_name] = format_value_with_significance(
                                abs_mean, abs_std, brevity, show_std=show_std_with_significance,
                                metric_name=display_name
                            )
                            rel_mean = np.mean(rel_improvements)
                            rel_std = np.std(rel_improvements)
                            rel_imp_row[display_name] = format_value_with_significance(
                                rel_mean, rel_std, brevity, show_std=show_std_with_significance,
                                metric_name=display_name
                            ) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'

                    # ; process quality metrics
                    for qm, (key, display_name) in quality_metrics.items():
                        values = quality_scores[qm]
                        if values:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            post_row[display_name] = format_value_with_significance(
                                mean_val, std_val, brevity, show_std=show_std_with_significance
                            )
                            abs_improvements = [v - 1.0 for v in values]
                            rel_improvements = [(v - 1.0) * 100 for v in values]
                            abs_mean = np.mean(abs_improvements)
                            abs_std = np.std(abs_improvements)
                            abs_imp_row[display_name] = format_value_with_significance(
                                abs_mean, abs_std, brevity, show_std=show_std_with_significance
                            )
                            rel_mean = np.mean(rel_improvements)
                            rel_std = np.std(rel_improvements)
                            rel_imp_row[display_name] = format_value_with_significance(
                                rel_mean, rel_std, brevity, show_std=show_std_with_significance
                            ) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'

                    post_rows.append(post_row)
                    abs_imp_rows.append(abs_imp_row)
                    rel_imp_rows.append(rel_imp_row)

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
        show_std_with_significance: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, DefenseStats]]]:
    """Generate defense analysis tables with Bayesian credible intervals.

    Args:
        base_dir: base directory containing results
        rq: research question to analyze
        corpora: list of corpora to analyze
        threat_models: mapping of model types to display names
        mode: "pre" for baseline only, "post" for full analysis
        brevity: if True, remove leading zeros in output
        show_std_with_significance: if True, show std even with credible intervals

    Returns:
        If mode=="pre": DataFrame with pre-defense values
        If mode=="post": Tuple of (post_values_df, absolute_improvements_df,
                                 relative_improvements_df, stats_dict)
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
        'pinc': ('pinc_overall_avg', 'PINC ↓'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↓')
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"

    if mode == "pre":
        return get_defense_tables(base_dir, rq, corpora, threat_models, mode, brevity)

    post_df, abs_imp_df, rel_imp_df = get_defense_tables(
        base_dir, rq, corpora, threat_models, mode, brevity)

    stats_dict = {}

    for corpus in corpora:
        for threat_model_key, threat_model_name in threat_models.items():
            corpus_path = Path(base_dir) / corpus / rq_main / rq
            if not corpus_path.exists():
                continue

            for model_dir in corpus_path.glob("*"):
                if not model_dir.is_dir():
                    continue

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

                stats = DefenseStats(corpus, threat_model_name, model_name)
                orig_metrics = defaultdict(list)
                post_metrics = defaultdict(list)
                quality_scores = defaultdict(list)

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
                    for qm, (key, _) in quality_metrics.items():
                        score = quality.get(qm, {}).get(key)
                        if score is not None:
                            quality_scores[qm].append(float(score))

                # Process effectiveness metrics with Bayesian estimation
                for metric_key, display_name in metrics_map.items():
                    orig_vals = orig_metrics[metric_key]
                    post_vals = post_metrics[metric_key]

                    if orig_vals and post_vals:
                        stats.add_estimate(
                            metric_key,
                            'effectiveness',
                            np.mean(orig_vals),
                            post_vals,
                            display_name
                        )

                # Process quality metrics with Bayesian estimation
                for qm, (key, display_name) in quality_metrics.items():
                    values = quality_scores[qm]
                    if values:
                        stats.add_estimate(
                            qm,
                            'quality',
                            1.0,  # baseline for quality metrics
                            values,
                            display_name
                        )

                stats_dict[(corpus, threat_model_name, model_name)] = stats

                # Update DataFrames with credible intervals
                key = (corpus, threat_model_name, model_name)
                if key in stats_dict:
                    stat_result = stats_dict[key]
                    idx = post_df[
                        (post_df['Corpus'] == corpus.upper()) &
                        (post_df['Threat Model'] == threat_model_name) &
                        (post_df['Defense Model'] == model_name)
                        ].index

                    if len(idx) > 0:
                        idx = idx[0]
                        # Update effectiveness metrics
                        for metric_key, display_name in metrics_map.items():
                            if metric_key in stat_result.effectiveness_estimates:
                                result = stat_result.effectiveness_estimates[metric_key]
                                if display_name in post_df.columns:
                                    post_df.at[idx, display_name] = format_value_with_significance(
                                        result.post_mean,
                                        std=None,
                                        brevity=brevity,
                                        show_std=False,
                                        ci_lower=result.ci_lower,
                                        ci_upper=result.ci_upper
                                    )

                        # Update quality metrics
                        for qm, (_, display_name) in quality_metrics.items():
                            if qm in stat_result.quality_estimates:
                                result = stat_result.quality_estimates[qm]
                                if display_name in post_df.columns:
                                    post_df.at[idx, display_name] = format_value_with_significance(
                                        result.post_mean,
                                        std=None,
                                        brevity=brevity,
                                        show_std=False,
                                        ci_lower=result.ci_lower,
                                        ci_upper=result.ci_upper
                                    )

    return post_df, abs_imp_df, rel_imp_df, stats_dict

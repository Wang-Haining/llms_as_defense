"""
defense evaluation script that performs bayesian analysis comparing pre/post defense
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
    """
    estimate beta distribution parameters with 95% hdi intervals for metrics
    naturally bounded in [0,1]
    """
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
            progressbar=False,
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
    """
    use the beta estimation function for all metrics (except for entropy change,
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
    """container for defense evaluation bayesian estimates for each metric"""

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
        add bayesian estimate for a metric based on post_values
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
            brevity=False,
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
    """convert rq identifier to a readable scenario name"""
    parts = rq.split('_')
    if len(parts) > 1:
        return ' '.join(part.capitalize() for part in parts[1:])
    return rq


def format_estimate_with_hdi(
        value: float,
        brevity: bool = False,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None
) -> str:
    """format a value with its 95% hdi interval (ignoring std)

    if brevity is true, leading zeros are removed
    """
    if brevity:
        val_str = f"{value:.3f}".lstrip('0')
        val_str = val_str if val_str.startswith('.') else f"1{val_str}"
    else:
        val_str = f"{value:.3f}"

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

    return val_str


def calculate_effect_size(pre_value: float, post_values: List[float]) -> float:
    """calculate glass's d effect size using post values"""
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
    """print bayesian summaries with optional filtering"""
    for key, stats in stats_dict.items():
        if ((corpus is None or stats.corpus == corpus) and
            (threat_model is None or stats.threat_model == threat_model) and
            (defense_model is None or stats.defense_model == defense_model)):
            stats.print_summary()


def display_copyable(df: pd.DataFrame) -> HTML:
    """display dataframe in a copyable format with a copy button"""
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
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    generate defense analysis tables

    for pre mode, quality measures are set to their ideal (1 for most, 0 for pinc)
    for post mode, the output includes the post values for effectiveness and quality metrics

    returns:
        if mode=="pre": dataframe with pre-defense values
        if mode=="post": tuple of (post_df, abs_imp_df, rel_imp_df)
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

    # quality metrics (sample-level): for a text compared to itself these yield 1 (except pinc yields 0)
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
                        row[display_name] = format_estimate_with_hdi(value, brevity=brevity)
                    for _, (_, display_name) in quality_metrics.items():
                        # pre-defense quality is ideal (1 for most, 0 for pinc)
                        baseline = 0.0 if _ == 'pinc' else 1.0
                        row[display_name] = format_estimate_with_hdi(baseline, brevity=brevity)
                    pre_rows.append(row)
                    break  # only need one model's data
        columns = ['Corpus', 'Scenario', 'Threat Model'] + list(
            metrics_map.values()) + [v[1] for v in quality_metrics.values()]
        return pd.DataFrame(pre_rows, columns=columns)

    else:  # mode == "post"
        post_rows = []
        abs_imp_rows = []
        rel_imp_rows = []
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
                    orig_metrics = defaultdict(list)
                    post_metrics = defaultdict(list)
                    quality_scores = defaultdict(list)
                    with open(eval_file) as f:
                        results = json.load(f)
                    for seed_results in results.values():
                        if threat_model_key not in seed_results:
                            continue
                        metrics = seed_results[threat_model_key]['attribution']
                        for metric_key, _ in metrics_map.items():
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
                    base_row = {'Corpus': corpus.upper(),
                                'Scenario': get_scenario_name(rq),
                                'Threat Model': threat_model_name,
                                'Defense Model': model_name}
                    post_row = base_row.copy()
                    abs_imp_row = base_row.copy()
                    rel_imp_row = base_row.copy()
                    for metric_key, display_name in metrics_map.items():
                        orig_vals = orig_metrics[metric_key]
                        post_vals = post_metrics[metric_key]
                        if orig_vals and post_vals:
                            post_mean = np.mean(post_vals)
                            post_row[display_name] = format_estimate_with_hdi(
                                post_mean, brevity=brevity,
                                ci_lower=np.mean(post_vals) - np.std(post_vals) * 1.96,
                                ci_upper=np.mean(post_vals) + np.std(post_vals) * 1.96)
                            # abs_imp is computed from raw differences (to be superseded by bayesian changes later)
                            abs_improvements = [p - o for o, p in zip(orig_vals, post_vals)]
                            rel_improvements = [((p - o) / o) * 100 if o != 0 else np.nan
                                                for o, p in zip(orig_vals, post_vals)]
                            abs_imp_row[display_name] = format_estimate_with_hdi(
                                np.mean(abs_improvements), brevity=brevity,
                                ci_lower=np.mean(abs_improvements) - np.std(abs_improvements) * 1.96,
                                ci_upper=np.mean(abs_improvements) + np.std(abs_improvements) * 1.96)
                            rel_imp_row[display_name] = format_estimate_with_hdi(
                                np.mean(rel_improvements), brevity=brevity,
                                ci_lower=np.mean(rel_improvements) - np.std(rel_improvements) * 1.96,
                                ci_upper=np.mean(rel_improvements) + np.std(rel_improvements) * 1.96) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'
                    for qm, (key, display_name) in quality_metrics.items():
                        values = quality_scores[qm]
                        if values:
                            mean_val = np.mean(values)
                            post_row[display_name] = format_estimate_with_hdi(
                                mean_val, brevity=brevity,
                                ci_lower=np.mean(values) - np.std(values) * 1.96,
                                ci_upper=np.mean(values) + np.std(values) * 1.96)
                            # for quality measures, baseline is 1 (except pinc baseline 0)
                            base_val = 0.0 if qm == 'pinc' else 1.0
                            abs_improvements = [v - base_val for v in values]
                            rel_improvements = [((v - base_val) * 100) for v in values]
                            abs_imp_row[display_name] = format_estimate_with_hdi(
                                np.mean(abs_improvements), brevity=brevity,
                                ci_lower=np.mean(abs_improvements) - np.std(abs_improvements) * 1.96,
                                ci_upper=np.mean(abs_improvements) + np.std(abs_improvements) * 1.96)
                            rel_imp_row[display_name] = format_estimate_with_hdi(
                                np.mean(rel_improvements), brevity=brevity,
                                ci_lower=np.mean(rel_improvements) - np.std(rel_improvements) * 1.96,
                                ci_upper=np.mean(rel_improvements) + np.std(rel_improvements) * 1.96) + "%"
                        else:
                            post_row[display_name] = '-'
                            abs_imp_row[display_name] = '-'
                            rel_imp_row[display_name] = '-'
                    post_rows.append(post_row)
                    abs_imp_rows.append(abs_imp_row)
                    rel_imp_rows.append(rel_imp_row)
        columns = ['Corpus', 'Scenario', 'Threat Model', 'Defense Model'] + list(metrics_map.values()) + [v[1] for v in quality_metrics.values()]
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
) -> Union[pd.DataFrame, Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[
        str, DefenseStats]]]:
    """
    generate defense analysis tables with bayesian credible intervals and additional
    dataframes for absolute and relative changes (with 95% hdi) computed from bayesian estimates
    progress bars are added to track processing

    returns a tuple of:
      - post_df: formatted post-intervention values (using bayesian estimates)
      - abs_imp_df: original absolute improvements computed from raw data
      - rel_imp_df: original relative improvements computed from raw data
      - abs_change_df: absolute change (bayesian estimate) = post_mean - pre_value, with its 95% hdi
      - rel_change_df: relative change (bayesian estimate) in percent, with its 95% hdi
      - stats_dict: dictionary mapping (corpus, threat_model, defense_model) to DefenseStats objects
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
    # quality metrics (sample-level) with additional sbert
    quality_metrics = {
        'bleu': ('bleu', 'BLEU ↑'),
        'meteor': ('meteor_avg', 'METEOR ↑'),
        'pinc': ('pinc_overall_avg', 'PINC ↑'),
        'bertscore': ('bertscore_f1_avg', 'BERTScore ↑'),
        'sbert': ('sbert_similarity_avg', 'SBERT ↑')
    }

    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"

    # get the basic tables from previous function
    post_df, abs_imp_df, rel_imp_df = get_defense_tables(base_dir, rq, corpora,
                                                         threat_models, mode, brevity)

    # prepare additional dataframes for bayesian change estimates
    abs_change_rows = []
    rel_change_rows = []

    stats_dict = {}

    for corpus in tqdm(corpora, desc="processing corpora"):
        for threat_model_key, threat_model_name in threat_models.items():
            corpus_path = Path(base_dir) / corpus / rq_main / rq
            if not corpus_path.exists():
                continue
            for model_dir in tqdm(list(corpus_path.glob("*")),
                                  desc=f"processing models for {corpus} {threat_model_name}",
                                  leave=False):
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
                    for metric_key, _ in metrics_map.items():
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
                # process effectiveness metrics
                for metric_key, display_name in metrics_map.items():
                    orig_vals = orig_metrics[metric_key]
                    post_vals = post_metrics[metric_key]
                    if orig_vals and post_vals:
                        stats.add_estimate(metric_key, 'effectiveness',
                                           np.mean(orig_vals), post_vals, display_name)
                # process quality metrics
                for qm, (key, display_name) in quality_metrics.items():
                    values = quality_scores[qm]
                    if values:
                        stats.add_estimate(qm, 'quality', 1.0 if qm != 'pinc' else 0.0,
                                           values, display_name)
                stats_dict[(corpus, threat_model_name, model_name)] = stats

                # for each metric in the DefenseStats, compute bayesian change estimates
                base_row = {'Corpus': corpus.upper(), 'Threat Model': threat_model_name,
                            'Defense Model': model_name}
                abs_row = base_row.copy()
                rel_row = base_row.copy()
                # process effectiveness
                for metric_key, display_name in metrics_map.items():
                    if metric_key in stats.effectiveness_estimates:
                        br = stats.effectiveness_estimates[metric_key]
                        abs_change = br.post_mean - br.pre_value
                        abs_ci_lower = br.ci_lower - br.pre_value
                        abs_ci_upper = br.ci_upper - br.pre_value
                        # for relative change, if effectiveness then use baseline as pre_value
                        rel_change = (abs_change / br.pre_value * 100) if br.pre_value != 0 else np.nan
                        rel_ci_lower = (abs_ci_lower / br.pre_value * 100) if br.pre_value != 0 else np.nan
                        rel_ci_upper = (abs_ci_upper / br.pre_value * 100) if br.pre_value != 0 else np.nan
                        abs_row[display_name] = format_estimate_with_hdi(
                            abs_change, brevity=brevity, ci_lower=abs_ci_lower,
                            ci_upper=abs_ci_upper)
                        rel_row[display_name] = format_estimate_with_hdi(
                            rel_change, brevity=brevity, ci_lower=rel_ci_lower,
                            ci_upper=rel_ci_upper) + "%"
                    else:
                        abs_row[display_name] = '-'
                        rel_row[display_name] = '-'
                # process quality metrics
                for qm, (key, display_name) in quality_metrics.items():
                    if qm in stats.quality_estimates:
                        br = stats.quality_estimates[qm]
                        # for quality, baseline is 1 for most metrics (0 for pinc)
                        base_val = 0.0 if qm == 'pinc' else 1.0
                        abs_change = br.post_mean - base_val
                        abs_ci_lower = br.ci_lower - base_val
                        abs_ci_upper = br.ci_upper - base_val
                        if base_val == 1.0:
                            rel_change = (abs_change / (1 - base_val) * 100) if (1 - base_val) != 0 else np.nan
                            rel_ci_lower = (abs_ci_lower / (1 - base_val) * 100) if (1 - base_val) != 0 else np.nan
                            rel_ci_upper = (abs_ci_upper / (1 - base_val) * 100) if (1 - base_val) != 0 else np.nan
                        else:
                            rel_change = (abs_change / base_val * 100) if base_val != 0 else np.nan
                            rel_ci_lower = (abs_ci_lower / base_val * 100) if base_val != 0 else np.nan
                            rel_ci_upper = (abs_ci_upper / base_val * 100) if base_val != 0 else np.nan
                        abs_row[display_name] = format_estimate_with_hdi(
                            abs_change, brevity=brevity, ci_lower=abs_ci_lower,
                            ci_upper=abs_ci_upper)
                        rel_row[display_name] = format_estimate_with_hdi(
                            rel_change, brevity=brevity, ci_lower=rel_ci_lower,
                            ci_upper=rel_ci_upper) + "%"
                    else:
                        abs_row[display_name] = '-'
                        rel_row[display_name] = '-'
                abs_change_rows.append(abs_row)
                rel_change_rows.append(rel_row)

    columns = ['Corpus', 'Threat Model', 'Defense Model'] + list(
        metrics_map.values()) + [v[1] for v in quality_metrics.values()]
    abs_change_df = pd.DataFrame(abs_change_rows, columns=columns)
    rel_change_df = pd.DataFrame(rel_change_rows, columns=columns)

    return post_df, abs_imp_df, rel_imp_df, abs_change_df, rel_change_df, stats_dict


def main():
    """example usage of defense evaluation; parse args and output csv tables in results folder"""
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

    # call the heavy-lifting function that returns six dataframes
    (post_df, abs_imp_df, rel_imp_df, abs_change_df, rel_change_df, stats_dict) = get_defense_tables_with_stats(
        base_dir=args.base_dir,
        rq=args.rq,
        corpora=[args.corpus] if args.corpus else None,
        threat_models=None,  # uses default if not provided
        mode="post",
        brevity=True,
    )

    # print the results to the console
    print("\npost-intervention results (bayesian estimates):")
    print(post_df)
    print("\nraw absolute improvements:")
    print(abs_imp_df)
    print("\nraw relative improvements:")
    print(rel_imp_df)
    print("\nbayesian absolute changes (post_mean - pre_value):")
    print(abs_change_df)
    print("\nbayesian relative changes (in percent):")
    print(rel_change_df)

    # create output folder if not exist
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # use the research question identifier as base filename
    base_filename = args.rq
    post_df.to_csv(output_folder / f"{base_filename}_post.csv", index=False)
    abs_imp_df.to_csv(output_folder / f"{base_filename}_abs_imp.csv", index=False)
    rel_imp_df.to_csv(output_folder / f"{base_filename}_rel_imp.csv", index=False)
    abs_change_df.to_csv(output_folder / f"{base_filename}_abs_change.csv", index=False)
    rel_change_df.to_csv(output_folder / f"{base_filename}_rel_change.csv", index=False)

    print(f"\nresults saved in folder '{output_folder.absolute()}'")


if __name__ == "__main__":
    main()

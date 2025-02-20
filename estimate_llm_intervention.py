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
│   │   │   │           quality: text quality metrics (aggregated and/or sample-level arrays)
│   │   │   │           [example_metrics and/or example_quality may be present]
│   │   │   └── {another_model}/
│   └── rq{N+1}/
└── {another_corpus}/

Results are saved as CSV files (pre, post, absolute change, and relative change)
in the specified output folder.
"""
"""
Defense evaluation script that performs Bayesian analysis comparing pre/post defense
performance across different metrics, corpora, and threat models.

This version focuses on core metrics and removes redundancy while maintaining proper
Bayesian hierarchical modeling. Raw observations are stored in the stats objects.
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


class BayesResult(NamedTuple):
    pre_value: float
    post_mean: float
    std: float
    ci_lower: float
    ci_upper: float


class DefenseStats:
    """Container for defense evaluation with raw observations and Bayesian estimates."""

    def __init__(self, corpus: str, threat_model: str, defense_model: str):
        self.corpus = corpus
        self.threat_model = threat_model
        self.defense_model = defense_model
        self.effectiveness_estimates = {}
        self.quality_estimates = {}
        # store raw observations
        self.run_level_observations = {}  # for metrics with 5 runs
        self.sample_level_observations = {}  # for metrics with per-example values

    def add_observations(self, metric_name: str, metric_type: str,
                         pre_value: float, values: List[float]) -> None:
        """add raw observations and compute bayesian estimates"""
        if not values:
            return

        # store raw observations
        if len(values) == 5:  # run-level metric
            self.run_level_observations[metric_name] = values
        else:  # sample-level metric
            self.sample_level_observations[metric_name] = values

        # compute estimates
        baseline = pre_value if metric_type == 'effectiveness' else (
            0.0 if metric_name == 'pinc' else 1.0)

        if metric_name == 'entropy':
            # normalize entropy by log2(num_classes)
            factor = np.log2(21) if self.corpus.lower() == 'rj' else np.log2(45)
            scaled_values = [v / factor for v in values]
            result = estimate_beta_metric(scaled_values)
            result = {k: v * factor for k, v in result.items()}
        else:
            result = estimate_beta_metric(values)

        bayes_result = BayesResult(
            pre_value=baseline,
            post_mean=result["mean"],
            std=result["std"],
            ci_lower=result["hdi_lower"],
            ci_upper=result["hdi_upper"]
        )

        if metric_type == 'effectiveness':
            self.effectiveness_estimates[metric_name] = bayes_result
        else:
            self.quality_estimates[metric_name] = bayes_result

    def print_debug_summary(self, metrics: List[str]):
        """print detailed debug information for specified metrics"""
        print(f"\nModel: {self.defense_model} vs {self.threat_model}")
        print(f"Corpus: {self.corpus}")

        print("\nStored observations:")
        print("Run-level metrics:", list(self.run_level_observations.keys()))
        print("Sample-level metrics:", list(self.sample_level_observations.keys()))

        for metric in metrics:
            print(f"\n{metric}:")
            if metric in self.run_level_observations:
                values = self.run_level_observations[metric]
                print(f"Run-level observations (n=5): {values}")
                if metric in self.effectiveness_estimates:
                    est = self.effectiveness_estimates[metric]
                    print(f"Pre-value: {est.pre_value:.4f}")
                    print(f"Post mean: {est.post_mean:.4f}")
                    print(f"95% CI: [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")

            elif metric in self.sample_level_observations:
                values = self.sample_level_observations[metric]
                print(f"Sample-level observations (n={len(values)})")
                print("First 20 raw values:")
                for i, v in enumerate(values[:20]):
                    print(f"  {i + 1}: {v:.4f}")
                print(f"Mean: {np.mean(values):.4f}")
                if metric in self.quality_estimates:
                    est = self.quality_estimates[metric]
                    print(f"Bayesian estimate mean: {est.post_mean:.4f}")
                    print(f"95% CI: [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")


def estimate_beta_metric(post_values: List[float]) -> dict:
    """estimate beta distribution with 95% HDI intervals"""
    epsilon = 1e-6
    post_values = np.clip(np.array(post_values), epsilon, 1 - epsilon)
    n_obs = len(post_values)

    # use informative prior for run-level metrics (n=5)
    if n_obs == 5:
        prior_mean = float(np.median(post_values))
        default_strength = 2.0
        alpha_prior = prior_mean * default_strength
        beta_prior = (1 - prior_mean) * default_strength
    else:
        # use flat prior for sample-level metrics
        alpha_prior, beta_prior = 1, 1

    with pm.Model() as model:
        mu = pm.Beta("mu", alpha=alpha_prior, beta=beta_prior)
        kappa = pm.HalfNormal("kappa", sigma=10)
        _ = pm.Beta("obs", alpha=mu * kappa, beta=(1 - mu) * kappa,
                    observed=post_values)
        trace = pm.sample(2000, tune=1000, cores=4, random_seed=42,
                          return_inferencedata=True)

    mu_samples = trace.posterior["mu"].values.flatten()
    return {
        "mean": float(np.mean(mu_samples)),
        "std": float(np.std(mu_samples)),
        "hdi_lower": float(max(0, az.hdi(mu_samples)[0])),
        "hdi_upper": float(min(1, az.hdi(mu_samples)[1]))
    }


def format_estimate(value: float, std: Optional[float] = None,
                    ci_lower: Optional[float] = None, ci_upper: Optional[float] = None,
                    as_percent: bool = False) -> str:
    """
    Format estimate with standard deviation and credible interval.

    Args:
        value: posterior mean
        std: posterior standard deviation
        ci_lower: lower bound of credible interval
        ci_upper: upper bound of credible interval
        as_percent: whether to format as percentage

    Returns:
        formatted string with mean ± std [ci_lower, ci_upper]
    """
    if as_percent:
        if std is not None and ci_lower is not None:
            return f"{value:.1f}% ± {std:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"
        elif std is not None:
            return f"{value:.1f}% ± {std:.1f}%"
        return f"{value:.1f}%"

    if std is not None and ci_lower is not None:
        return f"{value:.3f} ± {std:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
    elif std is not None:
        return f"{value:.3f} ± {std:.3f}"
    return f"{value:.3f}"


def get_defense_tables_with_stats(
        base_dir: str = "defense_evaluation",
        rqs: Union[str, List[str]] = "rq1.1_basic_paraphrase",
        corpora: Optional[List[str]] = None,
        threat_models: Optional[Dict[str, str]] = None,
        debug_mode: bool = False,
        debug_metrics: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], DefenseStats]]:
    """generate defense analysis table and stats dictionary"""
    if isinstance(rqs, str):
        rqs = [rqs]
    if corpora is None:
        corpora = ['ebg', 'rj']
    if threat_models is None:
        threat_models = {'logreg': 'LogReg', 'svm': 'SVM', 'roberta': 'RoBERTa'}

    # define metrics to analyze
    run_metrics = ['accuracy@1', 'accuracy@5', 'true_class_confidence']
    sample_metrics = ['entropy', 'pinc', 'bertscore', 'sbert']

    if debug_mode and debug_metrics:
        # filter metrics in debug mode
        run_metrics = [m for m in run_metrics if m in debug_metrics]
        sample_metrics = [m for m in sample_metrics if m in debug_metrics]

    stats_dict = {}
    results_rows = []

    # process each configuration
    for corpus in tqdm(corpora, desc="Processing corpora"):
        for threat_model_key, threat_model_name in threat_models.items():
            for rq in rqs:
                # extract metrics from files
                model_results = _process_model_results(
                    base_dir, corpus, rq, threat_model_key,
                    run_metrics, sample_metrics, stats_dict,
                    debug_mode
                )

                # add to results if any valid data found
                if model_results:
                    results_rows.extend(model_results)

    # create output dataframe
    columns = ['Corpus', 'Scenario', 'Threat Model', 'Defense Model'] + \
              [f"{m} ↓" for m in run_metrics if m != 'entropy'] + \
              ['entropy ↑'] + [f"{m} ↑" for m in sample_metrics if m != 'entropy']

    return pd.DataFrame(results_rows, columns=columns), stats_dict


def _extract_metrics(results: Dict, corpus: str, rq: str, threat_model_key: str,
                     model_name: str, run_metrics: List[str],
                     sample_metrics: List[str], stats_dict: Dict) -> Optional[Dict]:
    """extract both run-level and sample-level metrics from results"""
    # initialize row
    config_key = (corpus, threat_model_key, model_name)
    row = {
        'Corpus': corpus.upper(),
        'Scenario': 'Combined' if '_' not in rq else ' '.join(rq.split('_')[1:]),
        'Threat Model': threat_model_key,
        'Defense Model': model_name
    }

    # ensure stats object exists
    if config_key not in stats_dict:
        stats_dict[config_key] = DefenseStats(corpus, threat_model_key, model_name)

    # for each seed
    for seed_key, seed_results in results.items():
        if threat_model_key not in seed_results:
            continue

        run_pre = seed_results[threat_model_key]['attribution']['pre']
        run_post = seed_results[threat_model_key]['attribution']['post']

        # get pre/post values for run-level metrics
        for metric in run_metrics:
            if metric == 'entropy':
                continue  # handle separately below
            orig_val = run_pre.get(metric)
            post_val = run_post.get(metric)
            if orig_val is not None and post_val is not None:
                stats_dict[config_key].run_level_observations.setdefault(metric,
                                                                         []).append(
                    post_val)
                if len(stats_dict[config_key].run_level_observations[metric]) == 5:
                    stats_dict[config_key].add_observations(
                        metric, 'effectiveness', orig_val,
                        stats_dict[config_key].run_level_observations[metric]
                    )
                    display_name = f"{metric} ↓"
                    row[display_name] = format_estimate(
                        stats_dict[config_key].effectiveness_estimates[
                            metric].post_mean,
                        stats_dict[config_key].effectiveness_estimates[metric].ci_lower,
                        stats_dict[config_key].effectiveness_estimates[metric].ci_upper
                    )

        # process sample-level metrics from this seed
        # calculate and store entropy values
        if 'raw_predictions' in seed_results[threat_model_key]['attribution']:
            transformed_preds = \
            seed_results[threat_model_key]['attribution']['raw_predictions'][
                'transformed']
            for pred in transformed_preds:
                pred = np.array(pred)
                entropy = -np.sum(pred * np.log2(pred + 1e-10))
                stats_dict[config_key].sample_level_observations.setdefault('entropy',
                                                                            []).append(
                    entropy)

        # extract quality metrics
        if 'quality' in seed_results[threat_model_key]:
            quality_data = seed_results[threat_model_key]['quality']

            # handle PINC scores - average PINC1-4 for each sample
            if 'pinc' in quality_data:
                n_samples = len(quality_data["pinc"].get("pinc_1_scores", []))
                if n_samples > 0:
                    for i in range(n_samples):
                        sample_pinc_scores = []
                        for k in range(1, 5):  # PINC1 to PINC4
                            scores_key = f"pinc_{k}_scores"
                            if scores_key in quality_data["pinc"]:
                                scores = quality_data["pinc"][scores_key]
                                if i < len(scores):
                                    sample_pinc_scores.append(scores[i])

                        if sample_pinc_scores:  # if we have scores for this sample
                            avg_pinc = np.mean(sample_pinc_scores)
                            stats_dict[config_key].sample_level_observations.setdefault(
                                'pinc', []).append(avg_pinc)

            # handle BERTScore - extract F1 scores from individual results
            if 'bertscore' in quality_data and 'bertscore_individual' in quality_data[
                'bertscore']:
                bertscore_f1s = [item['f1'] for item in
                                 quality_data['bertscore']['bertscore_individual']]
                stats_dict[config_key].sample_level_observations.setdefault('bertscore',
                                                                            []).extend(
                    bertscore_f1s)

            # handle SBERT scores
            if 'sbert' in quality_data and 'sbert_similarity_scores' in quality_data[
                'sbert']:
                sbert_scores = [float(x) for x in
                                quality_data['sbert']['sbert_similarity_scores']]
                stats_dict[config_key].sample_level_observations.setdefault('sbert',
                                                                            []).extend(
                    sbert_scores)

    # process all sample-level metrics after collecting all samples
    # handle entropy
    if 'entropy' in sample_metrics and 'entropy' in stats_dict[
        config_key].sample_level_observations:
        entropy_values = stats_dict[config_key].sample_level_observations['entropy']
        pre_entropy = run_pre.get('entropy', 0.0)
        stats_dict[config_key].add_observations('entropy', 'effectiveness', pre_entropy,
                                                entropy_values)
        row['entropy ↑'] = format_estimate(
            stats_dict[config_key].effectiveness_estimates['entropy'].post_mean,
            stats_dict[config_key].effectiveness_estimates['entropy'].ci_lower,
            stats_dict[config_key].effectiveness_estimates['entropy'].ci_upper
        )

    # handle PINC
    if 'pinc' in sample_metrics and 'pinc' in stats_dict[
        config_key].sample_level_observations:
        pinc_values = stats_dict[config_key].sample_level_observations['pinc']
        stats_dict[config_key].add_observations('pinc', 'quality', 0.0, pinc_values)
        row['pinc ↑'] = format_estimate(
            stats_dict[config_key].quality_estimates['pinc'].post_mean,
            stats_dict[config_key].quality_estimates['pinc'].ci_lower,
            stats_dict[config_key].quality_estimates['pinc'].ci_upper
        )

    # handle BERTScore
    if 'bertscore' in sample_metrics and 'bertscore' in stats_dict[
        config_key].sample_level_observations:
        bertscore_values = stats_dict[config_key].sample_level_observations['bertscore']
        stats_dict[config_key].add_observations('bertscore', 'quality', 1.0,
                                                bertscore_values)
        row['bertscore ↑'] = format_estimate(
            stats_dict[config_key].quality_estimates['bertscore'].post_mean,
            stats_dict[config_key].quality_estimates['bertscore'].ci_lower,
            stats_dict[config_key].quality_estimates['bertscore'].ci_upper
        )

    # handle SBERT
    if 'sbert' in sample_metrics and 'sbert' in stats_dict[
        config_key].sample_level_observations:
        sbert_values = stats_dict[config_key].sample_level_observations['sbert']
        stats_dict[config_key].add_observations('sbert', 'quality', 1.0, sbert_values)
        row['sbert ↑'] = format_estimate(
            stats_dict[config_key].quality_estimates['sbert'].post_mean,
            stats_dict[config_key].quality_estimates['sbert'].ci_lower,
            stats_dict[config_key].quality_estimates['sbert'].ci_upper
        )

    return row


def print_debug_summary(self, metrics: List[str]):
    print(f"\nModel: {self.defense_model} vs {self.threat_model}")
    print(f"Corpus: {self.corpus}")

    print("\nStored observations:")
    print("Run-level metrics:", list(self.run_level_observations.keys()))
    print("Sample-level metrics:", list(self.sample_level_observations.keys()))

    for metric in metrics:
        print(f"\n{metric}:")
        if metric in self.run_level_observations:
            values = self.run_level_observations[metric]
            print(f"Run-level observations (n=5): {values}")
            if metric in self.effectiveness_estimates:
                est = self.effectiveness_estimates[metric]
                print(f"Pre-value: {est.pre_value:.4f}")
                print(f"Post mean: {est.post_mean:.4f}")
                print(f"95% CI: [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")

        elif metric in self.sample_level_observations:
            values = self.sample_level_observations[metric]
            print(f"Sample-level observations (n={len(values)})")
            print("First 20 raw values:")
            for i, v in enumerate(values[:20]):
                print(f"  {i + 1}: {v:.4f}")
            print(f"Mean: {np.mean(values):.4f}")
            if metric in self.quality_estimates:
                est = self.quality_estimates[metric]
                print(f"Bayesian estimate mean: {est.post_mean:.4f}")
                print(f"95% CI: [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")


def _process_model_results(base_dir: str, corpus: str, rq: str,
                           threat_model_key: str, run_metrics: List[str],
                           sample_metrics: List[str], stats_dict: dict,
                           debug_mode: bool = False) -> List[dict]:
    """process results for a specific model configuration"""
    rq_main = f"rq{rq.split('_')[0].split('.')[0].lstrip('rq')}"
    corpus_path = Path(base_dir) / corpus / rq_main / rq

    if not corpus_path.exists():
        return []

    results_rows = []
    for model_dir in corpus_path.glob("*"):
        if not model_dir.is_dir():
            continue

        # skip non-Claude models in debug mode
        if debug_mode and 'claude' not in model_dir.name.lower():
            continue

        # get model display name
        model_name = _get_model_display_name(model_dir.name)

        # process model results
        eval_file = model_dir / "evaluation.json"
        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            results = json.load(f)

        row = _extract_metrics(
            results, corpus, rq, threat_model_key, model_name,
            run_metrics, sample_metrics, stats_dict
        )

        if row:
            results_rows.append(row)

    return results_rows


def _get_model_display_name(model_dir_name: str) -> str:
    """get standardized display name for model"""
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
    """main entry point with argument parsing"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate defense effectiveness with Bayesian analysis")
    parser.add_argument('--base_dir', type=str, default='defense_evaluation')
    parser.add_argument('--rqs', type=str, nargs='+',
                        default=['rq1.1_basic_paraphrase'])
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--debug', action='store_true',
                        help='Run debug mode with RJ/EBG, Claude-3.5, LogReg')

    args = parser.parse_args()

    if args.debug:
        # debug mode - just print statistics
        threat_models = {'logreg': 'LogReg'}
        debug_metrics = ['accuracy@1',
                         'accuracy@5',
                         'true_class_confidence',
                         'entropy',
                         'pinc',
                         'bertscore',
                         'sbert']

        print("Running in debug mode...")
        print("Models: Claude-3.5 (defense) vs LogReg (threat)")
        print("Metrics:", debug_metrics)
        print("Corpora: RJ, EBG")

        results_df, stats_dict = get_defense_tables_with_stats(
            base_dir=args.base_dir,
            rqs=args.rqs,
            debug_mode=True,
            debug_metrics=debug_metrics,
            threat_models=threat_models
        )

        # print debug summaries
        for key, stats in stats_dict.items():
            if stats.defense_model == 'Claude-3.5':
                stats.print_debug_summary(metrics=debug_metrics)

        print("\nResults DataFrame:")
        print(results_df)

    else:
        # normal mode - save results and raw observations
        results_df, stats_dict = get_defense_tables_with_stats(
            base_dir=args.base_dir,
            rqs=args.rqs
        )

        output_folder = Path(args.output)
        output_folder.mkdir(parents=True, exist_ok=True)

        # save results dataframe
        rq_str = args.rqs[0] if len(args.rqs) == 1 else "combined"
        results_df.to_csv(output_folder / f"{rq_str}_results.csv", index=False)

        # save raw observations
        raw_observations = {}
        for key, stats in stats_dict.items():
            corpus, threat_model, defense_model = key
            model_key = f"{corpus}_{defense_model}_{threat_model}"
            raw_observations[model_key] = {
                "run_level": {
                    metric: values
                    for metric, values in stats.run_level_observations.items()
                },
                "sample_level": {
                    metric: values
                    for metric, values in stats.sample_level_observations.items()
                }
            }

        with open(output_folder / f"{rq_str}_raw_observations.json", "w") as f:
            json.dump(raw_observations, f, indent=2)

        print(f"\nResults and raw observations saved to {output_folder}")


if __name__ == "__main__":
    main()
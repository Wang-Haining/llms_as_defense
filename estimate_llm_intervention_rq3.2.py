#!/usr/bin/env python
"""
Script to analyze the relationship between exemplar length and defense metrics for RQ3.2.

This script models how exemplar length affects various metrics in defending against
authorship attribution attacks using Bayesian methods.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

# Constants
CORPORA = ['ebg', 'rj']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
METRICS = ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']
MAX_ENTROPY = {"ebg": np.log2(45), "rj": np.log2(21)}


def model_continuous_metric(df: pd.DataFrame,
                            metric: str,
                            higher_is_better: bool,
                            max_entropy: float) -> Optional[Dict]:
    """fit a bayesian model for a continuous metric.

    args:
        df: dataframe containing exemplar length and metric data
        metric: name of the metric column
        higher_is_better: whether higher values of the metric are better
        max_entropy: maximum possible entropy value for normalization

    returns:
        dictionary with model results or None if modeling failed
    """
    # drop rows with missing values
    df = df.dropna(subset=["exemplar_length", metric])

    if df.empty or df['exemplar_length'].nunique() <= 1:
        return None

    # prepare data
    x = df['exemplar_length'].values
    y = df[metric].values
    corpus = df['corpus'].iloc[0]

    # center and scale x for better inference
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000  # scale to thousands for numerical stability

    # normalize metrics if needed
    if metric == 'entropy':
        y = y / max_entropy  # normalize entropy to [0,1]

    # clip values to (0,1) for beta model
    epsilon = 1e-6
    y = np.clip(y, epsilon, 1 - epsilon)

    try:
        with pm.Model() as model:
            # priors
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)

            # linear predictor
            mu_est = alpha + beta * x_scaled

            # transform to probability scale
            theta = pm.Deterministic("theta", pm.math.invlogit(mu_est))

            # concentration parameter for beta distribution
            concentration = pm.HalfNormal("concentration", 10.0)

            # parameterize beta distribution
            a_beta = theta * concentration
            b_beta = (1 - theta) * concentration

            # likelihood
            _ = pm.Beta("likelihood", alpha=a_beta, beta=b_beta, observed=y)

            # sample posterior
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                              return_inferencedata=True)
    except Exception as e:
        print(f"Error in modeling {metric}: {str(e)}")
        return None

    # analyze results
    beta_samples = trace.posterior['beta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate highest density interval
    hdi = az.hdi(beta_samples, hdi_prob=0.95)

    # region of practical equivalence
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))

    # probability of benefit
    prob_benefit = float(np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))

    # determine conclusion
    conclusion = ("Practically Equivalent" if in_rope > 0.95 else
                  ("Significant Improvement" if prob_benefit > 0.95 else
                   ("Significant Deterioration" if prob_benefit < 0.05 else "Inconclusive")))

    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples))
    }


def model_binary_metric(df: pd.DataFrame,
                        metric: str,
                        higher_is_better: bool) -> Optional[Dict]:
    """fit a bayesian model for a binary metric.

    args:
        df: dataframe containing exemplar length and binary metric data
        metric: name of the binary metric column
        higher_is_better: whether higher values of the metric are better

    returns:
        dictionary with model results or None if modeling failed
    """
    # prepare binary data
    binary_metrics = []
    exemplar_lengths = []

    for idx, row in df.iterrows():
        if metric not in row or pd.isna(row[metric]):
            continue
        binary_metrics.append(float(row[metric]))
        exemplar_lengths.append(row['exemplar_length'])

    if not binary_metrics or len(set(binary_metrics)) < 2:
        return None

    # create dataframe with valid data points
    data_df = pd.DataFrame({
        'exemplar_length': exemplar_lengths,
        'binary_metric': binary_metrics
    })

    if data_df.empty or data_df['exemplar_length'].nunique() <= 1:
        return None

    # prepare data for modeling
    x = data_df['exemplar_length'].values
    y = data_df['binary_metric'].values

    # center and scale x for better inference
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000  # scale to thousands for numerical stability

    try:
        with pm.Model() as model:
            # priors
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.Cauchy("beta", alpha=0, beta=0.5)

            # linear predictor
            eta = alpha + beta * x_scaled

            # transform to probability scale
            theta = pm.Deterministic("theta", pm.math.sigmoid(eta))

            # likelihood
            _ = pm.Bernoulli("likelihood", p=theta, observed=y)

            # sample posterior
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                              return_inferencedata=True)
    except Exception as e:
        print(f"Error in modeling binary metric {metric}: {str(e)}")
        return None

    # analyze results
    beta_samples = trace.posterior['beta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate highest density interval
    hdi = az.hdi(beta_samples, hdi_prob=0.95)

    # region of practical equivalence
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))

    # probability of benefit
    prob_benefit = float(np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))

    # determine conclusion
    conclusion = ("Practically Equivalent" if in_rope > 0.95 else
                  ("Significant Improvement" if prob_benefit > 0.95 else
                   ("Significant Deterioration" if prob_benefit < 0.05 else "Inconclusive")))

    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples))
    }


def create_diagnostic_plot(df: pd.DataFrame,
                           metric: str,
                           model_results: Dict,
                           output_path: Path,
                           use_binary: bool = False) -> None:
    """create a diagnostic plot showing the relationship between exemplar length and metric.

    args:
        df: dataframe containing exemplar length and metric data
        metric: name of the metric column to plot
        model_results: dictionary with model results
        output_path: path to save the plot
        use_binary: whether to use binary version of the metric
    """
    plt.figure(figsize=(12, 8))

    # determine which metric to plot
    plot_metric = f"binary_{metric}" if use_binary and f"binary_{metric}" in df.columns else metric

    # set up the plot
    sns.set_style("whitegrid")

    # plot points for each seed with different colors/markers
    for seed_val in sorted(df['seed'].unique()):
        seed_data = df[df['seed'] == seed_val]
        if not seed_data.empty and plot_metric in seed_data.columns:
            plt.scatter(
                seed_data['exemplar_length'],
                seed_data[plot_metric],
                label=f"Seed {seed_val}",
                alpha=0.7,
                marker='o'
            )

    # draw the fitted curve
    x_min = df['exemplar_length'].min()
    x_max = df['exemplar_length'].max()
    x_vals = np.linspace(x_min, x_max, 100)
    x_mean = np.mean(df['exemplar_length'])
    x_scaled = (x_vals - x_mean) / 1000

    alpha_mean = model_results["alpha_mean"]
    beta_mean = model_results["beta_mean"]

    # calculate predicted values based on model type
    if use_binary:
        # logistic function for binary data
        y_pred = 1.0 / (1.0 + np.exp(-(alpha_mean + beta_mean * x_scaled)))
    else:
        # logistic function for beta model
        y_pred = 1.0 / (1.0 + np.exp(-(alpha_mean + beta_mean * x_scaled)))

    plt.plot(x_vals, y_pred, 'r-', linewidth=2, label="Model fit")

    # create plot title
    llm = df['llm'].iloc[0] if df['llm'].nunique() == 1 else "Various"
    threat_model = df['threat_model'].iloc[0] if df['threat_model'].nunique() == 1 else "Various"
    corpus = df['corpus'].iloc[0] if df['corpus'].nunique() == 1 else "Various"

    plt.title(f"{corpus.upper()} - {llm} vs {threat_model}: Impact on {plot_metric}", fontsize=14)
    plt.xlabel("Exemplar Length (words)", fontsize=12)
    plt.ylabel(f"{plot_metric}", fontsize=12)

    # add annotations
    plt.annotate(
        f"Slope: {model_results['slope_mean']:.4f}\n"
        f"95% HDI: [{model_results['slope_hdi'][0]:.4f}, {model_results['slope_hdi'][1]:.4f}]\n"
        f"P(Improvement): {model_results['prob_benefit']:.4f}\n"
        f"Conclusion: {model_results['conclusion']}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
        fontsize=10,
        verticalalignment='top'
    )

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_scenario_data(data_dir: Path,
                          output_dir: Path,
                          corpus: str,
                          experiment: str,
                          llm: str,
                          threat_model: str,
                          debug: bool) -> List[Dict]:
    """analyze data for a specific corpus/llm/threat_model scenario.

    args:
        data_dir: directory containing prepared data
        output_dir: directory to save results
        corpus: corpus name (ebg or rj)
        experiment: experiment name (e.g., rq3.2_imitation_variable_length)
        llm: language model name
        threat_model: attribution model name
        debug: whether to run in debug mode (generate additional diagnostics)

    returns:
        list of dictionaries with analysis results
    """
    # create output directories
    plots_dir = output_dir / "plots" / corpus / experiment / llm / threat_model
    plots_dir.mkdir(parents=True, exist_ok=True)

    # load the data for this scenario
    data_file = data_dir / corpus / experiment / f"{threat_model}_{llm}_data.csv"

    if not data_file.exists():
        print(f"Warning: Data file not found: {data_file}")
        return []

    df = pd.read_csv(data_file)

    if df.empty:
        print(f"Warning: Empty data file: {data_file}")
        return []

    print(f"Analyzing {corpus}/{experiment}/{llm} vs {threat_model}")
    print(f"Data shape: {df.shape}")
    print(f"Exemplar lengths: {sorted(df['exemplar_length'].unique())}")

    results = []

    # analyze each metric
    for metric in METRICS:
        print(f"  Analyzing metric: {metric}")

        # determine if higher is better for this metric
        higher_is_better = metric in ["entropy", "bertscore", "pinc"]

        # first check if we have binary data available for accuracy metrics
        binary_metric = None
        binary_results = None

        if metric == "accuracy@1" and "binary_acc1" in df.columns and not df["binary_acc1"].isna().all():
            binary_metric = "binary_acc1"
            print(f"    Using binary data for {metric}")
            binary_results = model_binary_metric(df, binary_metric, not higher_is_better)

        elif metric == "accuracy@5" and "binary_acc5" in df.columns and not df["binary_acc5"].isna().all():
            binary_metric = "binary_acc5"
            print(f"    Using binary data for {metric}")
            binary_results = model_binary_metric(df, binary_metric, not higher_is_better)

        # model the continuous metric
        continuous_results = model_continuous_metric(
            df, metric, higher_is_better, MAX_ENTROPY[corpus])

        # use binary results if available, otherwise use continuous results
        results_to_use = binary_results if binary_results is not None else continuous_results
        metric_display = binary_metric if binary_metric is not None else metric

        if results_to_use is None:
            print(f"    Skipping {metric_display}: Modeling failed")
            continue

        # record the results
        result = {
            "Corpus": corpus.upper(),
            "Threat Model": threat_model,
            "LLM": llm,
            "Metric": metric_display,
            "Higher is Better": higher_is_better,
            "Slope": results_to_use['slope_mean'],
            "Slope Std": results_to_use['slope_std'],
            "Slope HDI Lower": results_to_use['slope_hdi'][0],
            "Slope HDI Upper": results_to_use['slope_hdi'][1],
            "P(Improvement)": results_to_use['prob_benefit'],
            "In ROPE": results_to_use['in_rope'],
            "Conclusion": results_to_use['conclusion']
        }

        results.append(result)

        # create diagnostic plot if in debug mode or for all scenarios
        if debug or True:  # always create plots for now
            plot_path = plots_dir / f"{metric_display.replace('@', '_')}.png"
            create_diagnostic_plot(
                df,
                metric,
                results_to_use,
                plot_path,
                use_binary=(binary_metric is not None)
            )
            print(f"    Created diagnostic plot: {plot_path}")

    return results


def analyze_data(data_dir: Path, output_dir: Path, debug: bool = False):
    """analyze the relationship between exemplar length and metrics.

    args:
        data_dir: directory containing prepared data
        output_dir: directory to save results
        debug: whether to run in debug mode
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # filter corpora, models, and threat models for debug mode
    corpora_to_process = ['ebg'] if debug else CORPORA
    llms_to_process = ['ministral'] if debug else LLMS
    threat_models_to_process = ['roberta'] if debug else THREAT_MODELS

    # process each experiment in the data directory
    for corpus in corpora_to_process:
        corpus_dir = data_dir / corpus

        if not corpus_dir.exists():
            print(f"Warning: Corpus directory not found: {corpus_dir}")
            continue

        for experiment_dir in corpus_dir.glob("*"):
            if not experiment_dir.is_dir():
                continue

            experiment = experiment_dir.name
            print(f"Processing experiment: {corpus}/{experiment}")

            # process each LLM/threat model combination
            for llm in llms_to_process:
                for threat_model in threat_models_to_process:
                    # analyze this specific scenario
                    results = analyze_scenario_data(
                        data_dir, output_dir, corpus, experiment, llm, threat_model, debug)

                    all_results.extend(results)

    # save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = output_dir / "exemplar_length_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved combined results to {results_path}")

        # Fix for the non-numeric 'Conclusion' field - create summary without using pivot_table's aggregation
        conclusion_df = results_df[['Corpus', 'LLM', 'Threat Model', 'Metric', 'Conclusion']]

        # Reshape the data to have metrics as columns without aggregation
        conclusion_wide = conclusion_df.pivot(
            index=['Corpus', 'LLM', 'Threat Model'],
            columns='Metric',
            values='Conclusion'
        )

        summary_path = output_dir / "conclusion_summary.csv"
        conclusion_wide.to_csv(summary_path)
        print(f"Saved conclusion summary to {summary_path}")

        # Create effect direction summary without aggregation
        direction_df = results_df.copy()
        direction_df['Effect'] = direction_df.apply(
            lambda row: 'Positive' if ((row['Slope'] > 0 and row['Higher is Better']) or
                                       (row['Slope'] < 0 and not row['Higher is Better'])) else 'Negative',
            axis=1
        )

        direction_df['Significant'] = direction_df.apply(
            lambda row: row['Conclusion'] in ['Significant Improvement', 'Significant Deterioration'],
            axis=1
        )

        # Create separate dataframes for Effect and Significant
        effect_df = direction_df[['Corpus', 'LLM', 'Threat Model', 'Metric', 'Effect']]
        significant_df = direction_df[['Corpus', 'LLM', 'Threat Model', 'Metric', 'Significant']]

        # Reshape the data using pivot instead of pivot_table
        effect_wide = effect_df.pivot(
            index=['Corpus', 'LLM', 'Threat Model'],
            columns='Metric',
            values='Effect'
        )

        significant_wide = significant_df.pivot(
            index=['Corpus', 'LLM', 'Threat Model'],
            columns='Metric',
            values='Significant'
        )

        # Save effect directions
        effect_path = output_dir / "effect_direction.csv"
        effect_wide.to_csv(effect_path)

        # Save significance
        significant_path = output_dir / "significant_effects.csv"
        significant_wide.to_csv(significant_path)

        # Save combined summary with multi-level columns
        combined_summary = pd.concat([
            effect_wide.add_prefix('Effect_'),
            significant_wide.add_prefix('Significant_')
        ], axis=1)

        combined_path = output_dir / "effect_direction_summary.csv"
        combined_summary.to_csv(combined_path)
        print(f"Saved effect direction summaries to {output_dir}")
    else:
        print("Warning: No results were generated")


def main():
    """main function to analyze exemplar length data."""
    parser = argparse.ArgumentParser(description="Analyze exemplar length data")
    parser.add_argument("--data_dir", type=str, default="results/prepared_data_rq3.2",
                        help="Directory containing prepared data")
    parser.add_argument("--output_dir", type=str, default="results/exemplar_length_analysis_rq3.2",
                        help="Directory to save analysis results")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (EBG corpus, Ministral vs RoBERTa only)")

    args = parser.parse_args()

    analyze_data(Path(args.data_dir), Path(args.output_dir), args.debug)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
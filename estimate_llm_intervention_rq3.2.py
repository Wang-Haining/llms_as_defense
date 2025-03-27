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
from scipy.special import expit  # For sigmoid function

# Constants
CORPORA = ['ebg', 'rj']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
# Updated metrics to match the new column names in prepared data
METRICS = ['binary_acc1', 'binary_acc5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']
# Maximum entropy values based on corpus author counts
MAX_ENTROPY = {"ebg": np.log2(45), "rj": np.log2(21)}
# Chance levels for accuracy metrics
CHANCE_LEVELS = {
    "ebg": {
        "binary_acc1": 1/45,
        "binary_acc5": 5/45
    },
    "rj": {
        "binary_acc1": 1/21,
        "binary_acc5": 5/21
    }
}


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
    original_y = y.copy()  # store original values for later
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

            # add a non-linear term for more flexibility
            beta2 = pm.StudentT("beta2", nu=3, mu=0, sigma=0.3)

            # squared term for non-linearity
            x_squared = x_scaled**2

            # linear predictor with quadratic term
            mu_est = alpha + beta * x_scaled + beta2 * x_squared

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
    beta2_samples = trace.posterior['beta2'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate highest density interval
    hdi_beta = az.hdi(beta_samples, hdi_prob=0.95)
    hdi_beta2 = az.hdi(beta2_samples, hdi_prob=0.95)

    # combined effect - this is more complex with quadratic term
    # we'll use the linear term for simplicity in determining direction
    slope_mean = float(np.mean(beta_samples))

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
        "slope_mean": slope_mean,
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi_beta.tolist(),
        "slope2_mean": float(np.mean(beta2_samples)),
        "slope2_hdi": hdi_beta2.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples)),
        "beta2_mean": float(np.mean(beta2_samples)),
        "metric": metric,
        "is_entropy": metric == "entropy",
        "max_entropy": max_entropy,
        "x_mean": x_mean,  # store for plotting
        "original_values": original_y.tolist()  # store original values for plotting
    }


def model_binary_metric(df: pd.DataFrame,
                        metric: str,
                        higher_is_better: bool) -> Optional[Dict]:
    """fit a bayesian model for a binary metric using basis functions for flexibility.

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
    corpus = df['corpus'].iloc[0]

    # center and scale x for better inference
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000  # scale to thousands for numerical stability

    # create squared term for non-linearity
    x_squared = x_scaled**2

    try:
        with pm.Model() as model:
            # priors
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)

            # Add non-linear term 
            beta2 = pm.StudentT("beta2", nu=3, mu=0, sigma=0.3)

            # linear predictor with quadratic term
            eta = alpha + beta * x_scaled + beta2 * x_squared

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
    beta2_samples = trace.posterior['beta2'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate highest density interval
    hdi_beta = az.hdi(beta_samples, hdi_prob=0.95)
    hdi_beta2 = az.hdi(beta2_samples, hdi_prob=0.95)

    # use linear term for simplicity in determining direction
    slope_mean = float(np.mean(beta_samples))

    # region of practical equivalence
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))

    # probability of benefit (lower accuracy is better for defensive purposes)
    prob_benefit = float(np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))

    # determine conclusion
    conclusion = ("Practically Equivalent" if in_rope > 0.95 else
                  ("Significant Improvement" if prob_benefit > 0.95 else
                   ("Significant Deterioration" if prob_benefit < 0.05 else "Inconclusive")))

    # Get chance level for this metric in this corpus
    chance_level = CHANCE_LEVELS.get(corpus.lower(), {}).get(metric, None)

    return {
        "slope_mean": slope_mean,
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi_beta.tolist(),
        "slope2_mean": float(np.mean(beta2_samples)),
        "slope2_hdi": hdi_beta2.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples)),
        "beta2_mean": float(np.mean(beta2_samples)),
        "metric": metric,
        "is_binary": True,
        "x_mean": x_mean,  # store for plotting
        "corpus": corpus,
        "chance_level": chance_level
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

    # determine which metric to plot (no need to use binary_* prefix as the metric name already has it)
    plot_metric = metric

    # set up the plot
    sns.set_style("whitegrid")

    # plot all points together in one scatter (no separation by seed)
    if plot_metric in df.columns:
        plt.scatter(
            df['exemplar_length'],
            df[plot_metric],
            alpha=0.7,
            marker='o',
            color='blue',
            label='Data points'
        )

    # draw the fitted curve
    x_min = df['exemplar_length'].min()
    x_max = df['exemplar_length'].max()
    x_vals = np.linspace(x_min, x_max, 100)

    alpha_mean = model_results["alpha_mean"]
    beta_mean = model_results["beta_mean"]
    beta2_mean = model_results.get("beta2_mean", 0)  # Default to 0 if not present

    x_mean = model_results.get("x_mean", np.mean(df['exemplar_length']))
    x_scaled = (x_vals - x_mean) / 1000
    x_squared = x_scaled**2

    # Calculate posterior predictions using posterior samples
    n_samples = 1000
    alpha_samples = np.random.normal(alpha_mean, model_results["slope_std"] * 0.5, n_samples)
    beta_samples = np.random.normal(beta_mean, model_results["slope_std"], n_samples)

    # Get beta2 samples if available
    if "slope2_std" in model_results:
        beta2_std = model_results["slope2_std"]
    else:
        beta2_std = model_results["slope_std"] * 0.6  # Estimate

    beta2_samples = np.random.normal(beta2_mean, beta2_std, n_samples)

    # Create matrix to hold all predicted values
    pred_samples = np.zeros((n_samples, len(x_vals)))

    if use_binary or model_results.get("is_binary", False):
        # For binary metrics
        for i in range(n_samples):
            logit_val = alpha_samples[i] + beta_samples[i] * x_scaled + beta2_samples[i] * x_squared
            pred_samples[i, :] = expit(logit_val)  # Use sigmoid function

        # Calculate the mean prediction and HDI
        y_pred = np.mean(pred_samples, axis=0)
        hdi_lower = np.zeros(len(x_vals))
        hdi_upper = np.zeros(len(x_vals))

        for j in range(len(x_vals)):
            hdi = az.hdi(pred_samples[:, j], hdi_prob=0.95)
            hdi_lower[j] = hdi[0]
            hdi_upper[j] = hdi[1]

        # Plot curve and HDI
        plt.plot(x_vals, y_pred, 'r-', linewidth=2, label="Model fit")
        plt.fill_between(x_vals, hdi_lower, hdi_upper, color='red', alpha=0.2, label="95% HDI")

        # Add chance level if available
        chance_level = model_results.get("chance_level", None)
        if chance_level is not None:
            plt.axhline(y=chance_level, color='green', linestyle='--',
                        label=f"Chance level ({chance_level:.3f})")
    else:
        # For continuous metrics
        for i in range(n_samples):
            logit_val = alpha_samples[i] + beta_samples[i] * x_scaled + beta2_samples[i] * x_squared
            prob_val = expit(logit_val)

            # If this is entropy, scale it back up to original scale
            if model_results.get("is_entropy", False):
                max_entropy = model_results.get("max_entropy", 5.49)  # default to ebg
                pred_samples[i, :] = prob_val * max_entropy
            else:
                pred_samples[i, :] = prob_val

        # Calculate the mean prediction and HDI
        y_pred = np.mean(pred_samples, axis=0)
        hdi_lower = np.zeros(len(x_vals))
        hdi_upper = np.zeros(len(x_vals))

        for j in range(len(x_vals)):
            hdi = az.hdi(pred_samples[:, j], hdi_prob=0.95)
            hdi_lower[j] = hdi[0]
            hdi_upper[j] = hdi[1]

        # Plot curve and HDI
        plt.plot(x_vals, y_pred, 'r-', linewidth=2, label="Model fit")
        plt.fill_between(x_vals, hdi_lower, hdi_upper, color='red', alpha=0.2, label="95% HDI")

        # For entropy, also add the max possible entropy
        if model_results.get("is_entropy", False):
            max_entropy = model_results.get("max_entropy", 5.49)
            plt.axhline(y=max_entropy, color='green', linestyle='--',
                        label=f"Max entropy ({max_entropy:.2f})")

    # create plot title
    llm = df['llm'].iloc[0] if df['llm'].nunique() == 1 else "Various"
    threat_model = df['threat_model'].iloc[0] if df['threat_model'].nunique() == 1 else "Various"
    corpus = df['corpus'].iloc[0] if df['corpus'].nunique() == 1 else "Various"

    plt.title(f"{corpus.upper()} - {llm} vs {threat_model}: Impact on {plot_metric}", fontsize=14)
    plt.xlabel("Exemplar Length (words)", fontsize=12)
    plt.ylabel(f"{plot_metric}", fontsize=12)

    # add annotations with linear and quadratic effects
    effect_text = (
        f"Linear Effect: {model_results['slope_mean']:.4f}\n"
        f"95% HDI: [{model_results['slope_hdi'][0]:.4f}, {model_results['slope_hdi'][1]:.4f}]\n"
    )

    if "slope2_mean" in model_results:
        effect_text += (
            f"Quadratic Effect: {model_results['slope2_mean']:.4f}\n"
            f"95% HDI: [{model_results['slope2_hdi'][0]:.4f}, {model_results['slope2_hdi'][1]:.4f}]\n"
        )

    effect_text += (
        f"P(Improvement): {model_results['prob_benefit']:.4f}\n"
        f"Conclusion: {model_results['conclusion']}"
    )

    plt.annotate(
        effect_text,
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
    print(f"Available columns: {df.columns.tolist()}")

    results = []

    # analyze each metric
    for metric in METRICS:
        print(f"  Analyzing metric: {metric}")

        # Check if metric exists in dataframe
        if metric not in df.columns:
            print(f"    Skipping {metric}: Not found in data")
            continue

        # determine if higher is better for this metric
        higher_is_better = metric in ["entropy", "bertscore", "pinc"]

        # For binary metrics, use the binary model
        if metric.startswith('binary_'):
            print(f"    Using binary model for {metric}")
            model_results = model_binary_metric(df, metric, not higher_is_better)
        else:
            # For continuous metrics, use the continuous model
            model_results = model_continuous_metric(
                df, metric, higher_is_better, MAX_ENTROPY[corpus])

        if model_results is None:
            print(f"    Skipping {metric}: Modeling failed")
            continue

        # record the results
        result = {
            "Corpus": corpus.upper(),
            "Threat Model": threat_model,
            "LLM": llm,
            "Metric": metric,
            "Higher is Better": higher_is_better,
            "Slope": model_results['slope_mean'],
            "Slope Std": model_results['slope_std'],
            "Slope HDI Lower": model_results['slope_hdi'][0],
            "Slope HDI Upper": model_results['slope_hdi'][1],
            "P(Improvement)": model_results['prob_benefit'],
            "In ROPE": model_results['in_rope'],
            "Conclusion": model_results['conclusion']
        }

        # Add quadratic effect if available
        if "slope2_mean" in model_results:
            result["Slope2"] = model_results['slope2_mean']
            result["Slope2 HDI Lower"] = model_results['slope2_hdi'][0]
            result["Slope2 HDI Upper"] = model_results['slope2_hdi'][1]

        results.append(result)

        # create diagnostic plot if in debug mode or for all scenarios
        if debug or True:  # always create plots for now
            plot_path = plots_dir / f"{metric.replace('@', '_')}.png"
            create_diagnostic_plot(
                df,
                metric,
                model_results,
                plot_path,
                use_binary=metric.startswith('binary_')
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
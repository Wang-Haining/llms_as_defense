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

# constants
CORPORA = ['ebg', 'rj']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
# updated metrics to match the new column names in prepared data
METRICS = ['binary_acc1', 'binary_acc5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']
MAX_ENTROPY = {"ebg": np.log2(45), "rj": np.log2(21)}


def model_continuous_metric(df: pd.DataFrame,
                            metric: str,
                            higher_is_better: bool,
                            max_entropy: float) -> Optional[Dict]:
    """Fit a Bayesian model for a continuous metric and decide based on the 95% HDI vs. ROPE.

    The decision rule is:
      - "Significant Improvement" if the entire 95% HDI lies above ROPE for metrics where higher is better,
        or entirely below ROPE for metrics where lower is better.
      - "Practically Equivalent" if the entire 95% HDI lies within the ROPE.
      - "Inconclusive" otherwise.

    Args:
        df: DataFrame with exemplar length and metric data.
        metric: Name of the metric column.
        higher_is_better: Whether higher values indicate better performance.
        max_entropy: Maximum possible entropy value for normalization.

    Returns:
        A dictionary with model results and summary statistics, or None if modeling failed.
    """
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
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("beta", mu=0, sigma=5)
            mu_est = alpha + beta * x_scaled
            theta = pm.Deterministic("theta", pm.math.invlogit(mu_est))
            concentration = pm.HalfNormal("concentration", 10.0)
            a_beta = theta * concentration
            b_beta = (1 - theta) * concentration
            _ = pm.Beta("likelihood", alpha=a_beta, beta=b_beta, observed=y)
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                              return_inferencedata=True)
    except Exception as e:
        print(f"Error in modeling {metric}: {str(e)}")
        return None

    # analyze results
    beta_samples = trace.posterior['beta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate 95% highest density interval (HDI)
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    hdi_lower, hdi_upper = hdi[0], hdi[1]

    # calculate posterior standard deviation and define ROPE (symmetric around zero)
    beta_std = float(np.std(beta_samples))
    rope = 0.1 * beta_std

    # decide based on the 95% HDI relative to the ROPE
    if hdi_lower >= -rope and hdi_upper <= rope:
        conclusion = "Practically Equivalent"
    elif higher_is_better and hdi_lower > rope:
        conclusion = "Significant Improvement"
    elif not higher_is_better and hdi_upper < -rope:
        conclusion = "Significant Improvement"
    elif higher_is_better and hdi_upper < -rope:
        conclusion = "Significant Deterioration"
    elif not higher_is_better and hdi_lower > rope:
        conclusion = "Significant Deterioration"
    else:
        conclusion = "Inconclusive"

    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": beta_std,
        "slope_hdi_lower": float(hdi_lower),
        "slope_hdi_upper": float(hdi_upper),
        "rope": rope,
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples)),
        "metric": metric,
        "is_entropy": metric == "entropy",
        "max_entropy": max_entropy,
        "original_values": original_y.tolist()  # store original values for plotting
    }


def model_binary_metric(df: pd.DataFrame,
                        metric: str,
                        higher_is_better: bool) -> Optional[Dict]:
    """Fit a Bayesian model for a binary metric and decide based on the 95% HDI vs. ROPE.

    The decision rule is:
      - "Significant Improvement" if the entire 95% HDI lies above ROPE (for higher is better)
        or entirely below ROPE (for lower is better).
      - "Practically Equivalent" if the entire 95% HDI lies within the ROPE.
      - "Inconclusive" otherwise.

    Args:
        df: DataFrame with exemplar length and binary metric data.
        metric: Name of the binary metric column.
        higher_is_better: Whether higher values indicate better performance.

    Returns:
        A dictionary with model results and summary statistics, or None if modeling failed.
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

    # create DataFrame with valid data points
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
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)
            eta = alpha + beta * x_scaled
            theta = pm.Deterministic("theta", pm.math.sigmoid(eta))
            _ = pm.Bernoulli("likelihood", p=theta, observed=y)
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                              return_inferencedata=True)
    except Exception as e:
        print(f"Error in modeling binary metric {metric}: {str(e)}")
        return None

    # analyze results
    beta_samples = trace.posterior['beta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # calculate 95% HDI for beta samples
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    hdi_lower, hdi_upper = hdi[0], hdi[1]

    # calculate posterior standard deviation and define ROPE
    beta_std = float(np.std(beta_samples))
    rope = 0.1 * beta_std

    # decide based on the 95% HDI relative to the ROPE
    if hdi_lower >= -rope and hdi_upper <= rope:
        conclusion = "Practically Equivalent"
    elif higher_is_better and hdi_lower > rope:
        conclusion = "Significant Improvement"
    elif not higher_is_better and hdi_upper < -rope:
        conclusion = "Significant Improvement"
    elif higher_is_better and hdi_upper < -rope:
        conclusion = "Significant Deterioration"
    elif not higher_is_better and hdi_lower > rope:
        conclusion = "Significant Deterioration"
    else:
        conclusion = "Inconclusive"

    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": beta_std,
        "slope_hdi_lower": float(hdi_lower),
        "slope_hdi_upper": float(hdi_upper),
        "rope": rope,
        "conclusion": conclusion,
        "alpha_mean": float(np.mean(alpha_samples)),
        "beta_mean": float(np.mean(beta_samples)),
        "metric": metric,
        "is_binary": True,
        "x_mean": x_mean  # store for plotting
    }


def create_diagnostic_plot(df: pd.DataFrame,
                           metric: str,
                           model_results: Dict,
                           output_path: Path,
                           use_binary: bool = False) -> None:
    """Create a diagnostic plot showing the relationship between exemplar length and metric.

    This version includes annotations for the slope, HDI lower/upper bounds, ROPE, and conclusion.

    Args:
        df: DataFrame containing exemplar length and metric data.
        metric: Name of the metric column to plot.
        model_results: Dictionary with model results.
        output_path: Path to save the plot.
        use_binary: Whether to use binary version of the metric.
    """
    plt.figure(figsize=(12, 8))
    plot_metric = metric

    sns.set_style("whitegrid")

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

    if use_binary or model_results.get("is_binary", False):
        x_mean = model_results.get("x_mean", np.mean(df['exemplar_length']))
        x_scaled = (x_vals - x_mean) / 1000
        y_pred = 1.0 / (1.0 + np.exp(-(alpha_mean + beta_mean * x_scaled)))

        alpha_samples = np.random.normal(alpha_mean, model_results["slope_std"] * 0.5, 1000)
        beta_samples = np.random.normal(beta_mean, model_results["slope_std"], 1000)
        pred_samples = np.zeros((1000, len(x_vals)))
        for i in range(1000):
            pred_samples[i, :] = 1.0 / (1.0 + np.exp(-(alpha_samples[i] + beta_samples[i] * x_scaled)))
        lower_ci = np.percentile(pred_samples, 2.5, axis=0)
        upper_ci = np.percentile(pred_samples, 97.5, axis=0)

        plt.plot(x_vals, y_pred, 'r-', linewidth=2, label="Model fit")
        plt.fill_between(x_vals, lower_ci, upper_ci, color='red', alpha=0.2, label="95% CI")
    else:
        x_mean = np.mean(df['exemplar_length'])
        x_scaled = (x_vals - x_mean) / 1000
        logit_pred = alpha_mean + beta_mean * x_scaled
        prob_pred = 1.0 / (1.0 + np.exp(-logit_pred))
        if model_results.get("is_entropy", False):
            max_entropy = model_results.get("max_entropy", 5.49)
            y_pred = prob_pred * max_entropy
        else:
            y_pred = prob_pred

        alpha_samples = np.random.normal(alpha_mean, model_results["slope_std"] * 0.5, 1000)
        beta_samples = np.random.normal(beta_mean, model_results["slope_std"], 1000)
        pred_samples = np.zeros((1000, len(x_vals)))
        for i in range(1000):
            logit_val = alpha_samples[i] + beta_samples[i] * x_scaled
            prob_val = 1.0 / (1.0 + np.exp(-logit_val))
            if model_results.get("is_entropy", False):
                pred_samples[i, :] = prob_val * max_entropy
            else:
                pred_samples[i, :] = prob_val
        lower_ci = np.percentile(pred_samples, 2.5, axis=0)
        upper_ci = np.percentile(pred_samples, 97.5, axis=0)

        plt.plot(x_vals, y_pred, 'r-', linewidth=2, label="Model fit")
        plt.fill_between(x_vals, lower_ci, upper_ci, color='red', alpha=0.2, label="95% CI")

    # create plot title using available metadata
    llm = df['llm'].iloc[0] if df['llm'].nunique() == 1 else "Various"
    threat_model = df['threat_model'].iloc[0] if df['threat_model'].nunique() == 1 else "Various"
    corpus = df['corpus'].iloc[0] if df['corpus'].nunique() == 1 else "Various"
    plt.title(f"{corpus.upper()} - {llm} vs {threat_model}: Impact on {plot_metric}", fontsize=14)
    plt.xlabel("Exemplar Length (words)", fontsize=12)
    plt.ylabel(plot_metric, fontsize=12)

    # Annotate with new summary statistics
    annot_text = (
        f"Slope: {model_results['slope_mean']:.4f}\n"
        f"HDI: [{model_results['slope_hdi_lower']:.4f}, {model_results['slope_hdi_upper']:.4f}]\n"
        f"ROPE: ±{model_results['rope']:.4f}\n"
        f"Conclusion: {model_results['conclusion']}"
    )
    plt.annotate(
        annot_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
        fontsize=10,
        verticalalignment='top'
    )

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
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
    """Analyze data for a specific corpus/llm/threat_model scenario.

    Args:
        data_dir: Directory containing prepared data.
        output_dir: Directory to save results.
        corpus: Corpus name (ebg or rj).
        experiment: Experiment name (e.g., rq3.2_imitation_variable_length).
        llm: Language model name.
        threat_model: Attribution model name.
        debug: Whether to run in debug mode.

    Returns:
        List of dictionaries with analysis results.
    """
    plots_dir = output_dir / "plots" / corpus / experiment / llm / threat_model
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    for metric in METRICS:
        print(f"  Analyzing metric: {metric}")
        if metric not in df.columns:
            print(f"    Skipping {metric}: Not found in data")
            continue

        # determine if higher is better for this metric
        higher_is_better = metric in ["entropy", "bertscore", "pinc"]

        # use the appropriate model
        if metric.startswith('binary_'):
            print(f"    Using binary model for {metric}")
            model_results = model_binary_metric(df, metric, not higher_is_better)
        else:
            model_results = model_continuous_metric(df, metric, higher_is_better, MAX_ENTROPY[corpus])

        if model_results is None:
            print(f"    Skipping {metric}: Modeling failed")
            continue

        # record results including the new summary statistics
        result = {
            "Corpus": corpus.upper(),
            "Threat Model": threat_model,
            "LLM": llm,
            "Metric": metric,
            "Higher is Better": higher_is_better,
            "Slope": model_results['slope_mean'],
            "Slope Std": model_results['slope_std'],
            "Slope HDI Lower": model_results['slope_hdi_lower'],
            "Slope HDI Upper": model_results['slope_hdi_upper'],
            "ROPE": model_results['rope'],
            "Conclusion": model_results['conclusion']
        }
        results.append(result)

        # create diagnostic plot for each metric
        plot_path = plots_dir / f"{metric.replace('@', '_')}.png"
        create_diagnostic_plot(df, metric, model_results, plot_path, use_binary=metric.startswith('binary_'))
        print(f"    Created diagnostic plot: {plot_path}")

    return results


def analyze_data(data_dir: Path, output_dir: Path, debug: bool = False):
    """Analyze the relationship between exemplar length and metrics.

    Args:
        data_dir: Directory containing prepared data.
        output_dir: Directory to save analysis results.
        debug: Whether to run in debug mode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    corpora_to_process = ['ebg'] if debug else CORPORA
    llms_to_process = ['ministral'] if debug else LLMS
    threat_models_to_process = ['roberta'] if debug else THREAT_MODELS

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

            for llm in llms_to_process:
                for threat_model in threat_models_to_process:
                    results = analyze_scenario_data(data_dir, output_dir, corpus, experiment, llm, threat_model, debug)
                    all_results.extend(results)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = output_dir / "exemplar_length_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved combined results to {results_path}")

        # save a summary table with the new statistics
        summary_cols = ["Corpus", "Threat Model", "LLM", "Metric", "Higher is Better",
                        "Slope", "Slope Std", "Slope HDI Lower", "Slope HDI Upper", "ROPE", "Conclusion"]
        summary_df = results_df[summary_cols]
        summary_path = output_dir / "exemplar_length_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
    else:
        print("Warning: No results were generated")


def main():
    """Main function to analyze exemplar length data."""
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

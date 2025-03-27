import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

CORPORA = ['ebg', 'rj']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
METRICS = ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']


def extract_exemplar_lengths(prompt_dir: Path) -> Dict[int, int]:
    """Extract exemplar lengths from prompt files.

    Args:
        prompt_dir: Directory containing prompt JSON files

    Returns:
        Dictionary mapping prompt index to exemplar length
    """
    lengths = {}
    for prompt_file in prompt_dir.glob("*.json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)

        # extract prompt index from the metadata
        prompt_idx = prompt_data.get("metadata", {}).get("prompt_index")
        if prompt_idx is None:
            # fall back to extracting from filename if not in metadata
            match = re.search(r'prompt(\d+)\.json', prompt_file.name)
            if match:
                prompt_idx = int(match.group(1))
            else:
                print(f"Warning: Could not determine prompt index for {prompt_file}")
                continue

        # get the word_count from metadata
        word_count = prompt_data.get("metadata", {}).get("word_count")
        if word_count is not None:
            lengths[prompt_idx] = word_count
        else:
            print(
                f"Warning: No word_count found in metadata for prompt {prompt_idx}")

    return lengths


def prepare_exemplar_length_data(
        eval_base_dir: str,
        llm_outputs_dir: str,
        prompts_dir: str
) -> pd.DataFrame:
    """Prepare dataframe containing exemplar length data and evaluation metrics.

    Args:
        eval_base_dir: Base directory for evaluation results
        llm_outputs_dir: Directory containing LLM outputs
        prompts_dir: Directory containing prompt files

    Returns:
        DataFrame with exemplar lengths and evaluation metrics
    """
    rows = []
    for corpus in CORPORA:
        print(f"Processing corpus: {corpus}")
        for rq_folder in Path(llm_outputs_dir).joinpath(corpus, 'rq3').glob("rq3.2_imitation_variable_length"):
            # get the experiment name (e.g., "rq3.2_imitation_variable_length")
            experiment_name = rq_folder.name
            print(f"Processing experiment: {experiment_name}")

            # load prompt lengths from the corresponding prompts directory
            prompt_dir = Path(prompts_dir) / experiment_name
            if not prompt_dir.exists():
                print(f"Warning: Prompt directory not found: {prompt_dir}")
                continue

            prompt_lengths = extract_exemplar_lengths(prompt_dir)
            print(f"Extracted exemplar lengths for {len(prompt_lengths)} prompts")

            for model_dir in Path(eval_base_dir, corpus, 'rq3', rq_folder.name).glob("*"):
                model_name = model_dir.name.lower()
                print(f"Processing model: {model_name}")

                for eval_file in model_dir.glob("seed_*.json"):
                    seed = eval_file.stem.split("_")[-1]

                    # load detailed evaluation data for binary accuracy
                    binary_acc1 = None
                    binary_acc5 = None
                    try:
                        with open(eval_file, "r") as f:
                            detailed_eval_data = json.load(f)
                        if "example_metrics" in detailed_eval_data:
                            binary_acc1 = [
                                1 if ex["transformed_rank"] == 0 else 0
                                for ex in detailed_eval_data["example_metrics"]
                            ]
                            binary_acc5 = [
                                1 if ex["transformed_rank"] < 5 else 0
                                for ex in detailed_eval_data["example_metrics"]
                            ]
                    except Exception as e:
                        print(f"Warning: Could not extract binary metrics from {eval_file}: {e}")

                    # find the corresponding file in llm_outputs to get prompt_index
                    llm_output_file = Path(
                        llm_outputs_dir) / corpus / 'rq3' / rq_folder.name / model_name / f"seed_{seed}.json"

                    if not llm_output_file.exists():
                        print(f"Warning: LLM output file not found: {llm_output_file}")
                        continue

                    with open(llm_output_file) as f:
                        llm_data = json.load(f)

                    # extract prompt_index from the first entry in llm_data
                    prompt_idx = -1
                    if isinstance(llm_data, list) and llm_data:
                        if "prompt_index" in llm_data[0]:
                            prompt_idx = llm_data[0]["prompt_index"]
                        else:
                            for entry in llm_data:
                                if "prompt_index" in entry:
                                    prompt_idx = entry["prompt_index"]
                                    break

                    length = prompt_lengths.get(prompt_idx, -1)
                    if length == -1:
                        print(f"Warning: Could not find exemplar length for prompt index {prompt_idx}")
                        continue

                    # Load evaluation data
                    json_path = model_dir / "evaluation.json"
                    if not json_path.exists():
                        print(f"Warning: Evaluation JSON not found: {json_path}")
                        continue

                    with open(json_path) as f:
                        all_data = json.load(f)

                    if seed not in all_data:
                        print(f"Warning: Seed {seed} not found in {json_path}")
                        continue

                    record = all_data[seed]

                    for threat_model in THREAT_MODELS:
                        if threat_model not in record:
                            print(f"Warning: Threat model {threat_model} not found in record for seed {seed}")
                            continue

                        attr = record[threat_model].get("attribution", {}).get("post", {})
                        quality = record[threat_model].get("quality", {})

                        # use raw accuracy values (not binary conversion)
                        acc1 = float(attr.get("accuracy@1", 0.0))
                        acc5 = float(attr.get("accuracy@5", 0.0))

                        row = {
                            "corpus": corpus,
                            "llm": model_name,
                            "threat_model": threat_model,
                            "seed": seed,
                            "exemplar_length": length,
                            "accuracy@1": acc1,
                            "accuracy@5": acc5,
                            "true_class_confidence": attr.get("true_class_confidence"),
                            "entropy": attr.get("entropy"),
                            "bertscore": quality.get("bertscore", {}).get(
                                "bertscore_f1_avg"),
                            "pinc": np.mean([
                                quality.get("pinc", {}).get(f"pinc_{k}_avg", np.nan)
                                for k in range(1, 5)
                            ]),
                            "sample_id": prompt_idx
                        }

                        # add binary accuracy metrics if available
                        if binary_acc1 is not None:
                            row["binary_acc1"] = binary_acc1
                        if binary_acc5 is not None:
                            row["binary_acc5"] = binary_acc5

                        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows")
    return df


def model_continuous_metric(df, metric, higher_is_better, max_entropy_lookup):
    """Model the relationship between exemplar length and a continuous metric.

    Args:
        df: DataFrame with exemplar_length and metric columns
        metric: Name of the metric to model
        higher_is_better: Whether higher values of the metric are better
        max_entropy_lookup: Dictionary mapping corpus to max entropy value

    Returns:
        Dictionary with modeling results or None if modeling failed
    """
    # Drop rows with missing values
    df = df.dropna(subset=["exemplar_length", metric])

    if df.empty or df['exemplar_length'].nunique() <= 1:
        return None

    # Use all data points - each sample-seed combination is a unique observation
    x = df['exemplar_length'].values
    y = df[metric].values
    corpus = df['corpus'].iloc[0]

    # Scale x for better numerical stability
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000

    # Normalize entropy if needed
    if metric == 'entropy':
        y = y / max_entropy_lookup.get(corpus, np.log2(100))

    # Ensure y is in (0, 1) for beta regression
    epsilon = 1e-6
    y = np.clip(y, epsilon, 1 - epsilon)

    try:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)
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

    beta_samples = trace.posterior['beta'].values.flatten()
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))
    prob_benefit = float(
        np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))
    conclusion = "Practically Equivalent" if in_rope > 0.95 else (
        "Significant Improvement" if prob_benefit > 0.95 else (
            "Significant Deterioration" if prob_benefit < 0.05 else "Inconclusive"))
    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion
    }


def model_binary_metrics_from_examples(df, metric_name, higher_is_better):
    """Model binary metrics extracted from example-level data.

    Args:
        df: DataFrame with binary metrics
        metric_name: Base name of the metric (binary_acc1 or binary_acc5)
        higher_is_better: Whether higher values of the metric are better

    Returns:
        Dictionary with modeling results or None if modeling failed
    """
    # Expand the lists of binary metrics and corresponding exemplar lengths
    binary_metrics = []
    exemplar_lengths = []

    for idx, row in df.iterrows():
        if metric_name not in row or row[metric_name] is None:
            continue

        binary_list = row[metric_name]
        for binary_value in binary_list:
            binary_metrics.append(binary_value)
            exemplar_lengths.append(row['exemplar_length'])

    if not binary_metrics or len(set(binary_metrics)) < 2:
        return None

    # Create a new dataframe with expanded data
    expanded_df = pd.DataFrame({
        'exemplar_length': exemplar_lengths,
        'binary_metric': binary_metrics
    })

    if expanded_df.empty or expanded_df['exemplar_length'].nunique() <= 1:
        return None

    # Use all data points - each binary outcome is a distinct observation
    x = expanded_df['exemplar_length'].values
    y = expanded_df['binary_metric'].values

    # Scale x for better numerical stability
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000

    try:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.Cauchy("beta", alpha=0, beta=0.5)
            eta = alpha + beta * x_scaled
            theta = pm.Deterministic("theta", pm.math.sigmoid(eta))
            _ = pm.Bernoulli("likelihood", p=theta, observed=y)
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                              return_inferencedata=True)
    except Exception as e:
        print(f"Error in modeling {metric_name}: {str(e)}")
        return None

    beta_samples = trace.posterior['beta'].values.flatten()
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))
    prob_benefit = float(
        np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))
    conclusion = "Practically Equivalent" if in_rope > 0.95 else (
        "Significant Improvement" if prob_benefit > 0.95 else (
            "Significant Deterioration" if prob_benefit < 0.05 else "Inconclusive"))
    return {
        "slope_mean": float(np.mean(beta_samples)),
        "slope_std": float(np.std(beta_samples)),
        "slope_hdi": hdi.tolist(),
        "in_rope": float(in_rope),
        "prob_benefit": float(prob_benefit),
        "conclusion": conclusion
    }


def main():
    """Main function to run the exemplar length analysis."""
    import argparse
    output_dir = Path("results/exemplar_length_analysis_rq3.2")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Analyze the impact of exemplar length on defense performance")
    parser.add_argument("--eval_dir", type=str, default="defense_evaluation",
                        help="Directory containing evaluation results")
    parser.add_argument("--llm_dir", type=str, default="llm_outputs",
                        help="Directory containing LLM outputs")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Directory containing prompt files")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (EBG corpus, Ministral vs RoBERTa only)")
    args = parser.parse_args()

    print("Preparing exemplar length data...")
    # in debug mode, we still gather all data to verify exemplar lengths
    df = prepare_exemplar_length_data(args.eval_dir, args.llm_dir, args.prompt_dir)

    # check if we have varying exemplar lengths in the data
    print(f"\nExemplar length statistics:")
    print(f"Min length: {df['exemplar_length'].min()}")
    print(f"Max length: {df['exemplar_length'].max()}")
    print(f"Unique lengths: {sorted(df['exemplar_length'].unique())}")

    # save raw data before any processing
    df.to_csv(output_dir / "raw_data.csv", index=False)
    print(f"Raw data saved to {output_dir / 'raw_data.csv'}")

    # print the unique values for key columns to help with debugging
    print("\nUnique values in the dataset:")
    print(f"Corpus: {df['corpus'].unique()}")
    print(f"Threat Models: {df['threat_model'].unique()}")
    print(f"LLMs: {df['llm'].unique()}")

    # map model names to standardized names for analysis
    model_mapping = {
        'gpt-4o-2024-08-06': 'gpt-4o',
        'claude-3-5-sonnet-20241022': 'claude-3.5',
        'gemma-2-9b-it': 'gemma-2',
        'llama-3.1-8b-instruct': 'llama-3.1',
        'ministral-8b-instruct-2410': 'ministral'
    }

    # apply mapping to standardize model names
    df['llm_standardized'] = df['llm'].map(
        lambda x: next((v for k, v in model_mapping.items() if k in x), x))

    max_entropy_lookup = {"ebg": np.log2(45), "rj": np.log2(21)}
    records = []

    print("\nModeling the relationship between exemplar length and metrics...")

    # filter corpora, models, and threat models for debug mode
    corpora_to_process = ['ebg'] if args.debug else CORPORA
    llms_to_process = ['ministral'] if args.debug else LLMS
    threat_models_to_process = ['roberta'] if args.debug else THREAT_MODELS

    for corpus in corpora_to_process:
        for threat_model in threat_models_to_process:
            for llm in llms_to_process:
                # use the standardized model names for filtering
                subset = df[(df.corpus == corpus) &
                            (df.threat_model == threat_model) &
                            (df.llm_standardized == llm)]

                if subset.empty:
                    print(f"Warning: No data for {corpus}-{threat_model}-{llm}")
                    continue

                print(
                    f"Processing {corpus}-{threat_model}-{llm} with {len(subset)} samples")

                # In debug mode, print additional information about the subset
                if args.debug:
                    print(
                        f"  Exemplar lengths in subset: {sorted(subset['exemplar_length'].unique())}")
                    print(f"  Number of samples by exemplar length:")
                    print(subset['exemplar_length'].value_counts().sort_index())

                for metric in METRICS:
                    higher_is_better = metric in ["entropy", "bertscore", "pinc"]

                    if args.debug:
                        print(f"  Processing metric: {metric}")
                        print(f"  Metric values by exemplar length:")

                        # Group by exemplar length and show metric statistics
                        stats = subset.groupby('exemplar_length')[metric].agg(
                            ['mean', 'std', 'count'])
                        print(stats)

                    if metric == "accuracy@1":
                        # use binary accuracy if available
                        if "binary_acc1" in subset.columns and not subset[
                            "binary_acc1"].isna().all():
                            res = model_binary_metrics_from_examples(subset,
                                                                     "binary_acc1",
                                                                     not higher_is_better)
                            metric_display = "binary_accuracy@1"
                        else:
                            res = model_continuous_metric(subset, metric,
                                                          not higher_is_better,
                                                          max_entropy_lookup)
                            metric_display = metric
                    elif metric == "accuracy@5":
                        # use binary accuracy if available
                        if "binary_acc5" in subset.columns and not subset[
                            "binary_acc5"].isna().all():
                            res = model_binary_metrics_from_examples(subset,
                                                                     "binary_acc5",
                                                                     not higher_is_better)
                            metric_display = "binary_accuracy@5"
                        else:
                            res = model_continuous_metric(subset, metric,
                                                          not higher_is_better,
                                                          max_entropy_lookup)
                            metric_display = metric
                    else:
                        res = model_continuous_metric(subset, metric, higher_is_better,
                                                      max_entropy_lookup)
                        metric_display = metric

                    if res is None:
                        print(f"  Skipping {metric}: Modeling failed")
                        continue

                    records.append({
                        "Corpus": corpus.upper(),
                        "Threat Model": threat_model,
                        "LLM": llm,
                        "Metric": metric_display,
                        "Higher is Better": higher_is_better,
                        "Slope": res['slope_mean'],
                        "Slope Std": res['slope_std'],
                        "Slope HDI Lower": res['slope_hdi'][0],
                        "Slope HDI Upper": res['slope_hdi'][1],
                        "P(Improvement)": res['prob_benefit'],
                        "In ROPE": res['in_rope'],
                        "Conclusion": res['conclusion']
                    })
                    print(f"  Added results for {metric_display}")

                    # in debug mode, add a simple plot to visualize the relationship
                    # In debug mode, add a simple plot to visualize the relationship
                    # In debug mode, add a simple plot to visualize the relationship
                    if args.debug:
                        try:
                            import matplotlib.pyplot as plt

                            plt.figure(figsize=(12, 6))

                            # Scatter plot with points colored by seed
                            for seed in sorted(subset['seed'].unique()):
                                seed_data = subset[subset['seed'] == seed]
                                plt.scatter(seed_data['exemplar_length'],
                                            seed_data[metric],
                                            label=f"Seed {seed}", alpha=0.7)

                            # Add regression line
                            x = subset['exemplar_length'].values
                            y = subset[metric].values
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            sorted_x = np.sort(x)
                            plt.plot(sorted_x, p(sorted_x), "r--", linewidth=2,
                                     label="Linear trend")

                            plt.title(
                                f"{corpus.upper()} - {llm} vs {threat_model}: {metric_display}")
                            plt.xlabel("Exemplar Length")
                            plt.ylabel(metric_display)
                            plt.legend(loc='best')
                            plt.grid(True, alpha=0.3)

                            # Add slope and other statistics as text
                            plt.text(0.05, 0.95,
                                     f"Slope: {res['slope_mean']:.4f}\n"
                                     f"95% HDI: [{res['slope_hdi'][0]:.4f}, {res['slope_hdi'][1]:.4f}]\n"
                                     f"P(Improvement): {res['prob_benefit']:.4f}\n"
                                     f"Conclusion: {res['conclusion']}",
                                     transform=plt.gca().transAxes,
                                     verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white',
                                               alpha=0.8))

                            # Save the plot
                            plot_path = output_dir / f"debug_{corpus}_{llm}_{threat_model}_{metric_display.replace('@', '_')}.png"
                            plt.savefig(plot_path)
                            plt.close()
                            print(f"  Debug plot saved to {plot_path}")
                        except Exception as e:
                            print(f"  Could not generate debug plot: {e}")

    results_df = pd.DataFrame(records)
    results_file = output_dir / "exemplar_length_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()

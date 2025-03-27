import json
import re
from pathlib import Path
from typing import Dict, Optional

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

        # extract prompt index from filename
        match = re.search(r'prompt(\d+)\.json', prompt_file.name)
        if match:
            idx = int(match.group(1))
            lengths[idx] = prompt_data["metadata"]["word_count"]
    return lengths


def extract_exemplar_from_prompt(data: Dict) -> Optional[str]:
    """Extract exemplar text from a prompt.

    Args:
        data: Prompt data dictionary

    Returns:
        Extracted exemplar text or None if not found
    """
    user_instruction = data.get("user", "")
    start = user_instruction.find("expected to mimic:")
    end = user_instruction.find("Please rewrite", start)
    if start != -1 and end != -1:
        return user_instruction[start + 18:end].strip()
    return None


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
    found_data = False

    for corpus in CORPORA:
        print(f"Processing corpus: {corpus}")
        rq3_path = Path(llm_outputs_dir).joinpath(corpus, 'rq3')
        if not rq3_path.exists():
            print(f"Warning: Path not found: {rq3_path}")
            continue

        rq_folders = list(rq3_path.glob("rq3.2_*"))
        print(f"Found {len(rq_folders)} research question folders in {rq3_path}")

        for rq_folder in rq_folders:
            # Get the experiment name (e.g., "rq3.2_imitation_variable_length")
            experiment_name = rq_folder.name
            print(f"Processing experiment: {experiment_name}")

            # Load prompt lengths from the corresponding prompts directory
            prompt_dir = Path(prompts_dir) / experiment_name
            if not prompt_dir.exists():
                print(f"Warning: Prompt directory not found: {prompt_dir}")
                continue

            prompt_files = list(prompt_dir.glob("*.json"))
            print(f"Found {len(prompt_files)} prompt files in {prompt_dir}")

            prompt_lengths = extract_exemplar_lengths(prompt_dir)
            print(f"Extracted exemplar lengths for {len(prompt_lengths)} prompts")
            if not prompt_lengths:
                print(f"Warning: No exemplar lengths extracted from {prompt_dir}")

            eval_path = Path(eval_base_dir, corpus, 'rq3', rq_folder.name)
            if not eval_path.exists():
                print(f"Warning: Evaluation path not found: {eval_path}")
                continue

            model_dirs = list(eval_path.glob("*"))
            print(f"Found {len(model_dirs)} model directories in {eval_path}")

            for model_dir in model_dirs:
                model_name = model_dir.name.lower()
                print(f"Processing model: {model_name}")

                eval_files = list(model_dir.glob("seed_*.json"))
                print(f"Found {len(eval_files)} evaluation files for {model_name}")

                for eval_file in eval_files:
                    seed = eval_file.stem.split("_")[-1]
                    print(f"Processing seed file: {eval_file}")

                    # Find the corresponding file in llm_outputs to get prompt_index
                    llm_output_path = Path(
                        llm_outputs_dir) / corpus / 'rq3' / rq_folder.name / model_name
                    if not llm_output_path.exists():
                        print(f"Warning: LLM output path not found: {llm_output_path}")
                        continue

                    llm_output_file = llm_output_path / f"seed_{seed}.json"
                    if not llm_output_file.exists():
                        print(f"Warning: LLM output file not found: {llm_output_file}")
                        continue

                    try:
                        with open(llm_output_file) as f:
                            llm_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error: Could not parse JSON in {llm_output_file}")
                        continue

                    # Extract prompt_index from the first entry in llm_data
                    prompt_idx = -1
                    if isinstance(llm_data, list) and llm_data:
                        if "prompt_index" in llm_data[0]:
                            prompt_idx = llm_data[0]["prompt_index"]
                        else:
                            print(
                                f"Warning: 'prompt_index' not found in first entry of {llm_output_file}")
                            # Try to find it in any entry
                            for entry in llm_data:
                                if "prompt_index" in entry:
                                    prompt_idx = entry["prompt_index"]
                                    break
                    else:
                        print(
                            f"Warning: Expected list format in {llm_output_file}, got {type(llm_data)}")

                    if prompt_idx == -1:
                        print(
                            f"Warning: Could not find prompt_index in {llm_output_file}")
                        continue

                    length = prompt_lengths.get(prompt_idx, -1)
                    if length == -1:
                        print(
                            f"Warning: No exemplar length found for prompt index {prompt_idx}")
                        continue

                    # Load evaluation data
                    json_path = model_dir / "evaluation.json"
                    if not json_path.exists():
                        print(f"Warning: Evaluation JSON not found: {json_path}")
                        continue

                    try:
                        with open(json_path) as f:
                            all_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error: Could not parse JSON in {json_path}")
                        continue

                    if seed not in all_data:
                        print(f"Warning: Seed {seed} not found in {json_path}")
                        continue

                    record = all_data[seed]
                    found_data = True

                    for threat_model in THREAT_MODELS:
                        if threat_model not in record:
                            print(
                                f"Warning: Threat model {threat_model} not found in record for seed {seed}")
                            continue

                        attr = record[threat_model].get("attribution", {}).get("post",
                                                                               {})
                        if not attr:
                            print(
                                f"Warning: No attribution data for {threat_model} in seed {seed}")
                            continue

                        quality = record[threat_model].get("quality", {})
                        if not quality:
                            print(
                                f"Warning: No quality data for {threat_model} in seed {seed}")

                        row = {
                            "corpus": corpus,
                            "llm": model_name,
                            "threat_model": threat_model,
                            "seed": seed,
                            "exemplar_length": length,
                            "accuracy@1": int(attr.get("accuracy@1") == 1.0),
                            "accuracy@5": int(attr.get("accuracy@5") == 1.0),
                            "true_class_confidence": attr.get("true_class_confidence"),
                            "entropy": attr.get("entropy"),
                            "bertscore": quality.get("bertscore", {}).get(
                                "bertscore_f1_avg"),
                            "pinc": np.mean([
                                quality.get("pinc", {}).get(f"pinc_{k}_avg", np.nan)
                                for k in range(1, 5)
                            ])
                        }
                        rows.append(row)
                        print(
                            f"Added row for {corpus}-{threat_model}-{model_name}-{seed}")

    if not found_data:
        print(
            "ERROR: No data was found. Please check your directory structure and file formats.")

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
    df = df.dropna(subset=["exemplar_length", metric])
    if df.empty or df['exemplar_length'].nunique() <= 1:
        return None
    x = df['exemplar_length'].values
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000
    y = df[metric].values
    if metric == 'entropy':
        corpus = df['corpus'].iloc[0]
        y = y / max_entropy_lookup.get(corpus, np.log2(100))
    epsilon = 1e-6
    y = np.clip(y, epsilon, 1 - epsilon)
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
    beta_samples = trace.posterior['beta'].values.flatten()
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))
    prob_benefit = float(np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))
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


def model_binary_metric(df, metric, higher_is_better):
    """Model the relationship between exemplar length and a binary metric.

    Args:
        df: DataFrame with exemplar_length and metric columns
        metric: Name of the metric to model
        higher_is_better: Whether higher values of the metric are better

    Returns:
        Dictionary with modeling results or None if modeling failed
    """
    df = df.dropna(subset=["exemplar_length", metric])
    if df.empty or df['exemplar_length'].nunique() <= 1:
        return None
    x = df['exemplar_length'].values
    x_mean = np.mean(x)
    x_scaled = (x - x_mean) / 1000
    y = (df[metric] == 1).astype(int).values
    if len(np.unique(y)) < 2:
        return None
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 2)
        beta = pm.Cauchy("beta", alpha=0, beta=0.5)
        eta = alpha + beta * x_scaled
        theta = pm.Deterministic("theta", pm.math.sigmoid(eta))
        _ = pm.Bernoulli("likelihood", p=theta, observed=y)
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=4,
                          return_inferencedata=True)
    beta_samples = trace.posterior['beta'].values.flatten()
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    rope = 0.1 * np.std(beta_samples)
    in_rope = np.mean((beta_samples >= -rope) & (beta_samples <= rope))
    prob_benefit = float(np.mean(beta_samples > 0) if higher_is_better else np.mean(beta_samples < 0))
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

    parser = argparse.ArgumentParser(description="Analyze the impact of exemplar length on defense performance")
    parser.add_argument("--eval_dir", type=str, default="defense_evaluation",
                        help="Directory containing evaluation results")
    parser.add_argument("--llm_dir", type=str, default="llm_outputs",
                        help="Directory containing LLM outputs")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Directory containing prompt files")
    args = parser.parse_args()

    print("Preparing exemplar length data...")
    df = prepare_exemplar_length_data(args.eval_dir, args.llm_dir, args.prompt_dir)
    df.to_csv(output_dir / "raw_data.csv", index=False)
    print(f"Raw data saved to {output_dir / 'raw_data.csv'}")

    max_entropy_lookup = {"ebg": np.log2(45), "rj": np.log2(21)}
    records = []

    print("Modeling the relationship between exemplar length and metrics...")
    for corpus in CORPORA:
        for threat_model in THREAT_MODELS:
            for llm in LLMS:
                subset = df[(df.corpus == corpus) &
                            (df.threat_model == threat_model) &
                            (df.llm == llm)]

                if subset.empty:
                    print(f"Warning: No data for {corpus}-{threat_model}-{llm}")
                    continue

                print(f"Processing {corpus}-{threat_model}-{llm}...")

                for metric in METRICS:
                    higher_is_better = metric in ["entropy", "bertscore", "pinc"]
                    if metric in ["accuracy@1", "accuracy@5"]:
                        res = model_binary_metric(subset, metric, higher_is_better)
                    else:
                        res = model_continuous_metric(subset, metric, higher_is_better, max_entropy_lookup)

                    if res is None:
                        continue

                    records.append({
                        "Corpus": corpus.upper(),
                        "Threat Model": threat_model,
                        "LLM": llm,
                        "Metric": metric,
                        "Higher is Better": higher_is_better,
                        "Slope": res['slope_mean'],
                        "Slope Std": res['slope_std'],
                        "Slope HDI Lower": res['slope_hdi'][0],
                        "Slope HDI Upper": res['slope_hdi'][1],
                        "P(Improvement)": res['prob_benefit'],
                        "In ROPE": res['in_rope'],
                        "Conclusion": res['conclusion']
                    })

    results_df = pd.DataFrame(records)
    results_file = output_dir / "exemplar_length_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()

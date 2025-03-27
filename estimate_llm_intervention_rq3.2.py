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
    for corpus in CORPORA:
        for rq_folder in Path(llm_outputs_dir).joinpath(corpus, 'rq3').glob("rq3.2_*"):
            # get the experiment name (e.g., "rq3.2_imitation_variable_length")
            experiment_name = rq_folder.name

            # load prompt lengths from the corresponding prompts directory
            prompt_dir = Path(prompts_dir) / experiment_name
            if not prompt_dir.exists():
                print(f"Warning: Prompt directory not found: {prompt_dir}")
                continue

            prompt_lengths = extract_exemplar_lengths(prompt_dir)

            for model_dir in Path(eval_base_dir, corpus, 'rq3', rq_folder.name).glob(
                    "*"):
                model_name = model_dir.name.lower()
                for eval_file in model_dir.glob("seed_*.json"):
                    with open(eval_file) as f:
                        eval_data = json.load(f)
                    prompt_idx = eval_data[0].get("prompt_index", -1)
                    length = prompt_lengths.get(prompt_idx, -1)
                    seed = eval_file.stem.split("_")[-1]
                    json_path = model_dir / "evaluation.json"
                    if not json_path.exists():
                        continue
                    with open(json_path) as f:
                        all_data = json.load(f)
                    if seed not in all_data:
                        continue
                    record = all_data[seed]
                    for threat_model in THREAT_MODELS:
                        if threat_model not in record:
                            continue
                        attr = record[threat_model].get("attribution", {}).get("post",
                                                                               {})
                        quality = record[threat_model].get("quality", {})
                        rows.append({
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
                        })
    return pd.DataFrame(rows)


def model_continuous_metric(df, metric, higher_is_better, max_entropy_lookup):
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


if __name__ == "__main__":
    import argparse
    output_dir = Path("results/exemplar_length_analysis_rq3.2")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="defense_evaluation")
    parser.add_argument("--llm_dir", type=str, default="llm_outputs")
    parser.add_argument("--prompt_dir", type=str, default="prompts")
    args = parser.parse_args()

    df = prepare_exemplar_length_data(args.eval_dir, args.llm_dir, args.prompt_dir)
    df.to_csv(output_dir / "raw_data.csv", index=False)

    max_entropy_lookup = {"ebg": np.log2(45), "rj": np.log2(21)}
    records = []

    for corpus in CORPORA:
        for threat_model in THREAT_MODELS:
            for llm in LLMS:
                subset = df[(df.corpus == corpus) & (df.threat_model == threat_model) & (df.llm == llm)]
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

    pd.DataFrame(records).to_csv(output_dir / "exemplar_length_results.csv", index=False)

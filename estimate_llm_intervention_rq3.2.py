"""
Analyze the effect of exemplar length on LLM imitation defense performance.

This script analyzes how varying exemplar lengths (500/1000/2500 words) affect 
defense effectiveness against authorship attribution models, but now uses
the *actual* exemplar length extracted from each seed_{seed}.json file
rather than a fallback.

Usage:
    python estimate_llm_intervention_rq3.2.py
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sacremoses import MosesTokenizer
from scipy.special import expit  # for logistic transform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
METRICS = ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
CORPORA = ['ebg', 'rj']

# We no longer rely on [500, 1000, 2500] fallback for length, but
# we keep them if you still want to loop over subfolders. Otherwise you can remove.
EXEMPLAR_LENGTHS = [500, 1000, 2500]


def extract_exemplar_and_count(llm_output_file: str) -> dict:
    """
    Given a path to an LLM output JSON (like seed_XXXX.json),
    extract the exemplar text from the 'user' field and count its words.

    Returns a dict with:
      {
        "exemplar_text": str,
        "word_count": int,
        "file": str  # original file name
      }
    """
    marker_start = "Here is an example of the writing style you are expected to mimic:\n\n"
    marker_end   = "\n\nPlease rewrite the following text"

    with open(llm_output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_text = data.get("user", "")
    if not user_text:
        raise ValueError(f"No 'user' field found in {llm_output_file}")

    start_idx = user_text.find(marker_start)
    if start_idx == -1:
        raise ValueError(f"Could not find marker_start in 'user' text for {llm_output_file}")

    start_idx += len(marker_start)
    end_idx = user_text.find(marker_end, start_idx)
    if end_idx == -1:
        raise ValueError(f"Could not find marker_end in 'user' text for {llm_output_file}")

    exemplar_text = user_text[start_idx:end_idx].strip()

    tokenizer = MosesTokenizer(lang='en')
    tokens = tokenizer.tokenize(exemplar_text, escape=False)
    word_count = len(tokens)

    return {
        "exemplar_text": exemplar_text,
        "word_count": word_count,
        "file": llm_output_file
    }


def normalize_llm_name(model_name: str) -> str:
    """Standardize LLM names for consistency."""
    name = model_name.lower()
    if 'llama' in name:
        return 'llama-3.1'
    elif 'gemma' in name:
        return 'gemma-2'
    elif 'ministral' in name:
        return 'ministral'
    elif 'sonnet' in name or 'claude' in name:
        return 'claude-3.5'
    elif 'gpt' in name:
        return 'gpt-4o'
    return name


def prepare_data(
        evaluation_base_dir: str,
        llm_outputs_base_dir: str,
        debug: bool = False
) -> pd.DataFrame:
    """
    Combine metrics from defense_evaluation with actual exemplar text from llm_outputs.

    1) For each corpus, RQ folder (rq3.2_imitation_w_{length}words), and model_dir:
       - Parse evaluation.json and seed_{seed}.json in defense_evaluation to get metrics
       - Parse seed_{seed}.json in llm_outputs to get the 'user' prompt => exemplar length
    2) Merge them into a single row (exemplar_length + metrics).
    """

    import json
    import logging
    from pathlib import Path
    import pandas as pd

    logger = logging.getLogger(__name__)

    data = []
    total_rows = 0
    missing_eval_seed = 0
    missing_llm_seed = 0
    parse_errors = 0

    # We keep the same subfolder structure you have: rq3.2_imitation_w_500words, etc.
    # If you no longer want to loop over [500, 1000, 2500], remove it or adapt as needed.
    for target_length in [500, 1000, 2500]:
        rq_folder = f"rq3.2_imitation_w_{target_length}words"
        rq_main = "rq3"

        # For each corpus: ebg, rj
        for corpus in CORPORA:
            eval_rq_path = Path(evaluation_base_dir) / corpus / rq_main / rq_folder
            llm_rq_path = Path(llm_outputs_base_dir) / corpus / rq_main / rq_folder

            if debug:
                logger.info(f"[DEBUG] Checking corpus={corpus}, RQ folder={rq_folder}")
                logger.info(
                    f"[DEBUG] eval_rq_path={eval_rq_path}, llm_rq_path={llm_rq_path}")

            if not eval_rq_path.exists():
                logger.warning(f"Evaluation path not found: {eval_rq_path}")
                continue
            if not llm_rq_path.exists():
                logger.warning(f"LLM outputs path not found: {llm_rq_path}")
                continue

            # Loop over each model_dir in the EVAL folder
            for model_dir in eval_rq_path.glob("*"):
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name  # e.g. "gemma-2-9b-it"
                llm_eval_dir = model_dir  # path in defense_evaluation
                llm_out_dir = llm_rq_path / model_name  # corresponding folder in llm_outputs

                if not llm_out_dir.exists():
                    logger.warning(f"No matching LLM output dir for {llm_out_dir}")
                    continue

                # parse evaluation.json
                eval_file = llm_eval_dir / "evaluation.json"
                if not eval_file.exists():
                    logger.warning(f"evaluation.json not found: {eval_file}")
                    continue

                # read evaluation.json => results
                try:
                    with open(eval_file, "r") as f:
                        evaluation_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing {eval_file}: {e}")
                    continue

                # For each seed in evaluation_data, parse the post metrics
                for seed_str, seed_results in evaluation_data.items():
                    # We want metrics from "defense_evaluation" => seed_{seed_str}.json
                    # We want prompt from "llm_outputs" => seed_{seed_str}.json
                    eval_seed_file = llm_eval_dir / f"seed_{seed_str}.json"
                    llm_seed_file = llm_out_dir / f"seed_{seed_str}.json"

                    # If eval_seed_file missing => skip
                    if not eval_seed_file.exists():
                        missing_eval_seed += 1
                        if debug:
                            logger.warning(
                                f"[DEBUG] Missing eval seed file: {eval_seed_file}")
                        continue

                    # If llm_seed_file missing => skip (can't parse user prompt)
                    if not llm_seed_file.exists():
                        missing_llm_seed += 1
                        if debug:
                            logger.warning(
                                f"[DEBUG] Missing LLM seed file: {llm_seed_file}")
                        continue

                    # parse the prompt from llm_outputs
                    try:
                        extracted = extract_exemplar_and_count(str(llm_seed_file))
                        exemplar_length = extracted["word_count"]
                    except Exception as e:
                        parse_errors += 1
                        if debug:
                            logger.warning(
                                f"[DEBUG] error extracting from {llm_seed_file}: {e}")
                        continue

                    # Now parse the evaluation metrics from seed_results
                    # e.g. "logreg", "svm", "roberta"
                    for threat_model, tm_data in seed_results.items():
                        if threat_model not in THREAT_MODELS:
                            continue
                        # post metrics
                        post = tm_data["attribution"]["post"]

                        # Build a row
                        row = {
                            "corpus": corpus,
                            "llm": normalize_llm_name(model_name),
                            "threat_model": threat_model,
                            "seed": seed_str,
                            "exemplar_length": float(exemplar_length),
                            "accuracy@1": post.get("accuracy@1"),
                            "accuracy@5": post.get("accuracy@5"),
                            "true_class_confidence": post.get("true_class_confidence"),
                            "entropy": post.get("entropy")
                        }

                        # If "quality" is present
                        if "quality" in tm_data:
                            q = tm_data["quality"]
                            # PINC
                            if "pinc" in q:
                                pinc_scores = []
                                for k in range(1, 5):
                                    key = f"pinc_{k}_avg"
                                    if key in q["pinc"]:
                                        pinc_scores.append(q["pinc"][key])
                                if pinc_scores:
                                    row["pinc"] = float(np.mean(pinc_scores))

                            # BERTScore
                            if "bertscore" in q and "bertscore_f1_avg" in q[
                                "bertscore"]:
                                row["bertscore"] = q["bertscore"]["bertscore_f1_avg"]

                        # parse example_metrics from the EVAL seed file => binary outcomes
                        try:
                            with open(eval_seed_file, "r") as f:
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
                                row["binary_acc1"] = binary_acc1
                                row["binary_acc5"] = binary_acc5
                        except Exception as e:
                            logger.error(
                                f"Error reading eval seed file {eval_seed_file}: {e}")

                        data.append(row)
                        total_rows += 1

    df = pd.DataFrame(data)
    logger.info(
        f"Collected {len(df)} data points (rows) across "
        f"{df['corpus'].nunique()} corpora, "
        f"{df['llm'].nunique()} LLMs, "
        f"{df['threat_model'].nunique()} threat models, "
        f"and {df['exemplar_length'].nunique()} distinct exemplar lengths."
    )
    logger.info(
        f"Missing eval seed: {missing_eval_seed}, missing llm seed: {missing_llm_seed}, parse_errors: {parse_errors}")
    if debug:
        logger.info(
            f"[DEBUG] total_rows processed = {total_rows}, final df size = {len(df)}")

    return df


def analyze_exemplar_length_effect(
        data: pd.DataFrame,
        metric: str,
        corpus: str = "ebg",
        threat_model: str = "roberta",
        llm: str = "ministral",
        higher_is_better: Optional[bool] = None
) -> Dict:
    """
    Analyze the effect of exemplar length on a metric using Bayesian modeling.
    Uses robust Bayesian methods following Kruschke's approach.
    """
    df = data[(data['corpus'] == corpus) &
              (data['threat_model'] == threat_model) &
              (data['llm'] == llm)].copy()

    logger.info(f"Found {len(df)} data points for {corpus}-{threat_model}-{llm}-{metric}")
    logger.info(f"Data points by exemplar length: {df['exemplar_length'].value_counts().to_dict()}")

    if len(df) < 3 or metric not in df.columns or df[metric].isna().all():
        return {"error": f"Insufficient data for {corpus}-{threat_model}-{llm}-{metric}"}

    if higher_is_better is None:
        higher_is_better = metric in ['entropy', 'bertscore', 'pinc', 'meteor']

    binary_key = f'binary_{metric.replace("@", "")}'
    if metric in ['accuracy@1', 'accuracy@5'] and binary_key in df.columns:
        # If we have binary outcomes, do logistic modeling
        binary_rows = []
        for _, row in df.iterrows():
            if binary_key in row and isinstance(row[binary_key], list):
                for outcome in row[binary_key]:
                    binary_rows.append({
                        'corpus': row['corpus'],
                        'llm': row['llm'],
                        'threat_model': row['threat_model'],
                        'seed': row['seed'],
                        'exemplar_length': row['exemplar_length'],
                        'binary_outcome': outcome
                    })
        binary_df = pd.DataFrame(binary_rows)
        if len(binary_df) == 0:
            logger.warning(f"No binary outcomes found for {metric}, falling back to aggregate values")
        else:
            logger.info(f"Using {len(binary_df)} binary outcomes for {metric}")
            binary_df = binary_df.dropna(subset=['binary_outcome', 'exemplar_length'])
            x = binary_df['exemplar_length'].values
            y = binary_df['binary_outcome'].values
            x_mean = np.mean(x)
            x_range = np.linspace(min(x), max(x), 100)
            x_scaled = (x - x_mean) / 1000.0

            with pm.Model() as model:
                try:
                    alpha = pm.Normal("alpha", mu=0, sigma=2)
                    beta = pm.Cauchy("beta", alpha=0, beta=0.5)
                    eta = alpha + beta * x_scaled
                    theta = pm.Deterministic("theta", pm.math.invlogit(eta))
                    likelihood = pm.Bernoulli("likelihood", p=theta, observed=y)
                    trace = pm.sample(
                        2000, tune=1000, chains=4, random_seed=42,
                        target_accept=0.95, return_inferencedata=True, cores=4
                    )
                    alpha_samples = trace.posterior["alpha"].values.flatten()
                    beta_samples = trace.posterior["beta"].values.flatten()
                    x_range_scaled = (x_range - x_mean) / 1000.0
                    beta_mean = float(np.mean(beta_samples))
                    beta_std = float(np.std(beta_samples))
                    beta_hdi = az.hdi(beta_samples, hdi_prob=0.95)

                    if higher_is_better:
                        posterior_prob_beneficial = float(np.mean(beta_samples > 0))
                        direction = "positive" if posterior_prob_beneficial > 0.5 else "negative"
                    else:
                        posterior_prob_beneficial = float(np.mean(beta_samples < 0))
                        direction = "negative" if posterior_prob_beneficial > 0.5 else "positive"

                    effect_sd = beta_std
                    rope_bounds = (-0.05 * effect_sd, 0.05 * effect_sd)
                    in_rope = float(np.mean((beta_samples >= rope_bounds[0]) & (beta_samples <= rope_bounds[1])))

                    if (beta_hdi[0] > rope_bounds[1]) or (beta_hdi[1] < rope_bounds[0]):
                        significance = "Credible Effect"
                    elif (beta_hdi[0] >= rope_bounds[0]) and (beta_hdi[1] <= rope_bounds[1]):
                        significance = "Practically Equivalent to Zero"
                    else:
                        significance = "Not Credible"

                    # Predictions
                    pred_samples = []
                    for i in range(100):
                        idx = np.random.randint(0, len(alpha_samples))
                        a, b = alpha_samples[idx], beta_samples[idx]
                        logit = a + b * x_range_scaled
                        pred = 1 / (1 + np.exp(-logit))
                        pred_samples.append(pred)
                    pred_samples = np.array(pred_samples)
                    y_pred_mean = np.mean(pred_samples, axis=0)
                    y_pred_hdi = np.zeros((2, len(x_range)))
                    for i in range(len(x_range)):
                        y_pred_hdi[:, i] = az.hdi(pred_samples[:, i], hdi_prob=0.95)

                    y_pred_hdi_lower = y_pred_hdi[0, :]
                    y_pred_hdi_upper = y_pred_hdi[1, :]

                    # Conclusion
                    if in_rope > 0.95:
                        conclusion = "Practically Equivalent"
                    elif significance == "Credible Effect":
                        if (higher_is_better and beta_mean > 0) or (not higher_is_better and beta_mean < 0):
                            conclusion = "Significant Improvement"
                        else:
                            conclusion = "Significant Deterioration"
                    else:
                        conclusion = "Inconclusive"

                    effect_per_1000_words = beta_mean
                    effect_2000_words = beta_mean * 2
                    avg_prob = np.mean(y)
                    avg_logit = np.log(avg_prob / (1 - avg_prob))
                    prob_effect_1000 = expit(avg_logit + effect_per_1000_words) - avg_prob
                    prob_effect_2000 = expit(avg_logit + effect_2000_words) - avg_prob

                    return {
                        "corpus": corpus,
                        "threat_model": threat_model,
                        "llm": llm,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "data_points": len(binary_df),
                        "data_by_length": binary_df['exemplar_length'].value_counts().to_dict(),
                        "mean_value": np.mean(y),
                        "is_binary": True,
                        "slope": {
                            "mean": beta_mean,
                            "std": beta_std,
                            "hdi": beta_hdi.tolist(),
                            "posterior_prob_beneficial": posterior_prob_beneficial,
                            "prob_improvement": posterior_prob_beneficial,
                            "direction": direction,
                            "significance": significance,
                            "effect_per_1000_words": effect_per_1000_words,
                            "effect_2000_words": effect_2000_words,
                            "prob_effect_1000": float(prob_effect_1000),
                            "prob_effect_2000": float(prob_effect_2000),
                            "in_rope": in_rope,
                            "conclusion": conclusion
                        },
                        "predictions": {
                            "x_range": x_range.tolist(),
                            "y_pred_mean": y_pred_mean.tolist(),
                            "y_pred_hdi_lower": y_pred_hdi_lower.tolist(),
                            "y_pred_hdi_upper": y_pred_hdi_upper.tolist()
                        }
                    }
                except Exception as e:
                    logger.error(f"Error in binary model fitting: {str(e)}")
                    logger.error(f"Binary values: {np.bincount(y)}")
                    logger.warning("Falling back to aggregate modeling approach")

    # If no binary data or we fell back, do standard continuous approach
    df = df.dropna(subset=[metric, 'exemplar_length'])
    x = df['exemplar_length'].values
    y = df[metric].values

    # Some special handling for 'entropy'
    if corpus.lower() == 'ebg':
        max_entropy = np.log2(45)
    elif corpus.lower() == 'rj':
        max_entropy = np.log2(21)
    else:
        max_entropy = np.log2(100)

    x_mean = np.mean(x)
    x_range = np.linspace(min(x), max(x), 100)
    x_scaled = (x - x_mean) / 1000.0

    if metric == 'entropy':
        y_scaled = y / max_entropy
        y_mean = np.mean(y)
    elif metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
        y_scaled = np.clip(y, 0.001, 0.999)
        y_mean = np.mean(y)
    else:
        y_scaled = y
        y_mean = np.mean(y)

    with pm.Model() as model:
        try:
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)

            if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence', 'entropy']:
                sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.1)
            else:
                sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.5)

            mu_est = alpha + beta * x_scaled

            if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
                theta = pm.Deterministic("theta", pm.math.invlogit(mu_est))
                concentration = pm.HalfNormal("concentration", 10.0)
                alpha_beta = pm.Deterministic("alpha_beta", theta * concentration)
                beta_beta = pm.Deterministic("beta_beta", (1 - theta) * concentration)
                likelihood = pm.Beta("likelihood", alpha=alpha_beta, beta=beta_beta, observed=y_scaled)

            elif metric == 'entropy':
                likelihood = pm.TruncatedNormal(
                    "likelihood", mu=mu_est, sigma=sigma, lower=0, upper=1, observed=y_scaled
                )
            else:
                likelihood = pm.StudentT(
                    "likelihood", nu=4, mu=mu_est, sigma=sigma, observed=y_scaled
                )

            trace = pm.sample(
                2000, tune=1000, chains=4, random_seed=42,
                target_accept=0.95, return_inferencedata=True, cores=4
            )

            alpha_samples = trace.posterior["alpha"].values.flatten()
            beta_samples = trace.posterior["beta"].values.flatten()

            x_range_scaled = (x_range - x_mean) / 1000.0
            beta_mean = float(np.mean(beta_samples))
            beta_std = float(np.std(beta_samples))
            beta_hdi = az.hdi(beta_samples, hdi_prob=0.95)

            if higher_is_better:
                posterior_prob_beneficial = float(np.mean(beta_samples > 0))
                direction = "positive" if posterior_prob_beneficial > 0.5 else "negative"
            else:
                posterior_prob_beneficial = float(np.mean(beta_samples < 0))
                direction = "negative" if posterior_prob_beneficial > 0.5 else "positive"

            effect_sd = beta_std
            rope_bounds = (-0.1 * effect_sd, 0.1 * effect_sd)
            in_rope = float(np.mean((beta_samples >= rope_bounds[0]) & (beta_samples <= rope_bounds[1])))

            if (beta_hdi[0] > rope_bounds[1]) or (beta_hdi[1] < rope_bounds[0]):
                significance = "Credible Effect"
            elif (beta_hdi[0] >= rope_bounds[0]) and (beta_hdi[1] <= rope_bounds[1]):
                significance = "Practically Equivalent to Zero"
            else:
                significance = "Not Credible"

            pred_samples = []
            for i in range(100):
                idx = np.random.randint(0, len(alpha_samples))
                a, b = alpha_samples[idx], beta_samples[idx]

                if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
                    logit = a + b * x_range_scaled
                    pred = 1 / (1 + np.exp(-logit))
                elif metric == 'entropy':
                    pred = (a + b * x_range_scaled) * max_entropy
                    pred = np.clip(pred, 0, max_entropy)
                else:
                    pred = a + b * x_range_scaled
                pred_samples.append(pred)

            pred_samples = np.array(pred_samples)
            y_pred_mean = np.mean(pred_samples, axis=0)

            y_pred_hdi = np.zeros((2, len(x_range)))
            for i in range(len(x_range)):
                y_pred_hdi[:, i] = az.hdi(pred_samples[:, i], hdi_prob=0.95)

            y_pred_hdi_lower = y_pred_hdi[0, :]
            y_pred_hdi_upper = y_pred_hdi[1, :]

            if in_rope > 0.95:
                conclusion = "Practically Equivalent"
            elif significance == "Credible Effect":
                if (higher_is_better and beta_mean > 0) or (not higher_is_better and beta_mean < 0):
                    conclusion = "Significant Improvement"
                else:
                    conclusion = "Significant Deterioration"
            else:
                conclusion = "Inconclusive"

        except Exception as e:
            logger.error(f"Model fitting error: {str(e)}")
            logger.error(f"Data summary: min={np.min(y)}, max={np.max(y)}, mean={np.mean(y)}")
            return {"error": f"Model fitting failed for {corpus}-{threat_model}-{llm}-{metric}: {str(e)}"}

    effect_per_1000_words = beta_mean
    effect_2000_words = beta_mean * 2

    if metric == 'entropy':
        effect_per_1000_words *= max_entropy
        effect_2000_words *= max_entropy

    return {
        "corpus": corpus,
        "threat_model": threat_model,
        "llm": llm,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "data_points": len(df),
        "data_by_length": df['exemplar_length'].value_counts().to_dict(),
        "mean_value": y_mean,
        "is_binary": False,
        "slope": {
            "mean": beta_mean,
            "std": beta_std,
            "hdi": beta_hdi.tolist(),
            "posterior_prob_beneficial": posterior_prob_beneficial,
            "prob_improvement": posterior_prob_beneficial,
            "direction": direction,
            "significance": significance,
            "effect_per_1000_words": effect_per_1000_words,
            "effect_2000_words": effect_2000_words,
            "in_rope": in_rope,
            "conclusion": conclusion
        },
        "predictions": {
            "x_range": x_range.tolist(),
            "y_pred_mean": y_pred_mean.tolist(),
            "y_pred_hdi_lower": y_pred_hdi_lower.tolist(),
            "y_pred_hdi_upper": y_pred_hdi_upper.tolist()
        }
    }


def plot_exemplar_length_effect(
    data: pd.DataFrame,
    results: Dict,
    output_dir: str,
    format: str = 'png'
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path

    corpus = results['corpus']
    threat_model = results['threat_model']
    llm = results['llm']
    metric = results['metric']
    higher_is_better = results['higher_is_better']

    df = data[(data['corpus'] == corpus) &
              (data['threat_model'] == threat_model) &
              (data['llm'] == llm)].copy()
    df = df.dropna(subset=[metric, 'exemplar_length'])

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.stripplot(
        x='exemplar_length',
        y=metric,
        data=df,
        size=10,
        color='blue',
        alpha=0.5,
        jitter=0.3,
        ax=ax
    )
    ax.plot([], [], color='blue', label=f'Observations (n={len(df)})')

    data_min, data_max = df[metric].min(), df[metric].max()
    data_range = data_max - data_min
    for length in sorted(df['exemplar_length'].unique()):
        subset = df[df['exemplar_length'] == length]
        count = len(subset)
        max_y = subset[metric].max()
        offset = 0.02 * data_range
        ax.text(length, max_y + offset, f"n={count}", ha='center', va='bottom')

    x_range = np.array(results['predictions']['x_range'])
    y_pred_mean = np.array(results['predictions']['y_pred_mean'])
    y_pred_hdi_lower = np.array(results['predictions']['y_pred_hdi_lower'])
    y_pred_hdi_upper = np.array(results['predictions']['y_pred_hdi_upper'])

    ax.plot(x_range, y_pred_mean, color='red', linewidth=2, label='Bayesian model fit')
    ax.fill_between(x_range, y_pred_hdi_lower, y_pred_hdi_upper,
                    color='red', alpha=0.2, label='95% HDI')

    slope_info = (
        f"Slope: {results['slope']['mean']:.6f} "
        f"[95% HDI: {results['slope']['hdi'][0]:.6f}, {results['slope']['hdi'][1]:.6f}]"
    )
    effect_info = f"Effect (500→2500): {results['slope']['effect_2000_words']:.4f}"
    prob_info = f"Posterior prob. beneficial: {results['slope']['posterior_prob_beneficial']:.3f}"
    conclusion = f"Conclusion: {results['slope']['conclusion']}"
    text_box = f"{slope_info}\n{effect_info}\n{prob_info}\n{conclusion}"

    ax.annotate(text_box, xy=(0.05, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    metric_label = metric
    if metric == 'accuracy@1':
        metric_label = 'Top-1 Accuracy'
    elif metric == 'accuracy@5':
        metric_label = 'Top-5 Accuracy'
    elif metric == 'true_class_confidence':
        metric_label = 'True Class Confidence'
    elif metric == 'entropy':
        metric_label = 'Prediction Entropy'

    direction_str = "↑ (Higher is better)" if higher_is_better else "↓ (Lower is better)"
    ax.set_xlabel('Exemplar Length (words)', fontsize=14)
    ax.set_ylabel(f'{metric_label} {direction_str}', fontsize=14)
    ax.set_title(
        f'Effect of Exemplar Length on {metric_label}\n'
        f'({corpus.upper()}, {threat_model.upper()}, {llm.title()})',
        fontsize=16
    )

    if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    unique_lengths = sorted(df['exemplar_length'].unique())
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels([str(int(l)) for l in unique_lengths])
    ax.legend(loc='best', fontsize=12)

    output_path = Path(output_dir) / corpus / threat_model / llm
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{metric}_exemplar_effect.{format}"
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {file_path}")


def run_full_analysis(
        data: pd.DataFrame,
        output_dir: str,
        filter_corpus: Optional[str] = None,
        filter_model: Optional[str] = None,
        filter_threat_model: Optional[str] = None,
        filter_metric: Optional[str] = None
) -> pd.DataFrame:
    results = []
    corpora_to_analyze = [filter_corpus] if filter_corpus else CORPORA
    threat_models_to_analyze = [filter_threat_model] if filter_threat_model else THREAT_MODELS
    llms_to_analyze = [filter_model] if filter_model else LLMS
    metrics_to_analyze = [filter_metric] if filter_metric else METRICS

    for corpus in corpora_to_analyze:
        for threat_model in threat_models_to_analyze:
            for llm in llms_to_analyze:
                for metric in metrics_to_analyze:
                    subset = data[
                        (data['corpus'] == corpus) &
                        (data['threat_model'] == threat_model) &
                        (data['llm'] == llm)
                    ]
                    if len(subset) < 3 or metric not in subset.columns or subset[metric].isna().all():
                        logger.warning(f"Skipping {corpus}-{threat_model}-{llm}-{metric}: insufficient data")
                        continue
                    logger.info(f"Analyzing {corpus}-{threat_model}-{llm}-{metric}")

                    try:
                        analysis = analyze_exemplar_length_effect(data, metric, corpus, threat_model, llm)
                        if 'error' in analysis:
                            logger.warning(f"Analysis failed: {analysis['error']}")
                            continue

                        results.append({
                            'Corpus': corpus.upper(),
                            'Threat Model': threat_model.upper(),
                            'LLM': llm.title(),
                            'Metric': metric,
                            'Higher is Better': analysis['higher_is_better'],
                            'Data Points': analysis['data_points'],
                            'Mean Value': analysis['mean_value'],
                            'Slope': analysis['slope']['mean'],
                            'Slope HDI Lower': analysis['slope']['hdi'][0],
                            'Slope HDI Upper': analysis['slope']['hdi'][1],
                            'P(Improvement)': analysis['slope']['prob_improvement'],
                            'Effect per 1000 words': analysis['slope']['effect_per_1000_words'],
                            'Effect (500→2500)': analysis['slope']['effect_2000_words'],
                            'In ROPE': analysis['slope']['in_rope'],
                            'Conclusion': analysis['slope']['conclusion']
                        })

                        plot_exemplar_length_effect(data, analysis, output_dir)

                    except Exception as e:
                        logger.error(f"Error analyzing {corpus}-{threat_model}-{llm}-{metric}: {e}")

    results_df = pd.DataFrame(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(Path(output_dir) / "exemplar_length_analysis_results.csv", index=False)
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return results_df


def create_summary_report(results_df: pd.DataFrame, output_dir: str) -> None:
    from pathlib import Path
    report_path = Path(output_dir) / "exemplar_length_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("# Analysis of Exemplar Length Effect on LLM-based Imitation Defense\n\n")
        f.write("## Overview\n\n")
        f.write(
            f"This analysis examines the effect of *actual* exemplar length on the effectiveness "
            f"of LLM-based imitation as a defense against authorship attribution attacks. We extract "
            f"the exemplar text from each seed_{seed}.json, count its words, and model how that length "
            f"impacts various metrics.\n\n"
        )
        f.write(
            f"- {results_df['Corpus'].nunique()} corpora: {', '.join(sorted(results_df['Corpus'].unique()))}\n")
        f.write(
            f"- {results_df['Threat Model'].nunique()} threat models: {', '.join(sorted(results_df['Threat Model'].unique()))}\n")
        f.write(
            f"- {results_df['LLM'].nunique()} LLMs: {', '.join(sorted(results_df['LLM'].unique()))}\n")
        f.write(
            f"- {results_df['Metric'].nunique()} metrics: {', '.join(sorted(results_df['Metric'].unique()))}\n\n")

        # Additional details
        credible = results_df[results_df['Conclusion'].str.contains('Significant')]
        improvements = credible[credible['Conclusion'].str.contains('Improvement')]
        deteriorations = credible[credible['Conclusion'].str.contains('Deterioration')]
        inconclusive = results_df[results_df['Conclusion'] == 'Inconclusive']
        equivalent = results_df[results_df['Conclusion'] == 'Practically Equivalent']

        total_tests = len(results_df)
        f.write("## Overall Findings\n\n")
        f.write(f"Out of {total_tests} total tests:\n\n")
        f.write(f"- **Credible improvements with longer exemplars**: {len(improvements)} ({len(improvements) / total_tests:.1%})\n")
        f.write(f"- **Credible deteriorations with longer exemplars**: {len(deteriorations)} ({len(deteriorations) / total_tests:.1%})\n")
        f.write(f"- **Inconclusive results**: {len(inconclusive)} ({len(inconclusive) / total_tests:.1%})\n")
        f.write(f"- **Practically equivalent results**: {len(equivalent)} ({len(equivalent) / total_tests:.1%})\n\n")

    logger.info(f"Summary report saved to {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze the effect of exemplar length on LLM imitation defense performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--evaluation_dir', type=str, default='defense_evaluation')
    parser.add_argument('--llm_outputs_dir', type=str, default='llm_outputs')
    parser.add_argument('--output_dir', type=str, default='results/exemplar_length_analysis_rq3.2',
                        help='Directory to save analysis results')
    parser.add_argument('--corpus', type=str, choices=CORPORA, help='Specific corpus to analyze (default: all)')
    parser.add_argument('--model', type=str, help='Specific LLM to analyze (default: all)')
    parser.add_argument('--threat_model', type=str, choices=THREAT_MODELS, help='Specific threat model to analyze (default: all)')
    parser.add_argument('--metric', type=str, choices=METRICS, help='Specific metric to analyze (default: all)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable extra debug printing to diagnose length matching issues')

    return parser.parse_args()


def main():
    args = parse_arguments()
    logger.info("Starting exemplar length analysis")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing data for analysis")
    data = prepare_data(args.base_dir, debug=args.debug)

    # Save raw data
    data.to_csv(output_dir / "raw_data.csv", index=False)

    logger.info("Running analysis")
    results_df = run_full_analysis(
        data=data,
        output_dir=str(output_dir),
        filter_corpus=args.corpus,
        filter_model=args.model,
        filter_threat_model=args.threat_model,
        filter_metric=args.metric
    )

    logger.info("Creating summary report")
    create_summary_report(results_df, str(output_dir))
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()

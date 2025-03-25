"""
Analyze the effect of exemplar length on LLM imitation defense performance.

This script analyzes how varying exemplar lengths (500/1000/2500 words) affect 
defense effectiveness against authorship attribution models.

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
EXEMPLAR_LENGTHS = [500, 1000, 2500]


def get_actual_exemplar_lengths(base_dir: str) -> Dict[str, List[int]]:
    """
    Extract the actual word counts from each exemplar prompt file

    Args:
        base_dir: Base directory containing evaluation results

    Returns:
        Dictionary mapping RQ directories to lists of actual word counts
    """
    actual_lengths = {}

    for length in [500, 1000, 2500]:
        rq = f"rq3.2_imitation_w_{length}words"
        prompt_dir = Path('prompts') / rq

        if not prompt_dir.exists():
            logger.warning(f"Prompt directory not found: {prompt_dir}")
            continue

        actual_counts = []

        # Read all prompt files and extract actual word counts
        for prompt_file in prompt_dir.glob("prompt*.json"):
            try:
                with open(prompt_file) as f:
                    prompt_data = json.load(f)

                # Get the actual word count from metadata
                if "metadata" in prompt_data and "word_count" in prompt_data["metadata"]:
                    actual_counts.append(prompt_data["metadata"]["word_count"])
                else:
                    # Alternative: count words in the exemplar text
                    # This would require parsing the exemplar from the user prompt
                    pass

            except json.JSONDecodeError:
                logger.error(f"Error parsing {prompt_file}")
                continue

        if actual_counts:
            actual_lengths[rq] = actual_counts
            avg_count = sum(actual_counts) / len(actual_counts)
            logger.info(f"RQ {rq}: found {len(actual_counts)} prompts with average word count {avg_count:.1f}")

    return actual_lengths

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


def prepare_data(base_dir: str) -> pd.DataFrame:
    """
    Extract and organize data from evaluation results, including binary success/failure outcomes.
    """
    data = []

    # Get actual exemplar lengths
    actual_exemplar_lengths = get_actual_exemplar_lengths(base_dir)

    # Process each exemplar length category
    for target_length in [500, 1000, 2500]:
        rq = f"rq3.2_imitation_w_{target_length}words"
        rq_main = "rq3"

        # Get the actual length statistics for this RQ
        if rq in actual_exemplar_lengths:
            actual_length = sum(actual_exemplar_lengths[rq]) / len(
                actual_exemplar_lengths[rq])
        else:
            actual_length = target_length

        logger.info(f"Using actual exemplar length {actual_length:.1f} for {rq}")

        for corpus in CORPORA:
            corpus_path = Path(base_dir) / corpus / rq_main / rq
            if not corpus_path.exists():
                logger.warning(f"Path not found: {corpus_path}")
                continue

            logger.info(f"Processing {corpus} with exemplar length {actual_length:.1f}")

            for model_dir in corpus_path.glob("*"):
                if not model_dir.is_dir():
                    continue

                llm = normalize_llm_name(model_dir.name)

                # Process evaluation file
                eval_file = model_dir / "evaluation.json"
                if not eval_file.exists():
                    logger.warning(f"Evaluation file not found: {eval_file}")
                    continue

                try:
                    with open(eval_file) as f:
                        results = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing {eval_file}")
                    continue

                # For each seed, process the evaluation results
                for seed, seed_results in results.items():
                    for threat_model, tm_results in seed_results.items():
                        if threat_model not in THREAT_MODELS:
                            continue

                        # Extract aggregated attribution metrics
                        post = tm_results['attribution']['post']

                        # Create initial entry with actual length
                        entry = {
                            'corpus': corpus,
                            'llm': llm,
                            'threat_model': threat_model,
                            'seed': seed,
                            'exemplar_length': actual_length,
                            'accuracy@1': post.get('accuracy@1'),
                            'accuracy@5': post.get('accuracy@5'),
                            'true_class_confidence': post.get('true_class_confidence'),
                            'entropy': post.get('entropy')
                        }

                        # Add quality metrics if available
                        if 'quality' in tm_results:
                            quality = tm_results['quality']

                            # Add PINC score (average of PINC1-4)
                            if 'pinc' in quality:
                                pinc_scores = []
                                for k in range(1, 5):
                                    key = f"pinc_{k}_avg"
                                    if key in quality['pinc']:
                                        pinc_scores.append(quality['pinc'][key])
                                if pinc_scores:
                                    entry['pinc'] = np.mean(pinc_scores)

                            # Add BERTScore
                            if 'bertscore' in quality and 'bertscore_f1_avg' in quality[
                                'bertscore']:
                                entry['bertscore'] = quality['bertscore'][
                                    'bertscore_f1_avg']

                        data.append(entry)

                        # Now try to extract binary outcomes from example_metrics
                        # Look for a corresponding seed file with example_metrics
                        seed_file = model_dir / f"seed_{seed}.json"
                        if seed_file.exists():
                            try:
                                with open(seed_file) as f:
                                    detailed_data = json.load(f)

                                if 'example_metrics' in detailed_data:
                                    # Extract binary outcomes
                                    binary_acc1 = [
                                        1 if ex['transformed_rank'] == 0 else 0
                                        for ex in detailed_data['example_metrics']]
                                    binary_acc5 = [
                                        1 if ex['transformed_rank'] < 5 else 0
                                        for ex in detailed_data['example_metrics']]

                                    # Add to entry
                                    entry['binary_acc1'] = binary_acc1
                                    entry['binary_acc5'] = binary_acc5

                            except Exception as e:
                                logger.error(
                                    f"Error processing seed file {seed_file}: {e}")

    df = pd.DataFrame(data)
    logger.info(
        f"Collected {len(df)} data points across {df['corpus'].nunique()} corpora, "
        f"{df['llm'].nunique()} LLMs, {df['threat_model'].nunique()} threat models, "
        f"and {df['exemplar_length'].nunique()} exemplar lengths")

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

    Uses robust Bayesian methods following Kruschke's approach for modeling
    binary and continuous outcomes.

    Args:
        data: DataFrame with evaluation data
        metric: Metric to analyze
        corpus: Corpus to filter data for
        threat_model: Threat model to filter data for
        llm: LLM to filter data for
        higher_is_better: Whether higher values indicate better defense performance

    Returns:
        Dictionary with model results
    """
    # Filter data
    df = data[(data['corpus'] == corpus) &
              (data['threat_model'] == threat_model) &
              (data['llm'] == llm)].copy()

    # Print data point count for debugging
    logger.info(
        f"Found {len(df)} data points for {corpus}-{threat_model}-{llm}-{metric}")
    logger.info(
        f"Data points by exemplar length: {df['exemplar_length'].value_counts().to_dict()}")

    # Ensure we have enough data
    if len(df) < 3 or metric not in df.columns or df[metric].isna().all():
        return {
            "error": f"Insufficient data for {corpus}-{threat_model}-{llm}-{metric}"}

    # If higher_is_better not specified, determine from metric
    if higher_is_better is None:
        higher_is_better = metric in ['entropy', 'bertscore', 'pinc', 'meteor']

    # For accuracy metrics, use binary outcomes if available
    binary_key = f'binary_{metric.replace("@", "")}'
    if metric in ['accuracy@1', 'accuracy@5'] and binary_key in df.columns:
        # Unpack binary outcomes into a flatter structure
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

        # Create a new dataframe with binary outcomes
        binary_df = pd.DataFrame(binary_rows)
        if len(binary_df) == 0:
            logger.warning(
                f"No binary outcomes found for {metric}, falling back to aggregate values")
        else:
            logger.info(f"Using {len(binary_df)} binary outcomes for {metric}")

            # Extract x and y, dropping NaNs
            binary_df = binary_df.dropna(subset=['binary_outcome', 'exemplar_length'])
            x = binary_df['exemplar_length'].values
            y = binary_df['binary_outcome'].values

            # Scale inputs for numerical stability
            x_mean = np.mean(x)
            x_range = np.linspace(min(x), max(x), 100)
            x_scaled = (x - x_mean) / 1000.0  # Center and scale

            # Build a robust Bayesian logistic regression model for binary outcomes
            with pm.Model() as model:
                try:
                    # More robust priors following Kruschke's recommendations
                    # Normal prior for intercept with appropriate scale
                    alpha = pm.Normal("alpha", mu=0, sigma=2)

                    # Cauchy prior for slope - more robust to outliers and heavy tails
                    beta = pm.Cauchy("beta", alpha=0, beta=0.5)

                    # Linear predictor (log-odds)
                    eta = alpha + beta * x_scaled

                    # Convert log-odds to probability
                    theta = pm.Deterministic("theta", pm.math.invlogit(eta))

                    # Bernoulli likelihood for binary outcomes
                    likelihood = pm.Bernoulli("likelihood", p=theta, observed=y)

                    # Inference with robust settings
                    trace = pm.sample(
                        2000,
                        tune=1000,
                        chains=4,
                        random_seed=42,
                        target_accept=0.95,  # Higher target_accept for stability
                        return_inferencedata=True,
                        cores=4
                    )

                    # Predictions
                    alpha_samples = trace.posterior["alpha"].values.flatten()
                    beta_samples = trace.posterior["beta"].values.flatten()

                    # Calculate scaled x_range for predictions
                    x_range_scaled = (x_range - x_mean) / 1000.0

                    # Compute mean and HDI for beta
                    beta_mean = float(np.mean(beta_samples))
                    beta_std = float(np.std(beta_samples))
                    beta_hdi = az.hdi(beta_samples, hdi_prob=0.95)

                    # Compute posterior probability of improvement
                    if higher_is_better:
                        posterior_prob_beneficial = float(np.mean(beta_samples > 0))
                        direction = "positive" if posterior_prob_beneficial > 0.5 else "negative"
                    else:
                        posterior_prob_beneficial = float(np.mean(beta_samples < 0))
                        direction = "negative" if posterior_prob_beneficial > 0.5 else "positive"

                    # Adjust ROPE based on the scale of the data (Kruschke's approach)
                    # Define ROPE as 1% of the potential range for binary outcomes (0.01 on probability scale)
                    effect_sd = beta_std
                    rope_bounds = (-0.05 * effect_sd, 0.05 * effect_sd)
                    in_rope = float(np.mean(
                        (beta_samples >= rope_bounds[0]) & (
                                    beta_samples <= rope_bounds[1])))

                    # Determine significance using Kruschke's ROPE decision rules
                    if (beta_hdi[0] > rope_bounds[1]) or (beta_hdi[1] < rope_bounds[0]):
                        significance = "Credible Effect"
                    elif (beta_hdi[0] >= rope_bounds[0]) and (
                            beta_hdi[1] <= rope_bounds[1]):
                        significance = "Practically Equivalent to Zero"
                    else:
                        significance = "Not Credible"

                    # Generate predictions for the plot
                    pred_samples = []
                    for i in range(100):  # Use 100 posterior samples
                        idx = np.random.randint(0, len(alpha_samples))
                        a, b = alpha_samples[idx], beta_samples[idx]

                        # For binary outcomes, predict probability
                        logit = a + b * x_range_scaled
                        pred = 1 / (1 + np.exp(-logit))
                        pred_samples.append(pred)

                    pred_samples = np.array(pred_samples)
                    y_pred_mean = np.mean(pred_samples, axis=0)

                    # Compute HDI for predictions
                    y_pred_hdi = np.zeros((2, len(x_range)))
                    for i in range(len(x_range)):
                        y_pred_hdi[:, i] = az.hdi(pred_samples[:, i], hdi_prob=0.95)

                    y_pred_hdi_lower = y_pred_hdi[0, :]
                    y_pred_hdi_upper = y_pred_hdi[1, :]

                    # Determine conclusion
                    if in_rope > 0.95:
                        conclusion = "Practically Equivalent"
                    elif significance == "Credible Effect":
                        if (higher_is_better and beta_mean > 0) or (
                                not higher_is_better and beta_mean < 0):
                            conclusion = "Significant Improvement"
                        else:
                            conclusion = "Significant Deterioration"
                    else:
                        conclusion = "Inconclusive"

                    # Calculate effect sizes (scaled back to original units)
                    # Effect per 1000 words
                    effect_per_1000_words = beta_mean

                    # Effect from 500 to 2500 words (2000 word difference)
                    effect_2000_words = beta_mean * 2

                    # Convert log-odds effects to probability for interpretation
                    # Starting from average probability
                    avg_prob = np.mean(y)
                    avg_logit = np.log(avg_prob / (1 - avg_prob))

                    # Effect in probability space
                    prob_effect_1000 = expit(
                        avg_logit + effect_per_1000_words) - avg_prob
                    prob_effect_2000 = expit(avg_logit + effect_2000_words) - avg_prob

                    # Return results dictionary with additional information
                    return {
                        "corpus": corpus,
                        "threat_model": threat_model,
                        "llm": llm,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "data_points": len(binary_df),
                        "data_by_length": binary_df[
                            'exemplar_length'].value_counts().to_dict(),
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
                    # If binary modeling fails, fall back to aggregate approach
                    logger.warning("Falling back to aggregate modeling approach")

    # For non-binary metrics or if binary data not available/failed
    # Extract x and y, dropping NaNs
    df = df.dropna(subset=[metric, 'exemplar_length'])
    x = df['exemplar_length'].values
    y = df[metric].values

    # Get corpus-specific max entropy for proper scaling
    if corpus.lower() == 'ebg':
        max_entropy = np.log2(45)  # 45 authors in EBG
    elif corpus.lower() == 'rj':
        max_entropy = np.log2(21)  # 21 authors in RJ
    else:
        max_entropy = np.log2(100)  # Default fallback

    # Scale inputs for numerical stability
    x_mean = np.mean(x)
    x_range = np.linspace(min(x), max(x), 100)
    x_scaled = (x - x_mean) / 1000.0  # Center and scale

    # Special handling for bounded variables
    if metric == 'entropy':
        # Scale entropy to 0-1 range
        y_scaled = y / max_entropy
        y_mean = np.mean(y)
    elif metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
        # Avoid exact 0 and 1 values that cause problems with Beta
        y_scaled = np.clip(y, 0.001, 0.999)
        y_mean = np.mean(y)
    else:
        y_scaled = y
        y_mean = np.mean(y)

    # Build a Bayesian linear regression model with robust priors
    with pm.Model() as model:
        try:
            # Priors - use more robust Student's t for regression coefficients
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            beta = pm.StudentT("beta", nu=3, mu=0, sigma=0.5)

            # Different sigma prior based on metric type
            if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence',
                          'entropy']:
                # Bounded metrics need smaller sigma
                sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.1)
            else:
                # Less constrained for unbounded metrics
                sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.5)

            # Expected value
            mu = alpha + beta * x_scaled

            # Likelihood - tailored to variable type
            if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
                # For percentage metrics, use a Beta likelihood with more stable parameterization
                # Transform linear predictor to (0,1) interval
                theta = pm.Deterministic("theta", pm.math.invlogit(mu))
                # Use concentration parameterization
                concentration = pm.HalfNormal("concentration", 10.0)
                alpha_beta = pm.Deterministic("alpha_beta", theta * concentration)
                beta_beta = pm.Deterministic("beta_beta", (1 - theta) * concentration)
                # Beta likelihood with more stable parameterization
                likelihood = pm.Beta("likelihood", alpha=alpha_beta, beta=beta_beta,
                                     observed=y_scaled)

            elif metric == 'entropy':
                # For entropy (scaled to 0-1), use a bounded normal with transformation
                likelihood = pm.TruncatedNormal(
                    "likelihood",
                    mu=mu,
                    sigma=sigma,
                    lower=0,
                    upper=1,
                    observed=y_scaled
                )
            else:
                # For other metrics like bertscore, pinc, use robust t-distributed likelihood
                likelihood = pm.StudentT(
                    "likelihood",
                    nu=4,  # Degrees of freedom - more robust to outliers
                    mu=mu,
                    sigma=sigma,
                    observed=y_scaled
                )

            # Inference with robust settings
            trace = pm.sample(
                2000,
                tune=1000,
                chains=4,
                random_seed=42,
                target_accept=0.95,  # Higher target_accept for stability
                return_inferencedata=True,
                cores=4
            )

            # Predictions
            alpha_samples = trace.posterior["alpha"].values.flatten()
            beta_samples = trace.posterior["beta"].values.flatten()

            # Calculate scaled x_range for predictions
            x_range_scaled = (x_range - x_mean) / 1000.0

            # Compute mean and HDI for beta
            beta_mean = float(np.mean(beta_samples))
            beta_std = float(np.std(beta_samples))
            beta_hdi = az.hdi(beta_samples, hdi_prob=0.95)

            # Compute posterior probability of improvement
            if higher_is_better:
                posterior_prob_beneficial = float(np.mean(beta_samples > 0))
                direction = "positive" if posterior_prob_beneficial > 0.5 else "negative"
            else:
                posterior_prob_beneficial = float(np.mean(beta_samples < 0))
                direction = "negative" if posterior_prob_beneficial > 0.5 else "positive"

            # Adjust ROPE based on effect size
            effect_sd = beta_std
            rope_bounds = (-0.1 * effect_sd, 0.1 * effect_sd)
            in_rope = float(np.mean(
                (beta_samples >= rope_bounds[0]) & (beta_samples <= rope_bounds[1])))

            # Determine significance
            if (beta_hdi[0] > rope_bounds[1]) or (beta_hdi[1] < rope_bounds[0]):
                significance = "Credible Effect"
            elif (beta_hdi[0] >= rope_bounds[0]) and (beta_hdi[1] <= rope_bounds[1]):
                significance = "Practically Equivalent to Zero"
            else:
                significance = "Not Credible"

            # Generate predictions for the plot
            pred_samples = []
            for i in range(100):  # Use 100 posterior samples
                idx = np.random.randint(0, len(alpha_samples))
                a, b = alpha_samples[idx], beta_samples[idx]

                if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
                    # Transform back from logit scale
                    logit = a + b * x_range_scaled
                    pred = 1 / (1 + np.exp(-logit))
                elif metric == 'entropy':
                    # Transform back to original entropy scale
                    pred = (a + b * x_range_scaled) * max_entropy
                    pred = np.clip(pred, 0, max_entropy)
                else:
                    # Direct linear relationship
                    pred = a + b * x_range_scaled
                pred_samples.append(pred)

            pred_samples = np.array(pred_samples)
            y_pred_mean = np.mean(pred_samples, axis=0)

            # Compute HDI for predictions properly
            y_pred_hdi = np.zeros((2, len(x_range)))
            for i in range(len(x_range)):
                y_pred_hdi[:, i] = az.hdi(pred_samples[:, i], hdi_prob=0.95)

            y_pred_hdi_lower = y_pred_hdi[0, :]
            y_pred_hdi_upper = y_pred_hdi[1, :]

            # Determine conclusion
            if in_rope > 0.95:
                conclusion = "Practically Equivalent"
            elif significance == "Credible Effect":
                if (higher_is_better and beta_mean > 0) or (
                        not higher_is_better and beta_mean < 0):
                    conclusion = "Significant Improvement"
                else:
                    conclusion = "Significant Deterioration"
            else:
                conclusion = "Inconclusive"

        except Exception as e:
            logger.error(f"Model fitting error: {str(e)}")
            logger.error(
                f"Data summary: min={np.min(y)}, max={np.max(y)}, mean={np.mean(y)}")
            return {
                "error": f"Model fitting failed for {corpus}-{threat_model}-{llm}-{metric}: {str(e)}"
            }

    # Calculate effect sizes
    effect_per_1000_words = beta_mean
    effect_2000_words = beta_mean * 2

    # Transform effect to original scale for entropy
    if metric == 'entropy':
        effect_per_1000_words *= max_entropy
        effect_2000_words *= max_entropy

    # Return results dictionary
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
            "prob_improvement": posterior_prob_beneficial,  # Backward compatibility
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
    """
    Create and save plot of exemplar length effect on a metric.

    Args:
        data: DataFrame with raw data
        results: Results dictionary from analyze_exemplar_length_effect
        output_dir: Directory to save plot
        format: Output file format (png, pdf, svg)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path

    # Extract parameters from results
    corpus = results['corpus']
    threat_model = results['threat_model']
    llm = results['llm']
    metric = results['metric']
    higher_is_better = results['higher_is_better']

    # Filter data
    df = data[
        (data['corpus'] == corpus) &
        (data['threat_model'] == threat_model) &
        (data['llm'] == llm)
    ].copy()
    df = df.dropna(subset=[metric, 'exemplar_length'])

    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the raw data points using the actual exemplar_length values
    sns.stripplot(
        x='exemplar_length',
        y=metric,
        data=df,
        size=10,
        color='blue',
        alpha=0.5,
        jitter=0.3,
        label=f'Observations (n={len(df)})',
        ax=ax
    )

    # Add count labels for each exemplar length
    for length in sorted(df['exemplar_length'].unique()):
        subset = df[df['exemplar_length'] == length]
        if not subset.empty:
            count = len(subset)
            max_y = subset[metric].max()
            # Slight vertical offset so text doesn't overlap the highest point
            ax.text(length, max_y + 0.02, f"n={count}", ha='center', va='bottom')

    # Plot the model predictions with HDI
    x_range = np.array(results['predictions']['x_range'])
    y_pred_mean = np.array(results['predictions']['y_pred_mean'])
    y_pred_hdi_lower = np.array(results['predictions']['y_pred_hdi_lower'])
    y_pred_hdi_upper = np.array(results['predictions']['y_pred_hdi_upper'])

    ax.plot(x_range, y_pred_mean, color='red', linewidth=2, label='Bayesian model fit')
    ax.fill_between(
        x_range,
        y_pred_hdi_lower,
        y_pred_hdi_upper,
        color='red',
        alpha=0.2,
        label='95% HDI'
    )

    # Annotate slope information
    slope_info = (
        f"Slope: {results['slope']['mean']:.6f} "
        f"[95% HDI: {results['slope']['hdi'][0]:.6f}, {results['slope']['hdi'][1]:.6f}]"
    )
    effect_info = f"Effect (500→2500): {results['slope']['effect_2000_words']:.4f}"
    prob_info = f"Posterior prob. beneficial: {results['slope']['posterior_prob_beneficial']:.3f}"
    conclusion = f"Conclusion: {results['slope']['conclusion']}"

    text_box = f"{slope_info}\n{effect_info}\n{prob_info}\n{conclusion}"

    ax.annotate(
        text_box,
        xy=(0.05, 0.05),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
    )

    # Format metric name for display
    metric_label = metric
    if metric == 'accuracy@1':
        metric_label = 'Top-1 Accuracy'
    elif metric == 'accuracy@5':
        metric_label = 'Top-5 Accuracy'
    elif metric == 'true_class_confidence':
        metric_label = 'True Class Confidence'
    elif metric == 'entropy':
        metric_label = 'Prediction Entropy'

    # Add direction indicator for interpretation
    direction_str = "↑ (Higher is better)" if higher_is_better else "↓ (Lower is better)"

    # Set labels and title
    ax.set_xlabel('Exemplar Length (words)', fontsize=14)
    ax.set_ylabel(f'{metric_label} {direction_str}', fontsize=14)
    ax.set_title(
        f'Effect of Exemplar Length on {metric_label}\n'
        f'({corpus.upper()}, {threat_model.upper()}, {llm.title()})',
        fontsize=16
    )

    # Format y-axis as percentage for accuracy metrics
    if metric in ['accuracy@1', 'accuracy@5', 'true_class_confidence']:
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    # Optionally, set x-ticks explicitly if you only have 3 lengths
    # This ensures the x-axis has the correct tick labels at 500, 1000, 2500
    unique_lengths = sorted(df['exemplar_length'].unique())
    ax.set_xticks(unique_lengths)
    ax.set_xticklabels([str(int(l)) for l in unique_lengths])

    # Add legend
    ax.legend(loc='best', fontsize=12)

    # Create output directory structure
    output_path = Path(output_dir) / corpus / threat_model / llm
    output_path.mkdir(parents=True, exist_ok=True)

    # Save plot
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
    """
    Run analysis for all combinations and generate a summary table.

    Args:
        data: DataFrame with evaluation data
        output_dir: Directory to save results
        filter_*: Optional filters to limit analysis scope

    Returns:
        DataFrame with analysis results
    """
    results = []

    # Filter corpora if specified
    corpora_to_analyze = [filter_corpus] if filter_corpus else CORPORA

    # Filter threat models if specified
    threat_models_to_analyze = [filter_threat_model] if filter_threat_model else THREAT_MODELS

    # Filter LLMs if specified
    llms_to_analyze = [filter_model] if filter_model else LLMS

    # Filter metrics if specified
    metrics_to_analyze = [filter_metric] if filter_metric else METRICS

    for corpus in corpora_to_analyze:
        for threat_model in threat_models_to_analyze:
            for llm in llms_to_analyze:
                for metric in metrics_to_analyze:
                    # Skip if we don't have enough data points
                    subset = data[(data['corpus'] == corpus) &
                                  (data['threat_model'] == threat_model) &
                                  (data['llm'] == llm)]

                    if len(subset) < 3 or metric not in subset.columns or subset[metric].isna().all():
                        logger.warning(f"Skipping {corpus}-{threat_model}-{llm}-{metric}: insufficient data")
                        continue

                    logger.info(f"Analyzing {corpus}-{threat_model}-{llm}-{metric}")

                    try:
                        analysis = analyze_exemplar_length_effect(
                            data, metric, corpus, threat_model, llm)

                        # Skip if the analysis failed
                        if 'error' in analysis:
                            logger.warning(f"Analysis failed: {analysis['error']}")
                            continue

                        # Record results
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

                        # Generate and save plot
                        plot_exemplar_length_effect(data, analysis, output_dir)

                    except Exception as e:
                        logger.error(f"Error analyzing {corpus}-{threat_model}-{llm}-{metric}: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save full results
    results_df.to_csv(Path(output_dir) / "exemplar_length_analysis_results.csv", index=False)

    # Create and save summary tables

    # 1. Significant improvements
    improvements = results_df[results_df['Conclusion'].str.contains('Improvement')]
    if not improvements.empty:
        improvements.to_csv(Path(output_dir) / "significant_improvements.csv", index=False)
        with open(Path(output_dir) / "significant_improvements.txt", 'w') as f:
            f.write("# Significant Improvements with Longer Exemplars\n\n")
            for _, row in improvements.iterrows():
                f.write(f"- {row['LLM']} | {row['Threat Model']} | {row['Metric']}: "
                        f"Effect size = {row['Effect (500→2500)']:.4f}, "
                        f"p = {row['P(Improvement)']:.4f}\n")

    # 2. Significant deteriorations
    deteriorations = results_df[results_df['Conclusion'].str.contains('Deterioration')]
    if not deteriorations.empty:
        deteriorations.to_csv(Path(output_dir) / "significant_deteriorations.csv", index=False)
        with open(Path(output_dir) / "significant_deteriorations.txt", 'w') as f:
            f.write("# Significant Deteriorations with Longer Exemplars\n\n")
            for _, row in deteriorations.iterrows():
                f.write(f"- {row['LLM']} | {row['Threat Model']} | {row['Metric']}: "
                        f"Effect size = {row['Effect (500→2500)']:.4f}, "
                        f"p = {1-row['P(Improvement)']:.4f}\n")

    # 3. Effects by LLM
    with open(Path(output_dir) / "effects_by_llm.txt", 'w') as f:
        f.write("# Effects of Exemplar Length by LLM\n\n")
        for llm in llms_to_analyze:
            llm_results = results_df[results_df['LLM'] == llm.title()]
            if not llm_results.empty:
                f.write(f"## {llm.title()}\n\n")
                # Count significant effects
                significant = llm_results[llm_results['Conclusion'].str.contains('Significant')]
                improvements = significant[significant['Conclusion'].str.contains('Improvement')]
                deteriorations = significant[significant['Conclusion'].str.contains('Deterioration')]
                inconclusive = llm_results[llm_results['Conclusion'] == 'Inconclusive']
                equivalent = llm_results[llm_results['Conclusion'] == 'Practically Equivalent']

                f.write(f"- Total effects analyzed: {len(llm_results)}\n")
                f.write(f"- Significant improvements: {len(improvements)}\n")
                f.write(f"- Significant deteriorations: {len(deteriorations)}\n")
                f.write(f"- Inconclusive: {len(inconclusive)}\n")
                f.write(f"- Practically equivalent: {len(equivalent)}\n\n")

    # 4. Effects by metric
    with open(Path(output_dir) / "effects_by_metric.txt", 'w') as f:
        f.write("# Effects of Exemplar Length by Metric\n\n")
        for metric in metrics_to_analyze:
            metric_results = results_df[results_df['Metric'] == metric]
            if not metric_results.empty:
                # Format metric name
                metric_label = metric
                if metric == 'accuracy@1':
                    metric_label = 'Top-1 Accuracy'
                elif metric == 'accuracy@5':
                    metric_label = 'Top-5 Accuracy'
                elif metric == 'true_class_confidence':
                    metric_label = 'True Class Confidence'
                elif metric == 'entropy':
                    metric_label = 'Prediction Entropy'

                f.write(f"## {metric_label}\n\n")

                # Calculate average effect
                avg_effect = metric_results['Effect (500→2500)'].mean()
                higher_is_better = metric_results['Higher is Better'].iloc[0]
                direction = "increase" if avg_effect > 0 else "decrease"

                f.write(f"- Average effect (500→2500): {avg_effect:.4f} ({direction})\n")
                f.write(f"- Higher is better: {higher_is_better}\n")

                # Is this good or bad overall?
                if (avg_effect > 0 and higher_is_better) or (avg_effect < 0 and not higher_is_better):
                    f.write("- Overall impact: Positive (longer exemplars improve this metric)\n\n")
                else:
                    f.write("- Overall impact: Negative (longer exemplars worsen this metric)\n\n")

                # List results by LLM
                f.write("### Results by LLM:\n\n")
                for llm in llms_to_analyze:
                    llm_metric_results = metric_results[metric_results['LLM'] == llm.title()]
                    if not llm_metric_results.empty:
                        avg_llm_effect = llm_metric_results['Effect (500→2500)'].mean()
                        f.write(f"- {llm.title()}: {avg_llm_effect:.4f}\n")

                f.write("\n")

    logger.info(f"Analysis complete. Results saved to {output_dir}")

    return results_df


def create_summary_report(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary report of the analysis results.

    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save report
    """
    report_path = Path(output_dir) / "exemplar_length_analysis_report.md"

    with open(report_path, 'w') as f:
        f.write(
            "# Analysis of Exemplar Length Effect on LLM-based Imitation Defense\n\n")

        f.write("## Overview\n\n")
        f.write(
            f"This analysis examines the effect of exemplar length (500, 1000, and 2500 words) on the effectiveness of LLM-based imitation as a defense against authorship attribution attacks. The analysis covers:\n\n")
        f.write(
            f"- {results_df['Corpus'].nunique()} corpora: {', '.join(sorted(results_df['Corpus'].unique()))}\n")
        f.write(
            f"- {results_df['Threat Model'].nunique()} threat models: {', '.join(sorted(results_df['Threat Model'].unique()))}\n")
        f.write(
            f"- {results_df['LLM'].nunique()} LLMs: {', '.join(sorted(results_df['LLM'].unique()))}\n")
        f.write(
            f"- {results_df['Metric'].nunique()} metrics: {', '.join(sorted(results_df['Metric'].unique()))}\n\n")

        # Note on Bayesian interpretation
        f.write("## Note on Bayesian Interpretation\n\n")
        f.write(
            "This analysis uses Bayesian methods to estimate the effect of exemplar length on defense effectiveness. Key concepts:\n\n")
        f.write(
            "- **Posterior probability**: In Bayesian analysis, we calculate the probability of an effect based on observed data and prior beliefs.\n")
        f.write(
            "- **95% HDI (Highest Density Interval)**: The interval containing 95% of the posterior probability mass, showing where the true effect most likely lies.\n")
        f.write(
            "- **Credible Effect**: When the 95% HDI excludes zero, providing strong evidence that the effect is real.\n")
        f.write(
            "- **Posterior prob. beneficial**: The probability that longer exemplars improve a metric (either increasing metrics where higher is better, or decreasing metrics where lower is better).\n\n")

        # Overall findings
        f.write("## Overall Findings\n\n")

        # Count credible results
        credible = results_df[results_df['Conclusion'].str.contains('Significant')]
        improvements = credible[credible['Conclusion'].str.contains('Improvement')]
        deteriorations = credible[credible['Conclusion'].str.contains('Deterioration')]
        inconclusive = results_df[results_df['Conclusion'] == 'Inconclusive']
        equivalent = results_df[results_df['Conclusion'] == 'Practically Equivalent']

        total_tests = len(results_df)
        f.write(f"Out of {total_tests} total tests:\n\n")
        f.write(
            f"- **Credible improvements with longer exemplars**: {len(improvements)} ({len(improvements) / total_tests:.1%})\n")
        f.write(
            f"- **Credible deteriorations with longer exemplars**: {len(deteriorations)} ({len(deteriorations) / total_tests:.1%})\n")
        f.write(
            f"- **Inconclusive results**: {len(inconclusive)} ({len(inconclusive) / total_tests:.1%})\n")
        f.write(
            f"- **Practically equivalent results**: {len(equivalent)} ({len(equivalent) / total_tests:.1%})\n\n")

        # Key findings by metric
        f.write("## Key Findings by Metric\n\n")
        for metric in results_df['Metric'].unique():
            metric_results = results_df[results_df['Metric'] == metric]

            # Format metric name
            metric_label = metric
            if metric == 'accuracy@1':
                metric_label = 'Top-1 Accuracy'
            elif metric == 'accuracy@5':
                metric_label = 'Top-5 Accuracy'
            elif metric == 'true_class_confidence':
                metric_label = 'True Class Confidence'
            elif metric == 'entropy':
                metric_label = 'Prediction Entropy'

            f.write(f"### {metric_label}\n\n")

            # Calculate average effect and determine if it's positive
            avg_effect = metric_results['Effect (500→2500)'].mean()
            higher_is_better = metric_results['Higher is Better'].iloc[0]

            if (avg_effect > 0 and higher_is_better) or (
                    avg_effect < 0 and not higher_is_better):
                impact = "**Positive** (longer exemplars tend to improve this metric)"
            else:
                impact = "**Negative** (longer exemplars tend to worsen this metric)"

            f.write(f"- Average effect (500→2500): {avg_effect:.4f}\n")
            f.write(f"- Overall impact: {impact}\n")

            # Count credible results for this metric
            metric_improvements = improvements[improvements['Metric'] == metric]
            metric_deteriorations = deteriorations[deteriorations['Metric'] == metric]

            f.write(
                f"- Credible improvements: {len(metric_improvements)}/{len(metric_results)} ({len(metric_improvements) / len(metric_results):.1%})\n")
            f.write(
                f"- Credible deteriorations: {len(metric_deteriorations)}/{len(metric_results)} ({len(metric_deteriorations) / len(metric_results):.1%})\n\n")

            # Best and worst LLMs for this metric
            if not metric_results.empty:
                metric_by_llm = metric_results.groupby('LLM')[
                    'Effect (500→2500)'].mean().reset_index()

                if higher_is_better:
                    best_llm = metric_by_llm.iloc[
                        metric_by_llm['Effect (500→2500)'].argmax()]
                    worst_llm = metric_by_llm.iloc[
                        metric_by_llm['Effect (500→2500)'].argmin()]
                else:
                    best_llm = metric_by_llm.iloc[
                        metric_by_llm['Effect (500→2500)'].argmin()]
                    worst_llm = metric_by_llm.iloc[
                        metric_by_llm['Effect (500→2500)'].argmax()]

                f.write(
                    f"- Best LLM: **{best_llm['LLM']}** (average effect: {best_llm['Effect (500→2500)']:.4f})\n")
                f.write(
                    f"- Worst LLM: **{worst_llm['LLM']}** (average effect: {worst_llm['Effect (500→2500)']:.4f})\n\n")

            # List notable findings
            if len(metric_improvements) > 0:
                f.write("#### Notable Improvements:\n\n")
                for _, row in metric_improvements.nlargest(3,
                                                           'Effect (500→2500)').iterrows():
                    f.write(
                        f"- {row['LLM']} against {row['Threat Model']} on {row['Corpus']}: "
                        f"Effect = {row['Effect (500→2500)']:.4f}, "
                        f"Posterior prob. beneficial = {row['P(Improvement)']:.4f}\n")
                f.write("\n")

            if len(metric_deteriorations) > 0:
                f.write("#### Notable Deteriorations:\n\n")
                for _, row in metric_deteriorations.nlargest(3,
                                                             'Effect (500→2500)').iterrows():
                    f.write(
                        f"- {row['LLM']} against {row['Threat Model']} on {row['Corpus']}: "
                        f"Effect = {row['Effect (500→2500)']:.4f}, "
                        f"Posterior prob. detrimental = {1 - row['P(Improvement)']:.4f}\n")
                f.write("\n")

        # Key findings by LLM
        f.write("## Key Findings by LLM\n\n")
        for llm in results_df['LLM'].unique():
            llm_results = results_df[results_df['LLM'] == llm]
            f.write(f"### {llm}\n\n")

            # Count credible results
            llm_improvements = improvements[improvements['LLM'] == llm]
            llm_deteriorations = deteriorations[deteriorations['LLM'] == llm]
            llm_inconclusive = inconclusive[inconclusive['LLM'] == llm]
            llm_equivalent = equivalent[equivalent['LLM'] == llm]

            f.write(
                f"- Credible improvements: {len(llm_improvements)}/{len(llm_results)} ({len(llm_improvements) / len(llm_results):.1%})\n")
            f.write(
                f"- Credible deteriorations: {len(llm_deteriorations)}/{len(llm_results)} ({len(llm_deteriorations) / len(llm_results):.1%})\n")
            f.write(
                f"- Inconclusive results: {len(llm_inconclusive)}/{len(llm_results)} ({len(llm_inconclusive) / len(llm_results):.1%})\n")
            f.write(
                f"- Practically equivalent: {len(llm_equivalent)}/{len(llm_results)} ({len(llm_equivalent) / len(llm_results):.1%})\n\n")

            # Best and worst metrics for this LLM
            for is_better in [True, False]:
                metric_subset = llm_results[
                    llm_results['Higher is Better'] == is_better]

                if not metric_subset.empty:
                    metric_effect = metric_subset.groupby('Metric')[
                        'Effect (500→2500)'].mean().reset_index()

                    # For "higher is better" metrics, larger effects are better
                    # For "lower is better" metrics, smaller (negative) effects are better
                    if is_better:
                        best_metric = metric_effect.iloc[
                            metric_effect['Effect (500→2500)'].argmax()]
                        worst_metric = metric_effect.iloc[
                            metric_effect['Effect (500→2500)'].argmin()]
                    else:
                        best_metric = metric_effect.iloc[
                            metric_effect['Effect (500→2500)'].argmin()]
                        worst_metric = metric_effect.iloc[
                            metric_effect['Effect (500→2500)'].argmax()]

                    if is_better:
                        f.write(f"- For 'higher is better' metrics:\n")
                    else:
                        f.write(f"- For 'lower is better' metrics:\n")

                    f.write(
                        f"  - Most improved: **{best_metric['Metric']}** (avg effect: {best_metric['Effect (500→2500)']:.4f})\n")
                    f.write(
                        f"  - Least improved: **{worst_metric['Metric']}** (avg effect: {worst_metric['Effect (500→2500)']:.4f})\n\n")

            # Notable findings
            if len(llm_improvements) > 0:
                f.write("#### Notable Improvements:\n\n")
                for _, row in llm_improvements.nlargest(3,
                                                        'Effect (500→2500)').iterrows():
                    f.write(
                        f"- {row['Metric']} against {row['Threat Model']} on {row['Corpus']}: "
                        f"Effect = {row['Effect (500→2500)']:.4f}, "
                        f"Posterior prob. beneficial = {row['P(Improvement)']:.4f}\n")
                f.write("\n")

            if len(llm_deteriorations) > 0:
                f.write("#### Notable Deteriorations:\n\n")
                for _, row in llm_deteriorations.nlargest(3,
                                                          'Effect (500→2500)').iterrows():
                    f.write(
                        f"- {row['Metric']} against {row['Threat Model']} on {row['Corpus']}: "
                        f"Effect = {row['Effect (500→2500)']:.4f}, "
                        f"Posterior prob. detrimental = {1 - row['P(Improvement)']:.4f}\n")
                f.write("\n")

        # Key findings by threat model
        f.write("## Key Findings by Threat Model\n\n")
        for threat_model in results_df['Threat Model'].unique():
            tm_results = results_df[results_df['Threat Model'] == threat_model]
            f.write(f"### {threat_model}\n\n")

            # Count credible results
            tm_improvements = improvements[improvements['Threat Model'] == threat_model]
            tm_deteriorations = deteriorations[
                deteriorations['Threat Model'] == threat_model]

            f.write(
                f"- Credible improvements: {len(tm_improvements)}/{len(tm_results)} ({len(tm_improvements) / len(tm_results):.1%})\n")
            f.write(
                f"- Credible deteriorations: {len(tm_deteriorations)}/{len(tm_results)} ({len(tm_deteriorations) / len(tm_results):.1%})\n\n")

            # Average effect by metric
            f.write("#### Average Effect by Metric:\n\n")
            metrics_avg = tm_results.groupby('Metric')[
                'Effect (500→2500)'].mean().reset_index()
            for _, row in metrics_avg.iterrows():
                higher_is_better = tm_results[tm_results['Metric'] == row['Metric']][
                    'Higher is Better'].iloc[0]
                if (row['Effect (500→2500)'] > 0 and higher_is_better) or (
                        row['Effect (500→2500)'] < 0 and not higher_is_better):
                    impact = "positive"
                else:
                    impact = "negative"
                f.write(
                    f"- {row['Metric']}: {row['Effect (500→2500)']:.4f} ({impact})\n")
            f.write("\n")

            # Most affected LLM
            f.write("#### Effect by LLM:\n\n")
            llm_avg = tm_results.groupby('LLM')[
                'Effect (500→2500)'].mean().reset_index()
            llm_avg['Abs_Effect'] = llm_avg['Effect (500→2500)'].abs()
            most_affected = llm_avg.iloc[llm_avg['Abs_Effect'].argmax()]

            f.write(
                f"- Most affected LLM: **{most_affected['LLM']}** (avg effect: {most_affected['Effect (500→2500)']:.4f})\n\n")

        # Corpus-specific findings
        f.write("## Findings by Corpus\n\n")
        for corpus in results_df['Corpus'].unique():
            corpus_results = results_df[results_df['Corpus'] == corpus]
            f.write(f"### {corpus}\n\n")

            # Count credible results
            corpus_improvements = improvements[improvements['Corpus'] == corpus]
            corpus_deteriorations = deteriorations[deteriorations['Corpus'] == corpus]

            f.write(
                f"- Credible improvements: {len(corpus_improvements)}/{len(corpus_results)} ({len(corpus_improvements) / len(corpus_results):.1%})\n")
            f.write(
                f"- Credible deteriorations: {len(corpus_deteriorations)}/{len(corpus_results)} ({len(corpus_deteriorations) / len(corpus_results):.1%})\n\n")

            # Overall trend
            if len(corpus_improvements) > len(corpus_deteriorations):
                f.write(
                    "Overall, longer exemplars tend to be **more effective** on this corpus.\n\n")
            elif len(corpus_improvements) < len(corpus_deteriorations):
                f.write(
                    "Overall, longer exemplars tend to be **less effective** on this corpus.\n\n")
            else:
                f.write(
                    "Overall, the effects of longer exemplars are **mixed** on this corpus.\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        if len(improvements) > len(deteriorations):
            f.write(
                "Based on the Bayesian analysis, **longer exemplars generally improve the effectiveness** of LLM-based imitation as a defense against authorship attribution attacks. The improvement is most pronounced for:\n\n")

            # Find top improving metrics
            metric_improvement = results_df.groupby('Metric')[
                'Effect (500→2500)'].mean().reset_index()
            metric_improvement['Higher is Better'] = metric_improvement['Metric'].apply(
                lambda m:
                results_df[results_df['Metric'] == m]['Higher is Better'].iloc[0])
            metric_improvement['Positive_Effect'] = metric_improvement.apply(
                lambda row: (row['Effect (500→2500)'] > 0 and row[
                    'Higher is Better']) or
                            (row['Effect (500→2500)'] < 0 and not row[
                                'Higher is Better']), axis=1)

            top_metrics = metric_improvement[
                metric_improvement['Positive_Effect']].sort_values(
                by='Effect (500→2500)', ascending=False)

            for _, row in top_metrics.head(3).iterrows():
                f.write(
                    f"- **{row['Metric']}** (average effect: {row['Effect (500→2500)']:.4f})\n")

            f.write("\nAnd the LLMs that benefit most from longer exemplars are:\n\n")

            # Find top improving LLMs
            llm_improvement = improvements.groupby('LLM').size().reset_index(
                name='count')
            llm_improvement = llm_improvement.sort_values(by='count', ascending=False)

            for _, row in llm_improvement.head(3).iterrows():
                llm_total = len(results_df[results_df['LLM'] == row['LLM']])
                f.write(
                    f"- **{row['LLM']}** ({row['count']}/{llm_total} metrics improved, {row['count'] / llm_total:.1%})\n")

        elif len(improvements) < len(deteriorations):
            f.write(
                "Based on the Bayesian analysis, **longer exemplars generally reduce the effectiveness** of LLM-based imitation as a defense against authorship attribution attacks. The deterioration is most pronounced for:\n\n")

            # Find top deteriorating metrics
            metric_deterioration = results_df.groupby('Metric')[
                'Effect (500→2500)'].mean().reset_index()
            metric_deterioration['Higher is Better'] = metric_deterioration[
                'Metric'].apply(
                lambda m:
                results_df[results_df['Metric'] == m]['Higher is Better'].iloc[0])
            metric_deterioration['Negative_Effect'] = metric_deterioration.apply(
                lambda row: (row['Effect (500→2500)'] < 0 and row[
                    'Higher is Better']) or
                            (row['Effect (500→2500)'] > 0 and not row[
                                'Higher is Better']), axis=1)

            top_metrics = metric_deterioration[
                metric_deterioration['Negative_Effect']].sort_values(
                by='Effect (500→2500)', ascending=True)

            for _, row in top_metrics.head(3).iterrows():
                f.write(
                    f"- **{row['Metric']}** (average effect: {row['Effect (500→2500)']:.4f})\n")

            f.write(
                "\nAnd the LLMs that are most negatively affected by longer exemplars are:\n\n")

            # Find top deteriorating LLMs
            llm_deterioration = deteriorations.groupby('LLM').size().reset_index(
                name='count')
            llm_deterioration = llm_deterioration.sort_values(by='count',
                                                              ascending=False)

            for _, row in llm_deterioration.head(3).iterrows():
                llm_total = len(results_df[results_df['LLM'] == row['LLM']])
                f.write(
                    f"- **{row['LLM']}** ({row['count']}/{llm_total} metrics worsened, {row['count'] / llm_total:.1%})\n")

        else:
            f.write(
                "Based on the Bayesian analysis, **the effects of longer exemplars are mixed** for LLM-based imitation as a defense against authorship attribution attacks. Some metrics and models show improvement, while others show deterioration.\n\n")

        f.write(
            "\nThese findings suggest that the optimal exemplar length may depend on the specific LLM, threat model, and metric of interest. Users should consider these factors when choosing exemplar length for their specific defense scenario.\n")

        # Add information about data and methodology
        f.write("\n## Methodology Notes\n\n")
        f.write(
            "This analysis used Bayesian hierarchical modeling to estimate the relationship between exemplar length and defense effectiveness. For each LLM-threat model-metric combination, we:\n\n")
        f.write("1. Fitted a Bayesian model relating exemplar length to the metric\n")
        f.write("2. Estimated the slope parameter (effect per 1000 words)\n")
        f.write(
            "3. Calculated the 95% Highest Density Interval (HDI) for this parameter\n")
        f.write(
            "4. Determined the posterior probability that the effect is beneficial\n\n")

        f.write(
            "An effect was considered credible when the 95% HDI excluded zero. The analysis accounted for the bounded nature of metrics like accuracy and incorporated appropriate prior distributions where needed.\n")

    logger.info(f"Summary report saved to {report_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze the effect of exemplar length on LLM imitation defense performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base_dir',
        type=str,
        default='defense_evaluation',
        help='Base directory containing evaluation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/exemplar_length_analysis_rq3.2',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        choices=CORPORA,
        help='Specific corpus to analyze (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific LLM to analyze (default: all)'
    )
    parser.add_argument(
        '--threat_model',
        type=str,
        choices=THREAT_MODELS,
        help='Specific threat model to analyze (default: all)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=METRICS,
        help='Specific metric to analyze (default: all)'
    )

    return parser.parse_args()

def main():
    """Main entry point for script."""
    args = parse_arguments()

    logger.info("Starting exemplar length analysis")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    logger.info("Preparing data for analysis")
    data = prepare_data(args.base_dir)

    # Save raw data
    data.to_csv(output_dir / "raw_data.csv", index=False)

    # Run analysis
    logger.info("Running analysis")
    results_df = run_full_analysis(
        data=data,
        output_dir=str(output_dir),
        filter_corpus=args.corpus,
        filter_model=args.model,
        filter_threat_model=args.threat_model,
        filter_metric=args.metric
    )

    # Create summary report
    logger.info("Creating summary report")
    create_summary_report(results_df, str(output_dir))

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()

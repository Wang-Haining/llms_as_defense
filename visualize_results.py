
import json
import re
from collections import OrderedDict, defaultdict
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# set font to Times New Roman for all text elements
plt.rcParams['font.family'] = 'Times New Roman'


def clean_column_name(col: str) -> str:
    """
    Clean column names by removing arrows and whitespace.

    Args:
        col: raw column name potentially containing arrows (↑/↓) and whitespace

    Returns:
        cleaned column name
    """
    return re.sub(r'[↑↓\s]', '', col)


def parse_stat_entry(entry: str) -> tuple:
    """
    Parse statistical entry of form "mean ± std [ci_lower, ci_upper]"

    Args:
        entry: string of format "0.149 ± 0.075 [0.041, 0.223]"

    Returns:
        tuple of (mean, std, lower_bound, upper_bound)
    """
    # extract mean and std
    mean_std_match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', entry)
    if not mean_std_match:
        raise ValueError(f"Could not parse mean/std from: {entry}")
    mean = float(mean_std_match.group(1))
    std = float(mean_std_match.group(2))

    # extract CI bounds
    ci_match = re.search(r'\[([\d.]+),\s*([\d.]+)\]', entry)
    if not ci_match:
        raise ValueError(f"Could not parse CI bounds from: {entry}")

    # Return CI bounds directly instead of calculating relative to mean
    lower_bound = float(ci_match.group(1))
    upper_bound = float(ci_match.group(2))

    return mean, std, lower_bound, upper_bound


def calculate_rope_ranges(metric_name: str, pre_value: float, std: float,
                          corpus: str = None) -> dict:
    """
    Calculate ROPE (Region of Practical Equivalence) ranges for different metrics.

    Args:
        metric_name: name of the metric (case-insensitive)
        pre_value: pre-intervention value of the metric
        std: standard deviation of the posterior distribution
        corpus: name of corpus (needed for some metrics, e.g., 'rj' or 'ebg')

    Returns:
        dict containing:
            - pre_rope: tuple of (lower, upper) bounds for pre-value ROPE
            - ideal_rope: tuple of (lower, upper) bounds for ideal-value ROPE
            - ideal_value: the ideal/target value for this metric
    """
    if not metric_name:
        raise ValueError("metric_name cannot be empty")

    # standardize metric name and corpus for comparison
    metric_lower = metric_name.lower()
    corpus_lower = corpus.lower() if corpus else None

    # validate corpus for metrics that require it
    metrics_requiring_corpus = {
        "accuracy@1", "acc@1", "accuracy@5", "acc@5",
        "entropy", "true_class_confidence", "true_label_confidence"
    }
    if any(m in metric_lower for m in metrics_requiring_corpus):
        if not corpus:
            raise ValueError(f"corpus is required for metric '{metric_name}'")
        if corpus_lower not in {"rj", "ebg"}:
            raise ValueError(
                f"invalid corpus '{corpus}' for metric '{metric_name}'. Must be 'rj' or 'ebg'")

    # compute half rope range (0.1 * std is standard)
    half_rope_range = 0.1 * std

    # specific handling for PINC
    if 'pinc' in metric_lower:
        pre_rope_lower = 0  # PINC starts from 0
        pre_rope_upper = pre_value + half_rope_range
        ideal_rope_lower = 0.95 - half_rope_range  # High PINC is ideal
        ideal_rope_upper = 1.0
        ideal_value = 1.0
        return {
            'pre_rope': (pre_rope_lower, pre_rope_upper),
            'ideal_rope': (ideal_rope_lower, ideal_rope_upper),
            'ideal_value': ideal_value
        }

    # specific handling for SBert and BERTScore
    if any(keyword in metric_lower for keyword in ['sbert', 'bertscore']):
        pre_rope_lower = pre_value - half_rope_range
        pre_rope_upper = 1.0  # These metrics are bounded at 1.0
        return {
            'pre_rope': (pre_rope_lower, pre_rope_upper),
            'ideal_rope': None,  # No ideal ROPE for these metrics
            'ideal_value': None
        }

    # handling for other metrics
    if metric_lower in ["accuracy@1", "acc@1"]:
        ideal_value = 1 / 45 if corpus_lower == "ebg" else 1 / 21
    elif metric_lower in ["accuracy@5", "acc@5"]:
        ideal_value = 5 / 45 if corpus_lower == "ebg" else 5 / 21
    elif "entropy" in metric_lower:
        ideal_value = np.log2(45) if corpus_lower == "ebg" else np.log2(21)
    elif metric_lower in ["true_class_confidence", "true_label_confidence"]:
        ideal_value = 1 / 45 if corpus_lower == "ebg" else 1 / 21
    else:
        ideal_value = pre_value

    # standard ROPE calculations for other metrics
    pre_rope_lower = pre_value - half_rope_range
    pre_rope_upper = pre_value + half_rope_range

    # determine ideal-value ROPE ranges
    if "entropy" in metric_lower:
        ideal_rope_lower = max(0, ideal_value - half_rope_range)
        ideal_rope_upper = ideal_value
    else:
        ideal_rope_lower = max(0, ideal_value - half_rope_range)
        ideal_rope_upper = min(1, ideal_value + half_rope_range)

    return {
        'pre_rope': (pre_rope_lower, pre_rope_upper),
        'ideal_rope': (ideal_rope_lower, ideal_rope_upper),
        'ideal_value': ideal_value
    }


# def format_for_plotting(raw_df: pd.DataFrame, pre_df: pd.DataFrame,
#                         metrics: Union[str, List[str]],
#                         corpus: str = None) -> pd.DataFrame:
#     """
#     Format raw stats DataFrame for the HDI plotting functions.
#
#     Args:
#         raw_df: DataFrame with raw statistical entries for post values
#         pre_df: DataFrame with pre-values
#         metrics: metrics to process
#         corpus: optional corpus to filter by
#
#     Returns:
#         DataFrame formatted for plotting with correct pre and ideal values
#     """
#     # handle single metric case
#     if isinstance(metrics, str):
#         metrics = [metrics]
#
#     # list to store processed rows
#     processed_rows = []
#
#     for metric in metrics:
#         # find the actual column names that match this metric (ignoring arrows)
#         post_col = [col for col in raw_df.columns if
#                     clean_column_name(col) == clean_column_name(metric)][0]
#         pre_col = [col for col in pre_df.columns if
#                    clean_column_name(col) == clean_column_name(metric)][0]
#
#         # extract stats from the raw entries
#         for idx, row in raw_df.iterrows():
#             mean, std, ci_lower, ci_upper = parse_stat_entry(row[post_col])
#
#             # get pre-value from pre_df
#             pre_val = float(pre_df[
#                                 (pre_df['Corpus'] == row['Corpus']) &
#                                 (pre_df['Threat Model'] == row['Threat Model'])
#                                 ][pre_col].iloc[0])
#
#             # calculate ROPEs using pre-value
#             rope_ranges = calculate_rope_ranges(metric, pre_val, std, row['Corpus'])
#
#             row_data = {
#                 'Corpus': row['Corpus'],
#                 'Threat Model': row['Threat Model'],
#                 'Defense Model': row['Defense Model'],
#                 'Metric': clean_column_name(metric),
#                 'Post Mean': mean,
#                 'CI Lower': ci_lower,
#                 'CI Upper': ci_upper,
#                 'Pre ROPE Lower': rope_ranges['pre_rope'][0],
#                 'Pre ROPE Upper': rope_ranges['pre_rope'][1],
#             }
#
#             # Only add ideal ROPE if it exists
#             if rope_ranges['ideal_rope'] is not None:
#                 row_data.update({
#                     'Ideal ROPE Lower': rope_ranges['ideal_rope'][0],
#                     'Ideal ROPE Upper': rope_ranges['ideal_rope'][1],
#                     'Ideal Value': rope_ranges['ideal_value']
#                 })
#             else:
#                 row_data.update({
#                     'Ideal ROPE Lower': None,
#                     'Ideal ROPE Upper': None,
#                     'Ideal Value': None
#                 })
#
#             processed_rows.append(row_data)
#
#     result_df = pd.DataFrame(processed_rows)
#
#     # filter by corpus if specified
#     if corpus:
#         result_df = result_df[result_df['Corpus'].str.upper() == corpus.upper()]
#
#     return result_df


def format_for_plotting_attribution(raw_df: pd.DataFrame, pre_df: pd.DataFrame,
                                      metrics: Union[str, List[str]] = None,
                                      corpus: str = None) -> pd.DataFrame:
    """
    Format raw stats DataFrame for HDI plotting (effectiveness/robustness metrics).
    Defaults to metrics: ["accuracy@1", "accuracy@5", "true_class_confidence", "entropy"]

    Args:
        raw_df: DataFrame with raw statistical entries for post values.
        pre_df: DataFrame with pre-values.
        metrics: Metrics to process.
        corpus: Optional corpus to filter by.

    Returns:
        DataFrame formatted for plotting with correct pre and ideal values.
    """
    if metrics is None:
        metrics = ["accuracy@1", "accuracy@5", "true_class_confidence", "entropy"]
    if isinstance(metrics, str):
        metrics = [metrics]

    processed_rows = []

    for metric in metrics:
        post_col = [col for col in raw_df.columns
                    if clean_column_name(col) == clean_column_name(metric)][0]
        pre_col = [col for col in pre_df.columns
                   if clean_column_name(col) == clean_column_name(metric)][0]

        for idx, row in raw_df.iterrows():
            mean, std, ci_lower, ci_upper = parse_stat_entry(row[post_col])

            pre_val = float(pre_df[
                                (pre_df['Corpus'] == row['Corpus']) &
                                (pre_df['Threat Model'] == row['Threat Model'])
                                ][pre_col].iloc[0])

            rope_ranges = calculate_rope_ranges(metric, pre_val, std, row['Corpus'])

            row_data = {
                'Corpus': row['Corpus'],
                'Threat Model': row['Threat Model'],
                'Defense Model': row['Defense Model'],
                'Metric': clean_column_name(metric),
                'Post Mean': mean,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'Pre ROPE Lower': rope_ranges['pre_rope'][0],
                'Pre ROPE Upper': rope_ranges['pre_rope'][1],
            }

            if rope_ranges['ideal_rope'] is not None:
                row_data.update({
                    'Ideal ROPE Lower': rope_ranges['ideal_rope'][0],
                    'Ideal ROPE Upper': rope_ranges['ideal_rope'][1],
                    'Ideal Value': rope_ranges['ideal_value']
                })
            else:
                row_data.update({
                    'Ideal ROPE Lower': None,
                    'Ideal ROPE Upper': None,
                    'Ideal Value': None
                })

            processed_rows.append(row_data)

    result_df = pd.DataFrame(processed_rows)
    if corpus:
        result_df = result_df[result_df['Corpus'].str.upper() == corpus.upper()]

    return result_df


def format_for_plotting_quality(raw_df: pd.DataFrame, pre_df: pd.DataFrame,
                                metrics: Union[str, List[str]] = None,
                                corpus: str = None) -> pd.DataFrame:
    """
    Format raw stats DataFrame for HDI plotting (quality metrics).
    Defaults to metrics: ["bertscore", "sbert", "pinc"].

    Note: This function excludes the 'Threat Model' column, as quality assessments do not
    rely on threat model information. It also drops duplicate rows.

    Args:
        raw_df: DataFrame with raw statistical entries for post values.
        pre_df: DataFrame with pre-values.
        metrics: Metrics to process.
        corpus: Optional corpus to filter by.

    Returns:
        DataFrame formatted for plotting with deduplicated rows and without the threat model column.
    """
    if metrics is None:
        metrics = ["bertscore", "sbert", "pinc"]
    if isinstance(metrics, str):
        metrics = [metrics]

    processed_rows = []

    for metric in metrics:
        post_col = [col for col in raw_df.columns
                    if clean_column_name(col) == clean_column_name(metric)][0]
        pre_col = [col for col in pre_df.columns
                   if clean_column_name(col) == clean_column_name(metric)][0]

        for idx, row in raw_df.iterrows():
            mean, std, ci_lower, ci_upper = parse_stat_entry(row[post_col])

            # For quality metrics, filter only on Corpus since threat model is not applicable
            pre_val = float(pre_df[pre_df['Corpus'] == row['Corpus']][pre_col].iloc[0])
            rope_ranges = calculate_rope_ranges(metric, pre_val, std, row['Corpus'])

            row_data = {
                'Corpus': row['Corpus'],
                'Defense Model': row['Defense Model'],
                'Metric': clean_column_name(metric),
                'Post Mean': mean,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'Pre ROPE Lower': rope_ranges['pre_rope'][0],
                'Pre ROPE Upper': rope_ranges['pre_rope'][1],
            }

            if rope_ranges['ideal_rope'] is not None:
                row_data.update({
                    'Ideal ROPE Lower': rope_ranges['ideal_rope'][0],
                    'Ideal ROPE Upper': rope_ranges['ideal_rope'][1],
                    'Ideal Value': rope_ranges['ideal_value']
                })
            else:
                row_data.update({
                    'Ideal ROPE Lower': None,
                    'Ideal ROPE Upper': None,
                    'Ideal Value': None
                })

            processed_rows.append(row_data)

    result_df = pd.DataFrame(processed_rows)
    if corpus:
        result_df = result_df[result_df['Corpus'].str.upper() == corpus.upper()]

    # deduplicate the rows
    result_df = result_df.drop_duplicates()

    return result_df


def format_df_for_visualization(parsed_df: pd.DataFrame,
                                metrics: Union[str, List[str]],
                                corpus: str = None,
                                threat_model: str = None) -> pd.DataFrame:
    """
    Format parsed DataFrame into structure expected by plotting functions.
    Handles both single metric and multiple metrics.

    Args:
        parsed_df: DataFrame produced by prepare_stats_for_plotting
        metrics: single metric name or list of metric names (with or without arrows/spaces)
        corpus: optional corpus to filter by
        threat_model: optional threat model to filter by

    Returns:
        DataFrame with all metrics and data needed for plotting functions
    """
    # handle single metric case
    if isinstance(metrics, str):
        metrics = [metrics]

    # clean metric names
    metrics = [clean_column_name(m) for m in metrics]

    # list to store DataFrames for each metric
    dfs = []

    for metric in metrics:
        # create base DataFrame with identifying columns
        df = parsed_df[['Corpus', 'Threat Model', 'Defense Model']].copy()

        # add statistical columns for the metric
        df['Post Mean'] = parsed_df[f"{metric}_mean"]
        df['CI Lower'] = parsed_df[f"{metric}_ci_lower"]
        df['CI Upper'] = parsed_df[f"{metric}_ci_upper"]

        # calculate ROPE ranges
        for _, row in df.iterrows():
            ranges = calculate_rope_ranges(
                metric,
                row['Post Mean'],
                parsed_df.loc[_, f"{metric}_std"],
                row['Corpus']
            )
            df.loc[_, 'Pre ROPE Lower'] = ranges['pre_rope'][0]
            df.loc[_, 'Pre ROPE Upper'] = ranges['pre_rope'][1]
            df.loc[_, 'Ideal ROPE Lower'] = ranges['ideal_rope'][0]
            df.loc[_, 'Ideal ROPE Upper'] = ranges['ideal_rope'][1]
            df.loc[_, 'Ideal Value'] = ranges['ideal_value']

        # add metric name
        df['Metric'] = metric

        # apply filters if specified
        if corpus:
            df = df[df['Corpus'].str.upper() == corpus.upper()]
        if threat_model:
            df = df[df['Threat Model'] == threat_model]

        dfs.append(df)

    # combine all metric DataFrames
    return pd.concat(dfs, ignore_index=True)


def plot_single_bayesian_hdi(ax, stats_df, metric, corpus, threat_model=None,
                             arrow_offset=0, arrow_end_height=0.3):
    """
    Draws a single pivoted Bayesian HDI plot using the summarized stats DataFrame.
    For attribution metrics (which include threat model information), the plot groups by threat model.
    For quality metrics (which do not include threat model), the plot groups by defense model only,
    removes x-ticks and labels, and interleaves white and light gray background spans.
    """
    # mapping for defense model ordering and colors - order is important
    model_colors = {
        "Gemma-2": "#d62728",  # red
        "Llama-3.1": "#ff7f0e",  # orange
        "Ministral": "#e377c2",  # pink
        "Claude-3.5": "#2ca02c",  # green
        "GPT-4o": "#17becf"  # cyan
    }

    # caption dictionary for metrics
    caption_dict = {
        "accuracy@1": {"title": "Accuracy@1", "ylabel": "Accuracy@1"},
        "accuracy@5": {"title": "Accuracy@5", "ylabel": "Accuracy@5"},
        "true_class_confidence": {"title": "True Class Confidence",
                                  "ylabel": "Probability"},
        "entropy": {"title": "Entropy", "ylabel": "Bits"},
        "meteor": {"title": "Meteor", "ylabel": "Meteor"},
        "bleu": {"title": "Bleu", "ylabel": "Bleu"},
        "bertscore": {"title": "BERTScore", "ylabel": "BERTScore"},
        "sbert": {"title": "SBERT", "ylabel": "SBERT"},
        "pinc": {"title": "PINC", "ylabel": "PINC"}
    }
    m_lower = metric.lower()
    title_str = caption_dict[m_lower][
        "title"] if m_lower in caption_dict else metric.title()
    ylabel_str = caption_dict[m_lower][
        "ylabel"] if m_lower in caption_dict else metric.title()

    # Filter the DataFrame: if "Metric" column exists, filter by the given metric; otherwise, use all rows.
    if "Metric" in stats_df.columns:
        df = stats_df[(stats_df["Corpus"].str.upper() == corpus.upper()) &
                      (stats_df["Metric"] == metric)].copy()
    else:
        df = stats_df.copy()
        if "Corpus" in df.columns:
            df = df[df["Corpus"].str.upper() == corpus.upper()]

    if threat_model is not None and "Threat Model" in df.columns:
        df = df[df["Threat Model"] == threat_model]

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=16, fontfamily='Times New Roman')
        return ax

    # Determine whether we are plotting quality metrics (i.e. no "Threat Model" column)
    is_quality = "Threat Model" not in df.columns

    if not is_quality:
        # Attribution metrics: group by threat model.
        threat_order = ["logreg", "svm", "roberta"]
        threat_names = {"logreg": "Logistic Regression", "svm": "SVM",
                        "roberta": "RoBERTa"}
        df["Threat Group"] = df["Threat Model"].str.lower()

        group_width = 2.5  # width allocated for each threat group
        group_gap = 0.5  # gap between groups
        x_positions = {}
        current_pos = 0
        for threat in threat_order:
            group_df = df[df["Threat Group"] == threat]
            if not group_df.empty:
                model_spacing = group_width / (len(model_colors) + 1)
                position = 1
                for model in model_colors.keys():
                    model_indices = group_df[group_df["Defense Model"] == model].index
                    for idx in model_indices:
                        x_positions[idx] = current_pos + position * model_spacing
                    position += 1
            current_pos += group_width + group_gap
    else:
        # Quality metrics: group solely by defense model.
        defense_model_order = list(model_colors.keys())
        df["Defense Model"] = pd.Categorical(df["Defense Model"],
                                             categories=defense_model_order,
                                             ordered=True)
        df = df.sort_values("Defense Model")
        x_positions = {}
        unique_defense = df["Defense Model"].unique()
        for i, model in enumerate(unique_defense):
            for idx in df[df["Defense Model"] == model].index:
                x_positions[idx] = i + 1
        current_pos = len(unique_defense) + 1

        # Remove x ticks and labels for quality metrics.
        ax.set_xticks([])

        # Add interleaved background colors for each defense model group.
        for i, model in enumerate(unique_defense):
            group_rows = df[df["Defense Model"] == model]
            if not group_rows.empty:
                group_x_positions = [x_positions[idx] for idx in group_rows.index]
                group_min = min(group_x_positions) - 0.5
                group_max = max(group_x_positions) + 0.5
                # Alternate between white and light gray.
                color = "lightgrey" if i % 2 == 1 else "white"
                ax.axvspan(group_min, group_max, facecolor=color, alpha=0.3)

    # Set y-axis range.
    if "entropy" in m_lower:
        ax.set_ylim(0,
                    np.log2(45) + 0.1 if corpus.lower() == "ebg" else np.log2(21) + 0.1)
    else:
        ax.set_ylim(0, 1)

    # Remove top/right spines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Determine whether to show ideal ROPE (for attribution metrics).
    is_quality_metric = any(
        keyword in m_lower for keyword in ['pinc', 'sbert', 'bertscore'])
    show_ideal_rope = not is_quality_metric

    # Plot error bars and ROPE boxes.
    for idx, row in df.iterrows():
        x = x_positions[idx]
        y = row["Post Mean"]
        y_err_low = y - row["CI Lower"]
        y_err_high = row["CI Upper"] - y
        col = model_colors.get(row["Defense Model"], "gray")

        ax.errorbar(x, y, yerr=[[y_err_low], [y_err_high]], fmt=".", capsize=2,
                    color=col, elinewidth=0.5, ecolor=col, markeredgecolor="white")

        band_width = 0.2
        x_pre = x + arrow_offset
        ax.fill_between([x_pre - band_width / 2, x_pre + band_width / 2],
                        row["Pre ROPE Lower"], row["Pre ROPE Upper"],
                        color="grey", alpha=0.2)
        ax.plot([x_pre - band_width / 2, x_pre + band_width / 2],
                [row["Pre ROPE Lower"], row["Pre ROPE Lower"]],
                color="grey", lw=1.5, linestyle="--")
        ax.plot([x_pre - band_width / 2, x_pre + band_width / 2],
                [row["Pre ROPE Upper"], row["Pre ROPE Upper"]],
                color="grey", lw=1.5, linestyle="--")

        if show_ideal_rope and "Ideal ROPE Lower" in row and row[
            "Ideal ROPE Lower"] is not None:
            x_ideal = x - arrow_offset
            ax.fill_between([x_ideal - band_width / 2, x_ideal + band_width / 2],
                            row["Ideal ROPE Lower"], row["Ideal ROPE Upper"],
                            color="skyblue", alpha=0.2)
            ax.plot([x_ideal - band_width / 2, x_ideal + band_width / 2],
                    [row["Ideal ROPE Lower"], row["Ideal ROPE Lower"]],
                    color="skyblue", lw=1.5, linestyle="--")
            ax.plot([x_ideal - band_width / 2, x_ideal + band_width / 2],
                    [row["Ideal ROPE Upper"], row["Ideal ROPE Upper"]],
                    color="skyblue", lw=1.5, linestyle="--")

    # For attribution metrics, add threat group labels; for quality metrics, no x-labels are added.
    if not is_quality:
        ax.set_xticks([])
        current_pos = 0
        threat_order = ["logreg", "svm", "roberta"]
        threat_names = {"logreg": "Logistic Regression", "svm": "SVM",
                        "roberta": "RoBERTa"}
        for threat in threat_order:
            group_df = df[df["Threat Group"] == threat]
            if not group_df.empty:
                group_start = min([x_positions[idx] for idx in group_df.index]) - 0.5
                group_end = max([x_positions[idx] for idx in group_df.index]) + 0.5
                group_center = (group_start + group_end) / 2
                y_min, y_max = ax.get_ylim()
                ax.text(group_center, y_min - 0.15 * (y_max - y_min),
                        threat_names[threat],
                        ha="center", va="top", fontsize=12,
                        fontfamily='Times New Roman')
            current_pos += 2.5 + 0.5
    else:
        # For quality metrics, x ticks and labels have already been removed.
        pass

    ax.set_ylabel(ylabel_str, fontsize=14, fontfamily='Times New Roman')
    ax.set_title(title_str, fontsize=14, fontfamily='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    if is_quality:
        ax.set_xlim(min(x_positions.values()) - 0.5, max(x_positions.values()) + 0.5)
    else:
        ax.set_xlim(-0.5, current_pos - 0.5)
    return ax


def plot_pivoted_bayesian_hdi_row(stats_df, metrics_list, corpus, threat_model=None,
                                  arrow_offset=0, arrow_end_height=0.3, dpi=800,
                                  save_name=None):
    """
    Plots a row of pivoted Bayesian HDI interval plots for multiple metrics.
    """

    # define common legend elements with enforced order
    model_colors = {
        "Gemma-2": "#d62728",  # red
        "Llama-3.1": "#ff7f0e",  # orange
        "Ministral": "#e377c2",  # pink
        "Claude-3.5": "#2ca02c",  # green
        "GPT-4o": "#17becf"  # cyan
    }

    n = len(metrics_list)
    if n == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=dpi)
    elif n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), dpi=dpi)
        if n == 1:
            axes = [axes]

    for ax, metric in zip(axes, metrics_list):
        plot_single_bayesian_hdi(ax, stats_df, metric, corpus, threat_model,
                                 arrow_offset, arrow_end_height)

    # increase bottom margin to accommodate threat group labels and legend
    fig.tight_layout(rect=[0.12, 0.18, 0.9, 0.98])

    legend_elements = []

    # add model colors in specified order
    for model, color in model_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color=color,
                                      markerfacecolor=color, markeredgecolor='white',
                                      label=model, markersize=8, linestyle='None'))

    # check if any metrics should show ideal ROPE
    metrics_lower = [m.lower() for m in metrics_list]
    show_ideal_rope = not any(
        any(keyword in m for keyword in ['pinc', 'sbert', 'bertscore'])
        for m in metrics_lower)

    # add ROPE indicators
    legend_elements.append(Patch(facecolor='grey', alpha=0.2,
                                 label='ROPE of pre-intervention',
                                 edgecolor='grey', linestyle='--'))

    if show_ideal_rope:
        legend_elements.append(Patch(facecolor='skyblue', alpha=0.2,
                                     label='ROPE of ideal scenario',
                                     edgecolor='skyblue', linestyle='--'))

    # place legend closer to the bottom with Times New Roman font
    fig.legend(handles=legend_elements, loc='center',
               bbox_to_anchor=(0.5, 0.18),
               ncol=len(model_colors) + (2 if show_ideal_rope else 1),
               fontsize=10, prop={'family': 'Times New Roman'})

    if save_name:
        fig.savefig(save_name, format="jpg", dpi=dpi, bbox_inches='tight')

    return fig, axes


def get_pre_scores(pre_df):
    """
    Extract pre-intervention entropy scores from DataFrame.
    """
    pre_scores = {}
    for _, row in pre_df.iterrows():
        corpus = row['Corpus'].lower()
        model = row['Threat Model'].lower()
        entropy = row['entropy ↑']
        pre_scores[f"{corpus}_{model}"] = entropy
    return pre_scores


def process_entropy_data(json_data, pre_df):
    """
    Process raw JSON data into format needed for visualization.
    """
    distributions = defaultdict(lambda: defaultdict(dict))
    pre_scores = defaultdict(dict)

    pre_score_dict = get_pre_scores(pre_df)

    for exp_key, exp_data in json_data.items():
        corpus, defense_model, attack_model = exp_key.split('_')

        if 'sample_level' in exp_data and 'entropy' in exp_data['sample_level']:
            entropy_values = np.array(exp_data['sample_level']['entropy'])
            entropy_values = entropy_values[np.isfinite(entropy_values)]
            distributions[corpus][attack_model][defense_model] = entropy_values

            pre_key = f"{corpus}_{attack_model}"
            if pre_key in pre_score_dict:
                pre_scores[corpus][attack_model] = pre_score_dict[pre_key]

    return dict(distributions), dict(pre_scores)


def plot_entropy_distributions(
        json_data,
        pre_df,
        corpus='ebg',
        figsize=(15, 4),
        dpi=800,
        save_name=None
):
    """
    Create entropy distribution plots from JSON experiment data.
    """
    distributions_dict, pre_scores = process_entropy_data(json_data, pre_df)

    plt.rcParams['font.family'] = 'Times New Roman'

    # use OrderedDict to maintain fixed sequence
    defense_colors = OrderedDict([
        ("Gemma-2", "#d62728"),  # red
        ("Llama-3.1", "#ff7f0e"),  # orange
        ("Ministral", "#e377c2"),  # pink
        ("Claude-3.5", "#2ca02c"),  # green
        ("GPT-4o", "#17becf")  # cyan
    ])

    ideal_entropy = np.log2(21) if corpus.lower() == 'rj' else np.log2(45)

    # create figure and subplots with minimal spacing
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=0.1)

    titles = {
        'logreg': 'Logistic Regression',
        'svm': 'SVM',
        'roberta': 'RoBERTa'
    }

    for idx, (model_name, ax) in enumerate(zip(titles.keys(), axes)):
        if model_name in distributions_dict[corpus]:
            model_data = distributions_dict[corpus][model_name]

            # plot in fixed order from OrderedDict
            for defense_name in defense_colors.keys():
                if defense_name in model_data:
                    values = model_data[defense_name]
                    if len(values) > 5:
                        kernel = stats.gaussian_kde(values)
                        x_range = np.linspace(values.min(), values.max(), 200)
                        density = kernel(x_range)

                        ax.plot(x_range, density,
                                color=defense_colors[defense_name],
                                label=defense_name,
                                linewidth=2)

                        ax.plot(values, np.zeros_like(values) - 0.02,
                                '|', color=defense_colors[defense_name],
                                alpha=0.3, markersize=10)

            if model_name in pre_scores[corpus]:
                ax.axvline(x=pre_scores[corpus][model_name],
                           color='gray',
                           linestyle='--',
                           label='Initial entropy',
                           alpha=0.8)

            ax.axvline(x=ideal_entropy,
                       color='red',
                       linestyle=':',
                       label='Maximal entropy',
                       alpha=0.8)

            ax.set_title(titles[model_name], fontsize=14)
            if idx == 0:
                ax.set_ylabel('Density', fontsize=12)
            ax.set_xlabel('Entropy (bits)', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10)

            max_entropy = np.log2(45) if corpus.lower() == 'ebg' else np.log2(21)
            ax.set_xlim(0, max_entropy + 0.5)

            # add legend only to first subplot with fixed order
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # reorder handles and labels to put defense models first
                defense_order = list(defense_colors.keys())
                other_labels = ['Initial entropy', 'Maximal entropy']

                ordered_handles = []
                ordered_labels = []

                # add defense models in fixed order
                for defense_name in defense_order:
                    if defense_name in labels:
                        idx = labels.index(defense_name)
                        ordered_handles.append(handles[idx])
                        ordered_labels.append(defense_name)

                # add other labels
                for label in other_labels:
                    if label in labels:
                        idx = labels.index(label)
                        ordered_handles.append(handles[idx])
                        ordered_labels.append(label)

                ax.legend(ordered_handles, ordered_labels,
                          fontsize=10,
                          loc='upper center',
                          bbox_to_anchor=(1.8, -0.15),
                          ncol=len(defense_colors) + 2)

    # adjust layout with custom rect and no suptitle
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])

    if save_name:
        plt.savefig(save_name, format='jpg', bbox_inches='tight')

    return fig, axes


def parse_metric_cell(cell_str: str) -> tuple:
    """parses a cell containing mean ± std [hdi_lower, hdi_upper]."""
    pattern = r'([\d.]+)\s*±\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    match = re.match(pattern, cell_str)

    if match:
        mean = float(match.group(1))
        std = float(match.group(2))
        hdi_lower = float(match.group(3))
        hdi_upper = float(match.group(4))
        return mean, std, hdi_lower, hdi_upper
    return None, None, None, None


def format_cell(mean: float, hdi_lower: float, hdi_upper: float) -> str:
    """formats a cell with mean and HDI range."""
    return f"{mean:.3f} [{hdi_lower:.3f}, {hdi_upper:.3f}]"


def standardize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """standardizes the threat model and defense model names."""
    # standardize threat model names
    threat_model_map = {
        'logreg': 'Logistic Regression',
        'svm': 'SVM',
        'roberta': 'RoBERTa'
    }

    # standardize defense model names - matching exact strings from input
    defense_model_map = {
        'Gemma-2': 'Gemma-2',
        'Llama-3.1': 'Llama-3.1',
        'Ministral': 'Ministral',
        'Claude-3.5': 'Claude-3.5',
        'GPT-4o': 'GPT-4o'
    }

    df = df.copy()
    if 'Threat Model' in df.columns:
        df['Threat Model'] = df['Threat Model'].map(
            lambda x: threat_model_map.get(x, x))
    if 'Defense Model' in df.columns:
        df['Defense Model'] = df['Defense Model'].map(
            lambda x: defense_model_map.get(x, x))
    return df


def sort_dataframe(df: pd.DataFrame, include_threat_model: bool = True) -> pd.DataFrame:
    """sorts the dataframe by threat model and/or defense model in the specified order."""
    # define sort orders
    threat_model_order = ['Logistic Regression', 'SVM', 'RoBERTa']
    defense_model_order = ['Gemma-2', 'Llama-3.1', 'Ministral', 'Claude-3.5', 'GPT-4o']

    df = df.copy()

    # create categorical columns for sorting
    df['Defense Model'] = pd.Categorical(
        df['Defense Model'],
        categories=defense_model_order,
        ordered=True
    )

    if include_threat_model and 'Threat Model' in df.columns:
        df['Threat Model'] = pd.Categorical(
            df['Threat Model'],
            categories=threat_model_order,
            ordered=True
        )
        return df.sort_values(['Threat Model', 'Defense Model'])

    return df.sort_values('Defense Model')


def parse_results_df(df: pd.DataFrame) -> tuple:
    """Parses the results dataframe into four separate dataframes:
    - ebg_df: Metrics for EBG corpus (includes threat model information)
    - rj_df: Metrics for RJ corpus (includes threat model information)
    - ebg_quality_df: Quality metrics for EBG corpus (without threat model), columns reordered as:
                      'Defense Model', 'BERTScore ↑', 'SBERT ↑', 'PINC ↑'
    - rj_quality_df: Quality metrics for RJ corpus (without threat model), columns reordered as:
                      'Defense Model', 'BERTScore ↑', 'SBERT ↑', 'PINC ↑'
    """
    # create empty lists to store parsed data
    ebg_data = []
    rj_data = []
    ebg_quality_data = []
    rj_quality_data = []

    # standardize model names first
    df = standardize_model_names(df)

    for _, row in df.iterrows():
        corpus = row['Corpus']
        threat_model = row['Threat Model']
        defense_model = row['Defense Model']  # use the original value if mapping fails

        # parse metrics from the cells using the provided parse_metric_cell function
        acc1_mean, _, acc1_lower, acc1_upper = parse_metric_cell(row['accuracy@1 ↓'])
        acc5_mean, _, acc5_lower, acc5_upper = parse_metric_cell(row['accuracy@5 ↓'])
        conf_mean, _, conf_lower, conf_upper = parse_metric_cell(row['true_class_confidence ↓'])
        ent_mean, _, ent_lower, ent_upper = parse_metric_cell(row['entropy ↑'])
        pinc_mean, _, pinc_lower, pinc_upper = parse_metric_cell(row['pinc ↑'])
        bert_mean, _, bert_lower, bert_upper = parse_metric_cell(row['bertscore ↑'])
        sbert_mean, _, sbert_lower, sbert_upper = parse_metric_cell(row['sbert ↑'])

        # create metrics row with threat model included
        metrics_row = {
            'Threat Model': threat_model,
            'Defense Model': defense_model,
            'Accuracy@1 ↓': format_cell(acc1_mean, acc1_lower, acc1_upper),
            'Accuracy@5 ↓': format_cell(acc5_mean, acc5_lower, acc5_upper),
            'True Class Confidence ↓': format_cell(conf_mean, conf_lower, conf_upper),
            'Entropy ↑': format_cell(ent_mean, ent_lower, ent_upper)
        }

        # create quality metrics row without threat model info
        quality_row = {
            'Defense Model': defense_model,
            'PINC ↑': format_cell(pinc_mean, pinc_lower, pinc_upper),
            'BERTScore ↑': format_cell(bert_mean, bert_lower, bert_upper),
            'SBERT ↑': format_cell(sbert_mean, sbert_lower, sbert_upper)
        }

        # append rows to respective lists based on corpus value
        if corpus == 'EBG':
            ebg_data.append(metrics_row)
            ebg_quality_data.append(quality_row)
        elif corpus == 'RJ':
            rj_data.append(metrics_row)
            rj_quality_data.append(quality_row)

    # create DataFrames
    ebg_df = pd.DataFrame(ebg_data)
    rj_df = pd.DataFrame(rj_data)
    ebg_quality_df = pd.DataFrame(ebg_quality_data).drop_duplicates()
    rj_quality_df = pd.DataFrame(rj_quality_data).drop_duplicates()

    # sort dataframes with consistent ordering
    ebg_df = sort_dataframe(ebg_df, include_threat_model=True)
    rj_df = sort_dataframe(rj_df, include_threat_model=True)
    ebg_quality_df = sort_dataframe(ebg_quality_df, include_threat_model=False)
    rj_quality_df = sort_dataframe(rj_quality_df, include_threat_model=False)

    # Reorder quality metric columns: Defense Model, BERTScore, SBERT, PINC
    quality_order = ['Defense Model', 'BERTScore ↑', 'SBERT ↑', 'PINC ↑']
    ebg_quality_df = ebg_quality_df[quality_order]
    rj_quality_df = rj_quality_df[quality_order]

    return ebg_df, rj_df, ebg_quality_df, rj_quality_df



#!/usr/bin/env python
"""
Script to prepare data for RQ3.2 exemplar length analysis.

This script extracts exemplar lengths from prompt files and combines them
with evaluation metrics to create dataframes for each LLM vs attribution model
scenario on different corpora.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

# Constants
CORPORA = ['ebg', 'rj']
THREAT_MODELS = ['logreg', 'svm', 'roberta']
LLMS = ['gemma-2', 'llama-3.1', 'ministral', 'claude-3.5', 'gpt-4o']
METRICS = ['binary_acc1', 'binary_acc5', 'true_class_confidence', 'entropy', 'bertscore', 'pinc']


def extract_exemplar_lengths(prompt_dir: Path) -> Dict[int, int]:
    """extract exemplar lengths from prompt files.

    args:
        prompt_dir: directory containing prompt json files

    returns:
        dictionary mapping prompt index to exemplar length
    """
    lengths = {}
    for prompt_file in prompt_dir.glob("prompt*.json"):  # Only include prompt files, exclude metadata.json
        try:
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
                print(f"Warning: No word_count found in metadata for prompt {prompt_idx}")
        except Exception as e:
            print(f"Error processing {prompt_file}: {e}")

    return lengths


def get_sample_metrics(eval_file: Path):
    """extract sample-level metrics from evaluation file.

    args:
        eval_file: path to evaluation json file

    returns:
        tuple of binary accuracy@1, accuracy@5 lists, true confidence list, and entropy list
    """
    # Initialize with None values
    binary_acc1_list = None
    binary_acc5_list = None
    true_conf_list = None
    entropy_list = None

    try:
        with open(eval_file, "r") as f:
            detailed_eval_data = json.load(f)

        if "example_metrics" in detailed_eval_data:
            example_metrics = detailed_eval_data["example_metrics"]

            # Create binary accuracy metrics
            binary_acc1_list = [
                1 if ex["transformed_rank"] == 0 else 0
                for ex in example_metrics
            ]
            binary_acc5_list = [
                1 if ex["transformed_rank"] < 5 else 0
                for ex in example_metrics
            ]

            # Extract true class confidence and calculate entropy
            true_conf_list = []
            entropy_list = []

            for ex in example_metrics:
                true_label = ex["true_label"]
                trans_probs = np.array(ex["trans_probs"])

                # Get confidence for true class
                true_conf_list.append(float(trans_probs[true_label]))

                # Calculate entropy
                epsilon = 1e-10
                probs_safe = np.clip(trans_probs, epsilon, 1.0)
                entropy = -np.sum(probs_safe * np.log2(probs_safe))
                entropy_list.append(float(entropy))

    except Exception as e:
        print(f"Warning: Could not extract metrics from {eval_file}: {e}")

    return binary_acc1_list, binary_acc5_list, true_conf_list, entropy_list


def get_quality_metrics(seed_file: Path, threat_model: str, doc_idx: int):
    """extract quality metrics from seed file.

    args:
        seed_file: path to seed json file
        threat_model: the attribution model (logreg, svm, roberta)
        doc_idx: index of the document

    returns:
        tuple of bertscore_f1 and pinc values for the specified document
    """
    bertscore = None
    pinc = None

    try:
        with open(seed_file, "r") as f:
            data = json.load(f)

        quality = data.get('results', {}).get(threat_model, {}).get('quality', {})

        # Get BERTScore F1
        bertscore_data = quality.get('bertscore', {})
        if 'bertscore_individual' in bertscore_data and doc_idx < len(bertscore_data['bertscore_individual']):
            bertscore = bertscore_data['bertscore_individual'][doc_idx].get('f1')
        else:
            # Fall back to average if individual scores are not available
            bertscore = bertscore_data.get('bertscore_f1_avg')

        # Get PINC scores (average of pinc_1 through pinc_4)
        pinc_data = quality.get('pinc', {})
        pinc_values = []

        for k in range(1, 5):
            pinc_key = f'pinc_{k}_scores'
            if pinc_key in pinc_data and doc_idx < len(pinc_data[pinc_key]):
                pinc_values.append(pinc_data[pinc_key][doc_idx])

        if pinc_values:
            pinc = np.mean(pinc_values)
        else:
            # Fall back to average of averages
            pinc = np.mean([
                pinc_data.get(f'pinc_{k}_avg', np.nan)
                for k in range(1, 5)
            ])

    except Exception as e:
        print(f"Warning: Could not extract quality metrics from {seed_file} for document {doc_idx}: {e}")

    return bertscore, pinc


def prepare_data(eval_base_dir: str,
                llm_outputs_dir: str,
                prompts_dir: str,
                output_dir: str):
    """prepare dataframes containing exemplar length data and evaluation metrics.

    args:
        eval_base_dir: base directory for evaluation results
        llm_outputs_dir: directory containing llm outputs
        prompts_dir: directory containing prompt files
        output_dir: directory to save prepared data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # mapping from directory names to standardized model names
    model_mapping = {
        'gpt-4o-2024-08-06': 'gpt-4o',
        'claude-3-5-sonnet-20241022': 'claude-3.5',
        'gemma-2-9b-it': 'gemma-2',
        'llama-3.1-8b-instruct': 'llama-3.1',
        'ministral-8b-instruct-2410': 'ministral'
    }

    # reverse mapping for finding directories
    dir_mapping = {v: k for k, v in model_mapping.items()}

    for corpus in CORPORA:
        print(f"Processing corpus: {corpus}")

        for rq_folder in Path(llm_outputs_dir).joinpath(corpus, 'rq3').glob("rq3.2_imitation_variable_length"):
            # get the experiment name
            experiment_name = rq_folder.name
            print(f"Processing experiment: {experiment_name}")

            # load prompt lengths from the corresponding prompts directory
            prompt_dir = Path(prompts_dir) / experiment_name
            if not prompt_dir.exists():
                print(f"Warning: Prompt directory not found: {prompt_dir}")
                continue

            prompt_lengths = extract_exemplar_lengths(prompt_dir)
            print(f"Extracted exemplar lengths for {len(prompt_lengths)} prompts")

            # create a directory for scenario data files
            scenario_dir = Path(output_dir) / corpus / experiment_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # track all available data for summary
            summary_data = []

            # get the actual model directories available in the evaluation directory
            available_model_dirs = {}
            for model_dir in Path(eval_base_dir, corpus, 'rq3', experiment_name).glob("*"):
                if model_dir.is_dir():
                    # map directory name to standardized name if possible
                    for k, v in model_mapping.items():
                        if k in model_dir.name:
                            available_model_dirs[v] = model_dir.name
                            break

            print(f"Found model directories: {available_model_dirs}")

            for threat_model in THREAT_MODELS:
                for llm in LLMS:
                    print(f"Processing {corpus}/{threat_model}/{llm}")

                    # collect data for this specific scenario
                    scenario_data = []

                    # get the actual directory name from mapping or try original name
                    model_dir_name = available_model_dirs.get(llm, dir_mapping.get(llm, llm))

                    # check the evaluation directory for this scenario
                    eval_dir = Path(eval_base_dir) / corpus / 'rq3' / experiment_name / model_dir_name

                    if not eval_dir.exists():
                        print(f"Warning: Evaluation directory not found: {eval_dir}")
                        continue

                    # load evaluation summary data
                    eval_summary_path = eval_dir / "evaluation.json"
                    if not eval_summary_path.exists():
                        print(f"Warning: Evaluation summary not found: {eval_summary_path}")
                        continue

                    with open(eval_summary_path) as f:
                        all_eval_data = json.load(f)

                    # process each seed
                    for seed_file in eval_dir.glob("seed_*.json"):
                        seed = seed_file.stem.split("_")[-1]

                        if seed not in all_eval_data:
                            print(f"Warning: Seed {seed} not found in evaluation summary")
                            continue

                        # get sample-level metrics from detailed evaluation
                        binary_acc1_list, binary_acc5_list, true_conf_list, entropy_list = get_sample_metrics(seed_file)

                        # find corresponding llm output file using actual directory name
                        llm_output_file = Path(llm_outputs_dir) / corpus / 'rq3' / experiment_name / model_dir_name / f"seed_{seed}.json"

                        if not llm_output_file.exists():
                            print(f"Warning: LLM output file not found: {llm_output_file}")
                            continue

                        with open(llm_output_file) as f:
                            llm_data = json.load(f)

                        # extract evaluation metrics for this seed
                        record = all_eval_data[seed]

                        if threat_model not in record:
                            print(f"Warning: Threat model {threat_model} not found in seed {seed}")
                            continue

                        # process each document in the llm output
                        if isinstance(llm_data, list) and llm_data:
                            for doc_idx, document in enumerate(llm_data):
                                if "prompt_index" not in document:
                                    print(f"Warning: No prompt_index in document {doc_idx} for seed {seed}")
                                    continue

                                # fix mismatched prompt indices by subtracting 1
                                prompt_idx = document["prompt_index"]
                                if prompt_idx >= len(prompt_lengths):
                                    prompt_idx = prompt_idx - 1
                                length = prompt_lengths.get(prompt_idx, -1)

                                if length == -1:
                                    print(f"Warning: Could not find exemplar length for prompt index {prompt_idx} (original: {document['prompt_index']})")
                                    continue

                                # get quality metrics directly from the seed file
                                bertscore, pinc = get_quality_metrics(seed_file, threat_model, doc_idx)

                                # create row with all metrics
                                row = {
                                    "corpus": corpus,
                                    "llm": llm,  # use standardized name
                                    "llm_dir": model_dir_name,  # save the directory name for reference
                                    "threat_model": threat_model,
                                    "seed": seed,
                                    "document_idx": doc_idx,
                                    "exemplar_length": length,
                                    "binary_acc1": binary_acc1_list[doc_idx] if binary_acc1_list and doc_idx < len(binary_acc1_list) else None,
                                    "binary_acc5": binary_acc5_list[doc_idx] if binary_acc5_list and doc_idx < len(binary_acc5_list) else None,
                                    "true_class_confidence": true_conf_list[doc_idx] if true_conf_list and doc_idx < len(true_conf_list) else None,
                                    "entropy": entropy_list[doc_idx] if entropy_list and doc_idx < len(entropy_list) else None,
                                    "bertscore": bertscore,
                                    "pinc": pinc,
                                    "sample_id": prompt_idx
                                }

                                scenario_data.append(row)
                                summary_data.append(row)

                    # save data for this specific scenario if we have data points
                    if scenario_data:
                        scenario_df = pd.DataFrame(scenario_data)
                        file_path = scenario_dir / f"{threat_model}_{llm}_data.csv"
                        scenario_df.to_csv(file_path, index=False)
                        print(f"Saved {len(scenario_data)} data points to {file_path}")

            # save combined data for the corpus/experiment
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = scenario_dir / "all_data.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"Saved combined data with {len(summary_data)} points to {summary_path}")

                # also save summary statistics
                stats_df = summary_df.groupby(['llm', 'threat_model']).agg({
                    'exemplar_length': ['count', 'min', 'max', 'mean', 'nunique'],
                    'binary_acc1': ['mean'] if 'binary_acc1' in summary_df.columns else [],
                    'bertscore': ['mean', 'std']
                })

                stats_path = scenario_dir / "data_statistics.csv"
                stats_df.to_csv(stats_path)
                print(f"Saved data statistics to {stats_path}")


def main():
    """main function to prepare data for exemplar length analysis."""
    parser = argparse.ArgumentParser(description="Prepare data for exemplar length analysis")
    parser.add_argument("--eval_dir", type=str, default="defense_evaluation",
                       help="Directory containing evaluation results")
    parser.add_argument("--llm_dir", type=str, default="llm_outputs",
                       help="Directory containing LLM outputs")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                       help="Directory containing prompt files")
    parser.add_argument("--output_dir", type=str, default="results/prepared_data_rq3.2",
                       help="Directory to save prepared data")
    parser.add_argument("--verbose", action="store_true",
                       help="Print more detailed debug information")

    args = parser.parse_args()

    prepare_data(args.eval_dir, args.llm_dir, args.prompt_dir, args.output_dir)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
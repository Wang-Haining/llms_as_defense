"""
Validation tool for LLM defense experiment outputs.

This script validates experiment outputs by:
1. Checking if each experiment has the expected number of seed files
2. Verifying if each seed file contains the expected number of samples
3. Generating a detailed report of any missing data

Usage:
    python validate_llm_outputs.py  # checks default models
    python validate_llm_outputs.py --models "meta-llama/Llama-3.1-8B"
    python validate_llm_outputs.py --models claude-3-5-sonnet-20241022 gpt-4-0125-preview
    python validate_llm_outputs.py --rq rq1.1_basic_paraphrase  # check specific research question
    python validate_llm_outputs.py --rq rq1.1_basic_paraphrase --models "meta-llama/Llama-3.1-8B" "google/gemma-2-9b-it"

The tool will print a detailed validation report to console.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from utils import CORPORA, LLMS, load_corpus

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_model_name(model: str) -> str:
    """normalize model names for comparison with directory names"""
    # extract last part after / if present
    model_name = model.split('/')[-1].lower()
    return model_name


class ExperimentValidator:
    def __init__(
        self,
        base_dir: str,
        expected_seeds: int = 5,
        models: List[str] = None,
        research_question: str = None
    ):
        self.base_dir = Path(base_dir)
        self.expected_seeds = expected_seeds
        self.expected_counts = self.load_expected_counts()
        self.research_question = research_question

        # normalize model names for matching
        self.models_to_check = None
        if models:
            self.models_to_check = [normalize_model_name(m) for m in models]
            logger.info(f"Will check models: {', '.join(models)}")
        else:
            # use defaults
            self.models_to_check = [normalize_model_name(m) for m in LLMS]
            logger.info(
                f"Using default models for validation: {', '.join(LLMS)}"
            )

        if research_question:
            logger.info(f"Will only validate experiments for {research_question}")

    def should_check_model(self, model_dir: str) -> bool:
        """determine if a model directory should be validated"""
        model_name = normalize_model_name(model_dir)
        return any(target in model_name for target in self.models_to_check)

    def should_check_experiment(self, config: Dict) -> bool:
        """determine if an experiment should be validated based on RQ filter"""
        if not self.research_question:
            return True
        return config.get('sub_question', '').lower() == self.research_question.lower()

    @staticmethod
    def load_expected_counts() -> Dict[str, int]:
        """get expected sample counts for each corpus"""
        corpus_counts = {}
        for corpus in CORPORA:
            _, _, test_texts, _ = load_corpus(corpus=corpus, task="no_protection")
            corpus_counts[corpus] = len(test_texts)
        return corpus_counts

    def get_seed_files(self, experiment_dir: Path) -> Set[Path]:
        """get all seed files in an experiment directory"""
        return set(experiment_dir.glob('seed_*.json'))

    def validate_experiment(self, config_file: Path) -> List[Dict]:
        """validate a single experiment directory"""
        issues = []
        experiment_dir = config_file.parent

        try:
            # load experiment config
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # check if we should validate this experiment
            if not self.should_check_model(config['model']['name']):
                return []

            if not self.should_check_experiment(config):
                return []

            corpus = config['corpus']
            expected_count = self.expected_counts[corpus]

            # check number of seed files
            seed_files = self.get_seed_files(experiment_dir)
            if len(seed_files) != self.expected_seeds:
                issues.append({
                    'type': 'missing_seed_files',
                    'experiment': str(experiment_dir),
                    'corpus': corpus,
                    'research_question': config['research_question'],
                    'sub_question': config['sub_question'],
                    'model': config['model']['name'],
                    'expected_seeds': self.expected_seeds,
                    'actual_seeds': len(seed_files),
                    'missing_seeds': self.expected_seeds - len(seed_files)
                })

            # check sample counts in each seed file
            for seed_file in seed_files:
                try:
                    with open(seed_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)

                    actual_count = len(results)
                    if actual_count != expected_count:
                        issues.append({
                            'type': 'missing_samples',
                            'experiment': str(experiment_dir),
                            'corpus': corpus,
                            'research_question': config['research_question'],
                            'sub_question': config['sub_question'],
                            'model': config['model']['name'],
                            'seed_file': seed_file.name,
                            'expected_samples': expected_count,
                            'actual_samples': actual_count,
                            'missing_samples': expected_count - actual_count
                        })

                except Exception as e:
                    logger.error(f"Error processing {seed_file}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing experiment {experiment_dir}: {str(e)}")

        return issues

    def scan_all_experiments(self) -> List[Dict]:
        """scan all experiment outputs and validate"""
        all_issues = []

        # find all experiment_config.json files
        glob_pattern = '**/'
        if self.research_question:
            rq_base = self.research_question.split('.')[0]
            glob_pattern = f'**/{rq_base}/{self.research_question}/**/experiment_config.json'
        else:
            glob_pattern = '**/experiment_config.json'

        for config_file in self.base_dir.glob(glob_pattern):
            issues = self.validate_experiment(config_file)
            all_issues.extend(issues)

        return all_issues


def generate_report(issues: List[Dict]) -> Tuple[str, Dict]:
    """generate a summary report and statistics"""
    if not issues:
        return "No validation issues found.", {}

    # aggregate statistics
    stats = {
        'missing_seed_files': defaultdict(lambda: defaultdict(int)),
        'missing_samples': defaultdict(lambda: defaultdict(int))
    }

    for issue in issues:
        corpus = issue['corpus']
        rq = issue['sub_question']
        issue_type = issue['type']
        stats[issue_type][corpus][rq] += 1

    # format report
    report_lines = ["=== Validation Issues Report ===\n"]

    # report missing seed files
    if stats['missing_seed_files']:
        report_lines.append("\n=== Missing Seed Files ===")
        for corpus in sorted(stats['missing_seed_files'].keys()):
            report_lines.append(f"\nCorpus: {corpus}")
            for rq in sorted(stats['missing_seed_files'][corpus].keys()):
                report_lines.append(
                    f"  {rq}: {stats['missing_seed_files'][corpus][rq]} experiments affected"
                )

    # report missing samples
    if stats['missing_samples']:
        report_lines.append("\n=== Missing Samples ===")
        for corpus in sorted(stats['missing_samples'].keys()):
            report_lines.append(f"\nCorpus: {corpus}")
            for rq in sorted(stats['missing_samples'][corpus].keys()):
                report_lines.append(
                    f"  {rq}: {stats['missing_samples'][corpus][rq]} seed files affected"
                )

    # detailed issues
    report_lines.append("\n=== Detailed Issues ===")

    # group by issue type
    for issue in sorted(issues,
                       key=lambda x: (x['type'], x['corpus'], x['sub_question'])):
        if issue['type'] == 'missing_seed_files':
            report_lines.append(
                f"\n{issue['corpus']}/{issue['sub_question']}/{issue['model']}"
                f"\n  - Type: Missing seed files"
                f"\n  - Expected seeds: {issue['expected_seeds']}"
                f"\n  - Actual seeds: {issue['actual_seeds']}"
                f"\n  - Missing: {issue['missing_seeds']}"
            )
        else:  # missing_samples
            report_lines.append(
                f"\n{issue['corpus']}/{issue['sub_question']}/{issue['model']}"
                f"\n  - Type: Missing samples"
                f"\n  - Seed file: {issue['seed_file']}"
                f"\n  - Expected samples: {issue['expected_samples']}"
                f"\n  - Actual samples: {issue['actual_samples']}"
                f"\n  - Missing: {issue['missing_samples']}"
            )

    return "\n".join(report_lines), dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description='Validate LLM defense experiment outputs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--base_dir',
        default='llm_outputs',
        help='Base directory containing experiment outputs'
    )
    parser.add_argument(
        '--expected_seeds',
        type=int,
        default=5,
        help='Expected number of seed files per experiment'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        help='Space-separated list of models to check. Defaults to llama and gemma models.'
    )
    parser.add_argument(
        '--rq',
        type=str,
        help='Research question to validate (e.g., RQ1.1). If not provided, validates all RQs.'
    )
    args = parser.parse_args()

    logger.info("Starting validation scan...")
    validator = ExperimentValidator(
        args.base_dir,
        args.expected_seeds,
        args.models,
        args.rq
    )
    issues = validator.scan_all_experiments()

    report, stats = generate_report(issues)
    print(report)

    if issues:
        logger.warning(f"Validation complete. Found {len(issues)} issues.")
    else:
        logger.info("Validation complete. No issues found.")


if __name__ == "__main__":
    main()
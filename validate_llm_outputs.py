"""
validate.py - Validation tool for LLM defense experiment outputs.

This script validates experiment outputs by:
1. Checking if each experiment has the expected number of seed files
2. Verifying if each seed file contains the expected number of samples
3. Generating a detailed report of any missing data

Usage:
    python validate_llm_outputs.py

The tool will generate a validation_report.json with detailed findings.
"""

import json
import logging
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

from utils import load_corpus  # assuming same utils as main script

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentValidator:
    def __init__(self, base_dir: str, expected_seeds: int = 5):
        self.base_dir = Path(base_dir)
        self.expected_seeds = expected_seeds
        self.expected_counts = self.load_expected_counts()

    @staticmethod
    def load_expected_counts() -> Dict[str, int]:
        """get expected sample counts for each corpus"""
        corpus_counts = {}
        for corpus in ['rj', 'ebg', 'lcmc']:
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
        for config_file in self.base_dir.glob('**/experiment_config.json'):
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
    args = parser.parse_args()

    logger.info("Starting validation scan...")
    validator = ExperimentValidator(args.base_dir, args.expected_seeds)
    issues = validator.scan_all_experiments()

    report, stats = generate_report(issues)
    print(report)

    # save detailed report
    output_file = "validation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'issues': issues,
                'statistics': stats
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    if issues:
        logger.warning(
            f"Validation complete. Found {len(issues)} issues. "
            f"See {output_file} for details."
        )
    else:
        logger.info("Validation complete. No issues found.")


if __name__ == "__main__":
    main()
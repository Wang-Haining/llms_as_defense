# Analysis of Exemplar Length Effect on LLM-based Imitation Defense

## Overview

This analysis examines the effect of exemplar length (500, 1000, and 2500 words) on the effectiveness of LLM-based imitation as a defense against authorship attribution attacks. The analysis covers:

- 2 corpora: EBG, RJ
- 3 threat models: LOGREG, ROBERTA, SVM
- 5 LLMs: Claude-3.5, Gemma-2, Gpt-4O, Llama-3.1, Ministral
- 6 metrics: accuracy@1, accuracy@5, bertscore, entropy, pinc, true_class_confidence

## Note on Bayesian Interpretation

This analysis uses Bayesian methods to estimate the effect of exemplar length on defense effectiveness. Key concepts:

- **Posterior probability**: In Bayesian analysis, we calculate the probability of an effect based on observed data and prior beliefs.
- **95% HDI (Highest Density Interval)**: The interval containing 95% of the posterior probability mass, showing where the true effect most likely lies.
- **Credible Effect**: When the 95% HDI excludes zero, providing strong evidence that the effect is real.
- **Posterior prob. beneficial**: The probability that longer exemplars improve a metric.

## Overall Findings

Out of 180 total tests:

- **Credible improvements with longer exemplars**: 27 (15.0%)
- **Credible deteriorations with longer exemplars**: 21 (11.7%)
- **Inconclusive results**: 132 (73.3%)
- **Practically equivalent results**: 0 (0.0%)


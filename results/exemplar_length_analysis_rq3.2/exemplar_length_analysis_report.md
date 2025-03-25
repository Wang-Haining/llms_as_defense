# Analysis of Exemplar Length Effect on LLM-based Imitation Defense

## Overview

This analysis examines the effect of *actual* exemplar length on the effectiveness of LLM-based imitation as a defense against authorship attribution attacks. We extract the exemplar text from each seed_{seed}.json, count its words, and model how that length impacts various metrics.

- 2 corpora: EBG, RJ
- 3 threat models: LOGREG, ROBERTA, SVM
- 5 LLMs: Claude-3.5, Gemma-2, Gpt-4O, Llama-3.1, Ministral
- 6 metrics: accuracy@1, accuracy@5, bertscore, entropy, pinc, true_class_confidence

## Overall Findings

Out of 180 total tests:

- **Credible improvements with longer exemplars**: 0 (0.0%)
- **Credible deteriorations with longer exemplars**: 0 (0.0%)
- **Inconclusive results**: 180 (100.0%)
- **Practically equivalent results**: 0 (0.0%)


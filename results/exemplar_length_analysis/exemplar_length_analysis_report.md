# Analysis of Exemplar Length Effect on LLM-based Imitation Defense

## Overview

This analysis examines the effect of exemplar length (500, 1000, and 2500 words) on the effectiveness of LLM-based imitation as a defense against authorship attribution attacks. The analysis covers:

- 2 corpora: EBG, RJ
- 3 threat models: LOGREG, ROBERTA, SVM
- 5 LLMs: Claude-3.5, Gemma-2, Gpt-4O, Llama-3.1, Ministral
- 6 metrics: accuracy@1, accuracy@5, bertscore, entropy, pinc, true_class_confidence

## Overall Findings

Out of 180 total tests:

- **Significant improvements with longer exemplars**: 11 (6.1%)
- **Significant deteriorations with longer exemplars**: 1 (0.6%)
- **Inconclusive results**: 168 (93.3%)
- **Practically equivalent results**: 0 (0.0%)

## Key Findings by Metric

### Top-1 Accuracy

- Average effect (500→2500): 0.0579
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Significant improvements: 2/30 (6.7%)
- Significant deteriorations: 0/30 (0.0%)

- Best LLM: **Llama-3.1** (average effect: -0.0902)
- Worst LLM: **Gpt-4O** (average effect: 0.2751)

#### Notable Improvements:

- Gpt-4O against SVM on EBG: Effect = 0.9360, p = 0.0014
- Gpt-4O against LOGREG on EBG: Effect = 0.8575, p = 0.0025

### Top-5 Accuracy

- Average effect (500→2500): 0.0739
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Significant improvements: 2/30 (6.7%)
- Significant deteriorations: 0/30 (0.0%)

- Best LLM: **Ministral** (average effect: -0.0703)
- Worst LLM: **Gpt-4O** (average effect: 0.2326)

#### Notable Improvements:

- Gpt-4O against SVM on EBG: Effect = 0.8973, p = 0.0009
- Gpt-4O against LOGREG on EBG: Effect = 0.6916, p = 0.0029

### True Class Confidence

- Average effect (500→2500): 0.0405
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Significant improvements: 1/30 (3.3%)
- Significant deteriorations: 0/30 (0.0%)

- Best LLM: **Claude-3.5** (average effect: -0.0325)
- Worst LLM: **Gpt-4O** (average effect: 0.1542)

#### Notable Improvements:

- Gpt-4O against LOGREG on EBG: Effect = 0.5911, p = 0.0147

### Prediction Entropy

- Average effect (500→2500): 0.2938
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Significant improvements: 6/30 (20.0%)
- Significant deteriorations: 1/30 (3.3%)

- Best LLM: **Llama-3.1** (average effect: 1.4851)
- Worst LLM: **Gemma-2** (average effect: -0.9139)

#### Notable Improvements:

- Llama-3.1 against SVM on EBG: Effect = 3.4179, p = 0.9858
- Llama-3.1 against LOGREG on EBG: Effect = 3.3921, p = 0.9928
- Gpt-4O against SVM on EBG: Effect = 3.1859, p = 0.9926

#### Notable Deteriorations:

- Gemma-2 against ROBERTA on EBG: Effect = -2.7308, p = 0.9778

### bertscore

- Average effect (500→2500): 0.0414
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Significant improvements: 0/30 (0.0%)
- Significant deteriorations: 0/30 (0.0%)

- Best LLM: **Gpt-4O** (average effect: 0.1043)
- Worst LLM: **Ministral** (average effect: 0.0063)

### pinc

- Average effect (500→2500): -0.0301
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Significant improvements: 0/30 (0.0%)
- Significant deteriorations: 0/30 (0.0%)

- Best LLM: **Llama-3.1** (average effect: 0.0230)
- Worst LLM: **Gpt-4O** (average effect: -0.1323)

## Key Findings by LLM

### Gemma-2

- Significant improvements: 0/36 (0.0%)
- Significant deteriorations: 1/36 (2.8%)
- Inconclusive results: 35/36 (97.2%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0123)
  - Least improved: **entropy** (avg effect: -0.9139)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.0025)
  - Least improved: **accuracy@1** (avg effect: 0.0810)

#### Notable Deteriorations:

- entropy against ROBERTA on EBG: Effect = -2.7308, p = 0.9778

### Llama-3.1

- Significant improvements: 2/36 (5.6%)
- Significant deteriorations: 0/36 (0.0%)
- Inconclusive results: 34/36 (94.4%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 1.4851)
  - Least improved: **bertscore** (avg effect: 0.0074)

- For 'lower is better' metrics:
  - Most improved: **accuracy@1** (avg effect: -0.0902)
  - Least improved: **accuracy@5** (avg effect: 0.0615)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 3.4179, p = 0.9858
- entropy against LOGREG on EBG: Effect = 3.3921, p = 0.9928

### Ministral

- Significant improvements: 0/36 (0.0%)
- Significant deteriorations: 0/36 (0.0%)
- Inconclusive results: 36/36 (100.0%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0063)
  - Least improved: **entropy** (avg effect: -0.3936)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.0703)
  - Least improved: **accuracy@1** (avg effect: 0.0412)

### Claude-3.5

- Significant improvements: 2/36 (5.6%)
- Significant deteriorations: 0/36 (0.0%)
- Inconclusive results: 34/36 (94.4%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.9545)
  - Least improved: **pinc** (avg effect: -0.0094)

- For 'lower is better' metrics:
  - Most improved: **true_class_confidence** (avg effect: -0.0325)
  - Least improved: **accuracy@5** (avg effect: 0.1484)

#### Notable Improvements:

- entropy against ROBERTA on RJ: Effect = 1.9797, p = 0.9644
- entropy against SVM on RJ: Effect = 1.6026, p = 0.9505

### Gpt-4O

- Significant improvements: 7/36 (19.4%)
- Significant deteriorations: 0/36 (0.0%)
- Inconclusive results: 29/36 (80.6%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.3368)
  - Least improved: **pinc** (avg effect: -0.1323)

- For 'lower is better' metrics:
  - Most improved: **true_class_confidence** (avg effect: 0.1542)
  - Least improved: **accuracy@1** (avg effect: 0.2751)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 3.1859, p = 0.9926
- entropy against LOGREG on EBG: Effect = 2.1971, p = 0.9634
- accuracy@1 against SVM on EBG: Effect = 0.9360, p = 0.0014

## Key Findings by Threat Model

### LOGREG

- Significant improvements: 5/60 (8.3%)
- Significant deteriorations: 0/60 (0.0%)

#### Average Effect by Metric:

- accuracy@1: 0.1697 (negative)
- accuracy@5: 0.0863 (negative)
- bertscore: 0.0416 (positive)
- entropy: 0.3893 (positive)
- pinc: -0.0301 (negative)
- true_class_confidence: 0.0964 (negative)

#### Effect by LLM:

- Most affected LLM: **Gpt-4O** (avg effect: 0.3831)

### SVM

- Significant improvements: 5/60 (8.3%)
- Significant deteriorations: 0/60 (0.0%)

#### Average Effect by Metric:

- accuracy@1: 0.0386 (negative)
- accuracy@5: 0.1187 (negative)
- bertscore: 0.0413 (positive)
- entropy: 0.6862 (positive)
- pinc: -0.0301 (negative)
- true_class_confidence: 0.0192 (negative)

#### Effect by LLM:

- Most affected LLM: **Llama-3.1** (avg effect: 0.3696)

### ROBERTA

- Significant improvements: 1/60 (1.7%)
- Significant deteriorations: 1/60 (1.7%)

#### Average Effect by Metric:

- accuracy@1: -0.0347 (positive)
- accuracy@5: 0.0168 (negative)
- bertscore: 0.0413 (positive)
- entropy: -0.1941 (negative)
- pinc: -0.0301 (negative)
- true_class_confidence: 0.0060 (negative)

#### Effect by LLM:

- Most affected LLM: **Gemma-2** (avg effect: -0.3156)

## Findings by Corpus

### EBG

- Significant improvements: 9/90 (10.0%)
- Significant deteriorations: 1/90 (1.1%)

Overall, longer exemplars tend to be **more effective** on this corpus.

### RJ

- Significant improvements: 2/90 (2.2%)
- Significant deteriorations: 0/90 (0.0%)

Overall, longer exemplars tend to be **more effective** on this corpus.

## Conclusion

Based on the analysis, **longer exemplars generally improve the effectiveness** of LLM-based imitation as a defense against authorship attribution attacks. The improvement is most pronounced for:

- **entropy** (average effect: 0.2938)
- **bertscore** (average effect: 0.0414)

And the LLMs that benefit most from longer exemplars are:

- **Gpt-4O** (7/36 metrics improved, 19.4%)
- **Claude-3.5** (2/36 metrics improved, 5.6%)
- **Llama-3.1** (2/36 metrics improved, 5.6%)

These findings suggest that the optimal exemplar length may depend on the specific LLM, threat model, and metric of interest. Users should consider these factors when choosing exemplar length for their specific defense scenario.

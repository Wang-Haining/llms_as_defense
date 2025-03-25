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
- **Posterior prob. beneficial**: The probability that longer exemplars improve a metric (either increasing metrics where higher is better, or decreasing metrics where lower is better).

## Overall Findings

Out of 180 total tests:

- **Credible improvements with longer exemplars**: 27 (15.0%)
- **Credible deteriorations with longer exemplars**: 21 (11.7%)
- **Inconclusive results**: 132 (73.3%)
- **Practically equivalent results**: 0 (0.0%)

## Key Findings by Metric

### Top-1 Accuracy

- Average effect (500→2500): 0.0801
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 2/30 (6.7%)

- Best LLM: **Llama-3.1** (average effect: -0.1569)
- Worst LLM: **Gpt-4O** (average effect: 0.3625)

#### Notable Deteriorations:

- Gpt-4O against SVM on EBG: Effect = 1.2574, Posterior prob. detrimental = 0.9995
- Gpt-4O against LOGREG on EBG: Effect = 1.1456, Posterior prob. detrimental = 0.9994

### Top-5 Accuracy

- Average effect (500→2500): 0.0840
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 2/30 (6.7%)

- Best LLM: **Ministral** (average effect: -0.1030)
- Worst LLM: **Gpt-4O** (average effect: 0.2723)

#### Notable Deteriorations:

- Gpt-4O against SVM on EBG: Effect = 1.0741, Posterior prob. detrimental = 0.9998
- Gpt-4O against LOGREG on EBG: Effect = 0.8078, Posterior prob. detrimental = 0.9985

### True Class Confidence

- Average effect (500→2500): 0.0625
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 1/30 (3.3%)

- Best LLM: **Claude-3.5** (average effect: -0.0721)
- Worst LLM: **Gpt-4O** (average effect: 0.2029)

#### Notable Deteriorations:

- Gpt-4O against LOGREG on EBG: Effect = 0.7799, Posterior prob. detrimental = 0.9940

### Prediction Entropy

- Average effect (500→2500): 0.0934
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Credible improvements: 9/30 (30.0%)
- Credible deteriorations: 4/30 (13.3%)

- Best LLM: **Llama-3.1** (average effect: 0.4537)
- Worst LLM: **Gemma-2** (average effect: -0.2528)

#### Notable Improvements:

- Llama-3.1 against SVM on EBG: Effect = 1.1350, Posterior prob. beneficial = 0.9989
- Llama-3.1 against LOGREG on EBG: Effect = 1.0588, Posterior prob. beneficial = 1.0000
- Gpt-4O against SVM on EBG: Effect = 0.8667, Posterior prob. beneficial = 1.0000

#### Notable Deteriorations:

- Ministral against SVM on EBG: Effect = -0.2870, Posterior prob. detrimental = 0.9996
- Gpt-4O against ROBERTA on EBG: Effect = -0.3171, Posterior prob. detrimental = 1.0000
- Gpt-4O against SVM on RJ: Effect = -0.4397, Posterior prob. detrimental = 0.9990

### bertscore

- Average effect (500→2500): 0.0107
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Credible improvements: 18/30 (60.0%)
- Credible deteriorations: 0/30 (0.0%)

- Best LLM: **Gpt-4O** (average effect: 0.0229)
- Worst LLM: **Ministral** (average effect: 0.0016)

#### Notable Improvements:

- Gpt-4O against LOGREG on EBG: Effect = 0.0451, Posterior prob. beneficial = 1.0000
- Gpt-4O against SVM on EBG: Effect = 0.0451, Posterior prob. beneficial = 1.0000
- Gpt-4O against ROBERTA on EBG: Effect = 0.0451, Posterior prob. beneficial = 1.0000

### pinc

- Average effect (500→2500): -0.0090
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 12/30 (40.0%)

- Best LLM: **Llama-3.1** (average effect: 0.0025)
- Worst LLM: **Gpt-4O** (average effect: -0.0378)

#### Notable Deteriorations:

- Ministral against LOGREG on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971
- Ministral against SVM on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971
- Ministral against ROBERTA on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971

## Key Findings by LLM

### Gemma-2

- Credible improvements: 4/36 (11.1%)
- Credible deteriorations: 4/36 (11.1%)
- Inconclusive results: 28/36 (77.8%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0040)
  - Least improved: **entropy** (avg effect: -0.2528)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.0236)
  - Least improved: **accuracy@1** (avg effect: 0.1704)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 0.2385, Posterior prob. beneficial = 0.9870
- bertscore against LOGREG on EBG: Effect = 0.0093, Posterior prob. beneficial = 0.9950
- bertscore against SVM on EBG: Effect = 0.0093, Posterior prob. beneficial = 0.9950

#### Notable Deteriorations:

- pinc against LOGREG on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9809
- pinc against SVM on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9809
- pinc against ROBERTA on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9809

### Llama-3.1

- Credible improvements: 6/36 (16.7%)
- Credible deteriorations: 0/36 (0.0%)
- Inconclusive results: 30/36 (83.3%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.4537)
  - Least improved: **pinc** (avg effect: 0.0025)

- For 'lower is better' metrics:
  - Most improved: **accuracy@1** (avg effect: -0.1569)
  - Least improved: **accuracy@5** (avg effect: 0.0896)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 1.1350, Posterior prob. beneficial = 0.9989
- entropy against LOGREG on EBG: Effect = 1.0588, Posterior prob. beneficial = 1.0000
- entropy against ROBERTA on RJ: Effect = 0.1454, Posterior prob. beneficial = 0.9854

### Ministral

- Credible improvements: 3/36 (8.3%)
- Credible deteriorations: 4/36 (11.1%)
- Inconclusive results: 29/36 (80.6%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0016)
  - Least improved: **entropy** (avg effect: -0.1119)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.1030)
  - Least improved: **true_class_confidence** (avg effect: 0.0997)

#### Notable Improvements:

- bertscore against LOGREG on RJ: Effect = 0.0027, Posterior prob. beneficial = 0.9829
- bertscore against SVM on RJ: Effect = 0.0027, Posterior prob. beneficial = 0.9829
- bertscore against ROBERTA on RJ: Effect = 0.0027, Posterior prob. beneficial = 0.9829

#### Notable Deteriorations:

- pinc against LOGREG on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971
- pinc against SVM on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971
- pinc against ROBERTA on EBG: Effect = -0.0036, Posterior prob. detrimental = 0.9971

### Claude-3.5

- Credible improvements: 9/36 (25.0%)
- Credible deteriorations: 3/36 (8.3%)
- Inconclusive results: 24/36 (66.7%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.2594)
  - Least improved: **pinc** (avg effect: -0.0033)

- For 'lower is better' metrics:
  - Most improved: **true_class_confidence** (avg effect: -0.0721)
  - Least improved: **accuracy@5** (avg effect: 0.1847)

#### Notable Improvements:

- entropy against ROBERTA on RJ: Effect = 0.5631, Posterior prob. beneficial = 1.0000
- entropy against SVM on RJ: Effect = 0.4702, Posterior prob. beneficial = 1.0000
- entropy against ROBERTA on EBG: Effect = 0.1436, Posterior prob. beneficial = 0.9830

#### Notable Deteriorations:

- pinc against LOGREG on EBG: Effect = -0.0063, Posterior prob. detrimental = 0.9998
- pinc against SVM on EBG: Effect = -0.0063, Posterior prob. detrimental = 0.9998
- pinc against ROBERTA on EBG: Effect = -0.0063, Posterior prob. detrimental = 0.9998

### Gpt-4O

- Credible improvements: 5/36 (13.9%)
- Credible deteriorations: 10/36 (27.8%)
- Inconclusive results: 21/36 (58.3%)
- Practically equivalent: 0/36 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.1188)
  - Least improved: **pinc** (avg effect: -0.0378)

- For 'lower is better' metrics:
  - Most improved: **true_class_confidence** (avg effect: 0.2029)
  - Least improved: **accuracy@1** (avg effect: 0.3625)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 0.8667, Posterior prob. beneficial = 1.0000
- entropy against LOGREG on EBG: Effect = 0.6753, Posterior prob. beneficial = 1.0000
- bertscore against LOGREG on EBG: Effect = 0.0451, Posterior prob. beneficial = 1.0000

#### Notable Deteriorations:

- accuracy@1 against SVM on EBG: Effect = 1.2574, Posterior prob. detrimental = 0.9995
- accuracy@1 against LOGREG on EBG: Effect = 1.1456, Posterior prob. detrimental = 0.9994
- accuracy@5 against SVM on EBG: Effect = 1.0741, Posterior prob. detrimental = 0.9998

## Key Findings by Threat Model

### LOGREG

- Credible improvements: 8/60 (13.3%)
- Credible deteriorations: 7/60 (11.7%)

#### Average Effect by Metric:

- accuracy@1: 0.2548 (negative)
- accuracy@5: 0.0976 (negative)
- bertscore: 0.0108 (positive)
- entropy: 0.1330 (positive)
- pinc: -0.0090 (negative)
- true_class_confidence: 0.1785 (negative)

#### Effect by LLM:

- Most affected LLM: **Gpt-4O** (avg effect: 0.3345)

### SVM

- Credible improvements: 10/60 (16.7%)
- Credible deteriorations: 8/60 (13.3%)

#### Average Effect by Metric:

- accuracy@1: 0.0634 (negative)
- accuracy@5: 0.1449 (negative)
- bertscore: 0.0107 (positive)
- entropy: 0.2104 (positive)
- pinc: -0.0090 (negative)
- true_class_confidence: 0.0267 (negative)

#### Effect by LLM:

- Most affected LLM: **Gpt-4O** (avg effect: 0.1742)

### ROBERTA

- Credible improvements: 9/60 (15.0%)
- Credible deteriorations: 6/60 (10.0%)

#### Average Effect by Metric:

- accuracy@1: -0.0780 (positive)
- accuracy@5: 0.0096 (negative)
- bertscore: 0.0107 (positive)
- entropy: -0.0631 (negative)
- pinc: -0.0090 (negative)
- true_class_confidence: -0.0177 (positive)

#### Effect by LLM:

- Most affected LLM: **Gemma-2** (avg effect: -0.1563)

## Findings by Corpus

### EBG

- Credible improvements: 18/90 (20.0%)
- Credible deteriorations: 20/90 (22.2%)

Overall, longer exemplars tend to be **less effective** on this corpus.

### RJ

- Credible improvements: 9/90 (10.0%)
- Credible deteriorations: 1/90 (1.1%)

Overall, longer exemplars tend to be **more effective** on this corpus.

## Conclusion

Based on the Bayesian analysis, **longer exemplars generally improve the effectiveness** of LLM-based imitation as a defense against authorship attribution attacks. The improvement is most pronounced for:

- **entropy** (average effect: 0.0934)
- **bertscore** (average effect: 0.0107)

And the LLMs that benefit most from longer exemplars are:

- **Claude-3.5** (9/36 metrics improved, 25.0%)
- **Llama-3.1** (6/36 metrics improved, 16.7%)
- **Gpt-4O** (5/36 metrics improved, 13.9%)

These findings suggest that the optimal exemplar length may depend on the specific LLM, threat model, and metric of interest. Users should consider these factors when choosing exemplar length for their specific defense scenario.

## Methodology Notes

This analysis used Bayesian hierarchical modeling to estimate the relationship between exemplar length and defense effectiveness. For each LLM-threat model-metric combination, we:

1. Fitted a Bayesian model relating exemplar length to the metric
2. Estimated the slope parameter (effect per 1000 words)
3. Calculated the 95% Highest Density Interval (HDI) for this parameter
4. Determined the posterior probability that the effect is beneficial

An effect was considered credible when the 95% HDI excluded zero. The analysis accounted for the bounded nature of metrics like accuracy and incorporated appropriate prior distributions where needed.

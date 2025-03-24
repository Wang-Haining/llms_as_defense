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

Out of 158 total tests:

- **Credible improvements with longer exemplars**: 24 (15.2%)
- **Credible deteriorations with longer exemplars**: 15 (9.5%)
- **Inconclusive results**: 76 (48.1%)
- **Practically equivalent results**: 43 (27.2%)

## Key Findings by Metric

### Top-5 Accuracy

- Average effect (500→2500): 0.0577
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 3/30 (10.0%)
- Credible deteriorations: 4/30 (13.3%)

- Best LLM: **Ministral** (average effect: -0.0878)
- Worst LLM: **Gpt-4O** (average effect: 0.2125)

#### Notable Improvements:

- Llama-3.1 against LOGREG on RJ: Effect = -0.2514, Posterior prob. beneficial = 0.9872
- Gpt-4O against SVM on RJ: Effect = -0.3475, Posterior prob. beneficial = 0.9994
- Gemma-2 against ROBERTA on EBG: Effect = -0.3794, Posterior prob. beneficial = 0.9968

#### Notable Deteriorations:

- Gpt-4O against SVM on EBG: Effect = 0.8410, Posterior prob. detrimental = 1.0000
- Gpt-4O against LOGREG on EBG: Effect = 0.6397, Posterior prob. detrimental = 1.0000
- Claude-3.5 against SVM on RJ: Effect = 0.3453, Posterior prob. detrimental = 0.9990

### True Class Confidence

- Average effect (500→2500): -0.0082
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 2/30 (6.7%)

- Best LLM: **Claude-3.5** (average effect: -0.0849)
- Worst LLM: **Gpt-4O** (average effect: 0.1273)

#### Notable Deteriorations:

- Gpt-4O against LOGREG on EBG: Effect = 0.5545, Posterior prob. detrimental = 1.0000
- Ministral against LOGREG on RJ: Effect = 0.3764, Posterior prob. detrimental = 0.9801

### Prediction Entropy

- Average effect (500→2500): 0.0908
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Credible improvements: 11/30 (36.7%)
- Credible deteriorations: 4/30 (13.3%)

- Best LLM: **Llama-3.1** (average effect: 0.4259)
- Worst LLM: **Gemma-2** (average effect: -0.2432)

#### Notable Improvements:

- Llama-3.1 against LOGREG on EBG: Effect = 1.0392, Posterior prob. beneficial = 1.0000
- Llama-3.1 against SVM on EBG: Effect = 1.0045, Posterior prob. beneficial = 0.9969
- Gpt-4O against SVM on EBG: Effect = 0.8658, Posterior prob. beneficial = 1.0000

#### Notable Deteriorations:

- Ministral against SVM on EBG: Effect = -0.2868, Posterior prob. detrimental = 0.9988
- Gpt-4O against ROBERTA on EBG: Effect = -0.3163, Posterior prob. detrimental = 0.9999
- Gpt-4O against SVM on RJ: Effect = -0.4299, Posterior prob. detrimental = 0.9990

### bertscore

- Average effect (500→2500): 0.0107
- Overall impact: **Positive** (longer exemplars tend to improve this metric)
- Credible improvements: 9/30 (30.0%)
- Credible deteriorations: 0/30 (0.0%)

- Best LLM: **Gpt-4O** (average effect: 0.0227)
- Worst LLM: **Ministral** (average effect: 0.0018)

#### Notable Improvements:

- Gpt-4O against ROBERTA on EBG: Effect = 0.0447, Posterior prob. beneficial = 1.0000
- Gpt-4O against SVM on EBG: Effect = 0.0447, Posterior prob. beneficial = 1.0000
- Gpt-4O against LOGREG on EBG: Effect = 0.0446, Posterior prob. beneficial = 1.0000

### pinc

- Average effect (500→2500): -0.0090
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 3/30 (10.0%)

- Best LLM: **Llama-3.1** (average effect: 0.0026)
- Worst LLM: **Gpt-4O** (average effect: -0.0383)

#### Notable Deteriorations:

- Gpt-4O against ROBERTA on EBG: Effect = -0.0755, Posterior prob. detrimental = 1.0000
- Gpt-4O against LOGREG on EBG: Effect = -0.0757, Posterior prob. detrimental = 1.0000
- Gpt-4O against SVM on EBG: Effect = -0.0757, Posterior prob. detrimental = 1.0000

### Top-1 Accuracy

- Average effect (500→2500): 0.1316
- Overall impact: **Negative** (longer exemplars tend to worsen this metric)
- Credible improvements: 1/8 (12.5%)
- Credible deteriorations: 2/8 (25.0%)

- Best LLM: **Llama-3.1** (average effect: -0.1775)
- Worst LLM: **Gpt-4O** (average effect: 0.4150)

#### Notable Improvements:

- Claude-3.5 against SVM on RJ: Effect = -0.3189, Posterior prob. beneficial = 0.9845

#### Notable Deteriorations:

- Gpt-4O against SVM on EBG: Effect = 0.9214, Posterior prob. detrimental = 1.0000
- Gpt-4O against LOGREG on EBG: Effect = 0.8340, Posterior prob. detrimental = 1.0000

## Key Findings by LLM

### Gemma-2

- Credible improvements: 2/30 (6.7%)
- Credible deteriorations: 2/30 (6.7%)
- Inconclusive results: 14/30 (46.7%)
- Practically equivalent: 12/30 (40.0%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0042)
  - Least improved: **entropy** (avg effect: -0.2432)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.0134)
  - Least improved: **true_class_confidence** (avg effect: 0.0066)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 0.2365, Posterior prob. beneficial = 0.9871
- accuracy@5 against ROBERTA on EBG: Effect = -0.3794, Posterior prob. beneficial = 0.9968

#### Notable Deteriorations:

- accuracy@5 against SVM on RJ: Effect = 0.2467, Posterior prob. detrimental = 0.9798
- entropy against ROBERTA on EBG: Effect = -0.8714, Posterior prob. detrimental = 0.9989

### Llama-3.1

- Credible improvements: 5/31 (16.1%)
- Credible deteriorations: 0/31 (0.0%)
- Inconclusive results: 14/31 (45.2%)
- Practically equivalent: 12/31 (38.7%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.4259)
  - Least improved: **bertscore** (avg effect: 0.0026)

- For 'lower is better' metrics:
  - Most improved: **accuracy@1** (avg effect: -0.1775)
  - Least improved: **accuracy@5** (avg effect: 0.0422)

#### Notable Improvements:

- entropy against LOGREG on EBG: Effect = 1.0392, Posterior prob. beneficial = 1.0000
- entropy against SVM on EBG: Effect = 1.0045, Posterior prob. beneficial = 0.9969
- entropy against SVM on RJ: Effect = 0.3011, Posterior prob. beneficial = 0.9766

### Ministral

- Credible improvements: 0/30 (0.0%)
- Credible deteriorations: 2/30 (6.7%)
- Inconclusive results: 15/30 (50.0%)
- Practically equivalent: 13/30 (43.3%)

- For 'higher is better' metrics:
  - Most improved: **bertscore** (avg effect: 0.0018)
  - Least improved: **entropy** (avg effect: -0.1099)

- For 'lower is better' metrics:
  - Most improved: **accuracy@5** (avg effect: -0.0878)
  - Least improved: **true_class_confidence** (avg effect: -0.0130)

#### Notable Deteriorations:

- true_class_confidence against LOGREG on RJ: Effect = 0.3764, Posterior prob. detrimental = 0.9801
- entropy against SVM on EBG: Effect = -0.2868, Posterior prob. detrimental = 0.9988

### Claude-3.5

- Credible improvements: 11/33 (33.3%)
- Credible deteriorations: 1/33 (3.0%)
- Inconclusive results: 15/33 (45.5%)
- Practically equivalent: 6/33 (18.2%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.2598)
  - Least improved: **pinc** (avg effect: -0.0032)

- For 'lower is better' metrics:
  - Most improved: **accuracy@1** (avg effect: -0.1434)
  - Least improved: **accuracy@5** (avg effect: 0.1350)

#### Notable Improvements:

- entropy against ROBERTA on RJ: Effect = 0.5622, Posterior prob. beneficial = 1.0000
- entropy against SVM on RJ: Effect = 0.4682, Posterior prob. beneficial = 1.0000
- entropy against LOGREG on EBG: Effect = 0.1580, Posterior prob. beneficial = 0.9705

#### Notable Deteriorations:

- accuracy@5 against SVM on RJ: Effect = 0.3453, Posterior prob. detrimental = 0.9990

### Gpt-4O

- Credible improvements: 6/34 (17.6%)
- Credible deteriorations: 10/34 (29.4%)
- Inconclusive results: 18/34 (52.9%)
- Practically equivalent: 0/34 (0.0%)

- For 'higher is better' metrics:
  - Most improved: **entropy** (avg effect: 0.1215)
  - Least improved: **pinc** (avg effect: -0.0383)

- For 'lower is better' metrics:
  - Most improved: **true_class_confidence** (avg effect: 0.1273)
  - Least improved: **accuracy@1** (avg effect: 0.4150)

#### Notable Improvements:

- entropy against SVM on EBG: Effect = 0.8658, Posterior prob. beneficial = 1.0000
- entropy against LOGREG on EBG: Effect = 0.6746, Posterior prob. beneficial = 1.0000
- bertscore against ROBERTA on EBG: Effect = 0.0447, Posterior prob. beneficial = 1.0000

#### Notable Deteriorations:

- accuracy@1 against SVM on EBG: Effect = 0.9214, Posterior prob. detrimental = 1.0000
- accuracy@5 against SVM on EBG: Effect = 0.8410, Posterior prob. detrimental = 1.0000
- accuracy@1 against LOGREG on EBG: Effect = 0.8340, Posterior prob. detrimental = 1.0000

## Key Findings by Threat Model

### LOGREG

- Credible improvements: 7/52 (13.5%)
- Credible deteriorations: 5/52 (9.6%)

#### Average Effect by Metric:

- accuracy@1: 0.3283 (negative)
- accuracy@5: 0.0693 (negative)
- bertscore: 0.0107 (positive)
- entropy: 0.1320 (positive)
- pinc: -0.0090 (negative)
- true_class_confidence: 0.0537 (negative)

#### Effect by LLM:

- Most affected LLM: **Gpt-4O** (avg effect: 0.2519)

### SVM

- Credible improvements: 10/52 (19.2%)
- Credible deteriorations: 7/52 (13.5%)

#### Average Effect by Metric:

- accuracy@1: 0.3013 (negative)
- accuracy@5: 0.1019 (negative)
- bertscore: 0.0107 (positive)
- entropy: 0.1979 (positive)
- pinc: -0.0090 (negative)
- true_class_confidence: -0.0268 (positive)

#### Effect by LLM:

- Most affected LLM: **Gpt-4O** (avg effect: 0.1806)

### ROBERTA

- Credible improvements: 7/54 (13.0%)
- Credible deteriorations: 3/54 (5.6%)

#### Average Effect by Metric:

- accuracy@1: -0.0517 (positive)
- accuracy@5: 0.0019 (negative)
- bertscore: 0.0106 (positive)
- entropy: -0.0574 (negative)
- pinc: -0.0090 (negative)
- true_class_confidence: -0.0515 (positive)

#### Effect by LLM:

- Most affected LLM: **Gemma-2** (avg effect: -0.1617)

## Findings by Corpus

### EBG

- Credible improvements: 14/79 (17.7%)
- Credible deteriorations: 11/79 (13.9%)

Overall, longer exemplars tend to be **more effective** on this corpus.

### RJ

- Credible improvements: 10/79 (12.7%)
- Credible deteriorations: 4/79 (5.1%)

Overall, longer exemplars tend to be **more effective** on this corpus.

## Conclusion

Based on the Bayesian analysis, **longer exemplars generally improve the effectiveness** of LLM-based imitation as a defense against authorship attribution attacks. The improvement is most pronounced for:

- **entropy** (average effect: 0.0908)
- **bertscore** (average effect: 0.0107)
- **true_class_confidence** (average effect: -0.0082)

And the LLMs that benefit most from longer exemplars are:

- **Claude-3.5** (11/33 metrics improved, 33.3%)
- **Gpt-4O** (6/34 metrics improved, 17.6%)
- **Llama-3.1** (5/31 metrics improved, 16.1%)

These findings suggest that the optimal exemplar length may depend on the specific LLM, threat model, and metric of interest. Users should consider these factors when choosing exemplar length for their specific defense scenario.

## Methodology Notes

This analysis used Bayesian hierarchical modeling to estimate the relationship between exemplar length and defense effectiveness. For each LLM-threat model-metric combination, we:

1. Fitted a Bayesian model relating exemplar length to the metric
2. Estimated the slope parameter (effect per 1000 words)
3. Calculated the 95% Highest Density Interval (HDI) for this parameter
4. Determined the posterior probability that the effect is beneficial

An effect was considered credible when the 95% HDI excluded zero. The analysis accounted for the bounded nature of metrics like accuracy and incorporated appropriate prior distributions where needed.

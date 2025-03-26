# Analysis of Exemplar Length Effect on LLM-based Imitation Defense

## Overview

This analysis examines the effect of *actual* exemplar length on the effectiveness of LLM-based imitation as a defense against authorship attribution attacks. We extract the exemplar text from each seed_{seed}.json, count its words, and model how that length impacts various metrics.

- 2 corpora: EBG, RJ
- 3 threat models: LOGREG, ROBERTA, SVM
- 5 LLMs: Claude-3.5, Gemma-2, Gpt-4O, Llama-3.1, Ministral
- 6 metrics: accuracy@1, accuracy@5, bertscore, entropy, pinc, true_class_confidence

## Data Availability

Out of 180 total test combinations:

- **Valid for exemplar length analysis**: 180 (100.0%)
- **Insufficient exemplar length variation**: 0 (0.0%)

Many combinations have only a single exemplar length, making it impossible to analyze the effect of length variation.

## Overall Findings (among valid analyses)

Out of 180 valid analyses:

- **Credible improvements with longer exemplars**: 27 (15.0%)
- **Credible deteriorations with longer exemplars**: 21 (11.7%)
- **Inconclusive results**: 132 (73.3%)
- **Practically equivalent results**: 0 (0.0%)

### Significant Effects

#### Improvements with Longer Exemplars

- Gemma-2 | bertscore | LOGREG: Effect size = 0.0093, P(beneficial) = 0.9950
- Llama-3.1 | entropy | LOGREG: Effect size = 1.0588, P(beneficial) = 1.0000
- Llama-3.1 | bertscore | LOGREG: Effect size = 0.0049, P(beneficial) = 0.9999
- Claude-3.5 | bertscore | LOGREG: Effect size = 0.0150, P(beneficial) = 0.9998
- Gpt-4O | entropy | LOGREG: Effect size = 0.6753, P(beneficial) = 1.0000
- Gpt-4O | bertscore | LOGREG: Effect size = 0.0451, P(beneficial) = 1.0000
- Gemma-2 | entropy | SVM: Effect size = 0.2385, P(beneficial) = 0.9870
- Gemma-2 | bertscore | SVM: Effect size = 0.0093, P(beneficial) = 0.9950
- Llama-3.1 | entropy | SVM: Effect size = 1.1350, P(beneficial) = 0.9989
- Llama-3.1 | bertscore | SVM: Effect size = 0.0049, P(beneficial) = 0.9999
- Claude-3.5 | bertscore | SVM: Effect size = 0.0150, P(beneficial) = 0.9998
- Gpt-4O | entropy | SVM: Effect size = 0.8667, P(beneficial) = 1.0000
- Gpt-4O | bertscore | SVM: Effect size = 0.0451, P(beneficial) = 1.0000
- Gemma-2 | bertscore | ROBERTA: Effect size = 0.0093, P(beneficial) = 0.9950
- Llama-3.1 | bertscore | ROBERTA: Effect size = 0.0049, P(beneficial) = 0.9999
- Claude-3.5 | entropy | ROBERTA: Effect size = 0.1436, P(beneficial) = 0.9830
- Claude-3.5 | bertscore | ROBERTA: Effect size = 0.0150, P(beneficial) = 0.9998
- Gpt-4O | bertscore | ROBERTA: Effect size = 0.0451, P(beneficial) = 1.0000
- Ministral | bertscore | LOGREG: Effect size = 0.0027, P(beneficial) = 0.9829
- Claude-3.5 | bertscore | LOGREG: Effect size = 0.0298, P(beneficial) = 0.9952
- Ministral | bertscore | SVM: Effect size = 0.0027, P(beneficial) = 0.9829
- Claude-3.5 | entropy | SVM: Effect size = 0.4702, P(beneficial) = 1.0000
- Claude-3.5 | bertscore | SVM: Effect size = 0.0298, P(beneficial) = 0.9952
- Llama-3.1 | entropy | ROBERTA: Effect size = 0.1454, P(beneficial) = 0.9854
- Ministral | bertscore | ROBERTA: Effect size = 0.0027, P(beneficial) = 0.9829
- Claude-3.5 | entropy | ROBERTA: Effect size = 0.5631, P(beneficial) = 1.0000
- Claude-3.5 | bertscore | ROBERTA: Effect size = 0.0298, P(beneficial) = 0.9952

#### Deteriorations with Longer Exemplars

- Gemma-2 | pinc | LOGREG: Effect size = -0.0036, P(detrimental) = 0.9809
- Ministral | pinc | LOGREG: Effect size = -0.0036, P(detrimental) = 0.9971
- Claude-3.5 | pinc | LOGREG: Effect size = -0.0063, P(detrimental) = 0.9998
- Gpt-4O | accuracy@1 | LOGREG: Effect size = 1.1456, P(detrimental) = 0.9994
- Gpt-4O | accuracy@5 | LOGREG: Effect size = 0.8078, P(detrimental) = 0.9985
- Gpt-4O | true_class_confidence | LOGREG: Effect size = 0.7799, P(detrimental) = 0.9940
- Gpt-4O | pinc | LOGREG: Effect size = -0.0755, P(detrimental) = 1.0000
- Gemma-2 | pinc | SVM: Effect size = -0.0036, P(detrimental) = 0.9809
- Ministral | entropy | SVM: Effect size = -0.2870, P(detrimental) = 0.9996
- Ministral | pinc | SVM: Effect size = -0.0036, P(detrimental) = 0.9971
- Claude-3.5 | pinc | SVM: Effect size = -0.0063, P(detrimental) = 0.9998
- Gpt-4O | accuracy@1 | SVM: Effect size = 1.2574, P(detrimental) = 0.9995
- Gpt-4O | accuracy@5 | SVM: Effect size = 1.0741, P(detrimental) = 0.9998
- Gpt-4O | pinc | SVM: Effect size = -0.0755, P(detrimental) = 1.0000
- Gemma-2 | entropy | ROBERTA: Effect size = -0.9118, P(detrimental) = 0.9994
- Gemma-2 | pinc | ROBERTA: Effect size = -0.0036, P(detrimental) = 0.9809
- Ministral | pinc | ROBERTA: Effect size = -0.0036, P(detrimental) = 0.9971
- Claude-3.5 | pinc | ROBERTA: Effect size = -0.0063, P(detrimental) = 0.9998
- Gpt-4O | entropy | ROBERTA: Effect size = -0.3171, P(detrimental) = 1.0000
- Gpt-4O | pinc | ROBERTA: Effect size = -0.0755, P(detrimental) = 1.0000
- Gpt-4O | entropy | SVM: Effect size = -0.4397, P(detrimental) = 0.9990


## Conclusion

Based on the available data with varying exemplar lengths, there are some significant effects observed. However, the majority of data points have insufficient variation in exemplar length to draw firm conclusions about the effect of exemplar length on defense effectiveness.

There is a tendency toward improvement with longer exemplars, but more data with varying exemplar lengths would be needed to confirm this pattern.

# A Measurement Perspective on LLM-Generated Text Detection: An Item Response Theory Framework

Official code, data, detector outputs, and estimated parameters for the paper **"A Measurement Perspective on LLM-Generated Text Detection: An Item Response Theory Framework."**

**Authors:** Yanbin Zhao, Zhipeng Chen, Guangsheng Bao, Guandong Zhang, and Yan Li (corresponding author)

This work studies LLM-generated text detection as a measurement problem. Benchmark texts are treated as measurement items, detectors as respondents, and detector correctness as item responses. A multidimensional item response theory (MIRT) model jointly represents text-level discrimination and difficulty-location properties and detector-level latent abilities.

[Overview](#overview) | [Benchmark](#benchmark) | [Detectors](#detectors) | [IRT framework](#irt-framework) | [Compact benchmark](#compact-benchmark)

## Overview

Reliable benchmark evaluation of LLM-generated text detectors remains challenging because reported performance depends heavily on the test corpora, making cross-paper comparisons unreliable. Aggregate metrics such as accuracy, F1 score, and AUC treat all benchmark samples equally and provide little diagnostic information about the benchmark itself. Consequently, they cannot determine whether poor performance comes from a detector's intrinsic weakness or from biased, low-informative test samples.

The paper addresses this limitation with an IRT-based evaluation framework and makes four main contributions:

1. It models the complete detector-text response structure, enabling joint analysis of benchmark heterogeneity and detector behavior.
2. It shows that human-written and machine-generated texts occupy distinct but overlapping regions of the estimated IRT item-parameter space. The human-machine label is the strongest associated factor, while domain and text length further modulate the structure.
3. It characterizes detectors with multidimensional ability profiles, revealing shared, detector-specific, supervised-versus-zero-shot, and domain-dependent response patterns.
4. It constructs a compact benchmark of 2,088 texts, 6.5% of the full benchmark, that preserves the main comparative detector-performance pattern across 14 detectors.

## Benchmark

The full benchmark contains **32,000 texts** from four domains, balanced between **16,000 human-written** and **16,000 machine-generated** samples.

### Human-written texts

For each domain, 4,000 texts with no more than 512 words were randomly selected. The source texts predate ChatGPT, which limits potential contamination by LLM-generated content.

| Domain | Writing type | Original source size | Selected human texts |
|---|---|---:|---:|
| ArXiv | Scientific abstracts published from 2002 to 2017 | 18,891 | 4,000 |
| Writing Prompt | Creative stories | 87,485 | 4,000 |
| XSum | News articles and summaries | 2,040,445 | 4,000 |
| Yelp Reviews | Customer reviews | 650,000 | 4,000 |
| **Total** |  |  | **16,000** |

### Machine-generated texts

The machine-generated half of the benchmark was produced with four LLMs:

- GPT-5;
- Claude Sonnet 4.5;
- Qwen3-32B;
- Llama 3.3 70B.

Within each domain, the 4,000 human-written texts were randomly divided into four groups of 1,000, with one group assigned to each LLM. The first 30 words of each human-written text were used as the prompt prefix, and the model generated the continuation independently. This procedure produced 1,000 machine-generated texts per model per domain and 16,000 machine-generated texts in total.

### Final composition

| Domain | Human | Machine | Total |
|---|---:|---:|---:|
| ArXiv | 4,000 | 4,000 | 8,000 |
| Writing Prompt | 4,000 | 4,000 | 8,000 |
| XSum | 4,000 | 4,000 | 8,000 |
| Yelp Reviews | 4,000 | 4,000 | 8,000 |
| **Total** | **16,000** | **16,000** | **32,000** |

The final files are stored in `Benchmark/datasets`. Each file is a JSON object in which `original` contains human-written texts and `sampled` contains machine-generated texts:

```json
{
  "original": ["human-written text", "..."],
  "sampled": ["machine-generated text", "..."]
}
```

Source-data preparation files are organized by data category under `original_data/data_generation`, `original_data/domain_human`, and `original_data/domain_llm`.

## Detectors

The study evaluates 14 representative detectors: 11 zero-shot or training-free methods and three supervised RoBERTa classifiers.

| Detector | Type |
|---|---|
| DetectGPT | Zero-shot |
| Likelihood | Zero-shot |
| Entropy | Zero-shot |
| Lastde | Training-free |
| Rank | Zero-shot |
| NPR | Zero-shot |
| LogRank | Zero-shot |
| Fast-DetectGPT | Zero-shot |
| Binoculars | Zero-shot |
| DNA-GPT | Zero-shot |
| LRR | Zero-shot |
| RoBERTa-Base | Supervised |
| RoBERTa-Large | Supervised |
| RoBERTa-ChatGPT | Supervised |

For detectors that return continuous scores, the detector-specific threshold is selected by maximizing the Youden index:

```math
\tau_l^* = \underset{\tau}{\arg\max}\left[TPR_l(\tau)-FPR_l(\tau)\right].
```

The calibrated binary predictions are converted to correctness responses for the IRT analysis.

### Overall detection performance

| Detector | AUC | F1 | Accuracy |
|---|---:|---:|---:|
| DetectGPT | **0.742** | 0.626 | 0.625 |
| Likelihood | 0.642 | 0.672 | 0.627 |
| Lastde | 0.628 | 0.688 | 0.617 |
| LogRank | 0.621 | **0.703** | **0.656** |
| Rank | 0.614 | 0.680 | 0.595 |
| NPR | 0.604 | 0.653 | 0.576 |
| Entropy | 0.603 | 0.635 | 0.606 |
| Fast-DetectGPT | 0.600 | 0.686 | 0.608 |
| Binoculars | 0.590 | 0.679 | 0.602 |
| DNA-GPT | 0.587 | 0.638 | 0.591 |
| LRR | 0.538 | 0.672 | 0.573 |
| RoBERTa-ChatGPT | 0.507 | 0.637 | 0.522 |
| RoBERTa-Base | 0.507 | 0.661 | 0.515 |
| RoBERTa-Large | 0.421 | 0.665 | 0.505 |

DetectGPT obtains the highest overall AUC, while LogRank obtains the highest F1 score and accuracy. Detector rankings vary substantially across domains and metrics, which motivates the item-level analysis.


## IRT framework

For text $i$ and detector $l$, let $Y_i$ be the ground-truth label and $D_{il}$ the detector prediction. The binary response is

```math
R_{il}=I(D_{il}=Y_i).
```


The paper uses a compensatory multidimensional 2PL-style model:

```math
P(R_{il}=1 \mid \mathbf{\theta}_l,\mathbf{\alpha}_i,\mathbf{\beta}_i)
=\frac{1}{1+\exp\left[-\mathbf{\alpha}_i^T(\mathbf{\theta}_l-\mathbf{\beta}_i)\right]}.
```

where:

- $\mathbf{\theta}_l$ is the multidimensional latent ability of detector $l$;
- $\mathbf{\alpha}_i$ is the discrimination vector of text $i$;
- $\mathbf{\beta}_i$ is the relative difficulty-location vector of text $i$.

The parameters are estimated with hierarchical Bayesian priors and variational inference. Posterior means are used as point estimates, followed by a post-estimation rotation to improve interpretability.

The latent dimension is selected from

```text
{2, 3, 5, 10, 15}
```

using five-fold cross-validation and validation binary cross-entropy. The paper selects **five dimensions** to balance validation performance, parsimony, and interpretability.


### Main IRT findings

- Human-written and machine-generated texts form distinct but overlapping regions in the IRT item-parameter space, indicating a graded rather than completely separable distinction.
- The human-machine label has the strongest multivariate association with the item-parameter space; domain and text length also contribute significant variation.
- The third rotated ability dimension acts as the dominant common detector dimension and is strongly correlated with overall F1 score ($\rho=0.87$).
- The fifth ability dimension separates many score-based methods from the supervised RoBERTa detectors and correlates with overall AUC ($\rho=0.77$) and accuracy ($\rho=0.82$).
- Domain-specific IRT models reveal different latent ability-performance relationships across ArXiv, Writing Prompt, XSum, and Yelp Reviews.

## Compact benchmark

The paper represents each text with its estimated item parameters:

```math
E_i=(\hat{\mathbf{\alpha}}_i,\hat{\mathbf{\beta}}_i).
```

K-means is applied to these representations, and the texts closest to the cluster centers are selected as representatives. With $K=12$ and $n_i=174$, the compact benchmark contains **2,088 texts**, or **6.5%** of the full benchmark.

| Domain | Human | Machine | Total |
|---|---:|---:|---:|
| ArXiv | 258 | 191 | 449 |
| Writing Prompt | 324 | 190 | 514 |
| XSum | 324 | 250 | 574 |
| Yelp Reviews | 337 | 214 | 551 |
| **Total** | **1,243** | **845** | **2,088** |

The subset preserves the main detector-wise performance pattern of the full benchmark, although several detectors obtain higher absolute accuracy on the subset. The current repository release provides the full 32K benchmark; the 2,088-sample result is reported here as the compact benchmark constructed in the paper.

## Installation

Python 3.9 or newer is recommended. Most detector experiments require a CUDA-capable GPU.

Install PyTorch for the target CUDA or CPU platform, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```


## Repository structure

```text
Benchmark/
  datasets/                 full 32,000-text benchmark
  detectors/                detector implementations
  results/                  detector metrics and correctness sequences
irt/
  all_results_item_format.csv
  code/                      multidimensional IRT training code
  results/14-kfold/          retained parameter estimates
original_data/
  data_generation/           source-corpus collection
  domain_human/              human-text preparation
  domain_llm/                machine-generated text records
Reference/                   related papers
```

Some scripts retain paths from the original experiment machine. Update their configuration constants before running them in a new environment.

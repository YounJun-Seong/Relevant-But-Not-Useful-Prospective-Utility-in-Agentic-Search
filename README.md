# When Relevance Is Not Enough: Prospective Utility in Agentic Search

Official anonymized code and experimental artifacts for the paper **"When Relevance Is Not Enough: Prospective Utility in Agentic Search"**, submitted to WSDM 2027.

## Overview

Conventional retrieval systems primarily rank documents by relevance. In agentic search, however, the value of retrieved evidence also depends on how it affects subsequent search actions and final answers.

We study this distinction through **Prospective Interventional Utility (PIU)**.

From a frozen pre-consumption agent state, we inject an unseen candidate document, resume the same continuation policy, and measure its effect on downstream answer quality.

PIU is used as a diagnostic measurement framework rather than a deployable retrieval oracle.

## Research Questions and Key Findings

### RQ1. Does relevance leave prospective utility headroom?

Across HotpotQA, 2WikiMultiHopQA, MuSiQue, a preregistered HotpotQA lockbox, and Qwen2.5-14B scale transfer, the measured PIU oracle consistently outperforms cross-encoder (CE) relevance selection.

- Oracle–CE improvements range from **+0.082 to +0.142 F1**.
- The HotpotQA lockbox yields **+0.117 F1**, with a 95% confidence interval of [0.092, 0.142].
- Cross-fitted outcome evaluation retains a **+0.111 F1** improvement.
- Representative non-enriched samples, matched intervention controls, and candidate-pool sensitivity analyses support the robustness of the finding.
- A post-hoc Phi-4 replication yields **+0.081 F1** [0.063, 0.102].

These results demonstrate measurable prospective headroom beyond relevance-based selection under the evaluated continuation policies.

### RQ2. Does stronger conventional relevance eliminate prospective headroom?

We compare the primary MiniLM cross-encoder with query-only BGE, state-aware BGE, and a state-aware Qwen2.5 listwise relevance ranker.

State-aware BGE substantially improves conventional relevance quality across all five evaluation settings.

On the HotpotQA lockbox:

- Supporting-document Hit@1 increases from 0.320 to 0.765.
- nDCG@10 increases from 0.387 to 0.840.
- Residual oracle headroom remains nearly unchanged: +0.117 F1 for CE versus +0.116 F1 for state-aware BGE.

Additional choice-space analyses evaluate the stronger selectors over matched candidate sets and include post-hoc rollouts of missing full-pool argmax candidates.

These results indicate that large improvements in conventional relevance do not reliably translate into comparable improvements in downstream continuation quality.

### RQ3. Can prospective headroom be identified before intervention?

We evaluate feature-based predictors and fine-tuned neural gates using only information available before intervention.

- The full shallow gate achieves ROC-AUC 0.637 on the HotpotQA lockbox.
- Performance is weaker on several cross-dataset and cross-scale evaluations.
- Fine-tuned MiniLM and DeBERTa gates do not consistently improve discrimination.
- Utility-winning candidates have median CE rank 3–4 within the retained intervention set.
- Top-3 CE candidate rollouts expose 54–71% of total oracle headroom.

Simple answer aggregation and early-termination heuristics fail to recover this headroom reliably.

An exploratory post-rollout continuation scorer provides small improvements but recovers only a limited fraction of the available value.

These findings distinguish candidate localization from the harder problem of prospective continuation evaluation.

## Experimental Setup

| Component | Configuration |
|---|---|
| Datasets | HotpotQA, 2WikiMultiHopQA, MuSiQue |
| Primary agent | Qwen2.5-7B-Instruct |
| Scale transfer | Qwen2.5-14B-Instruct |
| Cross-family replication | Phi-4 (post-hoc) |
| Corpus | HotpotQA enwiki-20171001 abstracts |
| Sparse retrieval | BM25 |
| Dense retrieval | E5-base-v2 |
| Primary relevance selector | MS MARCO MiniLM cross-encoder |
| Stronger relevance selectors | BGE-reranker-v2-m3, state-aware Qwen2.5 listwise |
| Primary metric | Downstream answer F1 |
| Statistical inference | Question-level paired bootstrap |
| Primary continuation policy | Greedy decoding |
| Maximum search turns | 4 |

The primary confirmatory evaluations use frozen agent states and identical continuation policies across candidate interventions.

## Repository Structure

- `piu/` — Core agent, retrieval, intervention, utility, and prediction implementations.
- `scripts/` — Experiment drivers, evaluation scripts, relevance analyses, robustness checks, and figure generation.
- `results/` — Released experimental measurements and analysis reports.
- `figures/` — Paper figures.
- `paper/` — Results summaries, analysis notes, and supplementary source material.
- `supplementary.pdf` — Supplementary experimental and implementation details.

### Important result directories

| Directory | Contents |
|---|---|
| `results/lockbox/` | HotpotQA lockbox outcomes, evaluation split, and provenance |
| `results/wiki2/` | 2Wiki evaluation |
| `results/musique/` | MuSiQue evaluation |
| `results/xagent/` | Qwen2.5-14B evaluation |
| `results/xfamily_phi4/` | Post-hoc Phi-4 replication |
| `results/choice_space/` | Candidate choice-space analyses |
| `results/relevance_baselines/` | Conventional relevance evaluations |
| `results/state_aware_relevance/` | State-aware relevance analyses |
| `results/rq3_state_gate/` | Pre-intervention state identification |
| `results/rq3_neural_gate/` | Fine-tuned neural gate evaluation |
| `results/continuation_scorer/` | Exploratory post-rollout scoring |
| `results/crossfit/` | Cross-fitted oracle evaluation |
| `results/decoding_sensitivity/` | Continuation-policy sensitivity |
| `results/myopic/` | Forced-answer myopic utility analysis |

## Reproducibility

The repository provides core experimental code, selected intervention utility labels, evaluation splits, and analysis reports.

### Environment

The reported experiments used:

- Python 3.11
- PyTorch 2.8.0
- Transformers 4.55.2
- vLLM 0.10.2
- bm25s 0.3.10
- scikit-learn 1.9.0
- NVIDIA H100 80GB GPUs

Install the available dependencies using:

```bash
pip install -r requirements.txt
```

### Data Preparation

The experiments use the official HotpotQA enwiki-20171001 abstract corpus.

```bash
python scripts/build_corpus_parallel.py data/raw data/corpus.jsonl
python scripts/build_index.py data/corpus.jsonl data/bm25_index
python scripts/embed_corpus.py
```

The original corpus, retrieval indexes, model checkpoints, and large intermediate artifacts are not redistributed.

### Experiment Scripts

Relevant scripts include:

- `scripts/lockbox_eval.py`
- `scripts/wiki2_eval.py`
- `scripts/musique_eval.py`
- `scripts/xagent_eval.py`
- `scripts/fullpool_relevance.py`
- `scripts/choice_space_consistency.py`
- `scripts/expand_choice_space.py`
- `scripts/rq3_state_gate.py`
- `scripts/rq3_neural_gate.py`
- `scripts/crossfit_eval.py`
- `scripts/decoding_sensitivity.py`
- `scripts/myopic_vs_full.py`
- `scripts/continuation_scorer.py`

Experiment-specific configuration and execution details are documented in the corresponding scripts and supplementary material.

Reproducing full intervention trajectories requires reconstructing the retrieval environment and running the agent models. Released summary artifacts support inspection of reported results without repeating every agent rollout.

## Confirmatory and Exploratory Analyses

The study distinguishes preregistered confirmatory evaluations from post-hoc analyses.

The primary confirmatory evaluation includes the HotpotQA lockbox, downstream Oracle–CE comparisons, and designated replication and robustness protocols.

Post-hoc analyses include full-pool choice-space expansion, Phi-4 cross-family replication, sampled-decoding sensitivity, forced-answer myopic utility, and exploratory post-rollout scoring.

The lockbox provenance record is available at:

`results/lockbox/PROVENANCE.json`

The record contains hashes of frozen experiment artifacts and the preregistration document. Some referenced frozen artifacts and the preregistration source document are not included in the public snapshot.

Legacy H1/H2/H3 reports are retained as historical experiment records. Their identifiers should not be confused with the final paper's RQ1/RQ2/RQ3 organization.

## Limitations

PIU measures realized downstream utility under a specified frozen state, candidate set, and continuation policy.

Its measured value is not an intrinsic property of a document.

The evaluation focuses on Wikipedia-based multi-hop question answering and does not establish generalization to unrestricted web search or arbitrary tool-using agents.

The measured oracle requires observed intervention outcomes and is not directly deployable.

## License

Code is released under the MIT License. External models, corpora, and benchmark datasets remain subject to their respective licenses.

## Citation

This work is under review at WSDM 2027. Author-identifying citation information is omitted during anonymous review.

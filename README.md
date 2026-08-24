# Relevant, But Not Useful: Prospective Utility in Agentic Search

Code and released artifacts for the paper *"Relevant, But Not Useful: Prospective Utility in Agentic Search"*.

We measure the **prospective interventional utility (PIU)** of a retrieved document for a search agent: at states actually visited by the agent, we inject a candidate document into the trajectory, replay the agent's own continuation policy, and measure the resulting change in downstream answer quality. This separates *relevance* (what rankers score) from *utility* (what actually helps the agent's future trajectory).

## Key findings

- **H1 — Measurable actionable headroom beyond cross-encoder relevance.** A within-pool oracle that picks documents by interventional utility beats CE top-1 selection by ΔF1 **+0.117** (95% CI [+0.092, +0.142]) on a sealed HotpotQA lockbox set, replicated on 2WikiMultihopQA (**+0.142**), MuSiQue (**+0.097**), and a Qwen2.5-14B agent (**+0.097**). Cross-fitted (selection/evaluation split) oracle gain **+0.111** [+0.080, +0.144] rules out winner's-curse inflation.
- **H2 — Conditional structure transfers.** A frozen conflicting-condition specialist predicts utility direction at 66–69% accuracy across all settings (binomial p < 1e-39). Two signature coefficients (anti-alignment with the query, brevity) replicate in sign across every setting.
- **H3 — Identifiability limit.** Despite ~75% specialist ceilings, deployable gates converge at AUC 0.61–0.69 with ≈0 mixture recovery: doc-level utility is largely agent-specific (ρ(U_7B, U_14B) = 0.10), while conditional ordering structure transfers.

## Repository layout

```
piu/                Core package
  agent.py            vLLM search agent loop (search/answer tags, replay-safe)
  retriever.py        BM25 leg (bm25s over enwiki-20171001 abstracts)
  dense.py            Dense (E5) + cross-encoder legs
  intervene.py/2.py   Interventional arms: candidate injection, controls, matched distractors
  utility.py          Utility axes: U_task (raw/ctrl), U_cost, diagnostic axes
  predictor.py        Utility predictor variants
  pairwise.py         Pairwise label construction (τ-thresholded)
  state_delta.py      State featurization
scripts/            Experiment drivers and analyses
  build_corpus*.py, build_index.py, embed_corpus.py    Corpus/index preparation
  run_gate0.py, run_gate1.py, *_driver.sh              Gate execution (0→3, lockbox, stress)
  analyze_gate*.py, lockbox_eval.py                    Pre-registered analyses
  wiki2_eval.py, musique_eval.py, xagent_eval.py       Replications (2Wiki, MuSiQue, 14B agent)
  crossfit_eval.py, defense_analyses.py                Review-defense analyses (cross-fitted oracle etc.)
  train_predictor.py, gate_features.py, gate_A.py      Predictor / deployable-gate campaign
  make_figures.py                                      Paper figures
results/            Released measurement artifacts per setting
  */questions.json      Evaluation splits (HotpotQA/2Wiki/MuSiQue question ids + strata)
  */utilities.jsonl     Interventional utility labels (per state × candidate arm)
  */[SETTING]_REPORT.json  Pre-registered hypothesis test reports
  lockbox/PROVENANCE.json  Lockbox seal/open provenance
figures/            Paper figures F1–F4 (PDF + PNG)
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Requires one GPU capable of serving Qwen2.5-7B-Instruct with vLLM (H100 80GB used in the paper; the 14B replication needs the same).

### Data preparation (once)

```bash
# 1. Corpus: official HotpotQA enwiki-20171001 abstracts dump -> flat jsonl
#    https://hotpotqa.github.io/wiki-readme.html
python scripts/build_corpus_parallel.py data/raw data/corpus.jsonl
# 2. BM25 index (bm25s, ~5M docs)
python scripts/build_index.py data/corpus.jsonl data/bm25_index
# 3. Dense leg (for Gate 1+): E5 corpus embeddings
python scripts/embed_corpus.py
```

## Running

```bash
# Smoke test (mock retriever, no index needed)
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_agent.py

# Gate 0: pipeline health + cost measurement (30 questions, resumable)
CUDA_VISIBLE_DEVICES=0 python scripts/run_gate0.py --n-questions 30
python scripts/analyze_gate0.py results/gate0

# Gate 1..3, lockbox, replications: see the corresponding *_driver.sh
bash scripts/gate1_driver.sh
```

Each gate is pre-registered: the driver runs base trajectories, plans interventional arms, replays them under the frozen continuation policy, and the analysis script tests the registered hypotheses. `results/` in this repository contains the exact reports and utility labels produced by these runs.

## Released artifacts

- **Interventional utility labels** (`results/*/utilities.jsonl`): for every audited (state, candidate-document) pair — base/arm/control outcomes (EM/F1/turns/calls/tokens), task and cost utility axes, diagnostic axes, candidate type, and config hash. Documents are referenced by title (enwiki-20171001); no article text is redistributed.
- **Evaluation splits** (`results/*/questions.json`): question ids and strata for every setting, including the sealed lockbox split.
- **Hypothesis reports** (`results/*/*_REPORT.json`): the pre-registered H1/H2/H3 test outputs backing the paper's tables.
- **Review analyses** (`results/*/report.json`): choice-space, relevance-baseline, state-aware, lookahead, matching, placebo/null, and gate-ablation checks added after the initial release.
- **Post-hoc replications** (`results/xfamily_phi4/XFAMILY_REPORT.json`, `results/myopic/report.json`): a cross-family replication with `microsoft/phi-4` and a forced-answer myopic-utility comparison on the HotpotQA lockbox, both conducted after the confirmatory analyses.
- **Result ledger** (`paper/results_master.md`): consolidated experimental results and provenance notes.
- **Supplementary material** ([`supplementary.pdf`](supplementary.pdf)): compiled appendix, including the Phi-4 cross-family replication and myopic-utility appendix sections; the LaTeX source is in `paper/supplementary.tex`.

The repository intentionally excludes raw trajectories, retrieval indexes, model
checkpoints, and other large intermediate artifacts. The released labels and
summary reports are sufficient to audit the reported statistics, while the
scripts reproduce each analysis from the corresponding local experiment output.

## License

MIT (see `LICENSE`). The enwiki corpus and HotpotQA/2Wiki/MuSiQue datasets retain their original licenses and are not redistributed here.

## Citation

```bibtex
@article{piu2026,
  title  = {Relevant, But Not Useful: Prospective Utility in Agentic Search},
  author = {[authors]},
  year   = {2026},
  note   = {Under review}
}
```

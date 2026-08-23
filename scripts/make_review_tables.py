"""LaTeX tables for the reviewer-response analyses."""
import json
import os

OUT = "results/review_tables"
os.makedirs(OUT, exist_ok=True)
SET = ["HotpotQA dev", "HotpotQA lockbox", "2WikiMultiHopQA", "Qwen2.5-14B",
       "MuSiQue"]
SHORT = {"HotpotQA dev": "HotpotQA dev", "HotpotQA lockbox": "HotpotQA lockbox",
         "2WikiMultiHopQA": "2Wiki", "Qwen2.5-14B": "Qwen2.5-14B",
         "MuSiQue": "MuSiQue"}
L = []


def ci(d, p=3):
    return f"{d['mean']:+.{p}f} [{d['ci95'][0]:+.{p}f}, {d['ci95'][1]:+.{p}f}]"


# ---- T1 relevance baselines -------------------------------------------------
R = json.load(open("results/relevance_baselines/report.json"))
L += [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{Relevance baselines do not close the measured headroom.}",
      r"Oracle$-$selector difference in downstream answer F1 (question-level",
      r"paired bootstrap). \emph{CE} is the confirmatory baseline",
      r"(ms-marco-MiniLM-L6-v2 on $q_t$); \emph{BGE} is bge-reranker-v2-m3",
      r"(568M) scored either on $q_t$ or on a state-aware query",
      r"($q_0$ + $q_t$ + consumed titles); \emph{LLM} is Qwen2.5-7B listwise",
      r"ranking given the same state. Random is a within-state reference.}",
      r"\label{tab:relbaselines}",
      r"\begin{tabular}{lccccc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Random} & \textbf{CE} & \textbf{BGE $q_t$} & "
      r"\textbf{BGE state} & \textbf{LLM state} \\", r"\midrule"]
for s in SET:
    h = R[s]["headroom_oracle_minus"]
    cells = [f"{h[k]['mean']:+.3f}" if k in h else "---"
             for k in ("random", "ce", "strong_qt", "strong_state", "llm_state")]
    L.append(f"{SHORT[s]} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T2 null-shuffle --------------------------------------------------------
N = json.load(open("results/null_shuffle/report.json"))
L += [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The conditional signatures are not distinguishable from",
      r"a conditioning-only null.} Coefficients refit inside the CE-conflicting",
      r"subset, against a null that permutes measured outcomes among the",
      r"candidates of a state while holding CE scores and features fixed. The",
      r"null preserves the conditioning structure but destroys any true",
      r"feature--utility relation.}",
      r"\label{tab:nullshuffle}",
      r"\begin{tabular}{lcccc}", r"\toprule",
      r"& \multicolumn{2}{c}{\textbf{query\_align}} & "
      r"\multicolumn{2}{c}{\textbf{doc\_len}} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
      r"\textbf{Setting} & \textbf{Refit} & \textbf{Null 95\% CI} & "
      r"\textbf{Refit} & \textbf{Null 95\% CI} \\", r"\midrule"]
for s in SET:
    f = N[s]["features"]
    a, b = f["query_align"], f["doc_len"]
    L.append(f"{SHORT[s]} & {a['real_coef']:+.2f} & "
             f"[{a['null_ci95'][0]:+.2f}, {a['null_ci95'][1]:+.2f}] & "
             f"{b['real_coef']:+.2f} & "
             f"[{b['null_ci95'][0]:+.2f}, {b['null_ci95'][1]:+.2f}] \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T3 placebo oracle ------------------------------------------------------
G = json.load(open("results/placebo_oracle/greedy_vs_sampled_replicates.json"))
L += [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The headroom is not a maximum taken over noise.}",
      r"Both panels use the same 137 audited lockbox states, three rollouts each.",
      r"\emph{Split} selects the candidate on one rollout and evaluates it on the",
      r"held-out rollouts. \emph{Matched-noise null} gives every candidate in a",
      r"state the same expected outcome and perturbs it with the observed",
      r"within-candidate rollout deviations.}",
      r"\label{tab:placebo}",
      r"\begin{tabular}{lccc}", r"\toprule",
      r"\textbf{Policy} & \textbf{Same-rollout} & \textbf{Split} & "
      r"\textbf{Matched-noise null} \\", r"\midrule"]
for key, lab in (("greedy_replays", "Greedy ($T{=}0$, as reported)"),
                 ("sampled_same_states", "Sampled ($T{=}0.7$)")):
    d = G[key]
    o = d["observed_select_and_eval_same"]
    n = d["matched_noise_null"]
    L.append(f"{lab} & {o['mean']:+.3f} & {d['split_mean_over_folds']:+.3f} & "
             f"{n['null_mean']:+.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T4 stratum -------------------------------------------------------------
S = json.load(open("results/stratum_headroom/report.json"))
L += [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The headroom is not produced by failure enrichment.}",
      r"Oracle$-$CE by preregistered sampling stratum; the representative stratum",
      r"is an unenriched sample of the benchmark.}",
      r"\label{tab:stratum}",
      r"\begin{tabular}{lccc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Pooled} & \textbf{Representative} & "
      r"\textbf{Failure-enriched} \\", r"\midrule"]
for s in SET:
    d = S[s]
    L.append(f"{SHORT[s]} & {ci(d['all']['headroom'])} & "
             f"{ci(d['representative']['headroom'])} & "
             f"{ci(d['fail_enriched']['headroom'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex = "\n".join(L)
open(f"{OUT}/tables.tex", "w").write(tex + "\n")
print(tex)
print("\nREVIEW_TABLES_DONE")

# ---- T5 paired selector comparison (A + B) ---------------------------------
P = json.load(open("results/paired_selectors/report.json"))
L2 = [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{Relevance selection is informative, and the headroom",
      r"survives stronger and state-aware selectors.} Absolute downstream F1 of",
      r"each selector and the Oracle$-$selector headroom, question-level paired",
      r"bootstrap. \emph{Random} is the exact expectation of a uniform pick over",
      r"the candidate set. CE is ms-marco-MiniLM-L6-v2 on $q_t$; BGE is",
      r"bge-reranker-v2-m3 scored on $q_t$ or on a state-aware query.}",
      r"\label{tab:selectors}",
      r"\begin{tabular}{llcc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Selector} & \textbf{F1} & "
      r"\textbf{Oracle$-$selector} \\", r"\midrule"]
ROWS = [("random_expected", "Random"), ("ce", "CE (MiniLM, $q_t$)"),
        ("strong_qt", "BGE ($q_t$)"), ("strong_state", "BGE (state-aware)"),
        ("oracle", "Oracle")]
for si, s in enumerate(SET):
    d = P[s]
    for k, lab in ROWS:
        if k not in d["abs_f1"]:
            continue
        h = d["headroom_oracle_minus"][k]
        hs = "---" if k == "oracle" else \
            f"{h['mean']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
        first = SHORT[s] if k == ROWS[0][0] else ""
        L2.append(f"{first} & {lab} & {d['abs_f1'][k]['mean']:.3f} & {hs} \\\\")
    cr = d["paired_ce_minus_random"]
    L2.append(r"\multicolumn{2}{r}{\emph{CE$-$Random (paired)}} & "
              f"\\multicolumn{{2}}{{l}}{{{cr['mean']:+.3f} "
              f"[{cr['ci95'][0]:+.3f}, {cr['ci95'][1]:+.3f}]}} \\\\")
    L2.append(r"\midrule" if si < len(SET) - 1 else r"\bottomrule")
L2 += [r"\end{tabular}", r"\end{table}", ""]

# ---- T6 gate ablation (C) ---------------------------------------------------
A = json.load(open("results/rq3_gate_ablation/report.json"))["settings"]
L2 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{States containing prospective headroom are hard to",
       r"identify before intervention.} AUC of the state-level gate by feature",
       r"set, with prevalence-adjusted metrics for the full gate. AUC is",
       r"insensitive to prevalence; average precision has the prevalence itself",
       r"as its random baseline, so AP/prevalence is the achieved lift.}",
       r"\label{tab:gate}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"& & \multicolumn{3}{c}{\textbf{AUC}} & \\",
       r"\cmidrule(lr){3-5}",
       r"\textbf{Setting} & \textbf{Prev.} & \textbf{CE stats} & "
       r"\textbf{$\varphi$ agg.} & \textbf{Full} & \textbf{AP/prev.} \\",
       r"\midrule"]
for s in SET:
    d = A[s]
    f = d["feature_sets"]
    L2.append(f"{SHORT[s]} & {d['prevalence']:.3f} & {f['ce_only']['auc']:.3f} & "
              f"{f['phi_only']['auc']:.3f} & {f['full']['auc']:.3f} & "
              f"{f['full']['ap_over_prevalence']:.2f} \\\\")
L2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex2 = "\n".join(L2)
with open(f"{OUT}/tables.tex", "a") as fh:
    fh.write("\n" + tex2 + "\n")
print(tex2)
print("\nEXTRA_TABLES_DONE")

# ---- T7 selector relevance quality on the full pool (a) ---------------------
FP = json.load(open("results/selector_relevance/fullpool_report.json"))
L3 = [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The stronger selectors are genuinely stronger relevance",
      r"models, yet this does not transfer downstream.} Gold hit@1 and nDCG@10",
      r"over the full candidate pool $\mathcal{C}_t$ (about 110 unseen documents",
      r"per state), scored against the datasets' annotated supporting documents;",
      r"states whose pool contains no annotated document are excluded and",
      r"reported as coverage. The last column is the paired selector$-$CE",
      r"difference in hit@1 with question-level 95\% bootstrap intervals.}",
      r"\label{tab:relquality}",
      r"\begin{tabular}{llccc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Selector} & \textbf{hit@1} & "
      r"\textbf{nDCG@10} & \textbf{$\Delta$hit@1 vs.\ CE} \\", r"\midrule"]
RQ = [("random", "Random"), ("ce", "CE (MiniLM, $q_t$)"),
      ("bge_qt", "BGE ($q_t$)"), ("bge_state", "BGE (state-aware)")]
for si, s in enumerate(SET):
    d = FP[s]
    for k, lab in RQ:
        dh = d["paired_vs_ce"].get(k, {}).get("hit_at_1")
        ds = "---" if dh is None else \
            f"{dh['mean']:+.3f} [{dh['ci95'][0]:+.3f}, {dh['ci95'][1]:+.3f}]"
        first = f"{SHORT[s]} ({d['gold_coverage']:.0%})".replace("%", r"\%") \
            if k == RQ[0][0] else ""
        L3.append(f"{first} & {lab} & {d['hit_at_1'][k]['mean']:.3f} & "
                  f"{d['ndcg_at_10'][k]['mean']:.3f} & {ds} \\\\")
    L3.append(r"\midrule" if si < len(SET) - 1 else r"\bottomrule")
L3 += [r"\end{tabular}", r"\end{table}", ""]

# ---- T8 all gate families incl. learned gates (c) --------------------------
AG = json.load(open("results/rq3_gate_ablation/all_gates.json"))
L3 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{No gate family identifies headroom states well.}",
       r"ROC-AUC for the state-level label $Z(s_t)$. The learned gates are",
       r"fine-tuned encoders trained on the same 2{,}368 training-split states;",
       r"\emph{Best} takes the most favourable family per setting, chosen post",
       r"hoc, and is therefore an optimistic bound.}",
       r"\label{tab:allgates}",
       r"\begin{tabular}{lcccccc}", r"\toprule",
       r"\textbf{Setting} & \textbf{Prev.} & \textbf{CE stats} & "
       r"\textbf{$\varphi$ agg.} & \textbf{Full feat.} & "
       r"\textbf{Transformer} & \textbf{Best} \\", r"\midrule"]
for s in SET:
    d = AG[s]
    L3.append(f"{SHORT[s]} & {d['prevalence']:.3f} & {d['ce_only']:.3f} & "
              f"{d['phi_only']:.3f} & {d['full']:.3f} & "
              f"{d['neural_best']:.3f} & {d['best_any_gate']:.3f} \\\\")
L3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T9 the two opportunity definitions (d) --------------------------------
OD = json.load(open("results/opportunity_definitions/report.json"))
L3 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{Two distinct positive-state rates.}",
       r"\emph{Opp-base} counts states where some candidate beats the",
       r"pre-intervention trajectory; \emph{Opp-CE} counts states where some",
       r"candidate beats the CE-selected candidate, which is the state-level",
       r"label $Z(s_t)$ used for prospective identification. They are different",
       r"quantities and are never interchangeable.}",
       r"\label{tab:oppdefs}",
       r"\begin{tabular}{lcccc}", r"\toprule",
       r"& \multicolumn{2}{c}{\textbf{Opp-base}} & "
       r"\multicolumn{2}{c}{\textbf{Opp-CE}} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"\textbf{Setting} & \textbf{$>0$} & \textbf{$\ge$0.25} & "
       r"\textbf{$>0$} & \textbf{$\ge$0.25} \\", r"\midrule"]
for s in SET:
    d = OD[s]
    L3.append(f"{SHORT[s]} & {d['Opp_base']['gt0']:.1%} & "
              f"{d['Opp_base']['ge025']:.1%} & {d['Opp_CE']['gt0']:.1%} & "
              f"{d['Opp_CE']['ge025']:.1%} \\\\".replace("%", r"\%"))
L3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex3 = "\n".join(L3)
with open(f"{OUT}/tables.tex", "a") as fh:
    fh.write("\n" + tex3 + "\n")
print(tex3)
print("\nFINAL_TABLES_DONE")

# ---- T10 W1 trajectory decomposition ---------------------------------------
W1 = json.load(open("results/w1_trajectory/report.json"))
L4 = [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The headroom is concentrated where the intervention",
      r"changes subsequent search.} States are split by what the two compared",
      r"arms ($d_{\mathrm{PIU}}$ and $d_{\mathrm{CE}}$) do after the",
      r"intervention. \emph{No search}: neither issues a further search.",
      r"\emph{Any search}: at least one does. \emph{Query changed}: at least one",
      r"searches and the first post-intervention query differs between the arms.",
      r"Oracle$-$CE with question-level 95\% paired-bootstrap intervals.}",
      r"\label{tab:w1}",
      r"\begin{tabular}{llcc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Subset} & \textbf{States} & "
      r"\textbf{Oracle$-$CE} \\", r"\midrule"]
W1ROWS = [("no_search", "No further search"), ("all", "All states"),
          ("any_search", "Any further search"),
          ("query_changed", "Query changed")]
for si, s in enumerate(SET):
    d = W1[s]
    for k, lab in W1ROWS:
        h = d[k]["headroom"]
        first = SHORT[s] if k == W1ROWS[0][0] else ""
        L4.append(f"{first} & {lab} & {d[k]['n_states']} & "
                  f"{h['mean']:+.3f} [{h['ci95'][0]:+.3f}, "
                  f"{h['ci95'][1]:+.3f}] \\\\")
    L4.append(r"\midrule" if si < len(SET) - 1 else r"\bottomrule")
L4 += [r"\end{tabular}", r"\end{table}", ""]

# ---- T11 W6 controlled PIU by selected candidate ---------------------------
W6 = json.load(open("results/w6_controlled/report.json"))
L4 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{Controlled PIU for the candidate each rule selects.}",
       r"$Y(s_t,E_{<t}\cup\{d\})-Y(s_t,E_{<t}\cup\{\tilde d\})$ against the",
       r"candidate's own matched control. Averaging over all candidates (the",
       r"\emph{Random} row) is near zero by construction; the selected-candidate",
       r"rows are informative. The oracle row is chosen on the raw outcome and",
       r"scored on the controlled one, so the random and anti-oracle rows bound",
       r"its selection effect.}",
       r"\label{tab:w6}",
       r"\begin{tabular}{lcccc}", r"\toprule",
       r"\textbf{Setting} & \textbf{Anti-oracle} & \textbf{Random} & "
       r"\textbf{CE} & \textbf{Oracle} \\", r"\midrule"]
for s in SET:
    u = W6[s]["U_ctrl"]
    L4.append(f"{SHORT[s]} & " + " & ".join(
        f"{u[k]['mean']:+.3f}" for k in
        ("anti_oracle", "random", "ce", "oracle")) + r" \\")
L4 += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T12 lookahead cost-accuracy -------------------------------------------
LA = json.load(open("results/lookahead/report.json"))
DG = json.load(open("results/lookahead/diagnostics.json"))
L4 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{A budgeted lookahead concentrates most of the",
       r"headroom, but cheap gold-free scoring does not exploit it.}",
       r"Rolling out the CE top-$k$ candidates: \emph{Oracle-$k$} keeps the best",
       r"measured outcome (an upper bound; it needs a scored outcome),",
       r"\emph{Vote} answers by majority over the $k$ continuations and uses no",
       r"gold. \emph{Recovered} is the fraction of the full 8-candidate headroom.",
       r"The last column reports how often the useful candidate's answer is the",
       r"modal answer across the candidate set, on opportunity states.}",
       r"\label{tab:lookahead}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"& \multicolumn{2}{c}{\textbf{Oracle-$k$ recovered}} & "
       r"\multicolumn{2}{c}{\textbf{Vote F1}} & \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"\textbf{Setting} & $k{=}2$ & $k{=}3$ & $k{=}1$ (CE) & $k{=}3$ & "
       r"\textbf{Oracle modal} \\", r"\midrule"]
for s in SET:
    d, g = LA[s], DG[s]
    L4.append(f"{SHORT[s]} & "
              f"{d['oracle_k']['2']['recovered_fraction']:.0%} & "
              f"{d['oracle_k']['3']['recovered_fraction']:.0%} & "
              f"{d['vote']['1']['mean']:.3f} & {d['vote']['3']['mean']:.3f} & "
              f"{g['oracle_answer_is_modal']:.0%} \\\\".replace("%", r"\%"))
L4 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex4 = "\n".join(L4)
with open(f"{OUT}/tables.tex", "a") as fh:
    fh.write("\n" + tex4 + "\n")
print(tex4)
print("\nW1_W6_LOOKAHEAD_TABLES_DONE")

# ---- T13 continuation scorer (A급) ------------------------------------------
CS = json.load(open("results/continuation_scorer/report_v2.json"))
L5 = [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{A conservative continuation scorer converts lookahead",
      r"into small but significant end-to-end gains.} The scorer sees only",
      r"observable, gold-free signals of the CE top-3 continuations and",
      r"overrides the CE choice only when its score margin exceeds a threshold",
      r"tuned once on the development setting. \emph{Pairwise acc.} is its",
      r"accuracy on decided continuation pairs; \emph{Oracle@3} bounds what any",
      r"selector could achieve at this budget.}",
      r"\label{tab:scorer}",
      r"\begin{tabular}{lcccc}", r"\toprule",
      r"\textbf{Setting} & \textbf{Overrides} & \textbf{Gain vs.\ CE} & "
      r"\textbf{Oracle@3} & \textbf{Pairwise acc.} \\", r"\midrule"]
for s in SET:
    d = CS["settings"][s]
    g, k = d["gain_vs_ce"], d["oracle_at_k"]
    L5.append(f"{SHORT[s]} & {d['n_overrides']}/{d['n_states']} & "
              f"{g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] & "
              f"{k['mean']:+.3f} & {d['pairwise_acc_decided']:.3f} \\\\")
L5 += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ---- T14 oracle winner characterization (B급) -------------------------------
OW = json.load(open("results/oracle_winner/report.json"))
L5 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{Utility winners are near the top of the relevance",
       r"ranking.} On states with CE-relative headroom, the CE rank of the",
       r"oracle-winning candidate within $\mathcal{D}_t$ (rank 1 is impossible",
       r"by definition), and the share of failure states whose winner is",
       r"already in the CE top-2 / top-3.}",
       r"\label{tab:winnerrank}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"\textbf{Setting} & \textbf{States} & \textbf{Median rank} & "
       r"\textbf{Top-2} & \textbf{Top-3} & \textbf{Rank 5+} \\", r"\midrule"]
for s in SET:
    d = OW[s]
    r = d["ce_rank_of_oracle_winner"]
    L5.append(f"{SHORT[s]} & {d['n_failure_states']} & {r['median']:.0f} & "
              f"{r['share_in_top2']:.1%} & {r['share_in_top3']:.1%} & "
              f"{r['dist']['5+']}/{d['n_failure_states']} \\\\".replace("%", r"\%"))
L5 += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

L5 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{Null-controlled characterisation of utility winners.}",
       r"Paired feature difference $\varphi(d_{\mathrm{PIU}})-\varphi(d_{\mathrm{CE}})$",
       r"on failure states, against a within-state outcome-permutation null that",
       r"preserves the mechanical part of the contrast (the winner is never the",
       r"CE argmax). A cell is a finding only when the real value falls outside",
       r"the null band; $\uparrow$/$\downarrow$ marks the direction relative to",
       r"the null, and --- marks cells inside it.}",
       r"\label{tab:winnerchar}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"\textbf{Setting} & \textbf{query\_align} & \textbf{q0\_align} & "
       r"\textbf{doc\_len} & \textbf{redundancy} & \textbf{novelty} \\",
       r"\midrule"]
for s in SET:
    f = OW[s]["features"]
    cells = []
    for k in ("query_align", "q0_align", "doc_len", "redundancy", "novelty"):
        d = f[k]
        if not d["outside_null"]:
            cells.append("---")
        else:
            arrow = r"$\uparrow$" if d["real"] > d["null_ci95"][1] else r"$\downarrow$"
            cells.append(f"{arrow} {d['real']:+.3f}")
    L5.append(f"{SHORT[s]} & " + " & ".join(cells) + r" \\")
L5 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex5 = "\n".join(L5)
with open(f"{OUT}/tables.tex", "a") as fh:
    fh.write("\n" + tex5 + "\n")
print(tex5[-800:])
print("\nSCORER_WINNER_TABLES_DONE")

# ---- T16/T17 choice-space consistency (reviewer: C_t vs D_t) ----------------
CX = json.load(open("results/choice_space/report_expanded.json"))
L6 = [r"\begin{table}[t]", r"\centering", r"\small",
      r"\caption{\textbf{The relevance advantage is present on the exact",
      r"downstream choice space.} \emph{Retained}: fraction of states whose",
      r"full-pool argmax is contained in the retained intervention set",
      r"$\mathcal{D}_t$ (guaranteed for CE by construction). \emph{Hit@1}:",
      r"supporting-document hit@1 over $\mathcal{D}_t$ on states whose retained",
      r"set contains an annotated document, with the paired BGE-state$-$CE",
      r"difference.}",
      r"\label{tab:choicespace}",
      r"\begin{tabular}{lcccccc}", r"\toprule",
      r"& \multicolumn{2}{c}{\textbf{Retained}} & "
      r"\multicolumn{3}{c}{\textbf{Hit@1 over $\mathcal{D}_t$}} & \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
      r"\textbf{Setting} & \textbf{BGE} & \textbf{BGE-St} & \textbf{CE} & "
      r"\textbf{BGE} & \textbf{BGE-St} & \textbf{$\Delta$(BGE-St$-$CE)} \\",
      r"\midrule"]
for s in SET:
    d = CX[s]
    a = d["A_fullpool_argmax_retained_in_Dt"]
    h = d["B_hit_at_1_over_Dt"]
    p = d["B_paired_vs_ce"]["bge_state"]
    L6.append(f"{SHORT[s]} & {a['bge_qt']:.2f} & {a['bge_state']:.2f} & "
              f"{h['ce']['mean']:.3f} & {h['bge_qt']['mean']:.3f} & "
              f"{h['bge_state']['mean']:.3f} & "
              f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] \\\\")
L6 += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

L6 += [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{\textbf{Granting the stronger selectors their genuine",
       r"full-pool choice does not close the gap.} Each selector's full-pool",
       r"argmax was injected and rolled out under the frozen greedy policy",
       r"wherever it was not already retained (3{,}861 new arms; post-hoc",
       r"exploratory). All rows share the identical state set; the expanded",
       r"oracle maximises over $\mathcal{D}_t$ plus the new arms.}",
       r"\label{tab:expanded}",
       r"\begin{tabular}{lcccc}", r"\toprule",
       r"\textbf{Setting} & \textbf{BGE$-$CE} & \textbf{BGE-St$-$CE} & "
       r"\textbf{Oracle($\mathcal{D}_t$)$-$CE} & \textbf{Oracle(exp.)$-$CE} \\",
       r"\midrule"]
for s in SET:
    q = CX[s]["C_paired_vs_ce"]
    def fmt(k):
        g = q[k]
        return f"{g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]"
    L6.append(f"{SHORT[s]} & {fmt('bge_qt_full')} & {fmt('bge_state_full')} & "
              f"{q['oracle_Dt']['mean']:+.3f} & {q['oracle_expanded']['mean']:+.3f} \\\\")
L6 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex6 = "\n".join(L6)
with open(f"{OUT}/tables.tex", "a") as fh:
    fh.write("\n" + tex6 + "\n")
print(tex6[-600:])
print("\nCHOICE_SPACE_TABLES_DONE")

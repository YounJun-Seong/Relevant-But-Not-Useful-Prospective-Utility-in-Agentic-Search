"""Gate 1 report: the 5 headline numbers (R8), gold/non-gold usefulness rates,
per-ctype tables, search-case split, stratum split, token accounting."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gate1"
EPS = 1e-9

utils = read_jsonl(os.path.join(OUT, "utilities.jsonl"))
omits = read_jsonl(os.path.join(OUT, "omission_utilities.jsonl"))
ce_rank = {r["state_id"]: r for r in read_jsonl(os.path.join(OUT, "ce_rankings.jsonl"))}
arm_results = {r["arm_id"]: r for r in read_jsonl(os.path.join(OUT, "arm_results.jsonl"))}
with open(os.path.join(OUT, "questions.json")) as f:
    qmeta = json.load(f)
base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
stratum = {s["question_id"]: s["stratum"] for s in qmeta["selected"]}

by_state = defaultdict(list)
for u in utils:
    by_state[(u["question_id"], u["state_idx"])].append(u)

lines = ["# Gate 1 결과 보고 — PIU 5대 지표\n"]
lines.append(f"run dir: `{OUT}` · 개입 {len(utils)}건 · state {len(by_state)}개 · "
             f"질문 {len({u['question_id'] for u in utils})}개\n")

# ---------- ① Opportunity Rate ----------
def opp_rate(strata=None):
    states = [k for k in by_state if strata is None or stratum.get(k[0]) == strata]
    dec = [k for k in states
           if max(r["U_task_raw_f1"] for r in by_state[k]) > EPS]
    return len(dec), len(states)

d_all, n_all = opp_rate()
lines.append("## ① Opportunity Rate — P(max_d PIU(d) > 0)\n")
lines.append(f"- 전체: **{d_all}/{n_all} = {d_all/max(n_all,1):.1%}** (Gate 0: 6.5%)")
for s in ("representative", "fail_enriched"):
    d, n = opp_rate(s)
    lines.append(f"- {s}: {d}/{n} = {d/max(n,1):.1%}")
ctrl_dec = [k for k in by_state
            if max((r.get("U_task_ctrl_f1") or -1) for r in by_state[k]) > EPS]
lines.append(f"- (참고) ctrl 기준 결정적 state: {len(ctrl_dec)}/{len(by_state)} "
             f"= {len(ctrl_dec)/max(len(by_state),1):.1%}\n")

# ---------- ②③ Oracle vs CE (among the measured candidates) ----------
gains, disagree, decisive_states = [], 0, 0
for (qid, sidx), rows in by_state.items():
    state_id = f"{qid}::s{sidx}"
    cr = ce_rank.get(state_id)
    if not cr:
        continue
    order = {t: i for i, t in enumerate(cr["ce_ranking"])}
    ce_pick = min(rows, key=lambda r: order.get(r["doc"], 10**6))
    pi_pick = max(rows, key=lambda r: r["U_task_raw_f1"])
    if pi_pick["U_task_raw_f1"] > EPS:
        decisive_states += 1
        if pi_pick["doc"] != ce_pick["doc"]:
            disagree += 1
    gains.append(pi_pick["arm_d"]["f1"] - ce_pick["arm_d"]["f1"])
mean_gain = sum(gains) / max(len(gains), 1)
lines.append("## ② Oracle Gain — Y(Oracle PIU 선택) − Y(CE 선택)\n")
lines.append(f"- 평균 ΔF1: **{mean_gain:+.4f}** (state {len(gains)}개; "
             f"동률 state에서는 0에 수렴하는 보수적 추정)")
pos_g = sum(1 for g in gains if g > EPS)
lines.append(f"- Oracle이 CE를 이긴 state: {pos_g}/{len(gains)}\n")
lines.append("## ③ Selection Disagreement — 결정적 state에서 argmax 불일치\n")
lines.append(f"- **{disagree}/{decisive_states}** "
             f"(결정적 state 한정; 동률 state 제외)\n")

# ---------- ④ Recoverable Failure Rate ----------
fail_qids = [qid for qid, s in stratum.items()
             if base_pre[qid]["outcome"]["em"] == 0.0]
recovered = set()
for u in utils:
    if u["question_id"] in fail_qids and u["arm_d"]["em"] == 1.0:
        recovered.add(u["question_id"])
lines.append("## ④ Recoverable Failure Rate\n")
lines.append(f"- base 실패 질문 {len(fail_qids)}개 중 어떤 개입으로든 EM=1 복구: "
             f"**{len(recovered)}개 = {len(recovered)/max(len(fail_qids),1):.1%}**\n")

# ---------- ⑤ Controlled Utility Stability ----------
dual = [u for u in utils if u.get("U_task_ctrl2_f1") is not None
        and u.get("U_task_ctrl_f1") is not None]
import math
def sign(x):
    return 0 if abs(x) <= EPS else (1 if x > 0 else -1)
agree = sum(1 for u in dual
            if sign(u["U_task_ctrl_f1"]) == sign(u["U_task_ctrl2_f1"]))
nz = [u for u in dual if abs(u["U_task_ctrl_f1"]) > EPS
      or abs(u["U_task_ctrl2_f1"]) > EPS]
agree_nz = sum(1 for u in nz
               if sign(u["U_task_ctrl_f1"]) == sign(u["U_task_ctrl2_f1"]))
lines.append("## ⑤ Controlled Utility Stability — 독립 control 2개 일치도\n")
lines.append(f"- dual-control 개입: {len(dual)}건")
lines.append(f"- sign 일치 (전체): {agree}/{len(dual)} = {agree/max(len(dual),1):.1%}")
lines.append(f"- sign 일치 (둘 중 하나라도 ≠0인 {len(nz)}건): "
             f"{agree_nz}/{max(len(nz),1)} = {agree_nz/max(len(nz),1):.1%}\n")

# ---------- Gold/Non-gold usefulness ----------
gold = [u for u in utils if u["doc_is_gold"]]
nong = [u for u in utils if not u["doc_is_gold"]]
gnu = sum(1 for u in gold if u["U_task_raw_f1"] <= EPS)
ngu = sum(1 for u in nong if u["U_task_raw_f1"] > EPS)
lines.append("## Gold-but-Non-Useful / Non-Gold-but-Useful (Figure 후보)\n")
lines.append(f"- P(PIU≤0 | d∈Gold) = {gnu}/{len(gold)} = "
             f"{gnu/max(len(gold),1):.1%}")
lines.append(f"- P(PIU>0 | d∉Gold) = {ngu}/{len(nong)} = "
             f"{ngu/max(len(nong),1):.1%}\n")

# ---------- per-ctype ----------
lines.append("## 후보 유형별 (training 축)\n")
lines.append("| ctype | n | mean ΔF1 raw | raw>0 | mean ΔF1 ctrl | ctrl>0 |")
lines.append("|---|---|---|---|---|---|")
by_type = defaultdict(list)
for u in utils:
    by_type[u["ctype"]].append(u)
for ctype in ["bm25_top", "dense_top", "ce_top", "mid_rel", "novelty",
              "redundancy", "entity_bridge", "random"]:
    rows = by_type.get(ctype, [])
    if not rows:
        continue
    n = len(rows)
    ctrl = [r["U_task_ctrl_f1"] for r in rows if r["U_task_ctrl_f1"] is not None]
    lines.append(
        f"| {ctype} | {n} | {sum(r['U_task_raw_f1'] for r in rows)/n:+.4f} "
        f"| {sum(1 for r in rows if r['U_task_raw_f1'] > EPS)} "
        f"| {sum(ctrl)/max(len(ctrl),1):+.4f} "
        f"| {sum(1 for c in ctrl if c > EPS)} |")
lines.append("")

# ---------- search case split ----------
cases = defaultdict(int)
for u in utils:
    cases[u["diag"]["search_case"]] += 1
lines.append("## U_search 케이스 분리 (R8-7)\n")
for c, n in sorted(cases.items(), key=lambda x: -x[1]):
    lines.append(f"- {c}: {n} ({n/len(utils):.1%})")
both = [u for u in utils if u["diag"]["search_case"] == "both_search"]
if both:
    ms = sum(u["diag"]["U_search"] for u in both) / len(both)
    lines.append(f"- both_search에서의 mean U_search: {ms:+.4f}")
lines.append("")

# ---------- omission sanity ----------
if omits:
    g = [o for o in omits if o["doc_is_gold"]]
    n = [o for o in omits if not o["doc_is_gold"]]
    lines.append("## I1 Omission sanity\n")
    if g:
        lines.append(f"- gold 제거 CTU: {sum(o['ctu_f1'] for o in g)/len(g):+.4f} (n={len(g)})")
    if n:
        lines.append(f"- non-gold 제거 CTU: {sum(o['ctu_f1'] for o in n)/len(n):+.4f} (n={len(n)})")
    lines.append("")

# ---------- token accounting ----------
pt = sum(r.get("prompt_tokens", 0) for r in arm_results.values())
gt = sum(r["gen_tokens"] for r in arm_results.values())
lines.append("## 토큰 회계 (R8-8)\n")
lines.append(f"- arm 수: {len(arm_results)} · prompt tokens: {pt:,} · "
             f"completion tokens: {gt:,} · total: {pt+gt:,}")
lines.append(f"- rollout당: prompt {pt/max(len(arm_results),1):,.0f} / "
             f"completion {gt/max(len(arm_results),1):,.0f}")
try:
    with open(os.path.join(OUT, "timings.json")) as f:
        t = json.load(f)
    lines.append(f"- phase arms wall-clock: {t['phase_arms_sec']:.0f}s")
except FileNotFoundError:
    pass

path = os.path.join(OUT, "GATE1_REPORT.md")
with open(path, "w") as f:
    f.write("\n".join(lines))
print(path)

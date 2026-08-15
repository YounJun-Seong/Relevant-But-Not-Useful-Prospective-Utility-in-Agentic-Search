"""Gate 2 official analysis (R10 pre-registered):
P1 Opportunity (stratified) · P2 Oracle vs CE question-level paired bootstrap ·
P3 Recoverable Failure · P4 Pairwise Consistency (strict/majority/strength) ·
τ sensitivity {0.05, 0.1, 0.2} · Conditional Utility Analysis · secondary."""
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, f1_score, norm_title
from piu.pairwise import build_pairs

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gate2"
EPS = 1e-9
TAU_MAIN = 0.1

utils = read_jsonl(os.path.join(OUT, "utilities.jsonl"))
plan = read_jsonl(os.path.join(OUT, "arm_plan.jsonl"))
arm_results = {r["arm_id"]: r for r in read_jsonl(os.path.join(OUT, "arm_results.jsonl"))}
ce_rank = {r["state_id"]: r for r in read_jsonl(os.path.join(OUT, "ce_rankings.jsonl"))}
omits = read_jsonl(os.path.join(OUT, "omission_utilities.jsonl"))
with open(os.path.join(OUT, "questions.json")) as f:
    qmeta = json.load(f)
base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
stratum = {s["question_id"]: s["stratum"] for s in qmeta["selected"]}

by_state = defaultdict(list)
for u in utils:
    by_state[(u["question_id"], u["state_idx"])].append(u)

L = ["# Gate 2 공식 판정 보고 (R10 사전 등록 준수)\n"]
L.append(f"run dir: `{OUT}` · 질문 {len({u['question_id'] for u in utils})} · "
         f"state {len(by_state)} · 개입 {len(utils)}건 · τ_main = {TAU_MAIN} (사전 등록)\n")

# ---------------- P1 Opportunity ----------------
L.append("## P1. Opportunity Rate\n")
for name, pred in [("전체", lambda q: True),
                   ("representative", lambda q: stratum.get(q) == "representative"),
                   ("fail_enriched", lambda q: stratum.get(q) == "fail_enriched")]:
    ks = [k for k in by_state if pred(k[0])]
    dec = sum(1 for k in ks if max(r["U_task_raw_f1"] for r in by_state[k]) > EPS)
    L.append(f"- {name}: {dec}/{len(ks)} = {dec/max(len(ks),1):.1%}")
L.append("")

# ---------------- P2 Oracle vs CE, question-level paired bootstrap ----------
per_q_gain = defaultdict(list)
for (qid, sidx), rows in by_state.items():
    cr = ce_rank.get(f"{qid}::s{sidx}")
    if not cr:
        continue
    order = {t: i for i, t in enumerate(cr["ce_ranking"])}
    ce_pick = min(rows, key=lambda r: order.get(r["doc"], 10**6))
    pi_pick = max(rows, key=lambda r: r["U_task_raw_f1"])
    per_q_gain[qid].append(pi_pick["arm_d"]["f1"] - ce_pick["arm_d"]["f1"])
q_gains = [sum(v) / len(v) for v in per_q_gain.values()]
rng = random.Random(7)
n_b = 10000
boots = []
for _ in range(n_b):
    s = [q_gains[rng.randrange(len(q_gains))] for _ in q_gains]
    boots.append(sum(s) / len(s))
boots.sort()
mean_gain = sum(q_gains) / len(q_gains)
lo, hi = boots[int(0.025 * n_b)], boots[int(0.975 * n_b)]
L.append("## P2. Oracle Outcome Gain vs strongest CE (질문 단위 paired bootstrap)\n")
L.append(f"- 질문 수: {len(q_gains)} · mean ΔF1 = **{mean_gain:+.4f}**, "
         f"95% CI **[{lo:+.4f}, {hi:+.4f}]**"
         + (" — **CI가 0을 제외 → 유의**" if lo > 0 else " — CI가 0 포함"))
pos_q = sum(1 for g in q_gains if g > EPS)
neg_q = sum(1 for g in q_gains if g < -EPS)
L.append(f"- 질문별 gain: 양수 {pos_q} / 음수 {neg_q} / 0 {len(q_gains)-pos_q-neg_q}\n")

# ---------------- P3 Recoverable Failure ----------------
fail_qids = [q for q in stratum if base_pre[q]["outcome"]["em"] == 0.0]
recovered = {u["question_id"] for u in utils
             if u["question_id"] in fail_qids and u["arm_d"]["em"] == 1.0}
L.append("## P3. Recoverable Failure Rate\n")
L.append(f"- base 실패 {len(fail_qids)}문항 중 복구 가능: **{len(recovered)} = "
         f"{len(recovered)/max(len(fail_qids),1):.1%}**\n")

# ---------------- P4 Pairwise Consistency (3단계, R10) ----------------
for p in plan:
    if p["arm_type"] in ("I2_add", "I2_add_rep"):
        r = arm_results.get(p["arm_id"])
        if r is not None and "_f1" not in r:
            r["_f1"] = f1_score(r["answer"] or "",
                                base_pre[p["question_id"]]["answer_gold"])
reps = defaultdict(list)
for p in plan:
    if p["arm_type"] in ("I2_add", "I2_add_rep"):
        r = arm_results.get(p["arm_id"])
        if r is not None:
            reps[(p["question_id"], p["state_idx"], p["doc"])].append(r)
pairs_main = build_pairs(utils, tau=TAU_MAIN)
decided_main = [p for p in pairs_main if not p["tie"]]
strict = majority = checked = 0
strengths = []
for pair in decided_main:
    ki = (pair["question_id"], pair["state_idx"], pair["doc_i"])
    kj = (pair["question_id"], pair["state_idx"], pair["doc_j"])
    if len(reps.get(ki, [])) < 3 or len(reps.get(kj, [])) < 3:
        continue
    checked += 1
    match = 0
    total = 0
    for ri, rj in itertools.product(reps[ki][:3], reps[kj][:3]):
        m = ri["_f1"] - rj["_f1"]
        got = "i" if m > TAU_MAIN else ("j" if m < -TAU_MAIN else None)
        total += 1
        if got == pair["pref"]:
            match += 1
    strengths.append(match / total)
    strict += int(match == total)
    majority += int(match > total / 2)
L.append("## P4. Pairwise Utility Consistency (3×3 replay, 결정 쌍 한정)\n")
if checked:
    L.append(f"- 검증 가능 쌍: {checked}")
    L.append(f"- **Strict** (9/9 일치): {strict}/{checked} = {strict/checked:.1%}")
    L.append(f"- **Majority** (>50% 일치): {majority}/{checked} = {majority/checked:.1%}")
    hist = Counter(round(s, 1) for s in strengths)
    L.append(f"- **Preference Strength** P(d_i≻d_j) 분포: "
             f"{dict(sorted(hist.items(), reverse=True))}")
    ge67 = sum(1 for s in strengths if s >= 0.67)
    L.append(f"- P≥0.67 비율 (training filter 후보): {ge67}/{checked} = "
             f"{ge67/checked:.1%}\n")
else:
    L.append("- 검증 가능 쌍 없음 (reliability-by-state 설정 확인 필요)\n")

# ---------------- τ sensitivity ----------------
L.append("## τ Sensitivity (사전 등록: main=0.1 고정, 민감도 전용)\n")
L.append("| τ | decided | decided% | main(0.1) 라벨과 방향 일치 |")
L.append("|---|---|---|---|")
main_lab = {(p["question_id"], p["state_idx"], p["doc_i"], p["doc_j"]): p["pref"]
            for p in decided_main}
for tau in (0.05, 0.1, 0.2):
    ps = build_pairs(utils, tau=tau)
    dec = [p for p in ps if not p["tie"]]
    both = [p for p in dec
            if (p["question_id"], p["state_idx"], p["doc_i"], p["doc_j"]) in main_lab]
    agree = sum(1 for p in both
                if p["pref"] == main_lab[(p["question_id"], p["state_idx"],
                                          p["doc_i"], p["doc_j"])])
    L.append(f"| {tau} | {len(dec)} | {len(dec)/max(len(ps),1):.1%} | "
             f"{agree}/{len(both)}" + (f" = {agree/len(both):.1%}" if both else "") + " |")
L.append("")

# ---------------- Conditional Utility Analysis (R10) ----------------
L.append("## Conditional Utility Analysis — P(opportunity | state 조건) (분석 전용)\n")
def state_facets(qid, sidx):
    rec = base_pre[qid]
    gold = {norm_title(g) for g in rec["gold_titles"]}
    seen = set()
    turn = 0
    for e in rec["events"][: sidx + 1]:
        if e["type"] == "info":
            turn += 1
            seen |= {norm_title(t) for t in e.get("titles", [])}
    cov = len(seen & gold) / max(len(gold), 1)
    return turn, cov, rec["outcome"]["em"] == 0.0
buckets = defaultdict(lambda: [0, 0])
for k, rows in by_state.items():
    turn, cov, is_fail = state_facets(*k)
    opp = max(r["U_task_raw_f1"] for r in rows) > EPS
    for name in [f"search step={min(turn,3)}{'+' if turn>=3 else ''}",
                 f"gold coverage {'0%' if cov==0 else ('100%' if cov>=1 else '부분')}",
                 f"base {'실패' if is_fail else '성공'}"]:
        buckets[name][0] += int(opp)
        buckets[name][1] += 1
for name in sorted(buckets):
    o, n = buckets[name]
    L.append(f"- {name}: {o}/{n} = {o/n:.1%}")
L.append("")

# ---------------- Secondary ----------------
gold = [u for u in utils if u["doc_is_gold"]]
nong = [u for u in utils if not u["doc_is_gold"]]
L.append("## Secondary\n")
if gold:
    gnu = sum(1 for u in gold if u["U_task_raw_f1"] <= EPS)
    L.append(f"- Gold-but-Non-Useful: {gnu}/{len(gold)} = {gnu/len(gold):.1%}")
ngu = sum(1 for u in nong if u["U_task_raw_f1"] > EPS)
L.append(f"- Non-Gold-but-Useful: {ngu}/{len(nong)} = {ngu/len(nong):.1%}")
cases = Counter(u["diag"]["search_case"] for u in utils)
L.append(f"- search cases: {dict(cases.most_common())}")
pt = sum(r.get("prompt_tokens", 0) for r in arm_results.values())
gt = sum(r["gen_tokens"] for r in arm_results.values())
L.append(f"- 토큰: prompt {pt:,} / completion {gt:,} "
         f"(prompt 비중 {pt/(pt+gt):.1%})")
if omits:
    g = [o for o in omits if o["doc_is_gold"]]
    n = [o for o in omits if not o["doc_is_gold"]]
    if g and n:
        L.append(f"- I1 omission: gold {sum(o['ctu_f1'] for o in g)/len(g):+.4f} "
                 f"(n={len(g)}) vs non-gold "
                 f"{sum(o['ctu_f1'] for o in n)/len(n):+.4f} (n={len(n)})")

path = os.path.join(OUT, "GATE2_REPORT.md")
with open(path, "w") as f:
    f.write("\n".join(L))
print(path)

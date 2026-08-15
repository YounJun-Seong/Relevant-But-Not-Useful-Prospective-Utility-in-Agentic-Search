"""Produce GATE0_REPORT.md from a gate0 results directory: pipeline health,
utility distributions by candidate type, oracle preview, cost extrapolation."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gate0"

cost = json.load(open(os.path.join(OUT, "cost_report.json")))
utils = read_jsonl(os.path.join(OUT, "utilities.jsonl"))
omits = read_jsonl(os.path.join(OUT, "omission_utilities.jsonl"))
base = read_jsonl(os.path.join(OUT, "base_trajectories.jsonl"))
config = json.load(open(os.path.join(OUT, "config.json")))

lines = []
lines.append("# Gate 0 결과 보고 — PIU 파이프라인 검증 + 비용 실측\n")
lines.append(f"run dir: `{OUT}` · model: `{config['model']}` · "
             f"config_hash: `{config['config_hash']}` · 판정 없음(설계상 Gate 0은 검증 전용)\n")

lines.append("## 1. 파이프라인 건강 지표\n")
lines.append(f"- 질문 수: {cost['n_questions']} · states/question: "
             f"{cost['states_per_question']:.2f} · 총 arm 수: {cost['n_arms']}")
lines.append(f"- Base agent 성능: EM {cost['base_em']:.3f} / F1 {cost['base_f1']:.3f}")
rc = cost.get("replay_consistency")
ra = cost.get("reliability_answer_agreement")
lines.append(f"- Replay 일관성 (base replay = 원본 답): "
             f"{rc:.3f}" if rc is not None else "- Replay 일관성: n/a")
lines.append(f"- Intervention Reliability (중복 +d arm 답변 일치): "
             f"{ra:.3f}" if ra is not None else "- Intervention Reliability: n/a")
lines.append(f"- ΔEM=0 비율 (sparse reward 실측): {cost['frac_dEM_zero']:.3f} "
             f"({cost['n_utility_rows']} 개입 중)\n")

lines.append("## 2. 후보 유형별 PIU 분포 (training 축)\n")
lines.append("| ctype | n | mean ΔF1 (raw) | ΔF1>0 | ΔF1<0 | mean ΔF1 (ctrl) | mean Δtokens (raw) |")
lines.append("|---|---|---|---|---|---|---|")
by_type = defaultdict(list)
for u in utils:
    by_type[u["ctype"]].append(u)
for ctype in ["top_rel", "mid_rel", "novelty", "redundant", "random"]:
    rows = by_type.get(ctype, [])
    if not rows:
        continue
    n = len(rows)
    mraw = sum(r["U_task_raw_f1"] for r in rows) / n
    pos = sum(1 for r in rows if r["U_task_raw_f1"] > 0)
    neg = sum(1 for r in rows if r["U_task_raw_f1"] < 0)
    ctrl = [r["U_task_ctrl_f1"] for r in rows if r["U_task_ctrl_f1"] is not None]
    mctrl = sum(ctrl) / len(ctrl) if ctrl else float("nan")
    mtok = sum(r["U_cost_raw"]["tokens"] for r in rows) / n
    lines.append(f"| {ctype} | {n} | {mraw:+.4f} | {pos} | {neg} | {mctrl:+.4f} | {mtok:+.0f} |")
lines.append("")

lines.append("## 3. 진단 축 (gold 기반 — 분석 전용, 학습 라벨 아님)\n")
lines.append("| ctype | mean U_evi | mean U_search | U_search n/a 비율 |")
lines.append("|---|---|---|---|")
for ctype in ["top_rel", "mid_rel", "novelty", "redundant", "random"]:
    rows = by_type.get(ctype, [])
    if not rows:
        continue
    n = len(rows)
    mevi = sum(r["diag"]["U_evi"] for r in rows) / n
    us = [r["diag"]["U_search"] for r in rows if r["diag"]["U_search"] is not None]
    mus = sum(us) / len(us) if us else float("nan")
    na = 1 - len(us) / n
    lines.append(f"| {ctype} | {mevi:+.4f} | {mus:+.4f} | {na:.2f} |")
lines.append("")

lines.append("## 4. Oracle 미리보기 (state별 best-candidate 선택 시)\n")
by_state = defaultdict(list)
for u in utils:
    by_state[(u["question_id"], u["state_idx"])].append(u)
oracle_gain, best_types = [], defaultdict(int)
disagree = 0
for key, rows in by_state.items():
    best = max(rows, key=lambda r: r["U_task_raw_f1"])
    oracle_gain.append(max(0.0, best["U_task_raw_f1"]))
    best_types[best["ctype"]] += 1
    top_rel = next((r for r in rows if r["ctype"] == "top_rel"), None)
    if top_rel and best["ctype"] != "top_rel" and best["U_task_raw_f1"] > top_rel["U_task_raw_f1"]:
        disagree += 1
n_states = len(by_state)
lines.append(f"- Oracle 평균 이득 (state당 max(0, best ΔF1)): "
             f"{sum(oracle_gain) / max(n_states, 1):+.4f}")
lines.append(f"- Oracle best가 양수인 state 비율: "
             f"{sum(1 for g in oracle_gain if g > 0) / max(n_states, 1):.3f}")
lines.append(f"- **Selection Gap 예고편**: best ≠ top_rel(BM25 top-1)인 state 비율 "
             f"{disagree / max(n_states, 1):.3f} (Gate 1에서 CE 기준으로 공식 측정)")
lines.append(f"- best candidate 유형 분포: {dict(best_types)}\n")

lines.append("## 5. I1 Omission sanity (Bridge Evidence 재현 방향)\n")
if omits:
    n = len(omits)
    mean_ctu = sum(o["ctu_f1"] for o in omits) / n
    gold = [o for o in omits if o["doc_is_gold"]]
    nong = [o for o in omits if not o["doc_is_gold"]]
    lines.append(f"- omission 수: {n} · mean CTU(F1): {mean_ctu:+.4f}")
    if gold:
        lines.append(f"- gold 문서 제거 시 mean CTU: "
                     f"{sum(o['ctu_f1'] for o in gold) / len(gold):+.4f} (n={len(gold)})")
    if nong:
        lines.append(f"- non-gold 문서 제거 시 mean CTU: "
                     f"{sum(o['ctu_f1'] for o in nong) / len(nong):+.4f} (n={len(nong)})")
lines.append("")

lines.append("## 6. 비용 실측 및 외삽\n")
t = cost["timings_sec"]
total_wall = sum(v for k, v in t.items() if k.startswith("phase"))
tok = cost["total_gen_tokens"]
n_roll = cost["n_arms"] + cost["n_questions"]
lines.append(f"- 총 rollout: {n_roll} · 총 생성 토큰: {tok:,} · "
             f"tokens/rollout: {cost['tokens_per_rollout']:.0f}")
lines.append(f"- 실행 시간: phase A {t.get('phase_A_sec', 0):.0f}s + "
             f"phase B {t.get('phase_B_sec', 0):.0f}s = {total_wall:.0f}s "
             f"(H100 1장, 모델 로드 {t.get('model_load_sec', 0):.0f}s 별도)")
per_q_sec = total_wall / max(cost["n_questions"], 1)
lines.append(f"- 질문당 비용: {per_q_sec:.0f}s, "
             f"{cost['arms_per_question']:.1f} arms")
lines.append("")
lines.append("| 규모 | 예상 GPU-hours (H100 1장) |")
lines.append("|---|---|")
for n_q, label in [(100, "Gate 1"), (300, "Gate 2"), (1000, "Gate 3 하한"),
                   (3000, "Gate 3 상한")]:
    lines.append(f"| {label} ({n_q}문항) | {per_q_sec * n_q / 3600:.1f} |")
lines.append("")

report_path = os.path.join(OUT, "GATE0_REPORT.md")
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(report_path)

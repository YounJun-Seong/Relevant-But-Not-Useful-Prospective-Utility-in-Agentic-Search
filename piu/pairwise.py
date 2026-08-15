"""Pairwise preference labels from intervention arms (R9).

Within one frozen state, candidates d_i, d_j were each injected into the same
prefix, so comparing their arm outcomes cancels the shared base and (mostly)
the one-extra-document context effect. Labels:

  pref = i  if F1(d_i) - F1(d_j) >  tau
  pref = j  if F1(d_j) - F1(d_i) >  tau
  tie       otherwise  (excluded from training / zero weight)

`tau` guards against learning control/decoding noise as signal (R9: tie or
downweight small margins). Weight = |margin| by default.
"""
from collections import defaultdict
from itertools import combinations


def build_pairs(util_rows, tau=0.1, metric="f1"):
    """util_rows: utilities.jsonl rows (I2 arms). Returns pair dicts."""
    by_state = defaultdict(list)
    for u in util_rows:
        by_state[(u["question_id"], u["state_idx"])].append(u)
    pairs = []
    for (qid, sidx), rows in by_state.items():
        for a, b in combinations(rows, 2):
            ya, yb = a["arm_d"][metric], b["arm_d"][metric]
            margin = ya - yb
            if margin > tau:
                pref, tie = "i", False
            elif margin < -tau:
                pref, tie = "j", False
            else:
                pref, tie = None, True
            pairs.append({
                "question_id": qid, "state_idx": sidx,
                "doc_i": a["doc"], "doc_j": b["doc"],
                "ctype_i": a["ctype"], "ctype_j": b["ctype"],
                "y_i": ya, "y_j": yb, "margin": margin,
                "pref": pref, "tie": tie,
                "weight": 0.0 if tie else abs(margin),
                "stratum": a.get("stratum"),
            })
    return pairs


def replay_consistency(pairs, plan_rows, arm_results, tau=0.1, metric_key="f1"):
    """Pairwise Utility Consistency via reliability replays: for pairs where
    BOTH candidates have duplicated arms, does the preference direction hold
    across replay combinations? Returns (n_checked, n_consistent)."""
    from .utility import arm_outcome  # noqa: F401 (doc dependency)

    reps = defaultdict(list)  # (qid, sidx, doc) -> [answer-equivalent F1 arms]
    for p in plan_rows:
        if p["arm_type"] not in ("I2_add", "I2_add_rep"):
            continue
        r = arm_results.get(p["arm_id"])
        if r is not None:
            reps[(p["question_id"], p["state_idx"], p["doc"])].append(r)
    n_checked = n_consistent = 0
    for pair in pairs:
        if pair["tie"]:
            continue
        key_i = (pair["question_id"], pair["state_idx"], pair["doc_i"])
        key_j = (pair["question_id"], pair["state_idx"], pair["doc_j"])
        if len(reps.get(key_i, [])) < 2 or len(reps.get(key_j, [])) < 2:
            continue
        n_checked += 1
        # preference holds for every replay combination?
        import itertools

        ok = True
        for ri, rj in itertools.product(reps[key_i], reps[key_j]):
            m = ri["_f1"] - rj["_f1"]
            want = pair["pref"]
            got = "i" if m > tau else ("j" if m < -tau else None)
            if got != want:
                ok = False
                break
        n_consistent += int(ok)
    return n_checked, n_consistent

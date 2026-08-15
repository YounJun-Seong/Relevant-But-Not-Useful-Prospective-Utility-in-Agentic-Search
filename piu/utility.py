"""4-axis utility computation from replay arms.

Role separation (DESIGN_REVISIONS.md R1):
  TRAINING axes   — U_task (ΔEM/ΔF1), U_cost (Δturns/calls/tokens); no gold docs
  DIAGNOSTIC axes — U_evi (future gold-evidence recall), U_search (next-query
                    nDCG@10 vs gold) — analysis only, never training labels
"""
from .data_utils import exact_match, f1_score, ndcg_at_k, norm_title


def arm_outcome(arm_record, gold_answer):
    ans = arm_record["answer"] or ""
    return {
        "answer": ans,
        "em": exact_match(ans, gold_answer),
        "f1": f1_score(ans, gold_answer),
        "turns": arm_record["n_turns"],
        "calls": arm_record["search_calls"],
        "tokens": arm_record["gen_tokens"],
        "finish_reason": arm_record["finish_reason"],
    }


def future_evidence_titles(arm_record, state_idx):
    """Titles served by searches AFTER the intervention state (acquired, not given)."""
    titles = []
    for e in arm_record["events"][state_idx + 1:]:
        if e["type"] == "info":
            titles.extend(e.get("titles", []))
    return titles


def future_gold_recall(arm_record, state_idx, gold_titles, include_injected=False):
    acquired = {norm_title(t) for t in future_evidence_titles(arm_record, state_idx)}
    if include_injected:
        for e in arm_record["events"][: state_idx + 1]:
            if e["type"] == "info":
                acquired |= {norm_title(t) for t in e.get("injected_titles", [])}
    gold = {norm_title(g) for g in gold_titles}
    if not gold:
        return 0.0
    return len(acquired & gold) / len(gold)


def next_query(arm_record, state_idx):
    for e in arm_record["events"][state_idx + 1:]:
        if e["type"] == "info" and e.get("query"):
            return e["query"]
    return None


def next_query_ndcg(retriever, arm_record, state_idx, gold_titles, k=10):
    q = next_query(arm_record, state_idx)
    if q is None:
        return None, None
    ranked = [d["title"] for d in retriever.search(q, k=k)]
    return q, ndcg_at_k(ranked, gold_titles, k=k)


def compute_utilities(retriever, base, arm_d, arm_ctrl, state_idx, gold_answer,
                      gold_titles, compute_ndcg=True):
    """base / +d / +d̃ arm records → training + diagnostic utilities.
    compute_ndcg=False skips the BM25-backed U_search scoring (label-generation
    mode; search_case is still derived from next-query presence)."""
    yb = arm_outcome(base, gold_answer)
    yd = arm_outcome(arm_d, gold_answer)
    yc = arm_outcome(arm_ctrl, gold_answer) if arm_ctrl else None

    out = {
        "base": yb, "arm_d": yd, "arm_ctrl": yc,
        # --- TRAINING axes ---
        "U_task_raw_f1": yd["f1"] - yb["f1"],
        "U_task_raw_em": yd["em"] - yb["em"],
        "U_task_ctrl_f1": (yd["f1"] - yc["f1"]) if yc else None,
        "U_task_ctrl_em": (yd["em"] - yc["em"]) if yc else None,
        "U_cost_raw": {m: yb[m] - yd[m] for m in ("turns", "calls", "tokens")},
        "U_cost_info": {m: yc[m] - yd[m] for m in ("turns", "calls", "tokens")} if yc else None,
    }

    # --- DIAGNOSTIC axes (gold-based; analysis only) ---
    rb = future_gold_recall(base, state_idx, gold_titles)
    rd = future_gold_recall(arm_d, state_idx, gold_titles)
    if compute_ndcg:
        qb, nb = next_query_ndcg(retriever, base, state_idx, gold_titles)
        qd, nd = next_query_ndcg(retriever, arm_d, state_idx, gold_titles)
    else:
        qb, nb = next_query(base, state_idx), None
        qd, nd = next_query(arm_d, state_idx), None
    # search transition case split (R8-7): ΔnDCG only defined on both_search;
    # terminates/extends are behavioral effects, evaluated on answer/cost axes
    if qb is not None and qd is not None:
        search_case = "both_search"
        u_search = (nd - nb) if (nb is not None and nd is not None) else None
    elif qb is not None and qd is None:
        search_case = "intervention_terminates"
        u_search = None
    elif qb is None and qd is not None:
        search_case = "intervention_extends"
        u_search = None
    else:
        search_case = "both_answer"
        u_search = None
    out["diag"] = {
        "U_evi": rd - rb,
        "recall_future_base": rb,
        "recall_future_d": rd,
        "recall_future_d_incl_injected": future_gold_recall(
            arm_d, state_idx, gold_titles, include_injected=True),
        "U_search": u_search,
        "search_case": search_case,
        "next_query_base": qb, "ndcg_base": nb,
        "next_query_d": qd, "ndcg_d": nd,
    }
    return out

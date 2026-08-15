"""Candidate pool construction, selective-intervention candidate types,
matched distractors, and arm (replay episode) construction.

Candidate types per state (selective intervention, Gate 0):
  top_rel    — BM25 top-1 unseen w.r.t. current query q_t
  mid_rel    — BM25 rank ~10 unseen w.r.t. q_t
  novelty    — bridge-candidate proxy: reached via entity/q_0 expansion,
               NOT in BM25 top-20 of q_t (low current-query relevance)
  redundant  — deceptive-relevance proxy: high q_t relevance AND high lexical
               overlap with evidence already seen
  random     — random unseen pool member (control)

Note (Gate 0 scope): pool = BM25(q_t) ∪ BM25(q_0) ∪ BM25(seen evidence titles
as entity queries). Dense retriever leg of the pool is deferred to Gate 1.
"""
import random
import re


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def state_points(record):
    """Indices of info events usable as intervention states (served, non-forced)."""
    points = []
    for i, e in enumerate(record["events"]):
        if e["type"] == "info" and not e.get("forced") and e.get("titles"):
            points.append(i)
    return points


def seen_titles_upto(record, event_idx):
    seen = []
    for e in record["events"][: event_idx + 1]:
        if e["type"] == "info":
            seen.extend(e.get("titles", []))
    return seen


def build_pool(retriever, q0, qt, seen_titles, k=20):
    """Union pool with provenance flags. Excludes already-served docs."""
    queries = [qt, q0] + [t for t in seen_titles[:6]]
    results = retriever.search_batch(queries, k=k)
    qt_top = results[0]
    pool = {}
    for src, docs in zip(["qt", "q0"] + ["entity"] * (len(queries) - 2), results):
        for rank, d in enumerate(docs):
            if d["title"] in seen_titles:
                continue
            entry = pool.setdefault(
                d["title"],
                {"title": d["title"], "text": d["text"], "sources": set(),
                 "qt_rank": None, "qt_score": 0.0},
            )
            entry["sources"].add(src)
            if src == "qt":
                entry["qt_rank"] = rank
                entry["qt_score"] = d["score"]
    for e in pool.values():
        e["sources"] = sorted(e["sources"])
    return pool, qt_top


def pick_candidates(pool, seen_texts, rng, gold_titles=None):
    """Pick the 5 selective-intervention candidates. Returns {ctype: title}."""
    entries = list(pool.values())
    if not entries:
        return {}
    seen_tok = _tokens(" ".join(seen_texts)) if seen_texts else set()
    by_qt = sorted([e for e in entries if e["qt_rank"] is not None],
                   key=lambda e: e["qt_rank"])
    picks = {}
    used = set()

    def take(ctype, entry):
        if entry is not None and entry["title"] not in used:
            picks[ctype] = entry["title"]
            used.add(entry["title"])

    take("top_rel", by_qt[0] if by_qt else None)
    mid = [e for e in by_qt if 8 <= e["qt_rank"] <= 15 and e["title"] not in used]
    take("mid_rel", mid[0] if mid else (by_qt[len(by_qt) // 2] if len(by_qt) > 2 else None))
    novel = [e for e in entries
             if e["qt_rank"] is None and ("entity" in e["sources"] or "q0" in e["sources"])
             and e["title"] not in used]
    if novel:
        take("novelty", rng.choice(novel))
    if by_qt:
        red = sorted([e for e in by_qt[:10] if e["title"] not in used],
                     key=lambda e: -jaccard(_tokens(e["text"]), seen_tok))
        if red and jaccard(_tokens(red[0]["text"]), seen_tok) > 0.05:
            take("redundant", red[0])
    rest = [e for e in entries if e["title"] not in used]
    if rest:
        take("random", rng.choice(rest))
    return picks


def match_distractor(retriever, candidate_title, qt, pool, gold_titles, used_titles, rng):
    """Matched distractor: same qt-rank bucket, token length ±10% (relaxed
    stepwise), not gold, low entity (title-token) overlap with the candidate."""
    cand = retriever.doc_by_title(candidate_title)
    cand_len = max(len(cand["text"].split()), 1)
    cand_title_tok = _tokens(candidate_title)
    gold = {g.lower() for g in gold_titles}
    wide = retriever.search(qt, k=100)
    cand_rank = next((i for i, d in enumerate(wide) if d["title"] == candidate_title), None)

    def bucket(rank):
        if rank is None:
            return None
        return rank // 20  # quantile bucket over top-100

    cband = bucket(cand_rank)
    for tol in (0.10, 0.25, 0.50, 1.00):
        options = []
        for i, d in enumerate(wide):
            if d["title"] == candidate_title or d["title"] in used_titles:
                continue
            if d["title"].lower() in gold:
                continue
            if jaccard(_tokens(d["title"]), cand_title_tok) > 0.3:
                continue
            dlen = max(len(d["text"].split()), 1)
            if abs(dlen - cand_len) / cand_len > tol:
                continue
            if cband is not None and bucket(i) != cband:
                continue
            options.append(d["title"])
        if options:
            return rng.choice(options)
    # last resort: any non-gold doc from the pool
    fallback = [t for t in pool if t not in used_titles and t.lower() not in gold
                and t != candidate_title]
    return rng.choice(fallback) if fallback else None


def make_arm_events(record, state_idx, injected_titles=(), removed_titles=()):
    """Frozen-state prefix: events up to and including the state's info block,
    with injections/removals applied to that block."""
    import copy

    events = copy.deepcopy(record["events"][: state_idx + 1])
    info = events[-1]
    assert info["type"] == "info"
    if injected_titles:
        info["injected_titles"] = list(injected_titles)
    if removed_titles:
        info["removed_titles"] = list(removed_titles)
    return events

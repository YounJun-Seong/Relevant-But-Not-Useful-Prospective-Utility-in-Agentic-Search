"""Gate 1 candidate machinery: BM25 ∪ Dense pool, 8 candidate types with a
CE ranking baseline, and dual independent matched controls.

Candidate types (exclude already-exposed docs):
  bm25_top   — BM25 top-1 w.r.t. current query q_t
  dense_top  — E5 top-1 w.r.t. q_t
  ce_top     — cross-encoder argmax over the pool (Selection Gap baseline)
  mid_rel    — BM25 rank 8..15
  novelty    — in pool but outside BM25 top-20 AND outside Dense top-5 (low
               current-query relevance; reached via q0/dense-tail expansion)
  redundancy — high q_t relevance AND high lexical overlap with seen evidence
  entity_bridge — reached only via entity-title queries, no BM25(q_t) rank
  random     — random pool member (control)
"""
import re

from .intervene import _tokens, jaccard


def build_pool_v2(retriever, dense, q0, qt, seen_titles, k=20, dense_k=20):
    """BM25(qt,q0,entities) ∪ Dense(qt,q0) pool with provenance."""
    bm_queries = [qt, q0] + [t for t in seen_titles[:6]]
    bm_results = retriever.search_batch(bm_queries, k=k)
    de_results = dense.search_batch([qt, q0], k=dense_k)
    pool = {}

    def add(src, docs, rank_field=None):
        for rank, d in enumerate(docs):
            if d["title"] in seen_titles:
                continue
            e = pool.setdefault(
                d["title"],
                {"title": d["title"], "idx": d["idx"], "sources": set(),
                 "qt_rank": None, "qt_score": 0.0, "dense_rank": None},
            )
            e["sources"].add(src)
            if rank_field:
                if e[rank_field] is None or rank < e[rank_field]:
                    e[rank_field] = rank
            if src == "qt":
                e["qt_score"] = d["score"]

    add("qt", bm_results[0], "qt_rank")
    add("q0", bm_results[1])
    for docs in bm_results[2:]:
        add("entity", docs)
    add("dense_qt", de_results[0], "dense_rank")
    add("dense_q0", de_results[1])
    for e in pool.values():
        e["sources"] = sorted(e["sources"])
        if "text" not in e:
            e["text"] = retriever.get_text(e["idx"])
    return pool


def pick_candidates_v2(pool, ce, qt, seen_texts, rng):
    """Pick up to 8 candidates. Returns ({ctype: title}, ce_ranking)."""
    entries = list(pool.values())
    if not entries:
        return {}, []
    seen_tok = _tokens(" ".join(seen_texts)) if seen_texts else set()
    ce_scores = ce.score(qt, [e["title"] + ". " + e["text"] for e in entries])
    for e, s in zip(entries, ce_scores):
        e["ce_score"] = s
    ce_ranking = [e["title"] for e in sorted(entries, key=lambda e: -e["ce_score"])]

    by_qt = sorted([e for e in entries if e["qt_rank"] is not None],
                   key=lambda e: e["qt_rank"])
    by_dense = sorted([e for e in entries if e["dense_rank"] is not None],
                      key=lambda e: e["dense_rank"])
    picks, used = {}, set()

    def take(ctype, entry):
        if entry is not None and entry["title"] not in used:
            picks[ctype] = entry["title"]
            used.add(entry["title"])

    take("bm25_top", by_qt[0] if by_qt else None)
    take("dense_top", next((e for e in by_dense if e["title"] not in used), None))
    take("ce_top", next((e for e in sorted(entries, key=lambda e: -e["ce_score"])
                         if e["title"] not in used), None))
    mid = [e for e in by_qt if 8 <= e["qt_rank"] <= 15 and e["title"] not in used]
    take("mid_rel", mid[0] if mid else None)
    novel = [e for e in entries if e["qt_rank"] is None
             and (e["dense_rank"] is None or e["dense_rank"] >= 5)
             and e["title"] not in used]
    if novel:
        take("novelty", rng.choice(novel))
    red = sorted([e for e in by_qt[:10] if e["title"] not in used],
                 key=lambda e: -jaccard(_tokens(e["text"]), seen_tok))
    if red and jaccard(_tokens(red[0]["text"]), seen_tok) > 0.05:
        take("redundancy", red[0])
    bridge = [e for e in entries if "entity" in e["sources"]
              and e["qt_rank"] is None and e["title"] not in used]
    if bridge:
        take("entity_bridge", rng.choice(bridge))
    rest = [e for e in entries if e["title"] not in used]
    if rest:
        take("random", rng.choice(rest))
    return picks, ce_ranking


def match_distractors(retriever, candidate_title, qt, pool, gold_titles,
                      used_titles, rng, n=1, wide=None):
    """Up to n independent matched distractors (same qt-rank bucket, similar
    length, non-gold, low title overlap with the candidate). Pass `wide`
    (top-100 for qt) to avoid re-searching per candidate — it is identical for
    every candidate of the same state."""
    cand = retriever.doc_by_title(candidate_title)
    cand_len = max(len(cand["text"].split()), 1)
    cand_title_tok = _tokens(candidate_title)
    gold = {g.lower() for g in gold_titles}
    if wide is None:
        wide = retriever.search(qt, k=100)
    cand_rank = next((i for i, d in enumerate(wide)
                      if d["title"] == candidate_title), None)

    def bucket(rank):
        return None if rank is None else rank // 20

    cband = bucket(cand_rank)
    chosen = []
    excluded = set(used_titles)
    for _ in range(n):
        found = None
        for tol in (0.10, 0.25, 0.50, 1.00):
            options = []
            for i, d in enumerate(wide):
                if d["title"] == candidate_title or d["title"] in excluded:
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
                found = rng.choice(options)
                break
        if found is None:
            fallback = [t for t in pool if t not in excluded
                        and t.lower() not in gold and t != candidate_title]
            found = rng.choice(fallback) if fallback else None
        if found is None:
            break
        chosen.append(found)
        excluded.add(found)
    return chosen

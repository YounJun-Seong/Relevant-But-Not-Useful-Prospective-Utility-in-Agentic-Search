"""Explicit state-delta features φ(s,d) (R12): what does the candidate ADD
relative to current knowledge? Interpretable, lexical-level proxies.

  novelty        fraction of candidate content tokens unseen in evidence
  redundancy     max Jaccard overlap with any single seen evidence doc
  entity_gain    new capitalized-token/number types (entity proxy) per 100 tok
  query_align    candidate overlap with current query q_t
  q0_align       candidate overlap with the original question
  gap_coverage   candidate coverage of question tokens NOT yet in evidence
  doc_len        log length
  turn_idx       state position (search step)
  n_evidence     docs seen so far
"""
import math
import re

STOP = set("the a an of and or to in on for is was were are be been at by with "
           "from as it its this that which who whom whose what when where he "
           "she they his her their have has had do does did not no".split())


def content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOP and len(t) > 1}


def entity_tokens(text):
    ents = set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text))
    ents |= set(re.findall(r"\b\d{2,}\b", text))
    return ents


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_phi(question, qt, evidence_texts, doc_text):
    d_tok = content_tokens(doc_text)
    q_tok = content_tokens(question)
    qt_tok = content_tokens(qt)
    ev_toks = [content_tokens(t) for t in evidence_texts]
    e_all = set().union(*ev_toks) if ev_toks else set()
    d_ents = entity_tokens(doc_text)
    e_ents = set().union(*(entity_tokens(t) for t in evidence_texts)) \
        if evidence_texts else set()

    novelty = len(d_tok - e_all) / max(len(d_tok), 1)
    redundancy = max((jaccard(d_tok, e) for e in ev_toks), default=0.0)
    entity_gain = len(d_ents - e_ents) / max(len(doc_text.split()) / 100.0, 1e-6)
    query_align = len(d_tok & qt_tok) / max(len(qt_tok), 1)
    q0_align = len(d_tok & q_tok) / max(len(q_tok), 1)
    gap = q_tok - e_all
    gap_coverage = len(d_tok & gap) / max(len(gap), 1) if gap else 0.0

    return {
        "novelty": novelty,
        "redundancy": redundancy,
        "entity_gain": min(entity_gain / 20.0, 3.0),  # scale-tamed
        "query_align": query_align,
        "q0_align": q0_align,
        "gap_coverage": gap_coverage,
        "doc_len": math.log1p(len(doc_text.split())) / 7.0,
    }


PHI_KEYS = ["novelty", "redundancy", "entity_gain", "query_align",
            "q0_align", "gap_coverage", "doc_len", "turn_idx", "n_evidence"]


def state_context(base_rec, state_idx, retriever):
    """(qt, evidence_texts, turn_idx, n_evidence) for a frozen state."""
    queries, seen = [], []
    for e in base_rec["events"][: state_idx + 1]:
        if e["type"] == "info":
            if e.get("query"):
                queries.append(e["query"])
            seen.extend(e.get("titles", []))
    qt = queries[-1] if queries else base_rec["question"]
    texts = []
    for t in seen:
        doc = retriever.doc_by_title(t)
        if doc:
            texts.append(doc["text"])
    return qt, texts, len(queries), len(seen)


def phi_vector(question, qt, evidence_texts, doc_text, turn_idx, n_evidence):
    phi = compute_phi(question, qt, evidence_texts, doc_text)
    phi["turn_idx"] = turn_idx / 4.0
    phi["n_evidence"] = n_evidence / 12.0
    return [phi[k] for k in PHI_KEYS]

"""HotpotQA loading, answer metrics (official normalization), nDCG."""
import json
import re
import string
import unicodedata
from collections import Counter


def load_hotpot_dev(n=None, seed=42, split="validation", offset=0):
    """Fullwiki questions with gold answer + supporting titles. `offset` skips
    the first rows of the seeded shuffle (lockbox: exclude prior pools)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)
    except TypeError:
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split,
                          trust_remote_code=True)
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(offset, offset + n))
    out = []
    for ex in ds:
        gold_titles = sorted(set(ex["supporting_facts"]["title"]))
        out.append(
            {
                "question_id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "gold_titles": gold_titles,
                "type": ex["type"],
                "level": ex["level"],
            }
        )
    return out


def load_2wiki(n=None, seed=42, split="validation", offset=0,
               corpus_titles=None):
    """2WikiMultiHopQA questions in the same record shape as load_hotpot_dev.
    If corpus_titles is given, keep only questions whose gold titles are ALL
    present in the corpus (label-independent construction filter, R17)."""
    from datasets import load_dataset

    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
    ds = ds.shuffle(seed=seed)
    out = []
    for ex in ds:
        gold_titles = sorted(set(ex["supporting_facts"]["title"]))
        if corpus_titles is not None and any(g not in corpus_titles
                                             for g in gold_titles):
            continue
        out.append({
            "question_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": gold_titles,
            "type": ex["type"],
            "level": ex["type"],  # 2Wiki has no level; keep field shape
        })
        if n is not None and len(out) >= offset + n:
            break
    return out[offset: offset + n] if n is not None else out[offset:]


def load_musique(n=None, seed=42, split="validation", offset=0,
                 corpus_titles=None):
    """MuSiQue (answerable) in the same record shape. Gold titles = supporting
    paragraph titles; corpus-coverage construction filter as in R17."""
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    ds = ds.shuffle(seed=seed)
    out = []
    for ex in ds:
        if not ex.get("answerable", True):
            continue
        gold_titles = sorted({p["title"] for p in ex["paragraphs"]
                              if p["is_supporting"]})
        if corpus_titles is not None and any(g not in corpus_titles
                                             for g in gold_titles):
            continue
        hops = len(ex.get("question_decomposition") or []) or 2
        out.append({
            "question_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": gold_titles,
            "type": f"{hops}hop",
            "level": f"{hops}hop",
        })
        if n is not None and len(out) >= offset + n:
            break
    return out[offset: offset + n] if n is not None else out[offset:]


# --- HotpotQA official EM/F1 ---

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def norm_title(t):
    """Title normalization for gold matching (unicode + case + underscores)."""
    t = unicodedata.normalize("NFC", t)
    return t.replace("_", " ").strip().lower()


def ndcg_at_k(ranked_ids, gold_ids, k=10):
    """Binary-relevance nDCG@k over a ranked doc-id list."""
    import math

    gold = {norm_title(g) for g in gold_ids}
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if norm_title(doc_id) in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

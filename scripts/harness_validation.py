"""R18' step ② and ③.

--mode noop   : rerun Qwen on the first 100 lockbox-segment questions with the
                patched harness; compare per-question behavior against the
                original (pre-patch) base file. Requires >=99% agreement.
--mode smoke  : Mistral 30-question compliance smoke (dev offset 1700 —
                untouched segment). Criteria per R18' (EM is NOT a criterion).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import load_hotpot_dev, read_jsonl
from piu.agent import AgentRunner, Episode
from piu.utility import arm_outcome


def get_runner(model, retriever, gpu_mem):
    from vllm import LLM
    from transformers import AutoTokenizer

    llm = LLM(model=model, gpu_memory_utilization=gpu_mem,
              max_model_len=16384, seed=42)
    tok = AutoTokenizer.from_pretrained(model)
    return AgentRunner(llm, tok, retriever, max_turns=4, docs_per_search=3)


def queries_of(rec):
    return [e.get("query") for e in rec["events"]
            if e["type"] == "info" and e.get("query")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["noop", "smoke"])
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3",
                    help="smoke mode: agent-B model to validate")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    from piu.retriever import BM25Retriever
    retriever = BM25Retriever("data/bm25_index")

    if args.mode == "noop":
        orig = read_jsonl("results/base_validation_pre900_t4_d3.jsonl")[:100]
        questions = load_hotpot_dev(n=100, seed=42, offset=800)
        assert [q["question_id"] for q in questions] == \
            [r["question_id"] for r in orig], "question order mismatch"
        runner = get_runner("Qwen/Qwen2.5-7B-Instruct", retriever, args.gpu_mem)
        eps = [Episode(q["question_id"], q["question"]) for q in questions]
        runner.run(eps)
        agree = {"answer": 0, "queries": 0, "turns": 0, "finish": 0, "full": 0}
        for ep, o in zip(eps, orig):
            r = ep.to_record()
            a = (r["answer"] or "") == (o["answer"] or "")
            q = queries_of(r) == queries_of(o)
            t = r["n_turns"] == o["n_turns"]
            f = r["finish_reason"] == o["finish_reason"]
            agree["answer"] += a
            agree["queries"] += q
            agree["turns"] += t
            agree["finish"] += f
            agree["full"] += (a and q and t and f)
        n = len(orig)
        for k, v in agree.items():
            print(f"noop {k}: {v}/{n} = {v/n:.1%}")
        ok = agree["full"] / n >= 0.99
        print("NOOP_" + ("PASS" if ok else "FAIL"))

    else:
        questions = load_hotpot_dev(n=30, seed=42, offset=1700)
        runner = get_runner(args.model, retriever, args.gpu_mem)
        eps = [Episode(q["question_id"], q["question"]) for q in questions]
        runner.run(eps)
        n = len(eps)
        recs = [ep.to_record() for ep in eps]
        token_limit = sum(r["finish_reason"] == "token_limit" for r in recs)
        valid_ans = sum(1 for r in recs
                        if r["finish_reason"] == "answer" and (r["answer"] or "").strip())
        did_search = sum(1 for r in recs if r["search_calls"] >= 1)
        cons = sum(r["conservative_parses"] for r in recs)
        total_gens = sum(1 + r["n_turns"] for r in recs)  # approx gen segments
        info_hits = sum(r["info_stop_hits"] for r in recs)
        # leakage: fabricated <information> text remaining in transcripts
        leakage = 0
        for r in recs:
            for e in r["events"]:
                if e["type"] == "gen" and "<information>" in e["text"]:
                    leakage += 1
        crit = {
            "malformed_search_rate(cons/gen)": (cons / max(total_gens, 1), 0.05),
            "token_limit_rate": (token_limit / n, 0.10),
            "hallucinated_info_leakage": (leakage, 0),
        }
        print(f"valid final answer: {valid_ans}/{n} = {valid_ans/n:.1%} (기준 >=90%)")
        print(f"search 사용 질문: {did_search}/{n}")
        print(f"info-stop 개입 횟수: {info_hits} (가드 작동 횟수 — 실패 아님)")
        ok = valid_ans / n >= 0.90
        for k, (v, thr) in crit.items():
            flag = v <= thr
            ok = ok and flag
            print(f"{k}: {v if isinstance(v,int) else round(v,3)} (기준 <= {thr}) {'OK' if flag else 'VIOLATION'}")
        mean_f1 = sum(arm_outcome(r, q["answer"])["f1"]
                      for r, q in zip(recs, questions)) / n
        print(f"(참고용, 기준 아님) mean F1: {mean_f1:.3f}")
        print("SMOKE30_" + ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()

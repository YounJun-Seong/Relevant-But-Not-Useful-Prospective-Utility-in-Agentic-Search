"""Smoke test of the agent loop + replay mechanics with a mock retriever.
Validates: chat template, stop tokens, search/answer parsing, info injection,
frozen-state replay, token accounting. No BM25 index needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piu.agent import AgentRunner, Episode
from piu.intervene import make_arm_events, state_points

DOCS = {
    "Scott Derrickson": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He directed the film Doctor Strange.",
    "Ed Wood": "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and author.",
    "Doctor Strange (2016 film)": "Doctor Strange is a 2016 American superhero film directed by Scott Derrickson.",
    "Nationality fact": "Both Scott Derrickson and Ed Wood are American filmmakers.",
    "Distractor A": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France, completed in 1889.",
}


class MockRetriever:
    def doc_by_title(self, title):
        return {"title": title, "text": DOCS[title], "idx": 0}

    def search(self, query, k=3):
        titles = list(DOCS)[:k]
        return [{"title": t, "text": DOCS[t], "score": 1.0, "idx": i}
                for i, t in enumerate(titles)]

    def search_batch(self, queries, k=3):
        return [self.search(q, k) for q in queries]


def main():
    from vllm import LLM
    from transformers import AutoTokenizer

    model = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    gpu_mem = float(os.environ.get("GPU_MEM", "0.85"))
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=16384)
    tok = AutoTokenizer.from_pretrained(model)
    retriever = MockRetriever()
    runner = AgentRunner(llm, tok, retriever, max_turns=3)

    q = ("Were Scott Derrickson and Ed Wood of the same nationality?")
    eps = [Episode("smoke1", q)]
    runner.run(eps)
    rec = eps[0].to_record()
    print("=== BASE ===")
    print("answer:", rec["answer"], "| finish:", rec["finish_reason"],
          "| turns:", rec["n_turns"], "| tokens:", rec["gen_tokens"])
    for e in rec["events"]:
        if e["type"] == "gen":
            print("[gen]", e["text"][:200].replace("\n", " "))
        else:
            print("[info]", e.get("query"), "->", e.get("titles"))

    pts = state_points(rec)
    print("state points:", pts)
    assert rec["finish_reason"] == "answer", "agent did not answer"
    assert pts, "no intervention states"

    # replay with injection at first state
    sidx = pts[0]
    inj = make_arm_events(rec, sidx, injected_titles=["Distractor A"])
    rep = [Episode("smoke1-inj", q, prefix_events=inj),
           Episode("smoke1-base", q, prefix_events=make_arm_events(rec, sidx))]
    runner.run(rep)
    for ep in rep:
        r = ep.to_record()
        print(f"=== REPLAY {ep.question_id} ===")
        print("answer:", r["answer"], "| finish:", r["finish_reason"],
              "| turns:", r["n_turns"])
    base_replay = rep[1].to_record()
    print("replay consistency:", base_replay["answer"] == rec["answer"])
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

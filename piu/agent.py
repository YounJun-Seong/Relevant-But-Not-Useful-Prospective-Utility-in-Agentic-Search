"""Batched ReAct-style search agent on vLLM (greedy, lockstep turns).

Episode = assistant transcript built from events:
  {"type": "gen",  "text": str}                          # model-generated segment
  {"type": "info", "query": str, "titles": [...],
   "injected_titles": [...], "removed_titles": [...]}    # served evidence block

Replay from a frozen state = rebuild transcript from a prefix of events
(with optional injected/removed docs in the last info block), then continue
the loop. Greedy decoding throughout (temperature=0).
"""
import re
import time

SYSTEM_PROMPT = """You are a research assistant that answers questions by searching a Wikipedia corpus.

Protocol, follow it exactly:
- Think step by step inside <think> ... </think>.
- When you need information, issue exactly one search query inside <search> ... </search> and stop.
- Search results will be given to you inside <information> ... </information>.
- When you have enough information, give the final answer inside <answer> ... </answer>.
- The final answer must be a short span (a few words), not a sentence.
- You may search at most {max_turns} times."""

USER_TEMPLATE = "Question: {question}"

FORCE_ANSWER_MSG = (
    "\n<information>\nSearch limit reached. You must now provide your final answer "
    "inside <answer> ... </answer> based on what you know.\n</information>\n"
)


def render_info_block(docs):
    lines = []
    for i, d in enumerate(docs):
        lines.append(f"({i + 1}) Title: {d['title']}\nText: {d['text']}")
    return "\n<information>\n" + "\n\n".join(lines) + "\n</information>\n"


def extract_last(tag, text):
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


class Episode:
    def __init__(self, question_id, question, prefix_events=None, meta=None):
        self.question_id = question_id
        self.question = question
        self.events = list(prefix_events) if prefix_events else []
        self.meta = meta or {}
        self.done = False
        self.answer = None
        self.finish_reason = None  # "answer" | "no_tag" | "token_limit"
        self.gen_tokens = 0
        self.prompt_tokens = 0  # summed over generate calls (context re-reads count)
        self.search_calls = sum(1 for e in self.events if e["type"] == "info" and e.get("query"))
        self.forced = False
        self.wall_start = time.time()
        self.wall_time = None

    def n_turns(self):
        return sum(1 for e in self.events if e["type"] == "info")

    def assistant_text(self, retriever):
        parts = []
        for e in self.events:
            if e["type"] == "gen":
                parts.append(e["text"])
            else:
                docs = []
                removed = set(e.get("removed_titles", []))
                for t in e["titles"]:
                    if t in removed:
                        continue
                    docs.append({"title": t, "text": retriever.doc_by_title(t)["text"]})
                for t in e.get("injected_titles", []):
                    docs.append({"title": t, "text": retriever.doc_by_title(t)["text"]})
                if e.get("forced"):
                    parts.append(FORCE_ANSWER_MSG)
                else:
                    parts.append(render_info_block(docs))
        return "".join(parts)

    def to_record(self):
        return {
            "question_id": self.question_id,
            "question": self.question,
            "meta": self.meta,
            "events": self.events,
            "answer": self.answer,
            "finish_reason": self.finish_reason,
            "gen_tokens": self.gen_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.gen_tokens + self.prompt_tokens,
            "search_calls": self.search_calls,
            "n_turns": self.n_turns(),
            "wall_time": self.wall_time,
            "info_stop_hits": getattr(self, "info_stop_hits", 0),
            "conservative_parses": getattr(self, "conservative_parses", 0),
        }


class AgentRunner:
    def __init__(self, llm, tokenizer, retriever, max_turns=4, docs_per_search=3,
                 max_gen_tokens=768, max_model_len=None):
        from vllm import SamplingParams

        self.llm = llm
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_turns = max_turns
        self.docs_per_search = docs_per_search
        # R18': "<information>" is a stop condition — only the environment may
        # emit retrieval results; an agent starting that block is cut off
        # before fabricating evidence. Behaviorally inert for compliant agents.
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_gen_tokens,
            stop=["</search>", "</answer>", "<information>"],
            include_stop_str_in_output=True,
        )

    def base_prompt(self, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=self.max_turns)},
            {"role": "user", "content": USER_TEMPLATE.format(question=question)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def run(self, episodes, log_every=None):
        """Run episodes to completion in lockstep batches. Mutates episodes."""
        active = [ep for ep in episodes if not ep.done]
        while active:
            prompts = [
                self.base_prompt(ep.question) + ep.assistant_text(self.retriever)
                for ep in active
            ]
            outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
            searches = []  # (episode, query)
            for ep, out in zip(active, outputs):
                gen = out.outputs[0]
                text = gen.text
                ep.gen_tokens += len(gen.token_ids)
                if out.prompt_token_ids is not None:
                    ep.prompt_tokens += len(out.prompt_token_ids)
                ep.events.append({"type": "gen", "text": text})
                stripped = text.rstrip()
                if stripped.endswith("<information>"):
                    # R18': agent tried to emit environment markup; cut the
                    # marker and fall through to conservative parsing below
                    text = stripped[: -len("<information>")]
                    ep.events[-1]["text"] = text
                    ep.info_stop_hits = getattr(ep, "info_stop_hits", 0) + 1
                    stripped = text.rstrip()
                if stripped.endswith("</answer>"):
                    ep.answer = extract_last("answer", text) or ""
                    ep.finish_reason = "answer"
                    ep.done = True
                elif stripped.endswith("</search>"):
                    query = extract_last("search", text)
                    if query is None or ep.forced:
                        ep.answer = ""
                        ep.finish_reason = "no_tag"
                        ep.done = True
                    else:
                        searches.append((ep, query))
                else:
                    # R18': conservative parsing of an unclosed trailing tag
                    open_ans = text.rfind("<answer>")
                    open_sea = text.rfind("<search>")
                    if open_ans > open_sea and "</answer>" not in text[open_ans:]:
                        ep.answer = text[open_ans + len("<answer>"):].strip()
                        ep.finish_reason = "answer" if ep.answer else "token_limit"
                        ep.done = True
                        ep.conservative_parses = getattr(ep, "conservative_parses", 0) + 1
                    elif open_sea > open_ans and "</search>" not in text[open_sea:] \
                            and not ep.forced:
                        query = text[open_sea + len("<search>"):].strip()
                        if query:
                            ep.conservative_parses = getattr(ep, "conservative_parses", 0) + 1
                            searches.append((ep, query))
                        else:
                            ep.answer = ""
                            ep.finish_reason = "no_tag"
                            ep.done = True
                    else:
                        ep.answer = extract_last("answer", text) or ""
                        ep.finish_reason = "token_limit" if not ep.answer else "answer"
                        ep.done = True
                if ep.done:
                    ep.wall_time = time.time() - ep.wall_start
            if searches:
                queries = [q for _, q in searches]
                results = self.retriever.search_batch(queries, k=self.docs_per_search)
                for (ep, query), docs in zip(searches, results):
                    ep.search_calls += 1
                    if ep.search_calls > self.max_turns:
                        # over budget: deliver forced-answer notice instead
                        ep.events.append({"type": "info", "query": query, "titles": [],
                                          "forced": True})
                        ep.forced = True
                    else:
                        ep.events.append(
                            {"type": "info", "query": query,
                             "titles": [d["title"] for d in docs]}
                        )
            active = [ep for ep in episodes if not ep.done]
        return episodes

"""ReAct agent: decompose -> (thought, action, observation)* -> synthesize -> verify."""
from __future__ import annotations

import re

from agentic import prompts
from agentic.hybrid import HybridRetriever, lookup_title
from agentic.llm import LLM, parse_json
from agentic.memory import MemoryStore
from agentic.types import AgentResult, Hit, Step
from config import (
    AGENT_MAX_STEPS,
    ENABLE_SELF_CORRECTION,
    FINAL_TOP_K,
    MAX_SUB_QUESTIONS,
    OBSERVATION_CHAR_LIMIT,
)

_THINKING = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL | re.IGNORECASE)
_MAX_EVIDENCE = 8
_TOOLS = {"search_wiki", "lookup_entity", "final_answer"}


class EvidenceBook:
    """Deduplicated, numbered evidence shared by the loop, synthesis and verifier."""

    def __init__(self) -> None:
        self._order: list[Hit] = []
        self._numbers: dict[str, int] = {}

    def add(self, hits: list[Hit]) -> list[tuple[int, Hit]]:
        out = []
        for hit in hits:
            if hit.id not in self._numbers:
                self._order.append(hit)
                self._numbers[hit.id] = len(self._order)
            out.append((self._numbers[hit.id], hit))
        return out

    def __len__(self) -> int:
        return len(self._order)

    def render(self, limit: int = _MAX_EVIDENCE) -> str:
        return "\n\n".join(
            f"[E{i}] ({h.title}) {h.text}" for i, h in enumerate(self._order[:limit], 1)
        )

    def sources(self) -> list[dict[str, str]]:
        seen, out = set(), []
        for hit in self._order:
            if hit.title not in seen:
                seen.add(hit.title)
                out.append(hit.source())
        return out

    def topics(self) -> list[tuple[str, str]]:
        return list({(h.title, h.entity_type) for h in self._order})


class ReActAgent:
    def __init__(
        self,
        llm: LLM,
        retriever: HybridRetriever,
        memory: MemoryStore,
        *,
        max_steps: int = AGENT_MAX_STEPS,
        self_correct: bool = ENABLE_SELF_CORRECTION,
    ) -> None:
        self.llm, self.retriever, self.memory = llm, retriever, memory
        self.max_steps, self.self_correct = max_steps, self_correct

    # -- public ---------------------------------------------------------------
    def run(self, question: str, session_id: str) -> AgentResult:
        memory_ctx = self.memory.build_context(session_id)
        task, sub_questions = self._decompose(question, memory_ctx)
        book = EvidenceBook()
        steps = self._react_loop(task, sub_questions, memory_ctx, book)

        thinking, draft = self._synthesize(task, sub_questions, book, memory_ctx)
        answer, corrected = draft, False
        if self.self_correct and len(book) and not draft.startswith(prompts.NOT_FOUND):
            answer, corrected = self._verify(task, draft, book)

        self._remember(session_id, question, answer, book)
        return AgentResult(
            answer=answer,
            thinking=thinking,
            sub_questions=sub_questions,
            steps=steps,
            sources=book.sources(),
            corrected=corrected,
        )

    # -- phases ---------------------------------------------------------------
    def _decompose(self, question: str, memory_ctx: str) -> tuple[str, list[str]]:
        raw = self.llm.chat(
            prompts.DECOMPOSE_SYSTEM, prompts.decompose_user(question, memory_ctx), json_mode=True
        )
        data = parse_json(raw) or {}
        task = str(data.get("standalone_question") or question).strip() or question
        subs = data.get("sub_questions")
        if not isinstance(subs, list):
            subs = []
        cleaned = [str(s).strip() for s in subs if str(s).strip()][:MAX_SUB_QUESTIONS]
        return task, cleaned or [task]

    def _react_loop(
        self, task: str, sub_questions: list[str], memory_ctx: str, book: EvidenceBook
    ) -> list[Step]:
        steps: list[Step] = []
        tried: set[tuple[str, str]] = set()
        for _ in range(self.max_steps):
            raw = self.llm.chat(
                prompts.REACT_SYSTEM,
                prompts.react_user(task, sub_questions, memory_ctx, self._scratchpad(steps)),
                json_mode=True,
            )
            data = parse_json(raw) or {}
            thought = str(data.get("thought", "")).strip()
            action = str(data.get("action", "")).strip()
            arg = str(data.get("action_input", "")).strip()

            if action not in _TOOLS:  # malformed output: fall back to a plain search
                action, arg = "search_wiki", arg or task
            if action == "final_answer":
                if len(book):
                    steps.append(Step(thought, action, arg, "Research finished."))
                    break
                action, arg = "search_wiki", task  # never answer with zero evidence
            if (action, arg.lower()) in tried:
                steps.append(Step(thought, action, arg, "Already tried; stopping research."))
                break
            tried.add((action, arg.lower()))
            steps.append(Step(thought, action, arg, self._act(action, arg or task, book)))
        if not len(book):  # last resort so synthesis always has a chance
            self._act("search_wiki", task, book)
        return steps

    def _act(self, action: str, arg: str, book: EvidenceBook) -> str:
        if action == "lookup_entity":
            hits = lookup_title(arg, limit=FINAL_TOP_K)
        else:
            hits = self.retriever.search(arg, top_k=FINAL_TOP_K)
        if not hits:
            return "No results. Try different keywords or another tool."
        return "\n".join(
            f"[E{n}] ({h.title}) {h.text[:OBSERVATION_CHAR_LIMIT]}" for n, h in book.add(hits)
        )

    def _synthesize(
        self, task: str, sub_questions: list[str], book: EvidenceBook, memory_ctx: str
    ) -> tuple[str, str]:
        if not len(book):
            return "No evidence was retrieved.", prompts.NOT_FOUND
        raw = self.llm.chat(
            prompts.SYNTHESIS_SYSTEM,
            prompts.synthesis_user(task, sub_questions, book.render(), memory_ctx),
        )
        return parse_thinking_answer(raw)

    def _verify(self, task: str, draft: str, book: EvidenceBook) -> tuple[str, bool]:
        raw = self.llm.chat(
            prompts.VERIFY_SYSTEM, prompts.verify_user(task, draft, book.render()), json_mode=True
        )
        data = parse_json(raw) or {}
        revised = str(data.get("final_answer") or "").strip()
        if data.get("verdict") == "revise" and revised and revised != draft:
            return revised, True
        return draft, False

    # -- memory ---------------------------------------------------------------
    def _remember(self, session_id: str, question: str, answer: str, book: EvidenceBook) -> None:
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", answer)
        self.memory.note_topics(session_id, book.topics())
        self.memory.compress(session_id, self._summarize)

    def _summarize(self, previous: str, transcript: str) -> str:
        return self.llm.chat(
            prompts.SUMMARY_SYSTEM,
            f"PREVIOUS SUMMARY:\n{previous or '(none)'}\n\nNEW MESSAGES:\n{transcript}",
        )

    @staticmethod
    def _scratchpad(steps: list[Step]) -> str:
        return "\n".join(
            f"Step {i}\nThought: {s.thought}\nAction: {s.action}({s.action_input})\n"
            f"Observation: {s.observation}"
            for i, s in enumerate(steps, 1)
        )


def parse_thinking_answer(raw: str) -> tuple[str, str]:
    """Split `<thinking>..</thinking><answer>..</answer>`; tolerate missing tags."""
    thinking = "\n".join(m.strip() for m in _THINKING.findall(raw))
    match = _ANSWER.search(raw)
    answer = match.group(1).strip() if match else _THINKING.sub("", raw).strip()
    answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
    return thinking, answer or prompts.NOT_FOUND

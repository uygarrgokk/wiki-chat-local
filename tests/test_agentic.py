import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.agent import ReActAgent, parse_thinking_answer
from agentic.hybrid import BM25Index, HybridRetriever, reciprocal_rank_fusion
from agentic.llm import parse_json
from agentic.memory import MemoryStore
from agentic.types import Hit


def hit(i, text, title="T", etype="person"):
    return Hit(i, text, title, etype, "http://x")


class FakeStore:
    def __init__(self, ids):
        self.ids = ids

    def query(self, q, k, filters):
        docs = {"a": "Turing broke Enigma", "b": "Everest is tall"}
        return {
            "ids": [self.ids],
            "documents": [[docs[i] for i in self.ids]],
            "metadatas": [[{"title": "X", "entity_type": "person", "source_url": "u"}] * len(self.ids)],
        }


class ScriptedLLM:
    """Returns canned replies keyed by which system prompt is used."""

    def __init__(self):
        self.react_calls = 0

    def chat(self, system, user, *, json_mode=False):
        if "plan research" in system:
            return json.dumps({"standalone_question": "What did Turing do?", "sub_questions": ["Turing work"]})
        if "reasoning core" in system:
            self.react_calls += 1
            if self.react_calls == 1:
                return json.dumps({"thought": "need info", "action": "search_wiki", "action_input": "Turing"})
            return json.dumps({"thought": "enough", "action": "final_answer", "action_input": "done"})
        if "strictly from the numbered" in system:
            return "<thinking>E1 says he broke Enigma.</thinking><answer>He broke Enigma [E1]. Also he flew.</answer>"
        if "fact-checker" in system:
            return json.dumps({"verdict": "revise", "unsupported_claims": ["flew"], "final_answer": "He broke Enigma [E1]."})
        return "summary"


class RetrievalTests(unittest.TestCase):
    def test_bm25_ranks_matching_doc_first(self):
        idx = BM25Index([hit("1", "cats purr"), hit("2", "turing enigma machine")])
        self.assertEqual(idx.search("enigma", 5)[0].id, "2")

    def test_bm25_entity_filter(self):
        idx = BM25Index([hit("1", "tower", etype="place"), hit("2", "tower", etype="person")])
        self.assertEqual([h.id for h in idx.search("tower", 5, ["place"])], ["1"])

    def test_rrf_prefers_docs_in_both_rankings(self):
        fused = reciprocal_rank_fusion([[hit("a", ""), hit("b", "")], [hit("c", ""), hit("b", "")]])
        self.assertEqual(fused[0].id, "b")

    def test_hybrid_degrades_when_vector_fails(self):
        class Broken:
            def query(self, *a):
                raise RuntimeError

        r = HybridRetriever(Broken(), use_reranker=False)
        r._bm25 = BM25Index([hit("1", "enigma")])
        self.assertEqual([h.id for h in r.search("enigma")], ["1"])


class MemoryTests(unittest.TestCase):
    def test_compress_and_context(self):
        m = MemoryStore(":memory:")
        for i in range(12):
            m.add_message("s", "user" if i % 2 == 0 else "assistant", f"msg{i}")
        self.assertTrue(m.compress("s", lambda prev, t: "SUM"))
        ctx = m.build_context("s")
        self.assertIn("SUM", ctx)
        self.assertIn("msg11", ctx)
        self.assertNotIn("msg0\n", ctx)

    def test_long_term_and_topics(self):
        m = MemoryStore(":memory:")
        m.add_message("old", "user", "x" * 1)
        m.conn.execute("INSERT INTO summaries VALUES ('old','talked about Tesla',1,1)")
        m.note_topics("new", [("Alan Turing", "person")])
        ctx = m.build_context("new")
        self.assertIn("talked about Tesla", ctx)
        self.assertIn("Alan Turing", ctx)


class AgentTests(unittest.TestCase):
    def test_full_loop_with_self_correction(self):
        retriever = HybridRetriever(FakeStore(["a"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        mem = MemoryStore(":memory:")
        result = ReActAgent(ScriptedLLM(), retriever, mem).run("Turing?", "s")
        self.assertEqual(result.answer, "He broke Enigma [E1].")
        self.assertTrue(result.corrected)
        self.assertIn("Enigma", result.thinking)
        self.assertEqual(result.steps[0].action, "search_wiki")
        self.assertEqual(len(mem.recent_messages("s")), 2)

    def test_no_evidence_means_dont_know(self):
        class Empty:
            def query(self, *a):
                return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        retriever = HybridRetriever(Empty(), use_reranker=False)
        retriever._bm25 = BM25Index([])
        result = ReActAgent(ScriptedLLM(), retriever, MemoryStore(":memory:")).run("?", "s")
        self.assertTrue(result.answer.startswith("I don't know"))


class ParsingTests(unittest.TestCase):
    def test_parse_json_with_noise(self):
        self.assertEqual(parse_json('sure! {"a": 1} ok'), {"a": 1})
        self.assertIsNone(parse_json("no json"))

    def test_thinking_answer_variants(self):
        self.assertEqual(parse_thinking_answer("<thinking>t</thinking><answer>a</answer>"), ("t", "a"))
        self.assertEqual(parse_thinking_answer("plain"), ("", "plain"))
        self.assertEqual(parse_thinking_answer("<thinking>t</thinking><answer>cut off"), ("t", "cut off"))


if __name__ == "__main__":
    unittest.main()

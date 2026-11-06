import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

from agentic import agent as agent_mod
from agentic.agent import ReActAgent, is_grounded, parse_thinking_answer
from agentic.hybrid import (
    BM25Index,
    HybridRetriever,
    canonicalize_entities,
    find_entities,
    reciprocal_rank_fusion,
)
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
        retriever._titles = []
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
        retriever._titles = []
        result = ReActAgent(ScriptedLLM(), retriever, MemoryStore(":memory:")).run("?", "s")
        self.assertTrue(result.answer.startswith("I don't know"))


class EntityTests(unittest.TestCase):
    TITLES = [("Albert Einstein", "person"), ("Nikola Tesla", "person"),
              ("Niagara Falls", "place"), ("Mount Everest", "place")]

    def test_typo_tolerant_full_name(self):
        self.assertEqual(find_entities("who is albert eistain", self.TITLES), ["Albert Einstein"])

    def test_person_surname_only(self):
        self.assertEqual(find_entities("what did tesla invent", self.TITLES), ["Nikola Tesla"])

    def test_place_last_word_does_not_match(self):
        self.assertEqual(find_entities("which falls are tallest", self.TITLES), [])

    def test_unrelated_text(self):
        self.assertEqual(find_entities("what is the weather today", self.TITLES), [])

    def test_canonicalize_fixes_misspelled_names_only(self):
        self.assertEqual(canonicalize_entities("Who is Albert Eistain?", self.TITLES), "Who is Albert Einstein?")
        self.assertEqual(canonicalize_entities("who is a. einstein", self.TITLES), "who is Albert Einstein")
        self.assertEqual(canonicalize_entities("what did tesla invent", self.TITLES), "what did tesla invent")
        self.assertEqual(canonicalize_entities("Is Mount Everest tall?", self.TITLES), "Is Mount Everest tall?")


class SeedingTests(unittest.TestCase):
    def test_intro_seeded_and_refusal_revised(self):
        intro = Hit("person:Albert Einstein:0", "Albert Einstein was a physicist.", "Albert Einstein", "person")

        class RefusingLLM(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "strictly from the numbered" in system:
                    return "<thinking>x</thinking><answer>I don't know.</answer>"
                if "fact-checker" in system:
                    assert "Albert Einstein was a physicist." in user
                    return json.dumps({"verdict": "revise", "final_answer": "A physicist [E1]."})
                if "reasoning core" in system:
                    return json.dumps({"thought": "enough", "action": "final_answer", "action_input": "done"})
                return super().chat(system, user, json_mode=json_mode)

        retriever = HybridRetriever(FakeStore([]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = [("Albert Einstein", "person")]
        with mock.patch.object(agent_mod, "lookup_title", return_value=[intro]) as lookup:
            result = ReActAgent(RefusingLLM(), retriever, MemoryStore(":memory:")).run("who is albert eistain", "s")
        lookup.assert_called_once_with("Albert Einstein", limit=2, exact=True)
        self.assertEqual(result.steps[0].action, "lookup_entity")
        self.assertEqual(result.answer, "A physicist [E1].")
        self.assertTrue(result.corrected)


class GroundingTests(unittest.TestCase):
    EVIDENCE = "[E1] (Burj Khalifa) The building opened in 2010 in Downtown Dubai."

    def test_grounded_answer_passes(self):
        self.assertTrue(is_grounded("Burj Khalifa opened in 2010 in Dubai [E1].", self.EVIDENCE))

    def test_outside_knowledge_rejected(self):
        self.assertFalse(is_grounded("Elon Musk founded SpaceX, Neuralink and SolarCity.", self.EVIDENCE))

    def test_verifier_hallucination_is_discarded(self):
        class HallucinatingVerifier(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "strictly from the numbered" in system:
                    return json.dumps({"thinking": "nothing about Musk", "relevant_facts": [], "answer": "I don't know"})
                if "fact-checker" in system:
                    return json.dumps({"verdict": "revise", "final_answer": "Elon Musk founded SpaceX and Neuralink."})
                return super().chat(system, user, json_mode=json_mode)

        retriever = HybridRetriever(FakeStore(["b"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = []
        result = ReActAgent(HallucinatingVerifier(), retriever, MemoryStore(":memory:")).run("Who is Elon Musk?", "s")
        self.assertEqual(result.answer, "I don't know")
        self.assertFalse(result.corrected)
        self.assertIn("nothing about Musk", result.thinking)


def einstein_retriever(store=None):
    retriever = HybridRetriever(store or FakeStore([]), use_reranker=False)
    retriever._bm25 = BM25Index([])
    retriever._titles = [("Albert Einstein", "person"), ("Statue of Liberty", "place")]
    return retriever


class RoutingTests(unittest.TestCase):
    def test_small_talk_skips_retrieval(self):
        class ChatLLM(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "plan research" in system:
                    return json.dumps({"route": "chat", "standalone_question": user, "sub_questions": []})
                if "friendly local assistant" in system:
                    assert "Albert Einstein" in system
                    return "Evet, Türkçe konuşabilirim."
                raise AssertionError(f"unexpected call: {system[:40]}")

        mem = MemoryStore(":memory:")
        result = ReActAgent(ChatLLM(), einstein_retriever(), mem).run("do you know Turkish", "s")
        self.assertEqual(result.answer, "Evet, Türkçe konuşabilirim.")
        self.assertEqual(result.steps, [])
        self.assertEqual(mem.recent_topics("s"), [])

    def test_factual_question_is_never_small_talk(self):
        class MisroutingLLM(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "plan research" in system:
                    return json.dumps({"route": "chat", "standalone_question": "Who is the president of Mars?", "sub_questions": []})
                if "friendly local assistant" in system:
                    raise AssertionError("factual question answered without evidence")
                return super().chat(system, user, json_mode=json_mode)

        agent = ReActAgent(MisroutingLLM(), einstein_retriever(), MemoryStore(":memory:"), self_correct=False)
        self.assertEqual(agent._decompose("Who is the president of Mars?", "")[0], "wiki")

    def test_rewrite_cannot_inject_memory_topic(self):
        agent = ReActAgent(ScriptedLLM(), einstein_retriever(), MemoryStore(":memory:"))
        self.assertTrue(agent._hijacked("do you know Turkish", "Who is Albert Einstein?"))
        self.assertFalse(agent._hijacked("where was he born", "Where was Albert Einstein born?"))
        self.assertFalse(agent._hijacked("who is a. einstein", "Who is Albert Einstein?"))


class LoopTests(unittest.TestCase):
    def test_paraphrased_query_counts_as_repeat(self):
        self.assertEqual(
            agent_mod._action_key("search_wiki", "where did albert einstein born"),
            agent_mod._action_key("search_wiki", "where was Albert Einstein born"),
        )

    def test_loop_stops_when_search_adds_nothing(self):
        class Searcher(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "reasoning core" in system:
                    self.react_calls += 1
                    return json.dumps({"thought": "more", "action": "search_wiki", "action_input": f"q{self.react_calls} turing"})
                return super().chat(system, user, json_mode=json_mode)

        llm = Searcher()
        retriever = HybridRetriever(FakeStore(["a"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = []
        ReActAgent(llm, retriever, MemoryStore(":memory:"), max_steps=5).run("Turing?", "s")
        self.assertEqual(llm.react_calls, 2)  # second search returned only known evidence

    def test_lookup_with_question_maps_to_title(self):
        agent = ReActAgent(ScriptedLLM(), einstein_retriever(), MemoryStore(":memory:"))
        self.assertEqual(agent._resolve_lookup("where did albert einstein born"), ("lookup_entity", "Albert Einstein"))
        self.assertEqual(agent._resolve_lookup("tallest tower"), ("search_wiki", "tallest tower"))


class FilterTests(unittest.TestCase):
    def test_where_question_about_person_is_not_limited_to_places(self):
        retriever = einstein_retriever()
        self.assertEqual(retriever._filters("where was Albert Einstein born"), ["person"])
        self.assertEqual(retriever._filters("where is the tallest tower"), [])


class VerifyFallbackTests(unittest.TestCase):
    def test_ungrounded_draft_falls_back_to_quoted_facts(self):
        class Mixed(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "strictly from the numbered" in system:
                    return json.dumps({
                        "thinking": "x",
                        "relevant_facts": ["Turing broke Enigma [E1]"],
                        "answer": "Turing broke Enigma and later founded Microsoft with Bill Gates.",
                    })
                if "fact-checker" in system:
                    return json.dumps({"verdict": "revise", "final_answer": "I don't know"})
                return super().chat(system, user, json_mode=json_mode)

        retriever = HybridRetriever(FakeStore(["a"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = []
        result = ReActAgent(Mixed(), retriever, MemoryStore(":memory:")).run("Turing?", "s")
        self.assertEqual(result.answer, "Turing broke Enigma [E1]")
        self.assertTrue(result.corrected)


class ParsingTests(unittest.TestCase):
    def test_truncated_synthesis_json_is_hidden(self):
        class Truncating(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "strictly from the numbered" in system:
                    return '{"thinking": "x", "relevant_facts": ["Turing broke Enigma'
                return super().chat(system, user, json_mode=json_mode)

        retriever = HybridRetriever(FakeStore(["a"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = []
        agent = ReActAgent(Truncating(), retriever, MemoryStore(":memory:"), self_correct=False)
        self.assertEqual(agent.run("Turing?", "s").answer, "I don't know")

    def test_synthesis_without_answer_key_uses_facts(self):
        class FactsOnly(ScriptedLLM):
            def chat(self, system, user, *, json_mode=False):
                if "strictly from the numbered" in system:
                    return json.dumps({"thinking": "x", "relevant_facts": ["Turing broke Enigma [E1]"]})
                return super().chat(system, user, json_mode=json_mode)

        retriever = HybridRetriever(FakeStore(["a"]), use_reranker=False)
        retriever._bm25 = BM25Index([])
        retriever._titles = []
        agent = ReActAgent(FactsOnly(), retriever, MemoryStore(":memory:"), self_correct=False)
        self.assertEqual(agent.run("Turing?", "s").answer, "Turing broke Enigma [E1]")

    def test_parse_json_with_noise(self):
        self.assertEqual(parse_json('sure! {"a": 1} ok'), {"a": 1})
        self.assertIsNone(parse_json("no json"))

    def test_thinking_answer_variants(self):
        self.assertEqual(parse_thinking_answer("<thinking>t</thinking><answer>a</answer>"), ("t", "a"))
        self.assertEqual(parse_thinking_answer("plain"), ("", "plain"))
        self.assertEqual(parse_thinking_answer("<thinking>t</thinking><answer>cut off"), ("t", "cut off"))


if __name__ == "__main__":
    unittest.main()

"""Hybrid retrieval: BM25 + Chroma fused with RRF, then cross-encoder reranking."""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from agentic.types import Hit
from config import (
    CANDIDATES_PER_RETRIEVER,
    FINAL_TOP_K,
    RERANK_POOL,
    RERANKER_MODEL,
    RRF_K,
    SQLITE_PATH,
    USE_RERANKER,
)

_TOKEN = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "the a an of in on at to is are was were and or for with by from as it this that "
    "what who where when which how did does do his her their its be been".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _copy(hit: Hit, score: float) -> Hit:
    return Hit(hit.id, hit.text, hit.title, hit.entity_type, hit.source_url, score)


class BM25Index:
    """Minimal Okapi BM25 over in-memory Hit objects (no external dependency)."""

    def __init__(self, hits: list[Hit], k1: float = 1.5, b: float = 0.75) -> None:
        self.hits = hits
        self.k1, self.b = k1, b
        self.doc_tf: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: Counter[str] = Counter()
        for hit in hits:
            # Title tokens are included so entity-name queries match strongly.
            tokens = tokenize(f"{hit.title} {hit.text}")
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_len.append(len(tokens))
            df.update(tf.keys())
        n = len(hits)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(
        self, query: str, top_k: int, entity_types: Iterable[str] | None = None
    ) -> list[Hit]:
        allowed = set(entity_types) if entity_types else None
        q_tokens = [t for t in set(tokenize(query)) if t in self.idf]
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.doc_tf):
            if allowed and self.hits[i].entity_type not in allowed:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
            score = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t in q_tokens
                if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [_copy(self.hits[i], s) for s, i in scored[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """score(d) = sum over rankers of 1 / (k + rank(d)), rank starting at 1."""
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.id] += 1.0 / (k + rank)
            by_id.setdefault(hit.id, hit)
    ordered = sorted(fused, key=fused.__getitem__, reverse=True)
    return [_copy(by_id[i], fused[i]) for i in ordered]


class Reranker:
    """Lazy cross-encoder; disables itself if the model cannot be loaded."""

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits or self._failed:
            return hits[:top_k]
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict([(query, h.text) for h in hits])
        except Exception as exc:  # offline model download, OOM, ...
            print(f"[reranker disabled: {exc}]")
            self._failed = True
            return hits[:top_k]
        ranked = sorted(zip(scores, hits), key=lambda p: float(p[0]), reverse=True)
        return [_copy(h, float(s)) for s, h in ranked[:top_k]]


class HybridRetriever:
    def __init__(
        self, store, reranker: Reranker | None = None, *, use_reranker: bool = USE_RERANKER
    ) -> None:
        self.store = store
        if reranker is None and use_reranker:
            reranker = Reranker()
        self.reranker = reranker
        self._bm25: BM25Index | None = None
        self._titles: list[tuple[str, str]] | None = None

    def refresh(self) -> None:
        """Drop cached indexes; call after /ingest or /reset."""
        self._bm25 = None
        self._titles = None

    def titles(self) -> list[tuple[str, str]]:
        if self._titles is None:
            self._titles = load_titles()
        return self._titles

    def find_entities(self, text: str) -> list[str]:
        return find_entities(text, self.titles())

    def canonicalize(self, text: str) -> str:
        return canonicalize_entities(text, self.titles())

    def _load_bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(load_chunks())
        return self._bm25

    def _filters(self, query: str) -> list[str]:
        """Entity types of the articles the query names. Keyword hints are not used:
        "where was Einstein born" is about a person despite "where"."""
        named = set(self.find_entities(query))
        return sorted({etype for title, etype in self.titles() if title in named})

    def search(self, query: str, top_k: int = FINAL_TOP_K) -> list[Hit]:
        filters = self._filters(query)
        lexical = self._load_bm25().search(query, CANDIDATES_PER_RETRIEVER, filters)
        semantic = self._vector_search(query, filters)
        fused = reciprocal_rank_fusion([lexical, semantic])[:RERANK_POOL]
        if self.reranker:
            return self.reranker.rerank(query, fused, top_k)
        return fused[:top_k]

    def _vector_search(self, query: str, filters: list[str]) -> list[Hit]:
        try:
            result = self.store.query(query, CANDIDATES_PER_RETRIEVER, filters)
        except RuntimeError:
            return []  # unreadable index: degrade to BM25-only
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [
            Hit(
                id=i,
                text=d,
                title=str(m.get("title", "")),
                entity_type=str(m.get("entity_type", "")),
                source_url=str(m.get("source_url", "")),
            )
            for i, d, m in zip(ids, docs, metas)
        ]


_CHUNK_SELECT = """SELECT c.id, c.text, c.title, c.entity_type, d.source_url
                   FROM chunks c JOIN docs d ON d.title = c.title"""


def _query_hits(sql: str, params: tuple, db_path: Path) -> list[Hit]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        return [Hit(*row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def load_chunks(db_path: Path = SQLITE_PATH) -> list[Hit]:
    return _query_hits(_CHUNK_SELECT, (), db_path)


def lookup_title(
    title: str, limit: int = 3, db_path: Path = SQLITE_PATH, *, exact: bool = False
) -> list[Hit]:
    """SQLite tool: opening chunks of documents whose title matches `title`."""
    if exact:
        where, param = "lower(c.title) = ?", title.lower().strip()
    else:
        where, param = "lower(c.title) LIKE ?", f"%{title.lower().strip()}%"
    sql = _CHUNK_SELECT + f" WHERE {where} ORDER BY c.title, c.chunk_index LIMIT ?"
    return _query_hits(sql, (param, limit), db_path)


def load_titles(db_path: Path = SQLITE_PATH) -> list[tuple[str, str]]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT title, entity_type FROM docs").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def canonicalize_entities(
    text: str, titles: list[tuple[str, str]], cutoff: float = 0.8
) -> str:
    """Replace misspelled full article names with the title ("albert eistain" ->
    "Albert Einstein"); otherwise the answer step says no such person exists."""
    spans = [(m.start(), m.end(), m.group().lower()) for m in _TOKEN.finditer(text)]
    replacements: list[tuple[int, int, str]] = []
    for title, _ in titles:
        target, n = title.lower(), len(title.split())
        if n == 1 and len(target) < 5:
            continue
        for i in range(len(spans) - n + 1):
            window = " ".join(s[2] for s in spans[i : i + n])
            if window != target and SequenceMatcher(None, window, target).ratio() >= cutoff:
                replacements.append((spans[i][0], spans[i + n - 1][1], title))
    out, last_start = text, len(text) + 1
    for start, end, title in sorted(replacements, reverse=True):
        if end <= last_start:  # skip overlapping matches
            out = out[:start] + title + out[end:]
            last_start = start
    return out


def find_entities(
    text: str, titles: list[tuple[str, str]], cutoff: float = 0.8
) -> list[str]:
    """Fuzzy-match document titles mentioned in `text` (tolerates typos like "eistain").

    Compares each title against word windows of the same length; for people the
    surname alone also counts ("who was tesla"). Best matches first.
    """
    words = _TOKEN.findall(text.lower())
    found: dict[str, float] = {}
    for title, entity_type in titles:
        t_words = title.lower().split()
        candidates = [(title.lower(), len(t_words))]
        if entity_type == "person" and len(t_words) > 1 and len(t_words[-1]) >= 5:
            candidates.append((t_words[-1], 1))
        for target, n in candidates:
            for i in range(len(words) - n + 1):
                window = " ".join(words[i : i + n])
                ratio = SequenceMatcher(None, window, target).ratio()
                if ratio >= cutoff and ratio > found.get(title, 0):
                    found[title] = ratio
    return sorted(found, key=found.__getitem__, reverse=True)

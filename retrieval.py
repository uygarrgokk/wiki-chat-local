import re
import sqlite3
from typing import Dict, List, Tuple

from config import SQLITE_PATH, TOP_K
from entities import PEOPLE, PLACES
from vector_store import LocalVectorStore


PERSON_HINTS = {"who", "person", "scientist", "artist", "singer", "player"}
PLACE_HINTS = {"where", "located", "place", "city", "country", "monument", "mountain"}


def classify_query(query: str) -> List[str]:
    q = query.lower()
    has_person_name = any(name.lower() in q for name in PEOPLE)
    has_place_name = any(name.lower() in q for name in PLACES)
    has_person_hint = any(word in q for word in PERSON_HINTS)
    has_place_hint = any(word in q for word in PLACE_HINTS)

    if (has_person_name and has_place_name) or (has_person_hint and has_place_hint):
        return ["person", "place"]
    if has_person_name or has_person_hint:
        return ["person"]
    if has_place_name or has_place_hint:
        return ["place"]
    return ["person", "place"]


def _keyword_score(query: str, text: str, title: str) -> int:
    query_tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    text_tokens = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
    title_lower = title.lower()
    overlap = len(query_tokens.intersection(text_tokens))
    title_boost = 4 if any(tok in title_lower for tok in query_tokens) else 0
    return overlap + title_boost


def _sqlite_fallback(query: str, filters: List[str], top_k: int) -> Tuple[str, list]:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        base_sql = """
            SELECT c.text, c.title, c.entity_type, d.source_url
            FROM chunks c
            JOIN docs d
              ON d.title = c.title AND d.entity_type = c.entity_type
        """
        params: List[str] = []
        if len(filters) == 1:
            base_sql += " WHERE c.entity_type = ?"
            params.append(filters[0])
        elif len(filters) > 1:
            placeholders = ",".join("?" for _ in filters)
            base_sql += f" WHERE c.entity_type IN ({placeholders})"
            params.extend(filters)

        rows = conn.execute(base_sql, params).fetchall()
    finally:
        conn.close()

    scored: List[Tuple[int, Dict[str, str]]] = []
    for row in rows:
        score = _keyword_score(query, row["text"], row["title"])
        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "text": row["text"],
                    "title": row["title"],
                    "entity_type": row["entity_type"],
                    "source_url": row["source_url"],
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [item for _, item in scored[:top_k]]

    blocks = [f"[Chunk {i}] {item['text']}" for i, item in enumerate(top_items, start=1)]
    sources = [
        {
            "title": item["title"],
            "entity_type": item["entity_type"],
            "source_url": item["source_url"],
        }
        for item in top_items
    ]
    return "\n\n".join(blocks), sources


def retrieve_context(store: LocalVectorStore, query: str) -> Tuple[str, list]:
    filters = classify_query(query)
    try:
        result = store.query(query_text=query, top_k=TOP_K, entity_filters=filters)
    except RuntimeError:
        return _sqlite_fallback(query=query, filters=filters, top_k=TOP_K)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    blocks = []
    sources = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        blocks.append(f"[Chunk {i}] {doc}")
        sources.append(meta)
    return "\n\n".join(blocks), sources

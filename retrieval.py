from typing import List, Tuple

from config import TOP_K
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


def retrieve_context(store: LocalVectorStore, query: str) -> Tuple[str, list]:
    filters = classify_query(query)
    result = store.query(query_text=query, top_k=TOP_K, entity_filters=filters)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    blocks = []
    sources = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        blocks.append(f"[Chunk {i}] {doc}")
        sources.append(meta)
    return "\n\n".join(blocks), sources

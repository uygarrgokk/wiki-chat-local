from typing import Any, List, cast

import chromadb
from chromadb.api.types import Metadata, Where
from sentence_transformers import SentenceTransformer

from chunk_row import ChunkRow
from config import CHROMA_DIR, EMBEDDING_MODEL


class LocalVectorStore:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name="wiki_chunks")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def reset(self) -> None:
        try:
            self.client.delete_collection("wiki_chunks")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="wiki_chunks")

    def upsert_rows(self, rows: List[ChunkRow]) -> None:
        if not rows:
            return
        texts = [r["text"] for r in rows]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        ids = [r["id"] for r in rows]
        metadatas: List[Metadata] = [
            cast(
                Metadata,
                {
                    "title": r["title"],
                    "entity_type": r["entity_type"],
                    "source_url": r["source_url"],
                    "chunk_index": r["chunk_index"],
                },
            )
            for r in rows
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(
        self, query_text: str, top_k: int, entity_filters: List[str]
    ) -> Any:
        query_embedding = self.embedder.encode([query_text]).tolist()
        where: Where | None = None
        if len(entity_filters) == 1:
            where = cast(Where, {"entity_type": entity_filters[0]})
        elif len(entity_filters) > 1:
            where = cast(
                Where,
                {"$or": [cast(Where, {"entity_type": t}) for t in entity_filters]},
            )
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

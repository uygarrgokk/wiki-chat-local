from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

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

    def upsert_rows(self, rows: List[Dict]) -> None:
        if not rows:
            return
        texts = [r["text"] for r in rows]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        ids = [r["id"] for r in rows]
        metadatas = [
            {
                "title": r["title"],
                "entity_type": r["entity_type"],
                "source_url": r["source_url"],
                "chunk_index": r["chunk_index"],
            }
            for r in rows
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int, entity_filters: List[str]):
        query_embedding = self.embedder.encode([query_text]).tolist()
        where = None
        if len(entity_filters) == 1:
            where = {"entity_type": entity_filters[0]}
        elif len(entity_filters) > 1:
            where = {"$or": [{"entity_type": t} for t in entity_filters]}
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

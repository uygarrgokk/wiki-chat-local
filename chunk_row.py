from typing import TypedDict


class ChunkRow(TypedDict):
    """One text chunk ready for embedding and Chroma upsert."""

    id: str
    title: str
    entity_type: str
    source_url: str
    chunk_index: int
    text: str

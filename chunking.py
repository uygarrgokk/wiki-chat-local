from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

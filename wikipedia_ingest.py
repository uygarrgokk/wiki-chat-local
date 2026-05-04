import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

from chunking import chunk_text
from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, SQLITE_PATH, WIKIPEDIA_API
from entities import PEOPLE, PLACES


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            title TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            text_length INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def fetch_wikipedia_extract(title: str) -> Tuple[str, str]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    response = requests.get(WIKIPEDIA_API, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    page_title = page.get("title", title)
    source_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    return extract, source_url


def build_local_corpus() -> List[Dict[str, str]]:
    ensure_dirs()
    conn = init_sqlite()
    rows: List[Dict[str, str]] = []

    worklist = [("person", p) for p in PEOPLE] + [("place", p) for p in PLACES]
    for entity_type, title in tqdm(worklist, desc="Ingesting Wikipedia pages"):
        try:
            text, source_url = fetch_wikipedia_extract(title)
        except Exception:
            continue
        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        conn.execute(
            """
            INSERT OR REPLACE INTO docs(title, entity_type, source_url, text_length)
            VALUES (?, ?, ?, ?)
            """,
            (title, entity_type, source_url, len(text)),
        )
        for idx, ch in enumerate(chunks):
            chunk_id = f"{entity_type}:{title}:{idx}"
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks(id, title, entity_type, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, title, entity_type, idx, ch),
            )
            rows.append(
                {
                    "id": chunk_id,
                    "title": title,
                    "entity_type": entity_type,
                    "source_url": source_url,
                    "chunk_index": idx,
                    "text": ch,
                }
            )
    conn.commit()
    conn.close()
    return rows

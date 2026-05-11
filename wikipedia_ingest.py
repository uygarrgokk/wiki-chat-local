import sqlite3
import time
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

from chunking import chunk_text
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SQLITE_PATH,
    WIKIPEDIA_API,
    WIKIPEDIA_HEADERS,
)

from chunk_row import ChunkRow
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


def clear_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM docs")
    conn.commit()


def fetch_wikipedia_extract(title: str) -> Tuple[str, str]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        
    }
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(
                WIKIPEDIA_API, params=params, headers=WIKIPEDIA_HEADERS, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            extract = page.get("extract", "")
            page_title = page.get("title", title)
            source_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            return extract, source_url
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            # Retry only for temporary rate-limit/server issues.
            if status not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 5:
                raise
            time.sleep(1.0 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Wikipedia fetch failed for unknown reason.")


def build_local_corpus(verbose: bool = False) -> List[ChunkRow]:
    ensure_dirs()
    conn = init_sqlite()
    clear_sqlite(conn)
    rows: List[ChunkRow] = []
    stats: Dict[str, int] = {
        "person_success": 0,
        "place_success": 0,
        "person_failed": 0,
        "place_failed": 0,
    }

    worklist = [("person", p) for p in PEOPLE] + [("place", p) for p in PLACES]
    if verbose:
        print(f"Starting ingestion of {len(worklist)} entities...")

    for entity_type, title in tqdm(worklist, desc="Ingesting Wikipedia pages"):
        if verbose:
            print(f"Fetching: {title}...")
        try:
            text, source_url = fetch_wikipedia_extract(title)
        except Exception as exc:
            stats[f"{entity_type}_failed"] += 1
            if verbose:
                print(f"Failed to ingest {title}: {exc}")
            continue
        if not text.strip():
            stats[f"{entity_type}_failed"] += 1
            if verbose:
                print(f"Failed to ingest {title}: empty extract")
            continue

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            stats[f"{entity_type}_failed"] += 1
            if verbose:
                print(f"Failed to ingest {title}: no chunks produced")
            continue

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
            row: ChunkRow = {
                "id": chunk_id,
                "title": title,
                "entity_type": entity_type,
                "source_url": source_url,
                "chunk_index": idx,
                "text": ch,
            }
            rows.append(row)
        stats[f"{entity_type}_success"] += 1
        if verbose:
            print(f"Successfully ingested {title} ({len(chunks)} chunks).")

    conn.commit()
    conn.close()
    if verbose:
        print("\nIngestion summary")
        print(
            "People: "
            f"{stats['person_success']} success / {stats['person_failed']} failed"
        )
        print(
            "Places: "
            f"{stats['place_success']} success / {stats['place_failed']} failed"
        )
        print(f"Total chunks prepared: {len(rows)}")
    return rows

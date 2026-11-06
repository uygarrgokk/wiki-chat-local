"""SQLite context engine: short-term turns, rolling summary, long-term topics."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Callable

from config import MEMORY_DB_PATH, RECENT_MESSAGES, SUMMARY_TRIGGER

Summarizer = Callable[[str, str], str]  # (previous_summary, new_transcript) -> summary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
    summarized_upto INTEGER NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS topics (
    session_id TEXT NOT NULL, title TEXT NOT NULL, entity_type TEXT NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 1, last_ts REAL NOT NULL,
    PRIMARY KEY (session_id, title));
"""


class MemoryStore:
    def __init__(self, path: Path | str = MEMORY_DB_PATH) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    @staticmethod
    def new_session() -> str:
        return uuid.uuid4().hex[:12]

    # -- writes ---------------------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
            (session_id, role, content, time.time()),
        )
        self.conn.commit()

    def note_topics(self, session_id: str, topics: list[tuple[str, str]]) -> None:
        """Record (title, entity_type) pairs the conversation touched this turn."""
        now = time.time()
        for title, entity_type in topics:
            self.conn.execute(
                """INSERT INTO topics (session_id, title, entity_type, last_ts)
                   VALUES (?,?,?,?)
                   ON CONFLICT(session_id, title)
                   DO UPDATE SET mentions = mentions + 1, last_ts = excluded.last_ts""",
                (session_id, title, entity_type, now),
            )
        self.conn.commit()

    # -- reads ----------------------------------------------------------------
    def recent_messages(
        self, session_id: str, limit: int = RECENT_MESSAGES, after_id: int = 0
    ) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id DESC LIMIT ?",
            (session_id, after_id, limit),
        ).fetchall()
        return rows[::-1]

    def _summary_row(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM summaries WHERE session_id=?", (session_id,)
        ).fetchone()

    def summary(self, session_id: str) -> str:
        row = self._summary_row(session_id)
        return row["summary"] if row else ""

    def recent_topics(self, session_id: str, limit: int = 5) -> list[str]:
        rows = self.conn.execute(
            "SELECT title FROM topics WHERE session_id=? ORDER BY last_ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [r["title"] for r in rows]

    def past_summaries(self, session_id: str, limit: int = 2) -> list[str]:
        """Long-term memory: summaries of earlier sessions, newest first."""
        rows = self.conn.execute(
            "SELECT summary FROM summaries WHERE session_id != ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [r["summary"] for r in rows]

    # -- compression ----------------------------------------------------------
    def compress(self, session_id: str, summarize: Summarizer) -> bool:
        """Fold old messages into the rolling summary; recent ones stay verbatim."""
        row = self._summary_row(session_id)
        upto = row["summarized_upto"] if row else 0
        pending = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id",
            (session_id, upto),
        ).fetchall()
        if len(pending) < SUMMARY_TRIGGER:
            return False
        old = pending[:-RECENT_MESSAGES]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        try:
            new_summary = summarize(self.summary(session_id), transcript).strip()
        except Exception:
            return False  # keep raw messages; retry next turn
        if not new_summary:
            return False
        self.conn.execute(
            """INSERT INTO summaries (session_id, summary, summarized_upto, ts)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary,
               summarized_upto=excluded.summarized_upto, ts=excluded.ts""",
            (session_id, new_summary, old[-1]["id"], time.time()),
        )
        self.conn.commit()
        return True

    # -- prompt injection -----------------------------------------------------
    def build_context(self, session_id: str) -> str:
        parts: list[str] = []
        if past := self.past_summaries(session_id):
            parts.append("Earlier sessions (long-term memory):\n- " + "\n- ".join(past))
        if summary := self.summary(session_id):
            parts.append(f"Conversation summary so far:\n{summary}")
        if topics := self.recent_topics(session_id):
            parts.append("Topics currently being discussed: " + ", ".join(topics))
        row = self._summary_row(session_id)
        floor = row["summarized_upto"] if row else 0
        if recent := self.recent_messages(session_id, after_id=floor):
            lines = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            parts.append(f"Recent messages:\n{lines}")
        return "\n\n".join(parts)

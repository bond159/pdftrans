"""Persistent notes: what the proofreader found per paragraph, and manual corrections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .config import CONFIG_DIR


def text_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class Store:
    """sqlite tables:

    notes      source paragraph -> draft, final translation and reviewer problems,
               so the report still knows about problems when BabelDOC serves a
               cached translation on a later run.
    overrides  source paragraph -> translation the user typed in; always wins.
    """

    def __init__(self, path: Path | None = None):
        path = path or CONFIG_DIR / "pdftrans.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS notes (k TEXT PRIMARY KEY, source TEXT, draft TEXT, final TEXT, problems TEXT)"
            )
            self.db.execute("CREATE TABLE IF NOT EXISTS overrides (k TEXT PRIMARY KEY, source TEXT, translation TEXT)")
            self.db.commit()

    def save_note(self, source: str, draft: str, final: str, problems: list[str]) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?)",
                (text_key(source), source, draft, final, json.dumps(problems, ensure_ascii=False)),
            )
            self.db.commit()

    def note(self, source: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT draft, final, problems FROM notes WHERE k = ?", (text_key(source),)
            ).fetchone()
        if not row:
            return None
        return {"draft": row[0], "final": row[1], "problems": json.loads(row[2] or "[]")}

    def set_override(self, source: str, translation: str) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO overrides VALUES (?, ?, ?)", (text_key(source), source, translation)
            )
            self.db.commit()

    def remove_override(self, source: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM overrides WHERE k = ?", (text_key(source),))
            self.db.commit()

    def override(self, source: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT translation FROM overrides WHERE k = ?", (text_key(source),)).fetchone()
        return row[0] if row else None

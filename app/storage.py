import json
import sqlite3
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = data_dir / "app.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
              canonical TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chapters (
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
              title TEXT NOT NULL, text TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS glossary (
              id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
              source_term TEXT NOT NULL, target_term TEXT NOT NULL, category TEXT NOT NULL,
              aliases TEXT NOT NULL, status TEXT NOT NULL, context TEXT NOT NULL,
              first_seen_chapter TEXT NOT NULL, notes TEXT NOT NULL,
              UNIQUE(document_id, source_term), FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS translations (
              document_id TEXT NOT NULL, chapter_id TEXT NOT NULL, markdown TEXT NOT NULL,
              warnings TEXT NOT NULL, PRIMARY KEY(document_id, chapter_id)
            );
            """)

    def connect(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def add_document(self, doc: dict[str, Any], chapters: list[dict[str, Any]]):
        with self.connect() as c:
            c.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", tuple(doc.values()))
            c.executemany("INSERT INTO chapters VALUES (?, ?, ?, ?, ?)", [tuple(x.values()) for x in chapters])

    def document(self, doc_id: str):
        with self.connect() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def chapters(self, doc_id: str):
        with self.connect() as c:
            return [dict(x) for x in c.execute("SELECT * FROM chapters WHERE document_id=? ORDER BY ordinal", (doc_id,))]

    def chapter(self, doc_id: str, chapter_id: str):
        with self.connect() as c:
            row = c.execute("SELECT * FROM chapters WHERE document_id=? AND id=?", (doc_id, chapter_id)).fetchone()
            return dict(row) if row else None

    def glossary(self, doc_id: str):
        with self.connect() as c:
            rows = c.execute("SELECT * FROM glossary WHERE document_id=? ORDER BY id", (doc_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["aliases"] = json.loads(item["aliases"])
                result.append(item)
            return result

    def upsert_glossary(self, doc_id: str, item: dict[str, Any]):
        values = (doc_id, item["source_term"], item["target_term"], item.get("category", "general"),
                  json.dumps(item.get("aliases", [])), item.get("status", "suggested"),
                  item.get("context", ""), item.get("first_seen_chapter", ""), item.get("notes", ""))
        with self.connect() as c:
            c.execute("""INSERT INTO glossary(document_id,source_term,target_term,category,aliases,status,context,first_seen_chapter,notes)
                         VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(document_id,source_term) DO UPDATE SET
                         target_term=excluded.target_term, category=excluded.category, aliases=excluded.aliases,
                         status=excluded.status, context=excluded.context, first_seen_chapter=excluded.first_seen_chapter,
                         notes=excluded.notes""", values)

    def save_translation(self, doc_id, chapter_id, markdown, warnings):
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO translations VALUES(?,?,?,?)", (doc_id, chapter_id, markdown, json.dumps(warnings)))

    def translation(self, doc_id, chapter_id):
        with self.connect() as c:
            row = c.execute("SELECT * FROM translations WHERE document_id=? AND chapter_id=?", (doc_id, chapter_id)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["warnings"] = json.loads(result["warnings"])
            return result

"""SQLite state for episodes, stories and seen-URL deduplication.

The database holds *state*, not content: which episodes exist, what stage each
reached, and which URLs have already been covered so a story does not lead the
show two days running. The content itself lives in files under ``work/``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from .models import Episode, Story

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    date        TEXT NOT NULL,
    number      INTEGER NOT NULL DEFAULT 0,
    title       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'new',
    created_at  TEXT NOT NULL,
    artifacts   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS stories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   TEXT NOT NULL,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    summary      TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    score        REAL NOT NULL DEFAULT 0,
    cluster      INTEGER NOT NULL DEFAULT -1,
    used         INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_stories_url ON stories(url);
CREATE INDEX IF NOT EXISTS idx_stories_episode ON stories(episode_id);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    at         TEXT NOT NULL
);
"""


class Database:
    """Episode state.

    The connection is shared across threads because the review UI's sync request
    handlers run in the server's threadpool, not the thread that built the app.
    sqlite3 forbids that by default, so ``check_same_thread=False`` is paired
    with a lock around every statement — the access pattern here is a handful of
    small queries per request, so serializing them costs nothing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---- episodes -------------------------------------------------------

    def upsert_episode(self, episode: Episode) -> Episode:
        if not episode.created_at:
            episode.created_at = datetime.now().isoformat(timespec="seconds")
        if not episode.number:
            episode.number = self.next_episode_number()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO episodes (id, date, number, title, status, created_at, artifacts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    artifacts=excluded.artifacts
                """,
                (
                    episode.id,
                    episode.date,
                    episode.number,
                    episode.title,
                    episode.status,
                    episode.created_at,
                    json.dumps(episode.artifacts),
                ),
            )
        return episode

    def get_episode(self, episode_id: str) -> Episode | None:
        rows = self._query("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return self._row_to_episode(rows[0]) if rows else None

    def list_episodes(self, limit: int = 30) -> list[Episode]:
        rows = self._query("SELECT * FROM episodes ORDER BY date DESC LIMIT ?", (limit,))
        return [self._row_to_episode(r) for r in rows]

    def next_episode_number(self) -> int:
        rows = self._query("SELECT MAX(number) AS n FROM episodes")
        return int(rows[0]["n"] or 0) + 1

    def set_status(self, episode_id: str, status: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id)
            )

    def set_artifact(self, episode_id: str, key: str, value: str) -> None:
        episode = self.get_episode(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode {episode_id!r}")
        episode.artifacts[key] = str(value)
        with self._tx() as conn:
            conn.execute(
                "UPDATE episodes SET artifacts = ? WHERE id = ?",
                (json.dumps(episode.artifacts), episode_id),
            )

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            date=row["date"],
            title=row["title"],
            status=row["status"],
            number=row["number"],
            created_at=row["created_at"],
            artifacts=json.loads(row["artifacts"] or "{}"),
        )

    # ---- stories --------------------------------------------------------

    def add_stories(self, episode_id: str, stories: list[Story]) -> int:
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT INTO stories
                    (episode_id, url, title, source, summary, published_at, score, cluster)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        episode_id,
                        s.url,
                        s.title,
                        s.source,
                        s.summary,
                        s.published_at,
                        s.score,
                        s.cluster,
                    )
                    for s in stories
                ],
            )
        return len(stories)

    def seen_urls(self, within_days: int = 21) -> set[str]:
        """URLs already attached to an episode, within a window of history.

        Used to stop a story recurring. The default window is deliberately
        generous — a slow-moving story gets re-reported by several outlets over
        a couple of weeks and it should lead the show once, not four times.

        ``within_days <= 0`` means all history: no story is ever covered twice.
        That is the right setting for a show that must be genuinely new every
        run, and it costs nothing until the table is very large.

        Note this only sees stories that made it into a brief. Items fetched and
        scored but never selected stay eligible, which is deliberate — being
        passed over on a busy day should not bury a story forever.
        """
        if within_days <= 0:
            rows = self._query("SELECT DISTINCT url FROM stories")
            return {r["url"] for r in rows}

        cutoff = (datetime.now() - timedelta(days=within_days)).strftime("%Y-%m-%d")
        rows = self._query(
            """
            SELECT DISTINCT s.url FROM stories s
            JOIN episodes e ON e.id = s.episode_id
            WHERE e.date >= ?
            """,
            (cutoff,),
        )
        return {r["url"] for r in rows}

    # ---- events ---------------------------------------------------------

    def log(self, episode_id: str, stage: str, status: str, detail: str = "") -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO events (episode_id, stage, status, detail, at) VALUES (?, ?, ?, ?, ?)",
                (
                    episode_id,
                    stage,
                    status,
                    detail[:2000],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def events(self, episode_id: str) -> list[dict[str, str]]:
        rows = self._query(
            "SELECT stage, status, detail, at FROM events WHERE episode_id = ? ORDER BY id",
            (episode_id,),
        )
        return [dict(r) for r in rows]

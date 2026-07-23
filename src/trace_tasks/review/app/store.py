"""Local SQLite persistence for public task review decisions and issues."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

AUDIT_FIELDS = (
    "prompt",
    "rendering",
    "answer",
    "annotation",
    "verifier_trace",
    "distribution",
    "code_docs",
    "taxonomy",
)
ISSUE_CATEGORIES = frozenset((*AUDIT_FIELDS, "other"))
ISSUE_SEVERITIES = frozenset({"note", "issue", "blocker"})
ISSUE_STATUSES = frozenset({"open", "resolved"})
ASSET_DECISIONS = frozenset({"approve", "improve", "remove"})


class _ThreadLock(AbstractContextManager["_ThreadLock"]):
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> "_ThreadLock":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._lock.release()


def _default_lock_factory(path: str) -> AbstractContextManager[Any]:
    try:
        from filelock import FileLock
    except ImportError:  # pragma: no cover - review extra installs filelock.
        return _ThreadLock()
    return FileLock(path)


class ReviewStore:
    """Concurrency-safe local store with an injectable cross-platform lock."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        lock_factory: Callable[[str], AbstractContextManager[Any]] | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        factory = lock_factory or _default_lock_factory
        self._write_lock = factory(f"{self.database_path}.lock")
        self._initialize()

    def create_issue(
        self,
        *,
        title: str,
        body: str,
        author: str = "",
        category: str = "other",
        severity: str = "issue",
        domain: str = "",
        scene_id: str = "",
        task_id: str = "",
        sample_uid: str = "",
        recipe_digest: str = "",
        sample_semantic_hash: str = "",
    ) -> dict[str, Any]:
        title_text = str(title).strip()
        body_text = str(body).strip()
        if not title_text or not body_text:
            raise ValueError("issue title and body must not be empty")
        category = _choice(category, ISSUE_CATEGORIES, "category")
        severity = _choice(severity, ISSUE_SEVERITIES, "severity")
        task_text = str(task_id).strip()
        sample_text = str(sample_uid).strip()
        recipe_text = str(recipe_digest).strip()
        semantic_text = str(sample_semantic_hash).strip()
        if task_text and not recipe_text:
            raise ValueError("task issues require a recipe digest")
        if sample_text and not semantic_text:
            raise ValueError("sample issues require a semantic hash")
        now = _now()
        issue_id = uuid4().hex
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO issues (
                    id, title, body, author, category, severity, status,
                    domain, scene_id, task_id, sample_uid, recipe_digest,
                    sample_semantic_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    issue_id,
                    title_text,
                    body_text,
                    str(author).strip(),
                    category,
                    severity,
                    str(domain),
                    str(scene_id),
                    task_text,
                    sample_text,
                    recipe_text,
                    semantic_text,
                    now,
                    now,
                ),
            )
        return self.get_issue(issue_id)

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM issues WHERE id = ?", (str(issue_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown issue id: {issue_id}")
        return dict(row)

    def list_issues(
        self,
        *,
        status: str | None = None,
        task_id: str | None = None,
        sample_uid: str | None = None,
        recipe_digest: str | None = None,
        sample_semantic_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(_choice(status, ISSUE_STATUSES, "status"))
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(str(task_id))
        if sample_uid is not None:
            clauses.append("sample_uid = ?")
            values.append(str(sample_uid))
        if recipe_digest is not None:
            clauses.append("recipe_digest = ?")
            values.append(str(recipe_digest))
        if sample_semantic_hash is not None:
            clauses.append("sample_semantic_hash = ?")
            values.append(str(sample_semantic_hash))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM issues{where} ORDER BY updated_at DESC, id", values
            ).fetchall()
        return [dict(row) for row in rows]

    def add_comment(
        self,
        issue_id: str,
        *,
        body: str,
        author: str = "",
    ) -> dict[str, Any]:
        body_text = str(body).strip()
        if not body_text:
            raise ValueError("comment body must not be empty")
        self.get_issue(issue_id)
        comment_id = uuid4().hex
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO issue_comments (id, issue_id, body, author, created_at) VALUES (?, ?, ?, ?, ?)",
                (comment_id, str(issue_id), body_text, str(author).strip(), now),
            )
            connection.execute(
                "UPDATE issues SET updated_at = ? WHERE id = ?", (now, str(issue_id))
            )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM issue_comments WHERE id = ?", (comment_id,)
            ).fetchone()
        return dict(row) if row is not None else {}

    def issue_comments(self, issue_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM issue_comments WHERE issue_id = ? ORDER BY created_at, id",
                (str(issue_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_issue_status(self, issue_id: str, status: str) -> dict[str, Any]:
        self.get_issue(issue_id)
        normalized = _choice(status, ISSUE_STATUSES, "status")
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
                (normalized, _now(), str(issue_id)),
            )
        return self.get_issue(issue_id)

    def get_task_audit(self, task_id: str, recipe_digest: str) -> dict[str, Any]:
        digest = _required_identity(recipe_digest, "recipe_digest")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_audits WHERE task_id = ? AND recipe_digest = ?",
                (str(task_id), digest),
            ).fetchone()
        if row is None:
            return {
                "task_id": str(task_id),
                "recipe_digest": digest,
                **{field: False for field in AUDIT_FIELDS},
                "notes": "",
                "updated_by": "",
                "updated_at": "",
            }
        result = dict(row)
        for field in AUDIT_FIELDS:
            result[field] = bool(result[field])
        return result

    def update_task_audit(
        self,
        task_id: str,
        recipe_digest: str,
        *,
        values: Mapping[str, Any],
        notes: str = "",
        updated_by: str = "",
    ) -> dict[str, Any]:
        unknown = set(values) - set(AUDIT_FIELDS)
        if unknown:
            raise ValueError(f"unknown audit fields: {sorted(unknown)!r}")
        digest = _required_identity(recipe_digest, "recipe_digest")
        existing = self.get_task_audit(task_id, digest)
        normalized = {
            field: int(bool(values.get(field, existing[field])))
            for field in AUDIT_FIELDS
        }
        now = _now()
        columns = ", ".join(AUDIT_FIELDS)
        placeholders = ", ".join("?" for _ in AUDIT_FIELDS)
        updates = ", ".join(f"{field} = excluded.{field}" for field in AUDIT_FIELDS)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO task_audits (
                    task_id, recipe_digest, {columns}, notes, updated_by, updated_at
                ) VALUES (?, ?, {placeholders}, ?, ?, ?)
                ON CONFLICT(task_id, recipe_digest) DO UPDATE SET
                    {updates}, notes = excluded.notes,
                    updated_by = excluded.updated_by, updated_at = excluded.updated_at
                """,
                (
                    str(task_id),
                    digest,
                    *(normalized[field] for field in AUDIT_FIELDS),
                    str(notes).strip(),
                    str(updated_by).strip(),
                    now,
                ),
            )
        return self.get_task_audit(task_id, digest)

    def task_audits(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, recipe_digest FROM task_audits "
                "ORDER BY task_id, recipe_digest"
            ).fetchall()
        return [
            self.get_task_audit(str(row["task_id"]), str(row["recipe_digest"]))
            for row in rows
        ]

    def set_asset_review(
        self,
        asset_id: str,
        *,
        kind: str,
        decision: str,
        notes: str = "",
        updated_by: str = "",
    ) -> dict[str, Any]:
        normalized = _choice(decision, ASSET_DECISIONS, "decision")
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO asset_reviews (asset_id, kind, decision, notes, updated_by, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, asset_id) DO UPDATE SET
                    decision = excluded.decision,
                    notes = excluded.notes, updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (
                    str(asset_id),
                    str(kind),
                    normalized,
                    str(notes).strip(),
                    str(updated_by).strip(),
                    now,
                ),
            )
        return self.get_asset_review(asset_id, kind=kind)

    def get_asset_review(
        self, asset_id: str, *, kind: str | None = None
    ) -> dict[str, Any]:
        with self._connect() as connection:
            if kind is None:
                row = connection.execute(
                    "SELECT * FROM asset_reviews WHERE asset_id = ? ORDER BY kind LIMIT 1",
                    (str(asset_id),),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM asset_reviews WHERE kind = ? AND asset_id = ?",
                    (str(kind), str(asset_id)),
                ).fetchone()
        return dict(row) if row is not None else {}

    def asset_reviews(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM asset_reviews ORDER BY kind, asset_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def export_snapshot(self) -> dict[str, Any]:
        issues = self.list_issues()
        return {
            "issues": [
                {**issue, "comments": self.issue_comments(str(issue["id"]))}
                for issue in issues
            ],
            "task_audits": self.task_audits(),
            "asset_reviews": self.asset_reviews(),
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._write_lock, self._connect() as connection:
            _assert_compatible_existing_schema(connection)
            connection.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT '',
                    scene_id TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    sample_uid TEXT NOT NULL DEFAULT '',
                    recipe_digest TEXT NOT NULL DEFAULT '',
                    sample_semantic_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS issue_comments (
                    id TEXT PRIMARY KEY,
                    issue_id TEXT NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
                    body TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_audits (
                    task_id TEXT NOT NULL,
                    recipe_digest TEXT NOT NULL,
                    prompt INTEGER NOT NULL DEFAULT 0,
                    rendering INTEGER NOT NULL DEFAULT 0,
                    answer INTEGER NOT NULL DEFAULT 0,
                    annotation INTEGER NOT NULL DEFAULT 0,
                    verifier_trace INTEGER NOT NULL DEFAULT 0,
                    distribution INTEGER NOT NULL DEFAULT 0,
                    code_docs INTEGER NOT NULL DEFAULT 0,
                    taxonomy INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (task_id, recipe_digest)
                );
                CREATE TABLE IF NOT EXISTS asset_reviews (
                    asset_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (kind, asset_id)
                );
                CREATE INDEX IF NOT EXISTS idx_issues_task ON issues(task_id, recipe_digest, status);
                CREATE INDEX IF NOT EXISTS idx_issues_sample ON issues(sample_uid, sample_semantic_hash, status);
                CREATE INDEX IF NOT EXISTS idx_comments_issue ON issue_comments(issue_id, created_at);
                """)


def _choice(value: str, choices: frozenset[str], label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)!r}")
    return normalized


def _required_identity(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _assert_compatible_existing_schema(connection: sqlite3.Connection) -> None:
    """Fail closed instead of attaching old decisions to replacement recipes."""

    expected = {
        "issues": ({"recipe_digest", "sample_semantic_hash"}, None),
        "task_audits": ({"task_id", "recipe_digest"}, ("task_id", "recipe_digest")),
        "asset_reviews": ({"kind", "asset_id"}, ("kind", "asset_id")),
    }
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    for table, (required_columns, required_primary_key) in expected.items():
        if table not in existing:
            continue
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        columns = {str(row[1]) for row in rows}
        primary_key = tuple(
            str(row[1]) for row in sorted(rows, key=lambda row: int(row[5])) if row[5]
        )
        if not required_columns.issubset(columns) or (
            required_primary_key is not None and primary_key != required_primary_key
        ):
            raise RuntimeError(
                "review database schema predates recipe-bound review state; "
                "export or move the old database and restart"
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ASSET_DECISIONS",
    "AUDIT_FIELDS",
    "ISSUE_CATEGORIES",
    "ISSUE_SEVERITIES",
    "ISSUE_STATUSES",
    "ReviewStore",
]

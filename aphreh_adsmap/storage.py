"""Bounded-cache SQLite results and explicitly closed disk-backed numerical arrays."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator
import numpy as np


def clean(value: Any) -> Any:
    """Unavailable internal floating values become JSON null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        raise TypeError("Do not serialize numerical arrays into JSON; store an .npy file")
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value: Any) -> str:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(clean(payload), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close_map(array: np.ndarray | None) -> None:
    """Windows-safe explicit flush and close; no mapping may survive scratch removal."""
    if isinstance(array, np.memmap):
        if array.flags.writeable:
            array.flush()
        mm = getattr(array, "_mmap", None)
        if mm is not None and not mm.closed:
            mm.close()


@contextlib.contextmanager
def mapped(path: Path, *, shape: tuple[int, ...] | None = None,
           dtype: str = "float64", mode: str = "r") -> Iterator[np.memmap]:
    if shape is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    else:
        a = np.load(path, mmap_mode=mode, allow_pickle=False)
    try:
        yield a
    finally:
        close_map(a)


class Store:
    """No object cache: every completed cell and realization is committed on disk."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("""CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL, scope TEXT NOT NULL, key TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(kind, scope, key)) WITHOUT ROWID""")
        self.db.commit()

    def put(self, kind: str, scope: str, key: str, payload: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?)",
                        (kind, scope, key, dumps(payload)))

    def get(self, kind: str, scope: str, key: str) -> Any | None:
        row = self.db.execute("SELECT payload FROM records WHERE kind=? AND scope=? AND key=?",
                              (kind, scope, key)).fetchone()
        return None if row is None else json.loads(row[0])

    def rows(self, kind: str, scope: str | None = None):
        if scope is None:
            cur = self.db.execute("SELECT payload FROM records WHERE kind=? ORDER BY scope,key", (kind,))
        else:
            cur = self.db.execute("SELECT payload FROM records WHERE kind=? AND scope=? ORDER BY key",
                                  (kind, scope))
        for row in cur:
            yield json.loads(row[0])

    def delete_scope(self, scope: str) -> None:
        self.db.execute("DELETE FROM records WHERE scope=?", (scope,))

    def import_run_from(self, source: Path, run: str) -> None:
        """Copy resumable records for one run from another Store database.

        Calibration workers execute in isolated SQLite files.  Importing the
        run namespace first preserves completed cell/null checkpoints from an
        interrupted parent project without allowing concurrent writers.
        """
        source = Path(source).resolve()
        if not source.is_file() or source == self.path.resolve():
            return
        self.db.commit()
        self.db.execute("ATTACH DATABASE ? AS source_store", (str(source),))
        try:
            self.db.execute(
                """INSERT OR REPLACE INTO records(kind,scope,key,payload)
                   SELECT kind,scope,key,payload FROM source_store.records
                   WHERE scope=? OR scope LIKE ? OR scope LIKE ?""",
                (run, run + "/%", run + "__NULL_%"),
            )
            self.db.commit()
        finally:
            self.db.execute("DETACH DATABASE source_store")

    def merge_from(self, source: Path) -> None:
        """Atomically merge every committed record from an isolated worker."""
        source = Path(source).resolve()
        if source == self.path.resolve():
            return
        if not source.is_file():
            raise FileNotFoundError(source)
        self.db.commit()
        self.db.execute("ATTACH DATABASE ? AS worker_store", (str(source),))
        try:
            self.db.execute(
                """INSERT OR REPLACE INTO records(kind,scope,key,payload)
                   SELECT kind,scope,key,payload FROM worker_store.records"""
            )
            self.db.commit()
        finally:
            self.db.execute("DETACH DATABASE worker_store")

    def commit(self) -> None:
        self.db.commit()

    def export_csv(self, kind: str, path: Path, scope: str | None = None, run_filter: str | None = None) -> None:
        # Two cursor passes, not list(cursor): RAM is independent of row count.
        def selected():
            for row in self.rows(kind, scope):
                if run_filter is None or row.get("run") == run_filter:
                    yield row
        fields: list[str] = []
        for row in selected():
            for name in row:
                if name not in fields:
                    fields.append(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in selected():
                writer.writerow({k: dumps(v) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
        os.replace(tmp, path)

    def close(self) -> None:
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()

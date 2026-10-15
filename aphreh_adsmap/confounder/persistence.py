"""Dedicated CA3 persistence and resume checkpoints for confounder execution."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from ..storage import Store, dumps


CONFOUNDER_PERSISTENCE_SCHEMA = "APHREH_CONFOUNDER_PERSISTENCE_CA3_V1"


class ConfounderRunState(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ConfounderCheckpointState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConfounderPersistence:
    """Candidate-confounder tables in the existing project ``results.sqlite``."""

    def __init__(self, store: Store):
        self.store = store
        self.db = store.db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ca_runs (
                run_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                reference_identity TEXT NOT NULL,
                confounder_identity TEXT NOT NULL,
                payload TEXT NOT NULL,
                state TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS ca_runs_candidate_idx
                ON ca_runs(candidate_id, updated_utc);
            CREATE TABLE IF NOT EXISTS ca_checkpoints (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                state TEXT NOT NULL,
                payload TEXT,
                updated_utc TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY(run_id, stage, scope, key),
                FOREIGN KEY(run_id) REFERENCES ca_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS ca_checkpoints_state_idx
                ON ca_checkpoints(run_id, state, stage);
            """
        )
        self.db.commit()

    def register_run(
        self,
        *,
        run_id: str,
        candidate_id: str,
        reference_identity: str,
        confounder_identity: str,
        payload: Any,
    ) -> None:
        now = _utc_now()
        encoded = dumps(payload)
        row = self.db.execute(
            "SELECT confounder_identity,payload FROM ca_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is not None:
            if row[0] != confounder_identity or row[1] != encoded:
                raise ValueError(f"confounder run identity collision for {run_id}")
            self.db.execute("UPDATE ca_runs SET updated_utc=? WHERE run_id=?", (now, run_id))
            self.db.commit()
            return
        self.db.execute(
            """INSERT INTO ca_runs(
                run_id,schema_version,candidate_id,reference_identity,confounder_identity,
                payload,state,created_utc,updated_utc,error
            ) VALUES(?,?,?,?,?,?,?,?,?,NULL)""",
            (
                run_id, CONFOUNDER_PERSISTENCE_SCHEMA, candidate_id,
                reference_identity, confounder_identity, encoded,
                ConfounderRunState.PLANNED.value, now, now,
            ),
        )
        self.db.commit()

    def run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """SELECT run_id,schema_version,candidate_id,reference_identity,confounder_identity,
                      payload,state,created_utc,updated_utc,error
               FROM ca_runs WHERE run_id=?""", (run_id,)
        ).fetchone()
        if row is None:
            return None
        keys = (
            "run_id", "schema_version", "candidate_id", "reference_identity",
            "confounder_identity", "payload", "state", "created_utc", "updated_utc", "error",
        )
        out = dict(zip(keys, row))
        out["payload"] = json.loads(out["payload"])
        return out

    def set_run_state(
        self, run_id: str, state: ConfounderRunState, *, error: str | None = None
    ) -> None:
        cur = self.db.execute(
            "UPDATE ca_runs SET state=?,updated_utc=?,error=? WHERE run_id=?",
            (state.value, _utc_now(), error, run_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"unknown confounder run {run_id}")
        self.db.commit()

    def put_checkpoint(
        self,
        run_id: str,
        stage: str,
        scope: str,
        key: str,
        state: ConfounderCheckpointState,
        *,
        payload: Any = None,
        error: str | None = None,
    ) -> None:
        if self.run(run_id) is None:
            raise KeyError(f"unknown confounder run {run_id}")
        self.db.execute(
            """INSERT OR REPLACE INTO ca_checkpoints(
                   run_id,stage,scope,key,state,payload,updated_utc,error
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                run_id, stage, scope, key, state.value,
                None if payload is None else dumps(payload), _utc_now(), error,
            ),
        )
        self.db.commit()

    def checkpoint(self, run_id: str, stage: str, scope: str, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """SELECT state,payload,updated_utc,error FROM ca_checkpoints
               WHERE run_id=? AND stage=? AND scope=? AND key=?""",
            (run_id, stage, scope, key),
        ).fetchone()
        if row is None:
            return None
        return {
            "state": row[0],
            "payload": None if row[1] is None else json.loads(row[1]),
            "updated_utc": row[2],
            "error": row[3],
        }

    def checkpoints(self, run_id: str) -> Iterable[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT stage,scope,key,state,payload,updated_utc,error
               FROM ca_checkpoints WHERE run_id=? ORDER BY stage,scope,key""",
            (run_id,),
        )
        for row in rows:
            yield {
                "stage": row[0], "scope": row[1], "key": row[2], "state": row[3],
                "payload": None if row[4] is None else json.loads(row[4]),
                "updated_utc": row[5], "error": row[6],
            }

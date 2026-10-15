"""SQLite persistence and resume checkpoints for robustness realizations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

from ..storage import Store, dumps
from .models import (
    CheckpointState,
    RealizationSpec,
    RealizationState,
    SensitivityPlan,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RobustnessPersistence:
    """Dedicated RS tables stored in the project's existing ``results.sqlite``.

    The tables are intentionally separate from the generic ``records`` table so
    future long-running sensitivity execution can query realization and
    checkpoint state without deserializing unrelated scientific outputs.
    """

    def __init__(self, store: Store):
        self.store = store
        self.db = store.db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS rs_plans (
                plan_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rs_realizations (
                realization_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                step_index INTEGER,
                trajectory_coordinate REAL,
                comparison_mode TEXT NOT NULL,
                invalidates_from TEXT NOT NULL,
                reference_realization_id TEXT,
                resolved_settings_hash TEXT NOT NULL,
                specification_hash TEXT NOT NULL,
                specification TEXT NOT NULL,
                state TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                error TEXT,
                FOREIGN KEY(plan_id) REFERENCES rs_plans(plan_id)
            );
            CREATE INDEX IF NOT EXISTS rs_realizations_plan_idx
                ON rs_realizations(plan_id, analysis_id, scenario_id, step_index);
            CREATE TABLE IF NOT EXISTS rs_plan_realizations (
                plan_id TEXT NOT NULL,
                realization_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY(plan_id, realization_id),
                FOREIGN KEY(plan_id) REFERENCES rs_plans(plan_id),
                FOREIGN KEY(realization_id) REFERENCES rs_realizations(realization_id)
            );
            CREATE INDEX IF NOT EXISTS rs_plan_realizations_order_idx
                ON rs_plan_realizations(plan_id, ordinal, realization_id);
            CREATE TABLE IF NOT EXISTS rs_realization_factors (
                realization_id TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                config_path TEXT NOT NULL,
                value_json TEXT NOT NULL,
                invalidates_from TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY(realization_id, factor_name),
                FOREIGN KEY(realization_id) REFERENCES rs_realizations(realization_id)
            );
            CREATE TABLE IF NOT EXISTS rs_checkpoints (
                realization_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                state TEXT NOT NULL,
                payload TEXT,
                updated_utc TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY(realization_id, stage, scope, key),
                FOREIGN KEY(realization_id) REFERENCES rs_realizations(realization_id)
            );
            CREATE INDEX IF NOT EXISTS rs_checkpoints_state_idx
                ON rs_checkpoints(realization_id, state, stage);
            CREATE TABLE IF NOT EXISTS rs_comparisons (
                reference_realization_id TEXT NOT NULL,
                sensitivity_realization_id TEXT NOT NULL,
                comparison_mode TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY(reference_realization_id, sensitivity_realization_id, comparison_mode)
            );
            CREATE TABLE IF NOT EXISTS rs_analysis_outputs (
                realization_id TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                output_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY(realization_id, analysis_id, output_kind, scope, key),
                FOREIGN KEY(realization_id) REFERENCES rs_realizations(realization_id)
            );
            CREATE INDEX IF NOT EXISTS rs_analysis_outputs_idx
                ON rs_analysis_outputs(realization_id, analysis_id, output_kind, scope);
            """
        )
        # Backfill membership for databases created by RSinit 2.3.0, where
        # rs_realizations.plan_id represented the only plan membership.
        self.db.execute(
            """INSERT OR IGNORE INTO rs_plan_realizations(plan_id,realization_id,ordinal)
               SELECT plan_id,realization_id,
                      ROW_NUMBER() OVER (PARTITION BY plan_id ORDER BY analysis_id,scenario_id,step_index,realization_id)-1
               FROM rs_realizations"""
        )
        self.db.commit()

    def register_plan(self, plan: SensitivityPlan) -> None:
        now = _utc_now()
        existing = self.db.execute(
            "SELECT payload FROM rs_plans WHERE plan_id=?", (plan.plan_id,)
        ).fetchone()
        payload = dumps(plan.to_dict())
        if existing is not None and existing[0] != payload:
            raise ValueError(f"robustness plan identity collision for {plan.plan_id}")
        self.db.execute(
            "INSERT OR IGNORE INTO rs_plans(plan_id,schema_version,payload,created_utc,updated_utc) VALUES(?,?,?,?,?)",
            (plan.plan_id, plan.schema_version, payload, now, now),
        )
        self.db.execute("UPDATE rs_plans SET updated_utc=? WHERE plan_id=?", (now, plan.plan_id))
        for ordinal, spec in enumerate(plan.all_realizations):
            self.register_realization(plan.plan_id, spec)
            self.db.execute(
                "INSERT OR IGNORE INTO rs_plan_realizations(plan_id,realization_id,ordinal) VALUES(?,?,?)",
                (plan.plan_id, spec.realization_id, ordinal),
            )
        self.db.commit()

    def register_realization(self, plan_id: str, spec: RealizationSpec) -> None:
        now = _utc_now()
        payload = dumps(spec.to_dict())
        existing = self.db.execute(
            "SELECT specification_hash,specification FROM rs_realizations WHERE realization_id=?",
            (spec.realization_id,),
        ).fetchone()
        if existing is not None:
            # A deterministic realization may legitimately belong to more than
            # one robustness plan.  Identity is defined by the realization
            # specification, not by the plan that happens to reference it.
            if existing[0] != spec.specification_hash or existing[1] != payload:
                raise ValueError(f"realization identity collision for {spec.realization_id}")
            return
        self.db.execute(
            """INSERT INTO rs_realizations(
                realization_id,plan_id,kind,scenario_id,analysis_id,step_index,trajectory_coordinate,
                comparison_mode,invalidates_from,reference_realization_id,
                resolved_settings_hash,specification_hash,specification,state,
                created_utc,updated_utc,error
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
            (
                spec.realization_id, plan_id, spec.kind.value, spec.scenario_id,
                spec.analysis_id, spec.step_index, spec.trajectory_coordinate, spec.comparison_mode.value,
                spec.invalidates_from.name, spec.reference_realization_id,
                spec.resolved_settings_hash, spec.specification_hash, payload,
                RealizationState.PLANNED.value, now, now,
            ),
        )
        for ordinal, factor in enumerate(spec.factors):
            self.db.execute(
                "INSERT INTO rs_realization_factors VALUES(?,?,?,?,?,?)",
                (
                    spec.realization_id, factor.name, factor.config_path,
                    dumps(factor.value), factor.invalidates_from.name, ordinal,
                ),
            )

    def realization(self, realization_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """SELECT realization_id,plan_id,kind,scenario_id,analysis_id,step_index,trajectory_coordinate,
                      comparison_mode,invalidates_from,reference_realization_id,
                      resolved_settings_hash,specification_hash,specification,state,
                      created_utc,updated_utc,error
               FROM rs_realizations WHERE realization_id=?""",
            (realization_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "realization_id","plan_id","kind","scenario_id","analysis_id","step_index","trajectory_coordinate",
            "comparison_mode","invalidates_from","reference_realization_id",
            "resolved_settings_hash","specification_hash","specification","state",
            "created_utc","updated_utc","error",
        )
        out = dict(zip(keys, row))
        out["specification"] = json.loads(out["specification"])
        return out

    def list_realizations(self, plan_id: str | None = None) -> Iterable[dict[str, Any]]:
        if plan_id is None:
            ids = self.db.execute(
                "SELECT realization_id FROM rs_realizations ORDER BY created_utc,realization_id"
            )
        else:
            ids = self.db.execute(
                """SELECT r.realization_id
                   FROM rs_plan_realizations m
                   JOIN rs_realizations r ON r.realization_id=m.realization_id
                   WHERE m.plan_id=?
                   ORDER BY m.ordinal,r.realization_id""",
                (plan_id,),
            )
        for (rid,) in ids:
            row = self.realization(rid)
            if row is not None:
                yield row

    def set_realization_state(
        self,
        realization_id: str,
        state: RealizationState,
        *,
        error: str | None = None,
    ) -> None:
        cur = self.db.execute(
            "UPDATE rs_realizations SET state=?,updated_utc=?,error=? WHERE realization_id=?",
            (state.value, _utc_now(), error, realization_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"unknown robustness realization {realization_id}")
        self.db.commit()

    def put_checkpoint(
        self,
        realization_id: str,
        stage: str,
        scope: str,
        key: str,
        state: CheckpointState,
        *,
        payload: Any = None,
        error: str | None = None,
    ) -> None:
        if self.realization(realization_id) is None:
            raise KeyError(f"unknown robustness realization {realization_id}")
        self.db.execute(
            """INSERT OR REPLACE INTO rs_checkpoints(
                   realization_id,stage,scope,key,state,payload,updated_utc,error
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                realization_id, stage, scope, key, state.value,
                None if payload is None else dumps(payload), _utc_now(), error,
            ),
        )
        self.db.commit()

    def checkpoint(self, realization_id: str, stage: str, scope: str, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """SELECT state,payload,updated_utc,error FROM rs_checkpoints
               WHERE realization_id=? AND stage=? AND scope=? AND key=?""",
            (realization_id, stage, scope, key),
        ).fetchone()
        if row is None:
            return None
        return {
            "state": row[0],
            "payload": None if row[1] is None else json.loads(row[1]),
            "updated_utc": row[2],
            "error": row[3],
        }

    def put_comparison(
        self,
        reference_realization_id: str,
        sensitivity_realization_id: str,
        comparison_mode: str,
        payload: Any,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO rs_comparisons VALUES(?,?,?,?,?)",
            (
                reference_realization_id, sensitivity_realization_id,
                comparison_mode, dumps(payload), _utc_now(),
            ),
        )
        self.db.commit()

    def comparison(
        self,
        reference_realization_id: str,
        sensitivity_realization_id: str,
        comparison_mode: str,
    ) -> Any | None:
        row = self.db.execute(
            """SELECT payload FROM rs_comparisons
               WHERE reference_realization_id=? AND sensitivity_realization_id=? AND comparison_mode=?""",
            (reference_realization_id, sensitivity_realization_id, comparison_mode),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def comparisons(
        self, reference_realization_id: str | None = None, comparison_mode: str | None = None
    ) -> Iterable[dict[str, Any]]:
        sql = "SELECT reference_realization_id,sensitivity_realization_id,comparison_mode,payload,updated_utc FROM rs_comparisons WHERE 1=1"
        params: list[Any] = []
        if reference_realization_id is not None:
            sql += " AND reference_realization_id=?"; params.append(reference_realization_id)
        if comparison_mode is not None:
            sql += " AND comparison_mode=?"; params.append(comparison_mode)
        sql += " ORDER BY sensitivity_realization_id,comparison_mode"
        for ref_id, sen_id, mode, payload, updated in self.db.execute(sql, tuple(params)):
            yield {
                "reference_realization_id": ref_id,
                "sensitivity_realization_id": sen_id,
                "comparison_mode": mode,
                "payload": json.loads(payload),
                "updated_utc": updated,
            }
    def clear_analysis_outputs(self, realization_id: str, analysis_id: str) -> None:
        self.db.execute(
            "DELETE FROM rs_analysis_outputs WHERE realization_id=? AND analysis_id=?",
            (realization_id, analysis_id),
        )
        self.db.commit()

    def put_analysis_output(
        self, realization_id: str, analysis_id: str, output_kind: str,
        scope: str, key: str, payload: Any,
    ) -> None:
        if self.realization(realization_id) is None:
            raise KeyError(f"unknown robustness realization {realization_id}")
        self.db.execute(
            "INSERT OR REPLACE INTO rs_analysis_outputs VALUES(?,?,?,?,?,?,?)",
            (realization_id, analysis_id, output_kind, scope, key, dumps(payload), _utc_now()),
        )
        self.db.commit()

    def put_analysis_outputs(
        self, realization_id: str, analysis_id: str,
        rows: Iterable[tuple[str, str, str, Any]],
    ) -> None:
        """Persist several analysis outputs in one transaction.

        This is used for diagnostic tables such as Analysis 3 BSA-year
        episode summaries, where committing every row separately would add
        substantial SQLite overhead without improving checkpoint semantics.
        """
        if self.realization(realization_id) is None:
            raise KeyError(f"unknown robustness realization {realization_id}")
        now = _utc_now()
        values = [
            (realization_id, analysis_id, output_kind, scope, key, dumps(payload), now)
            for output_kind, scope, key, payload in rows
        ]
        if not values:
            return
        self.db.executemany(
            "INSERT OR REPLACE INTO rs_analysis_outputs VALUES(?,?,?,?,?,?,?)",
            values,
        )
        self.db.commit()

    def analysis_output(
        self, realization_id: str, analysis_id: str, output_kind: str,
        scope: str, key: str,
    ) -> Any | None:
        row = self.db.execute(
            """SELECT payload FROM rs_analysis_outputs
               WHERE realization_id=? AND analysis_id=? AND output_kind=? AND scope=? AND key=?""",
            (realization_id, analysis_id, output_kind, scope, key),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def analysis_outputs(
        self, realization_id: str, analysis_id: str, output_kind: str | None = None,
        scope: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        sql = "SELECT output_kind,scope,key,payload FROM rs_analysis_outputs WHERE realization_id=? AND analysis_id=?"
        params: list[Any] = [realization_id, analysis_id]
        if output_kind is not None:
            sql += " AND output_kind=?"; params.append(output_kind)
        if scope is not None:
            sql += " AND scope=?"; params.append(scope)
        sql += " ORDER BY output_kind,scope,key"
        for kind, row_scope, key, payload in self.db.execute(sql, tuple(params)):
            yield {
                "output_kind": kind, "scope": row_scope, "key": key,
                "payload": json.loads(payload),
            }


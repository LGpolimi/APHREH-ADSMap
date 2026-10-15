"""RS Step 9B: explicit realization-specific search-adjusted null searches.

Each selected sensitivity realization receives an independent synchronized
within-year null search over its complete threshold-lag domain.  The same
production null-search implementation used by the reference optimizer is
called, but results are promoted to first-class robustness checkpoints and
exports.  Reference null cutoffs are never reused.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import FixedParametersEngine, SurfaceRunner, cell_key
from ..numerics import finite_quantile
from ..optimization import REFERENCE_METRIC, admissible, ensure_null_search
from ..storage import Store, atomic_json, clean
from .models import CheckpointState, ComparisonMode, RealizationState
from .planner import materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "REALIZATION_SPECIFIC_NULL_SEARCH"
CHECKPOINT_STAGE = "RS_REALIZATION_NULL_SEARCH"
SCHEMA_VERSION = "APHREH_RS_REALIZATION_NULL_SEARCH_V1"
SENSITIVITY_ANALYSES = {
    "ANALYSIS_2_SUPPORT_SENSITIVITY",
    "ANALYSIS_3_BSA_SPECIFIC_EXPOSURE",
    "ANALYSIS_4_BASELINE_SENSITIVITY",
    "ANALYSIS_5_DEPENDENCE_SENSITIVITY",
}


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(row.get(k)) for k in fields})


class RealizationNullSearchRunner:
    """Run and persist independent maximum-statistic null searches per realization."""

    def __init__(self, data, store: Store, persistence: RobustnessPersistence, root: Path, log=print):
        self.data = data
        self.store = store
        self.persistence = persistence
        self.root = Path(root)
        self.log = log

    @staticmethod
    def eligible_specs(plan):
        return [
            r for r in plan.realizations
            if r.analysis_id in SENSITIVITY_ANALYSES
            and r.comparison_mode in (ComparisonMode.REOPTIMIZED, ComparisonMode.BOTH)
        ]

    def _run_id(self, spec) -> str:
        record = self.persistence.realization(spec.realization_id)
        if not record or record.get("state") != RealizationState.COMPLETED.value:
            raise RuntimeError(f"Sensitivity realization {spec.realization_id} must complete before null search")
        summary = self.persistence.analysis_output(
            spec.realization_id, spec.analysis_id, "SUMMARY", "REALIZATION", "result"
        )
        if not summary or not summary.get("run_id"):
            raise RuntimeError(f"Sensitivity realization {spec.realization_id} has no persisted run_id")
        return str(summary["run_id"])

    def ensure_spec(self, reference_config: dict[str, Any], spec) -> dict[str, Any]:
        run_id = self._run_id(spec)
        realized = materialize_realization(reference_config, spec)
        engine = FixedParametersEngine(self.data, realized, self.store, self.root, self.log)
        runner = SurfaceRunner(engine)
        requested = int(realized["optimization"]["null_realizations_R"])

        self.persistence.put_checkpoint(
            spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING
        )
        try:
            def on_replicate(rep: int, row: Mapping[str, Any]) -> None:
                key = f"{int(rep):05d}"
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "run_id": run_id,
                    **dict(row),
                }
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "NULL_MAXIMUM", "REPLICATE", key, payload
                )
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REPLICATE", key,
                    CheckpointState.COMPLETED,
                    payload={
                        "replicate": int(rep),
                        "maximum_wmarm_cs": row.get("maximum_wmarm_cs"),
                        "valid_cells": row.get("valid_cells"),
                    },
                )

            maxima = ensure_null_search(runner, run_id, replicate_callback=on_replicate)

            # Persist cell-level null surfaces as first-class Step-9 products.
            null_cell_rows: list[dict[str, Any]] = []
            for row in self.store.rows("null_cell", run_id):
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    **row,
                }
                key = f"{int(row['replicate']):05d}/{cell_key(float(row['q']), int(row['lag']))}"
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "NULL_CELL", "REPLICATE_CELL", key, payload
                )
                null_cell_rows.append(payload)

            valid_maxima = [float(v) for v in maxima if v is not None and math.isfinite(float(v))]
            null_complete = len(maxima) == requested and len(valid_maxima) == requested
            cutoff = (
                finite_quantile(np.asarray(valid_maxima, dtype=float), 1 - realized["optimization"]["search_adjusted_alpha"])
                if null_complete else None
            )
            observed_cells = [
                self.store.get("cell", run_id, cell_key(float(q), int(lag)))
                for q in realized["thresholds"] for lag in realized["lags"]
            ]
            observed_eligible = [r for r in observed_cells if r is not None and admissible(r, realized["optimization"])]
            observed_max = max((r[REFERENCE_METRIC] for r in observed_eligible), default=None)
            p_max = (
                (1 + sum(v >= observed_max for v in valid_maxima)) / (len(valid_maxima) + 1)
                if null_complete and observed_max is not None else None
            )
            summary = {
                "schema_version": SCHEMA_VERSION,
                "analysis_id": ANALYSIS_ID,
                "realization_id": spec.realization_id,
                "sensitivity_analysis_id": spec.analysis_id,
                "scenario_id": spec.scenario_id,
                "step_index": spec.step_index,
                "run_id": run_id,
                "status": "COMPLETED" if null_complete else "INCOMPLETE_NULL_SEARCH",
                "reference_metric": "WMARM_CS",
                "null_realizations_requested": requested,
                "null_realizations_valid": len(valid_maxima),
                "null_complete": null_complete,
                "observed_admissible_maximum_wmarm_cs": observed_max,
                "null_cutoff_wmarm_cs": cutoff,
                "p_max": p_max,
                "monte_carlo_minimum_p": 1 / (requested + 1),
                "null_cell_row_count": len(null_cell_rows),
                "randomization": realized["optimization"]["null_randomization_rule"],
                "reference_null_cutoff_reused": False,
            }
            self.persistence.put_analysis_output(
                spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", summary
            )
            self.persistence.put_checkpoint(
                spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                CheckpointState.COMPLETED if null_complete else CheckpointState.FAILED,
                payload=summary,
                error=None if null_complete else "null search did not produce all requested valid maxima",
            )
            if not null_complete:
                raise RuntimeError(
                    f"Realization-specific null search incomplete for {spec.realization_id}: "
                    f"{len(valid_maxima)}/{requested} valid maxima"
                )
            return summary
        except BaseException as exc:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint is None or checkpoint.get("state") != CheckpointState.FAILED.value:
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}",
                )
            raise

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        specs = self.eligible_specs(plan)
        summaries = [self.ensure_spec(reference_config, spec) for spec in specs]
        maxima_rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        for spec in specs:
            maxima_rows.extend(
                r["payload"] for r in self.persistence.analysis_outputs(
                    spec.realization_id, ANALYSIS_ID, "NULL_MAXIMUM", "REPLICATE"
                )
            )
            cell_rows.extend(
                r["payload"] for r in self.persistence.analysis_outputs(
                    spec.realization_id, ANALYSIS_ID, "NULL_CELL", "REPLICATE_CELL"
                )
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if specs else "NOT_REQUESTED",
            "requested_realization_count": len(specs),
            "completed_realization_count": len(summaries),
            "summary_rows": summaries,
            "null_maximum_rows": maxima_rows,
            "null_cell_rows": cell_rows,
            "scope_note": (
                "Each selected sensitivity realization receives an independent synchronized within-year maximum-statistic null search. "
                "Reference null cutoffs are never reused."
            ),
        }
        export_realization_null(result, self.root / "results")
        return result


def export_realization_null(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_realization_null_summary.json", {
        k: v for k, v in result.items() if k not in ("null_maximum_rows", "null_cell_rows")
    })
    _write_csv(output_dir / "robustness_realization_null_maxima.csv", list(result.get("null_maximum_rows") or []))
    _write_csv(output_dir / "robustness_realization_null_cells.csv", list(result.get("null_cell_rows") or []))

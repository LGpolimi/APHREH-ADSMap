"""RS Step 9A: explicit realization-specific leave-one-year-out surfaces.

This component promotes LOYO reconstruction from an internal prerequisite of
complete reoptimization to a first-class, independently checkpointed and
exported robustness product.  It reuses the already-persisted yearly rows from
each sensitivity realization and therefore does not rerun exposure,
baseline, yearly RC estimation, or bootstrap merely to omit one year from
multi-year pooling.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from ..engine import FixedParametersEngine, cell_key
from ..storage import Store, atomic_json, clean
from .models import CheckpointState, ComparisonMode, RealizationState
from .planner import materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "REALIZATION_SPECIFIC_LOYO"
CHECKPOINT_STAGE = "RS_REALIZATION_LOYO"
SCHEMA_VERSION = "APHREH_RS_REALIZATION_LOYO_V1"
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


class RealizationLoyoRunner:
    """Build and persist LOYO surfaces for reoptimized sensitivity realizations."""

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
            raise RuntimeError(f"Sensitivity realization {spec.realization_id} must complete before LOYO")
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
        expected = len(realized["thresholds"]) * len(realized["lags"]) * len(self.data.years)
        completed = 0

        self.persistence.put_checkpoint(
            spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING
        )
        try:
            for q in realized["thresholds"]:
                for lag in realized["lags"]:
                    for year in self.data.years:
                        key = f"{cell_key(float(q), int(lag))}/{int(year)}"
                        checkpoint = self.persistence.checkpoint(
                            spec.realization_id, CHECKPOINT_STAGE, "CELL_YEAR", key
                        )
                        row = self.store.get("loyo", run_id, key)
                        if row is None:
                            row = engine.pool_cell(run_id, float(q), int(lag), excluded_year=int(year), save=False)
                            self.store.put("loyo", run_id, key, row)
                            self.store.commit()
                        payload = {
                            "schema_version": SCHEMA_VERSION,
                            "analysis_id": ANALYSIS_ID,
                            "realization_id": spec.realization_id,
                            "sensitivity_analysis_id": spec.analysis_id,
                            "scenario_id": spec.scenario_id,
                            "step_index": spec.step_index,
                            "run_id": run_id,
                            **row,
                        }
                        if checkpoint is None or checkpoint.get("state") != CheckpointState.COMPLETED.value:
                            self.persistence.put_analysis_output(
                                spec.realization_id, ANALYSIS_ID, "LOYO", "CELL_YEAR", key, payload
                            )
                            self.persistence.put_checkpoint(
                                spec.realization_id, CHECKPOINT_STAGE, "CELL_YEAR", key,
                                CheckpointState.COMPLETED, payload={"q": float(q), "lag": int(lag), "excluded_year": int(year)},
                            )
                        completed += 1

            summary = {
                "schema_version": SCHEMA_VERSION,
                "analysis_id": ANALYSIS_ID,
                "realization_id": spec.realization_id,
                "sensitivity_analysis_id": spec.analysis_id,
                "scenario_id": spec.scenario_id,
                "step_index": spec.step_index,
                "run_id": run_id,
                "status": "COMPLETED",
                "threshold_count": len(realized["thresholds"]),
                "lag_count": len(realized["lags"]),
                "year_count": len(self.data.years),
                "expected_cell_year_rows": expected,
                "completed_cell_year_rows": completed,
                "years": list(self.data.years),
                "method": "REPOOL_EXISTING_YEARLY_ROWS_WITH_ONE_YEAR_EXCLUDED",
                "core_engine_rerun": False,
            }
            self.persistence.put_analysis_output(
                spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", summary
            )
            self.persistence.put_checkpoint(
                spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                CheckpointState.COMPLETED, payload=summary,
            )
            return summary
        except BaseException as exc:
            self.persistence.put_checkpoint(
                spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        specs = self.eligible_specs(plan)
        summaries = [self.ensure_spec(reference_config, spec) for spec in specs]
        rows: list[dict[str, Any]] = []
        for spec in specs:
            rows.extend(
                r["payload"] for r in self.persistence.analysis_outputs(
                    spec.realization_id, ANALYSIS_ID, "LOYO", "CELL_YEAR"
                )
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if specs else "NOT_REQUESTED",
            "requested_realization_count": len(specs),
            "completed_realization_count": len(summaries),
            "summary_rows": summaries,
            "loyo_rows": rows,
            "scope_note": (
                "LOYO surfaces are reconstructed from the sensitivity realization's already-persisted yearly rows; "
                "no exposure, baseline, yearly RC, or bootstrap rerun is performed solely for year deletion."
            ),
        }
        export_realization_loyo(result, self.root / "results")
        return result


def export_realization_loyo(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_realization_loyo_summary.json", {
        k: v for k, v in result.items() if k != "loyo_rows"
    })
    _write_csv(output_dir / "robustness_realization_loyo.csv", list(result.get("loyo_rows") or []))

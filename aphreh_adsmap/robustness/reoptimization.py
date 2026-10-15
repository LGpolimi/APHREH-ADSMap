"""RS Step 8: complete realization-specific threshold-lag reoptimization.

Each sensitivity realization selected for REOPTIMIZED or BOTH comparison uses
its already-computed observed threshold-lag surface, reconstructs LOYO surfaces
from the persisted yearly rows, and then enters the same production Optimizer
used by the reference analysis.  The optimizer therefore performs a fresh,
realization-specific synchronized null search, maximum-statistic calibration,
robust-region construction, and representative selection using WMARMcs.

This module is orchestration/post-processing around the existing estimator and
optimizer.  It does not define an alternate optimization rule.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

from ..engine import FixedParametersEngine, SurfaceRunner, cell_key
from ..optimization import Optimizer, admissible
from ..storage import Store, atomic_json, clean
from .models import CheckpointState, ComparisonMode, RealizationState
from .planner import materialize_realization
from .persistence import RobustnessPersistence
from .realization_loyo import RealizationLoyoRunner
from .realization_null import RealizationNullSearchRunner

ANALYSIS_ID = "COMPLETE_REOPTIMIZATION"
CHECKPOINT_STAGE = "RS_COMPLETE_REOPTIMIZATION"
SCHEMA_VERSION = "APHREH_RS_COMPLETE_REOPTIMIZATION_V1"
SENSITIVITY_ANALYSES = {
    "ANALYSIS_2_SUPPORT_SENSITIVITY",
    "ANALYSIS_3_BSA_SPECIFIC_EXPOSURE",
    "ANALYSIS_4_BASELINE_SENSITIVITY",
    "ANALYSIS_5_DEPENDENCE_SENSITIVITY",
}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(row.get(k)) for k in fields})


def _status_from_optimization(result: Mapping[str, Any]) -> str:
    maximum = _finite(result.get("observed_admissible_maximum_wmarm_cs"))
    if maximum is not None and maximum <= 1e-15:
        return "ZERO_SUPPORTED_SURFACE"
    return str(result.get("status") or "UNKNOWN")


def _summary_row(spec, run_id: str, result: Mapping[str, Any], fingerprint: Mapping[str, Any] | None) -> dict[str, Any]:
    representative = result.get("representative") or {}
    region = result.get("robust_region") or []
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "realization_id": spec.realization_id,
        "reference_realization_id": spec.reference_realization_id,
        "sensitivity_analysis_id": spec.analysis_id,
        "scenario_id": spec.scenario_id,
        "step_index": spec.step_index,
        "trajectory_coordinate": spec.trajectory_coordinate,
        "comparison_mode": spec.comparison_mode.value,
        "run_id": run_id,
        "status": _status_from_optimization(result),
        "optimizer_status": result.get("status"),
        "observed_admissible_maximum_wmarm_cs": result.get("observed_admissible_maximum_wmarm_cs"),
        "observed_peak_q": (result.get("observed_peak") or {}).get("q"),
        "observed_peak_lag": (result.get("observed_peak") or {}).get("lag"),
        "observed_peak_wmarm_cs": (result.get("observed_peak") or {}).get("wmarm_cs"),
        "observed_peak_wmarm": (result.get("observed_peak") or {}).get("wmarm"),
        "p_max": result.get("p_max"),
        "null_cutoff_wmarm_cs": result.get("null_cutoff_wmarm_cs"),
        "null_realizations_requested": result.get("null_realizations_requested"),
        "null_realizations_valid": result.get("null_realizations_valid"),
        "null_complete": result.get("null_complete"),
        "monte_carlo_minimum_p": result.get("monte_carlo_minimum_p"),
        "robust_region_cell_count": len(region),
        "robust_component_count": len(result.get("components") or []),
        "representative_q": representative.get("q"),
        "representative_lag": representative.get("lag"),
        "representative_wmarm_cs": representative.get("wmarm_cs"),
        "representative_wmarm": representative.get("wmarm"),
        "surface_coherence": result.get("surface_coherence"),
        "fingerprint_E_CS_max": ((fingerprint or {}).get("effect") or {}).get("E_CS_max"),
        "fingerprint_E_CS_mean": ((fingerprint or {}).get("effect") or {}).get("E_CS_mean"),
        "fingerprint_R_CI": ((fingerprint or {}).get("statistical") or {}).get("R_CI"),
        "fingerprint_T_surface_CS": ((fingerprint or {}).get("temporal") or {}).get("T_surface_CS"),
    }


class ReoptimizationRunner:
    """Execute complete PO reoptimization for selected sensitivity realizations."""

    def __init__(self, data, store: Store, persistence: RobustnessPersistence, root: Path, log=print):
        self.data = data
        self.store = store
        self.persistence = persistence
        self.root = Path(root)
        self.log = log

    def _eligible_specs(self, plan):
        return [
            r for r in plan.realizations
            if r.analysis_id in SENSITIVITY_ANALYSES
            and r.comparison_mode in (ComparisonMode.REOPTIMIZED, ComparisonMode.BOTH)
        ]

    def _load_cells(self, run_id: str, realized: Mapping[str, Any]) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        for q in realized["thresholds"]:
            for lag in realized["lags"]:
                row = self.store.get("cell", run_id, cell_key(float(q), int(lag)))
                if row is None:
                    raise RuntimeError(
                        f"Sensitivity realization {run_id} is missing observed cell q={q}, lag={lag}; "
                        "complete reoptimization requires the full observed alternative surface"
                    )
                cells.append(row)
        return cells

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        specs = self._eligible_specs(plan)
        # Step 9 promotes realization-specific LOYO and null searches to
        # independently checkpointed/exported products. Reoptimization consumes
        # those exact persisted products rather than implementing them again.
        RealizationLoyoRunner(self.data, self.store, self.persistence, self.root, self.log).run(reference_config, plan)
        RealizationNullSearchRunner(self.data, self.store, self.persistence, self.root, self.log).run(reference_config, plan)
        summaries: list[dict[str, Any]] = []
        optimization_cells: list[dict[str, Any]] = []
        loyo_rows: list[dict[str, Any]] = []
        null_rows: list[dict[str, Any]] = []
        region_rows: list[dict[str, Any]] = []

        for spec in specs:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result")
                if stored is not None:
                    summaries.append(stored)
                    optimization_cells.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "OPTIMIZATION_CELL", "CELL"))
                    loyo_rows.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "LOYO", "CELL_YEAR"))
                    null_rows.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "NULL_MAXIMUM", "REPLICATE"))
                    region_rows.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "ROBUST_REGION", "CELL"))
                    continue

            record = self.persistence.realization(spec.realization_id)
            if not record or record.get("state") != RealizationState.COMPLETED.value:
                raise RuntimeError(f"Sensitivity realization {spec.realization_id} must complete before reoptimization")

            analysis_summary = self.persistence.analysis_output(spec.realization_id, spec.analysis_id, "SUMMARY", "REALIZATION", "result")
            if not analysis_summary or not analysis_summary.get("run_id"):
                raise RuntimeError(f"Sensitivity realization {spec.realization_id} has no persisted run_id")
            run_id = str(analysis_summary["run_id"])
            realized = materialize_realization(reference_config, spec)
            cells = self._load_cells(run_id, realized)
            engine = FixedParametersEngine(self.data, realized, self.store, self.root, self.log)
            self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING)
            try:
                self.log(
                    f"RS Step 8: reoptimizing {spec.analysis_id}/{spec.scenario_id} step={spec.step_index}; "
                    f"null R={realized['optimization']['null_realizations_R']}"
                )
                optimizer = Optimizer(SurfaceRunner(engine))
                result = optimizer.run(run_id, cells)
                fingerprint = self.store.get("fingerprint", run_id, "result")
                row = _summary_row(spec, run_id, result, fingerprint)

                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row)

                for diag in result.get("diagnostics") or []:
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "sensitivity_analysis_id": spec.analysis_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "run_id": run_id,
                        **diag,
                    }
                    key = cell_key(float(diag["q"]), int(diag["lag"]))
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "OPTIMIZATION_CELL", "CELL", key, payload)
                    optimization_cells.append(payload)

                for loo in self.store.rows("loyo", run_id):
                    payload = {"schema_version": SCHEMA_VERSION, "analysis_id": ANALYSIS_ID, "realization_id": spec.realization_id, **loo}
                    key = f"{cell_key(float(loo['q']), int(loo['lag']))}/{int(loo['excluded_year'])}"
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "LOYO", "CELL_YEAR", key, payload)
                    loyo_rows.append(payload)

                for null in self.store.rows("null_maximum", run_id):
                    payload = {"schema_version": SCHEMA_VERSION, "analysis_id": ANALYSIS_ID, "realization_id": spec.realization_id, **null}
                    key = f"{int(null['replicate']):05d}"
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "NULL_MAXIMUM", "REPLICATE", key, payload)
                    null_rows.append(payload)

                for region in result.get("robust_region") or []:
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "sensitivity_analysis_id": spec.analysis_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "run_id": run_id,
                        **region,
                    }
                    key = cell_key(float(region["q"]), int(region["lag"]))
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "ROBUST_REGION", "CELL", key, payload)
                    region_rows.append(payload)

                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.COMPLETED, payload=row,
                )
                summaries.append(row)
                self.log(
                    f"RS Step 8 complete: {spec.analysis_id}/{spec.scenario_id} step={spec.step_index}; "
                    f"status={row['status']}; robust cells={row['robust_region_cell_count']}"
                )
            except BaseException as exc:
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}",
                )
                raise

        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if specs else "NOT_REQUESTED",
            "requested_realization_count": len(specs),
            "completed_realization_count": len(summaries),
            "summary_rows": summaries,
            "optimization_cell_rows": optimization_cells,
            "loyo_rows": loyo_rows,
            "null_maximum_rows": null_rows,
            "robust_region_rows": region_rows,
            "scope_note": (
                "Every selected sensitivity realization uses its own observed WMARMcs surface, LOYO surfaces, "
                "search-adjusted maximum-null distribution, robust region, and representative configuration. "
                "Reference null cutoffs are never reused. Cross-realization surface/fingerprint comparison is a later step."
            ),
        }
        export_reoptimization(result, self.root / "results")
        return result


def export_reoptimization(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_reoptimization_summary.json", {
        k: v for k, v in result.items()
        if k not in ("optimization_cell_rows", "loyo_rows", "null_maximum_rows", "robust_region_rows")
    })
    _write_csv(output_dir / "robustness_reoptimization_realizations.csv", list(result.get("summary_rows") or []))
    _write_csv(output_dir / "robustness_reoptimization_cells.csv", list(result.get("optimization_cell_rows") or []))
    _write_csv(output_dir / "robustness_reoptimization_loyo.csv", list(result.get("loyo_rows") or []))
    _write_csv(output_dir / "robustness_reoptimization_null_maxima.csv", list(result.get("null_maximum_rows") or []))
    _write_csv(output_dir / "robustness_reoptimization_regions.csv", list(result.get("robust_region_rows") or []))

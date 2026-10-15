"""RS Analysis 2: sparse-data and effective-support sensitivity.

This module adds the support-policy layer and orchestrates support-threshold
sensitivity realizations through the existing fixed-parameter APHREH engine.
It does not define an alternate RC estimator. Every sensitivity realization uses the same FixedParametersEngine and pooling/WMARMcs code as the reference analysis; Steps 7 and 8 add comparison and reoptimization around these stored surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import FixedParametersEngine, SurfaceRunner
from ..storage import Store, atomic_json, clean, dumps
from .models import (
    CheckpointState,
    ComparisonMode,
    FactorKind,
    InvalidationStage,
    RealizationSpec,
    RealizationState,
)
from .planner import FactorSpec, RealizationPlanner, ScenarioSpec, materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "ANALYSIS_2_SUPPORT_SENSITIVITY"
CHECKPOINT_STAGE = "RS_ANALYSIS_2"
SCHEMA_VERSION = "APHREH_RS_ANALYSIS_2_V1"

SUPPORT_FACTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "min_exposed_observations": {
        "path": "core.support.min_exposed_observations",
        "integer": True,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 1.0,
        "maximum": None,
    },
    "min_nonexposed_observations": {
        "path": "core.support.min_nonexposed_observations",
        "integer": True,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 1.0,
        "maximum": None,
    },
    "min_event_contribution": {
        "path": "core.support.min_event_contribution",
        "integer": True,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 0.0,
        "maximum": None,
    },
    "min_baseline_support": {
        "path": "core.support.min_baseline_support",
        "integer": True,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 1.0,
        "maximum": None,
    },
    "min_effective_weighted_sample_size": {
        "path": "core.support.min_effective_weighted_sample_size",
        "integer": False,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 0.0,
        "maximum": None,
    },
    "min_bootstrap_validity_rate": {
        "path": "core.support.min_bootstrap_validity_rate",
        "integer": False,
        "invalidates_from": InvalidationStage.POOLING,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "min_contributing_years": {
        "path": "core.support.min_contributing_years",
        "integer": True,
        "invalidates_from": InvalidationStage.POOLING,
        "minimum": 2.0,
        "maximum": None,
    },
    "min_local_concordance": {
        "path": "core.support.min_local_concordance",
        "integer": False,
        "invalidates_from": InvalidationStage.YEARLY_RC,
        "minimum": 0.0,
        "maximum": 1.0,
    },
}

from ..support import (
    active_support_policy, validate_support_policy, apply_baseline_support_filter,
    evaluate_prebootstrap_support, evaluate_postbootstrap_support, enforce_pooled_support,
)

def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

def support_scenarios_from_config(config: Mapping[str, Any]) -> tuple[ScenarioSpec, ...]:
    section = config.get("robustness", {}).get("analysis_2_support_sensitivity", {})
    if not section.get("enabled", False):
        return ()
    scenarios: list[ScenarioSpec] = []
    reference_policy = active_support_policy(config)
    for raw in section.get("scenarios", []):
        factors: list[FactorSpec] = []
        for name, bounds in raw.get("factors", {}).items():
            meta = SUPPORT_FACTOR_REGISTRY[name]
            factors.append(FactorSpec(
                name=name,
                config_path=meta["path"],
                analysis_id=ANALYSIS_ID,
                kind=FactorKind.NUMERICAL,
                invalidates_from=meta["invalidates_from"],
                reference_value=reference_policy[name],
                lower=bounds["lower"],
                upper=bounds["upper"],
                integer=bool(meta["integer"]),
            ))
        scenarios.append(ScenarioSpec(
            scenario_id=raw["scenario_id"],
            analysis_id=ANALYSIS_ID,
            factors=tuple(factors),
            numerical_steps=int(raw["steps"]),
            comparison_mode=ComparisonMode(raw.get("comparison_mode", "FIXED_PARAMETER")),
            label=raw.get("label", raw["scenario_id"]),
        ))
    return tuple(scenarios)


def build_robustness_plan(config: Mapping[str, Any]):
    # Central RS plan: include every alternative analysis currently implemented.
    from .baseline_sensitivity import baseline_scenarios_from_config
    from .dependence_sensitivity import dependence_scenarios_from_config
    from .exposure_sensitivity import exposure_scenarios_from_config
    scenarios = (
        *support_scenarios_from_config(config),
        *exposure_scenarios_from_config(config),
        *baseline_scenarios_from_config(config),
        *dependence_scenarios_from_config(config),
    )
    return RealizationPlanner(config).build(scenarios)


def realization_run_id(spec: RealizationSpec) -> str:
    return f"RS_A2_{spec.realization_id.split('.')[-1]}"


def _surface_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in cells if r.get("valid") and _finite(r.get("wmarm_cs")) is not None]
    cs = np.asarray([float(r["wmarm_cs"]) for r in valid], dtype=float) if valid else np.asarray([], dtype=float)
    point = np.asarray([float(r["wmarm"]) for r in valid if _finite(r.get("wmarm")) is not None], dtype=float)
    kcs = np.asarray([float(r["confidence_support_retention_ratio"]) for r in valid if _finite(r.get("confidence_support_retention_ratio")) is not None], dtype=float)
    bsa = np.asarray([float(r["bsa_coverage"]) for r in valid if _finite(r.get("bsa_coverage")) is not None], dtype=float)
    pop = np.asarray([float(r["population_coverage"]) for r in valid if _finite(r.get("population_coverage")) is not None], dtype=float)
    return {
        "candidate_cell_count": len(cells),
        "valid_cell_count": len(valid),
        "wmarm_cs_max": float(np.max(cs)) if cs.size else None,
        "wmarm_cs_mean": float(np.mean(cs)) if cs.size else None,
        "diagnostic_wmarm_max": float(np.max(point)) if point.size else None,
        "diagnostic_wmarm_mean": float(np.mean(point)) if point.size else None,
        "k_cs_mean": float(np.mean(kcs)) if kcs.size else None,
        "bsa_coverage_mean": float(np.mean(bsa)) if bsa.size else None,
        "population_coverage_mean": float(np.mean(pop)) if pop.size else None,
        "zero_supported_surface": bool(cs.size and np.max(cs) <= 1e-15),
    }


def _exclusion_summary(store: Store, run: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    baseline_excluded_days = 0
    for row in store.rows("yearly"):
        if row.get("run") != run:
            continue
        baseline_excluded_days += int(row.get("baseline_support_excluded_days") or 0)
        reasons = list(row.get("support_failures") or [])
        if not row.get("eligible") and row.get("reason") and row.get("reason") not in reasons:
            reasons.append(str(row.get("reason")))
        if not reasons:
            reasons = ["ELIGIBLE"]
        for reason in reasons:
            counts[("YEARLY", reason)] = counts.get(("YEARLY", reason), 0) + 1
    for row in store.rows("pooled"):
        if row.get("run") != run:
            continue
        reason = "ELIGIBLE" if row.get("eligible") else str(row.get("reason") or "INELIGIBLE")
        counts[("POOLED", reason)] = counts.get(("POOLED", reason), 0) + 1
    if baseline_excluded_days:
        counts[("TARGET_DAY", "RS_MIN_BASELINE_SUPPORT")] = baseline_excluded_days
    return [
        {"level": level, "reason": reason, "count": count}
        for (level, reason), count in sorted(counts.items())
    ]


@dataclass
class SupportSensitivityRunner:
    data: Any
    store: Store
    persistence: RobustnessPersistence
    root: Path
    log: Any = print

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        """RS6.10: execute synchronized support-sensitivity realizations."""
        realizations = [r for r in plan.realizations if r.analysis_id == ANALYSIS_ID]
        summary_rows: list[dict[str, Any]] = []
        surface_rows: list[dict[str, Any]] = []
        exclusion_rows: list[dict[str, Any]] = []
        for spec in realizations:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result")
                if stored is not None:
                    summary_rows.append(stored)
                    surface_rows.extend(
                        r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "SURFACE", "CELL")
                    )
                    exclusion_rows.extend(
                        r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "EXCLUSION", "COUNT")
                    )
                    continue
            self.persistence.set_realization_state(spec.realization_id, RealizationState.RUNNING)
            self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING)
            try:
                realized = materialize_realization(reference_config, spec)
                run_id = realization_run_id(spec)
                self.log(f"RS Analysis 2: {spec.scenario_id} step={spec.step_index} support={active_support_policy(realized)}")
                engine = FixedParametersEngine(self.data, realized, self.store, self.root, self.log)
                cells = SurfaceRunner(engine).run(run_id, loyo=False, retain=False, rng_identity_run="REFERENCE")
                surface_summary = _surface_summary(cells)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "reference_realization_id": spec.reference_realization_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "run_id": run_id,
                    "support_policy": active_support_policy(realized),
                    "invalidates_from": spec.invalidates_from.name,
                    **surface_summary,
                }
                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row)
                for cell in cells:
                    payload = {
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        "run_id": run_id,
                        "support_policy": active_support_policy(realized),
                        **cell,
                    }
                    key = f"q{float(cell['q']):g}_lag{int(cell['lag']):04d}"
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SURFACE", "CELL", key, payload)
                    surface_rows.append(payload)
                for index, exclusion in enumerate(_exclusion_summary(self.store, run_id)):
                    payload = {
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        **exclusion,
                    }
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "EXCLUSION", "COUNT", f"{index:04d}", payload)
                    exclusion_rows.append(payload)
                self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.COMPLETED, payload=row)
                self.persistence.set_realization_state(spec.realization_id, RealizationState.COMPLETED)
                summary_rows.append(row)
            except BaseException as exc:
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}",
                )
                self.persistence.set_realization_state(spec.realization_id, RealizationState.FAILED, error=f"{type(exc).__name__}: {exc}")
                raise
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if realizations else "NOT_REQUESTED",
            "realization_count": len(realizations),
            "reference_support_policy": active_support_policy(reference_config),
            "summary_rows": summary_rows,
            "surface_rows": surface_rows,
            "exclusion_rows": exclusion_rows,
            "scope_note": "Step 3 executes support-sensitivity surfaces; Step 7 provides fixed-parameter comparison and Step 8 optionally performs complete realization-specific reoptimization when comparison_mode is REOPTIMIZED or BOTH.",
        }
        export_support_sensitivity(result, self.root / "results")
        return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: dumps(v) if isinstance(v, (dict, list)) else clean(v) for k, v in row.items()})


def export_support_sensitivity(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_analysis2_summary.json", {
        k: v for k, v in result.items() if k not in ("surface_rows", "exclusion_rows")
    })
    _write_csv(output_dir / "robustness_analysis2_realizations.csv", list(result.get("summary_rows", [])))
    _write_csv(output_dir / "robustness_analysis2_surfaces.csv", list(result.get("surface_rows", [])))
    _write_csv(output_dir / "robustness_analysis2_exclusions.csv", list(result.get("exclusion_rows", [])))

"""RS Analysis 4: baseline-window and baseline-matching sensitivity.

Each baseline realization reuses the existing FixedParametersEngine with a
realization-specific baseline policy.  The reference exposure definition,
incidence definition, RC estimator, bootstrap, pooling, WMARMcs and optimizer
mathematics are not duplicated here.  Step 5 executes alternative baseline
surfaces through the production estimator; Steps 7 and 8 subsequently provide fixed-point comparison and complete reoptimization when requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math, hashlib, shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from .. import numerics as nm

from ..engine import FixedParametersEngine
from ..storage import Store, atomic_json, clean, dumps, mapped
from .models import (
    CheckpointState,
    ComparisonMode,
    FactorKind,
    InvalidationStage,
    RealizationSpec,
    RealizationState,
)
from .planner import FactorSpec, ScenarioSpec, materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "ANALYSIS_4_BASELINE_SENSITIVITY"
CHECKPOINT_STAGE = "RS_ANALYSIS_4"
SCHEMA_VERSION = "APHREH_RS_ANALYSIS_4_V1"

BASELINE_FACTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "initial_half_width": {
        "path": "core.baseline.initial_half_width",
        "kind": FactorKind.NUMERICAL,
        "integer": True,
        "invalidates_from": InvalidationStage.BASELINE,
    },
    "maximum_half_width": {
        "path": "core.baseline.maximum_half_width",
        "kind": FactorKind.NUMERICAL,
        "integer": True,
        "invalidates_from": InvalidationStage.BASELINE,
    },
    "minimum_eligible_dates": {
        "path": "core.baseline.minimum_eligible_dates",
        "kind": FactorKind.NUMERICAL,
        "integer": True,
        "invalidates_from": InvalidationStage.BASELINE,
    },
    "dynamic_enlargement": {
        "path": "core.baseline.dynamic_enlargement",
        "kind": FactorKind.BINARY,
        "integer": False,
        "invalidates_from": InvalidationStage.BASELINE,
    },
    "day_of_week_matching": {
        "path": "core.baseline.day_of_week_matching",
        "kind": FactorKind.BINARY,
        "integer": False,
        "invalidates_from": InvalidationStage.BASELINE,
    },
    "holiday_matching": {
        "path": "core.baseline.holiday_matching",
        "kind": FactorKind.BINARY,
        "integer": False,
        "invalidates_from": InvalidationStage.BASELINE,
    },
}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def baseline_scenarios_from_config(config: Mapping[str, Any]) -> tuple[ScenarioSpec, ...]:
    section = config.get("robustness", {}).get("analysis_4_baseline_sensitivity", {})
    if not section.get("enabled", False):
        return ()
    reference = config["core"]["baseline"]
    scenarios: list[ScenarioSpec] = []
    for raw in section.get("scenarios", []):
        factors: list[FactorSpec] = []
        for name, definition in raw.get("factors", {}).items():
            meta = BASELINE_FACTOR_REGISTRY[name]
            if meta["kind"] is FactorKind.NUMERICAL:
                factors.append(FactorSpec(
                    name=name,
                    config_path=meta["path"],
                    analysis_id=ANALYSIS_ID,
                    kind=FactorKind.NUMERICAL,
                    invalidates_from=meta["invalidates_from"],
                    reference_value=reference.get(name),
                    lower=definition["lower"],
                    upper=definition["upper"],
                    integer=True,
                ))
            else:
                factors.append(FactorSpec(
                    name=name,
                    config_path=meta["path"],
                    analysis_id=ANALYSIS_ID,
                    kind=FactorKind.BINARY,
                    invalidates_from=meta["invalidates_from"],
                    reference_value=bool(reference.get(name, False)),
                    alternative_value=bool(definition["alternative"]),
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


def realization_run_id(spec: RealizationSpec) -> str:
    return f"RS_A4_{spec.realization_id.split('.')[-1]}"


def _surface_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in cells if r.get("valid") and _finite(r.get("wmarm_cs")) is not None]
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in valid if _finite(r.get(key)) is not None], dtype=float)
    cs, wm, kcs, bsa, pop = (arr(k) for k in (
        "wmarm_cs", "wmarm", "confidence_support_retention_ratio", "bsa_coverage", "population_coverage"
    ))
    return {
        "candidate_cell_count": len(cells),
        "valid_cell_count": len(valid),
        "wmarm_cs_max": float(np.max(cs)) if cs.size else None,
        "wmarm_cs_mean": float(np.mean(cs)) if cs.size else None,
        "diagnostic_wmarm_max": float(np.max(wm)) if wm.size else None,
        "diagnostic_wmarm_mean": float(np.mean(wm)) if wm.size else None,
        "k_cs_mean": float(np.mean(kcs)) if kcs.size else None,
        "bsa_coverage_mean": float(np.mean(bsa)) if bsa.size else None,
        "population_coverage_mean": float(np.mean(pop)) if pop.size else None,
        "zero_supported_surface": bool(cs.size and np.max(cs) <= 1e-15),
    }


def _yearly_rows(store: Store, run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in store.rows("yearly"):
        if row.get("run") != run:
            continue
        rows.append({
            "run": run,
            "q": row.get("q"),
            "lag": row.get("lag"),
            "bsa_id": row.get("bsa_id"),
            "year": row.get("year"),
            "eligible": row.get("eligible"),
            "reason": row.get("reason"),
            "rc": row.get("rc"),
            "variance": row.get("variance"),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "baseline_donor_count_mean": row.get("baseline_donor_count_mean"),
            "baseline_half_width_mean": row.get("baseline_half_width_mean"),
            "baseline_half_width_max": row.get("baseline_half_width_max"),
            "unavailable_baseline_days": row.get("unavailable_baseline_days"),
            "baseline_support_excluded_days": row.get("baseline_support_excluded_days"),
            "n_exposed": row.get("n_exposed"),
            "n_nonexposed": row.get("n_nonexposed"),
            "effective_exposed": row.get("effective_exposed"),
            "effective_nonexposed": row.get("effective_nonexposed"),
            "bootstrap_valid": row.get("bootstrap_valid"),
            "bootstrap_requested": row.get("bootstrap_requested"),
            "invalid_proportion": row.get("invalid_proportion"),
            "block_length": row.get("block_length"),
        })
    return rows


def _exclusion_summary(store: Store, run: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    unavailable = 0
    for row in store.rows("yearly"):
        if row.get("run") != run:
            continue
        unavailable += int(row.get("unavailable_baseline_days") or 0)
        reason = "ELIGIBLE" if row.get("eligible") else str(row.get("reason") or "INELIGIBLE")
        counts[("YEARLY", reason)] = counts.get(("YEARLY", reason), 0) + 1
    for row in store.rows("pooled"):
        if row.get("run") != run:
            continue
        reason = "ELIGIBLE" if row.get("eligible") else str(row.get("reason") or "INELIGIBLE")
        counts[("POOLED", reason)] = counts.get(("POOLED", reason), 0) + 1
    if unavailable:
        counts[("TARGET_DAY", "BASELINE_UNAVAILABLE")] = unavailable
    return [
        {"level": level, "reason": reason, "count": count}
        for (level, reason), count in sorted(counts.items())
    ]


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


def export_baseline_sensitivity(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_analysis4_summary.json", {
        k: v for k, v in result.items() if k not in ("surface_rows", "yearly_rows", "exclusion_rows")
    })
    _write_csv(output_dir / "robustness_analysis4_realizations.csv", list(result.get("summary_rows", [])))
    _write_csv(output_dir / "robustness_analysis4_surfaces.csv", list(result.get("surface_rows", [])))
    _write_csv(output_dir / "robustness_analysis4_yearly.csv", list(result.get("yearly_rows", [])))
    _write_csv(output_dir / "robustness_analysis4_exclusions.csv", list(result.get("exclusion_rows", [])))


@dataclass
class BaselineSensitivityRunner:
    data: Any
    store: Store
    persistence: RobustnessPersistence
    root: Path
    log: Any = print

    def _prepare_prebaseline_cache(self, reference_config: dict[str, Any], q: float, lag: int, folder: Path) -> Path:
        """Cache exposure/incidence/relevance quantities invariant to RS Analysis 4."""
        folder.mkdir(parents=True, exist_ok=True)
        required = [folder/"labels.npy", folder/"territory_threshold.npy", folder/"incidence.npy",
                    folder/"event_windows.npy", folder/"raw_weights.npy", folder/"local_thresholds.npy"]
        if all(p.exists() for p in required):
            return folder
        d=self.data; c=reference_config
        territory=d.series("territory")
        labels, territory_threshold = nm.exposure(territory,d.year_index,q)
        np.save(folder/"labels.npy",labels,allow_pickle=False)
        np.save(folder/"territory_threshold.npy",territory_threshold,allow_pickle=False)
        with mapped(folder/"incidence.npy",shape=(d.B,d.n)) as incidence, \
             mapped(folder/"event_windows.npy",shape=(d.B,d.n)) as event_windows, \
             mapped(folder/"raw_weights.npy",shape=(d.B,d.n)) as raw_weights, \
             mapped(folder/"local_thresholds.npy",shape=(d.B,len(d.years))) as local_thresholds:
            for b in range(d.B):
                source=d.series("outcome",b)
                incidence[b]=nm.cumulative_incidence(source,d.daily_population(b),lag,
                    c["core"]["incidence"]["scaling_multiplier"],"EVENT_COUNTS",d.segment_index)
                event_windows[b]=nm.cumulative_incidence(source,np.ones(d.n,dtype=float),lag,1.0,"EVENT_COUNTS",d.segment_index)
                local=d.series("hazard",b)
                w,thr=nm.relevance(local,labels,d.year_index,q)
                raw_weights[b]=w; local_thresholds[b]=thr
        return folder

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        realizations = [r for r in plan.realizations if r.analysis_id == ANALYSIS_ID]
        summaries: list[dict[str, Any]] = []
        surfaces: list[dict[str, Any]] = []
        yearly: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        cache_root=self.root/"scratch"/"rs_analysis4_prebaseline"
        cache_root.mkdir(parents=True,exist_ok=True)
        for spec in realizations:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result")
                if stored is not None:
                    summaries.append(stored)
                    surfaces.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "SURFACE", "CELL"))
                    yearly.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "YEARLY", "BSA_YEAR"))
                    exclusions.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "EXCLUSION", "COUNT"))
                    continue
            self.persistence.set_realization_state(spec.realization_id, RealizationState.RUNNING)
            self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING)
            try:
                realized = materialize_realization(reference_config, spec)
                baseline_policy = dict(realized["core"]["baseline"])
                if int(baseline_policy["initial_half_width"]) < 1:
                    raise ValueError("realized baseline initial_half_width must be >= 1")
                if int(baseline_policy["maximum_half_width"]) < int(baseline_policy["initial_half_width"]):
                    raise ValueError("realized baseline maximum_half_width must be >= initial_half_width")
                minimum = baseline_policy.get("minimum_eligible_dates")
                if minimum is not None and (type(minimum) is not int or minimum < 1):
                    raise ValueError("realized baseline minimum_eligible_dates must be null or >= 1")
                if baseline_policy.get("holiday_matching") and not reference_config.get("sources",{}).get("holiday_calendar_file"):
                    raise ValueError("realized holiday matching requires an explicit holiday calendar")
                run_id = realization_run_id(spec)
                self.log(f"RS Analysis 4: {spec.scenario_id} step={spec.step_index} baseline={baseline_policy}")
                # The same production estimator is used.  Reference RNG identity
                # is retained so bootstrap differences arise from changed baseline
                # panels/dependence requirements rather than arbitrary reseeding.
                engine = FixedParametersEngine(self.data, realized, self.store, self.root, self.log)
                cells=[]
                for q in realized["thresholds"]:
                    for lag in realized["lags"]:
                        key=hashlib.sha256(f"q={float(q):.17g}|lag={int(lag)}".encode()).hexdigest()[:24]
                        cache=self._prepare_prebaseline_cache(reference_config,float(q),int(lag),cache_root/key)
                        cells.append(engine.run(run_id,float(q),int(lag),loyo=False,retain=False,
                                                rng_identity_run="REFERENCE",prebaseline_cache=cache))
                yr = _yearly_rows(self.store, run_id)
                donor = [_finite(r.get("baseline_donor_count_mean")) for r in yr]; donor = [x for x in donor if x is not None]
                width = [_finite(r.get("baseline_half_width_mean")) for r in yr]; width = [x for x in width if x is not None]
                unavailable = sum(int(r.get("unavailable_baseline_days") or 0) for r in yr)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "reference_realization_id": spec.reference_realization_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "run_id": run_id,
                    "baseline_policy": baseline_policy,
                    "invalidates_from": spec.invalidates_from.name,
                    "mean_baseline_donor_count": float(np.mean(donor)) if donor else None,
                    "mean_adopted_half_width": float(np.mean(width)) if width else None,
                    "unavailable_baseline_days_total": unavailable,
                    **_surface_summary(cells),
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
                        "baseline_policy": baseline_policy,
                        **cell,
                    }
                    key = f"q{float(cell['q']):g}_lag{int(cell['lag']):04d}"
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SURFACE", "CELL", key, payload)
                    surfaces.append(payload)
                for r in yr:
                    payload = {
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        "baseline_policy": baseline_policy,
                        **r,
                    }
                    key = f"{float(r['q']):g}/{int(r['lag']):04d}/{r['bsa_id']}/{r['year']}"
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "YEARLY", "BSA_YEAR", key, payload)
                    yearly.append(payload)
                for i, ex in enumerate(_exclusion_summary(self.store, run_id)):
                    payload = {
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        **ex,
                    }
                    self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "EXCLUSION", "COUNT", f"{i:04d}", payload)
                    exclusions.append(payload)
                self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.COMPLETED, payload=row)
                self.persistence.set_realization_state(spec.realization_id, RealizationState.COMPLETED)
                summaries.append(row)
            except BaseException as exc:
                self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}")
                self.persistence.set_realization_state(spec.realization_id, RealizationState.FAILED, error=f"{type(exc).__name__}: {exc}")
                raise
        shutil.rmtree(cache_root,ignore_errors=True)
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if realizations else "NOT_REQUESTED",
            "realization_count": len(realizations),
            "reference_baseline_policy": dict(reference_config["core"]["baseline"]),
            "summary_rows": summaries,
            "surface_rows": surfaces,
            "yearly_rows": yearly,
            "exclusion_rows": exclusions,
            "scope_note": "Step 5 executes baseline-sensitivity surfaces; Step 7 provides fixed-parameter comparison and Step 8 optionally performs complete realization-specific reoptimization.",
        }
        export_baseline_sensitivity(result, self.root / "results")
        return result

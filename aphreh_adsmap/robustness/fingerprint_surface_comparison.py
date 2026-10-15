"""RS Step 10: reoptimized fingerprint and complete-surface comparison.

This module is pure post-processing.  It compares the reference realization
with every completed sensitivity realization that requested REOPTIMIZED or BOTH
comparison.  No estimator, bootstrap, LOYO, null search, pooling, or optimizer
is rerun here.

Primary comparisons use WMARMcs on the common admissible threshold-lag domain.
Standard WMARM remains a diagnostic point-estimate surface.  Robust-region
Jaccard overlap, representative displacement, surface-state transitions, and
scalar output-fingerprint differences are retained separately.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import cell_key, finite_mean
from ..optimization import REFERENCE_METRIC, DIAGNOSTIC_METRIC, admissible
from ..storage import Store, atomic_json, clean
from .models import CheckpointState, ComparisonMode
from .planner import materialize_realization
from .persistence import RobustnessPersistence
from .reoptimization import ANALYSIS_ID as REOPTIMIZATION_ANALYSIS_ID, SENSITIVITY_ANALYSES

ANALYSIS_ID = "FINGERPRINT_SURFACE_COMPARISON"
CHECKPOINT_STAGE = "RS_FINGERPRINT_SURFACE_COMPARISON"
SCHEMA_VERSION = "APHREH_RS_FINGERPRINT_SURFACE_COMPARISON_V1"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _delta(sensitivity: Any, reference: Any) -> float | None:
    sv, rv = _finite(sensitivity), _finite(reference)
    return None if sv is None or rv is None else sv - rv


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


def _average_ranks(values: list[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> tuple[float | None, str | None]:
    if len(x) != len(y) or len(x) < 2:
        return None, "INSUFFICIENT_COMMON_CELLS"
    rx, ry = _average_ranks(x), _average_ranks(y)
    if float(np.std(rx)) <= 0 or float(np.std(ry)) <= 0:
        return None, "CONSTANT_SURFACE"
    value = float(np.corrcoef(rx, ry)[0, 1])
    if not math.isfinite(value):
        return None, "NONFINITE_CORRELATION"
    return value, None


def _rmse(reference: list[float], sensitivity: list[float]) -> float | None:
    if not reference or len(reference) != len(sensitivity):
        return None
    a = np.asarray(reference, dtype=float)
    b = np.asarray(sensitivity, dtype=float)
    return float(np.sqrt(np.mean(np.square(b - a))))


def _surface_state(optimization: Mapping[str, Any] | None) -> str:
    if not optimization:
        return "UNAVAILABLE"
    maximum = _finite(optimization.get("observed_admissible_maximum_wmarm_cs"))
    if maximum is None:
        return "NO_ADMISSIBLE_SURFACE"
    if maximum <= 1e-15:
        return "ZERO_SUPPORTED_SURFACE"
    if optimization.get("robust_region"):
        return "ROBUST_REGION_IDENTIFIED"
    return "POSITIVE_NO_ROBUST_REGION"


def _region_keys(optimization: Mapping[str, Any] | None) -> set[str]:
    return {
        cell_key(float(row["q"]), int(row["lag"]))
        for row in ((optimization or {}).get("robust_region") or [])
    }


def _representative(optimization: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    rep = (optimization or {}).get("representative")
    return rep if isinstance(rep, Mapping) and rep.get("q") is not None and rep.get("lag") is not None else None


def _fingerprint_on_common_domain(
    cells: Mapping[str, Mapping[str, Any]],
    optimization: Mapping[str, Any],
    common_keys: set[str],
) -> dict[str, Any]:
    """Recompute fingerprint surface summaries on the common admissible domain.

    Search-adjusted p_max and overall LOYO surface coherence remain the values
    from the complete realization-specific optimization because they are scalar
    inferential outputs of that realization rather than simple cell averages.
    Robust-region summaries are restricted to robust cells that also belong to
    the common admissible comparison domain.
    """
    domain = [cells[k] for k in sorted(common_keys) if k in cells]
    region_keys = _region_keys(optimization) & common_keys
    region_cells = [cells[k] for k in sorted(region_keys) if k in cells]
    diagnostics = {
        cell_key(float(row["q"]), int(row["lag"])): row
        for row in (optimization.get("diagnostics") or [])
    }

    def mean(key: str) -> float | None:
        return finite_mean(row.get(key) for row in domain)

    diag_region = [diagnostics[k] for k in sorted(region_keys) if k in diagnostics]
    diagnostic_wmarm_values = [
        float(row[DIAGNOSTIC_METRIC]) for row in domain if _finite(row.get(DIAGNOSTIC_METRIC)) is not None
    ]
    return {
        "domain": "common admissible threshold-lag cells; equal cell weights",
        "reference_metric": "WMARM_CS",
        "domain_cell_count": len(domain),
        "effect": {
            "E_CS_max": max((float(row[REFERENCE_METRIC]) for row in domain), default=None),
            "E_CS_mean": mean(REFERENCE_METRIC),
            "E_CS_region": finite_mean(row.get(REFERENCE_METRIC) for row in region_cells),
            "A_region": (len(region_cells) / len(domain)) if domain and region_cells else None,
            "D_CS_signed_balance": mean("directional_balance_cs"),
            "K_CS_mean": mean("confidence_support_retention_ratio"),
            "diagnostic_WMARM_max": max(diagnostic_wmarm_values, default=None),
            "diagnostic_WMARM_mean": mean(DIAGNOSTIC_METRIC),
            "diagnostic_D_signed_balance": mean("directional_balance"),
        },
        "statistical": {
            "R_CI": mean("ci_excluding_zero_population_share"),
            "R_DS": mean("directional_stability"),
            "R_precision": mean("relative_precision"),
            "C_BSA": mean("bsa_coverage"),
            "C_POP": mean("population_coverage"),
            "R_validity": mean("bootstrap_validity"),
            "p_max": _finite(optimization.get("p_max")),
        },
        "temporal": {
            "T_sign": mean("year_pooled_sign_concordance"),
            "I_squared_percent": mean("mean_i_squared_percent"),
            "T_surface_CS": _finite(optimization.get("surface_coherence")),
            "T_optimal": finite_mean(row.get("persistence") for row in diag_region),
        },
        "spatial": {
            "S_direction_CS": mean("spatial_directional_coherence_cs"),
            "S_dispersion_CS": mean("spatial_magnitude_dispersion_cs"),
            "diagnostic_S_direction": mean("spatial_directional_coherence"),
            "diagnostic_S_dispersion": mean("spatial_magnitude_dispersion"),
            "S_concordance_E": mean("local_concordance_exposed"),
            "S_concordance_N": mean("local_concordance_nonexposed"),
            "C_BSA": mean("bsa_coverage"),
            "C_POP": mean("population_coverage"),
        },
        "unavailable_policy": "null with explicit comparison status; no combined robustness score",
    }


def _component_rows(
    reference_fp: Mapping[str, Any],
    sensitivity_fp: Mapping[str, Any],
    *,
    spec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in ("effect", "statistical", "temporal", "spatial"):
        rdom = reference_fp.get(domain) or {}
        sdom = sensitivity_fp.get(domain) or {}
        for component in sorted(set(rdom) | set(sdom)):
            rv = _finite(rdom.get(component))
            sv = _finite(sdom.get(component))
            special_local = (
                spec.analysis_id == "ANALYSIS_3_BSA_SPECIFIC_EXPOSURE"
                and domain == "spatial"
                and component in {"S_concordance_E", "S_concordance_N"}
            )
            if special_local:
                status = "NOT_COMPARABLE_METHOD_SPECIFIC"
                reason = "LOCAL_CONCORDANCE_NOT_APPLICABLE_UNDER_BSA_SPECIFIC_EXPOSURE"
                delta = None
            elif rv is None and sv is None:
                status = "UNAVAILABLE_BOTH"
                reason = "COMPONENT_UNAVAILABLE_IN_BOTH_REALIZATIONS"
                delta = None
            elif rv is None or sv is None:
                status = "NOT_COMPARABLE_MISSING_VALUE"
                reason = "COMPONENT_UNAVAILABLE_IN_ONE_REALIZATION"
                delta = None
            else:
                status = "COMPARABLE"
                reason = None
                delta = sv - rv
            role = "DIAGNOSTIC_POINT_ESTIMATE" if component.startswith("diagnostic_") else "REFERENCE_OR_SUPPORT"
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "analysis_id": ANALYSIS_ID,
                "reference_realization_id": spec.reference_realization_id,
                "sensitivity_realization_id": spec.realization_id,
                "sensitivity_analysis_id": spec.analysis_id,
                "scenario_id": spec.scenario_id,
                "step_index": spec.step_index,
                "trajectory_coordinate": spec.trajectory_coordinate,
                "fingerprint_domain": domain,
                "component": component,
                "metric_role": role,
                "status": status,
                "reason": reason,
                "reference_value": rv,
                "sensitivity_value": sv,
                "delta": delta,
            })
    return rows


class FingerprintSurfaceComparator:
    """Compare reference and reoptimized sensitivity realizations."""

    def __init__(self, store: Store, persistence: RobustnessPersistence, root: Path, log=print):
        self.store = store
        self.persistence = persistence
        self.root = Path(root)
        self.log = log

    @staticmethod
    def _eligible_specs(plan):
        return [
            r for r in plan.realizations
            if r.analysis_id in SENSITIVITY_ANALYSES
            and r.comparison_mode in (ComparisonMode.REOPTIMIZED, ComparisonMode.BOTH)
        ]

    def _reference_cells(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for q in config["thresholds"]:
            for lag in config["lags"]:
                key = cell_key(float(q), int(lag))
                row = self.store.get("cell", "REFERENCE", key)
                if row is not None:
                    out[key] = row
        return out

    def _sensitivity_cells(self, run_id: str, realized: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for q in realized["thresholds"]:
            for lag in realized["lags"]:
                key = cell_key(float(q), int(lag))
                row = self.store.get("cell", run_id, key)
                if row is not None:
                    out[key] = row
        return out

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        specs = self._eligible_specs(plan)
        reference_optimization = self.store.get("optimization", "REFERENCE", "result")
        if reference_optimization is None:
            raise RuntimeError("Reference optimization is required for Step 10 comparison")
        reference_cells = self._reference_cells(reference_config)
        reference_admissible = {
            key for key, row in reference_cells.items()
            if admissible(row, reference_config["optimization"])
        }

        summary_rows: list[dict[str, Any]] = []
        surface_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []

        for spec in specs:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.analysis_output(
                    spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result"
                )
                if stored is not None:
                    summary_rows.append(stored)
                    surface_rows.extend(
                        r["payload"] for r in self.persistence.analysis_outputs(
                            spec.realization_id, ANALYSIS_ID, "SURFACE_CELL", "CELL"
                        )
                    )
                    component_rows.extend(
                        r["payload"] for r in self.persistence.analysis_outputs(
                            spec.realization_id, ANALYSIS_ID, "FINGERPRINT_COMPONENT", "COMPONENT"
                        )
                    )
                    continue

            reopt_summary = self.persistence.analysis_output(
                spec.realization_id, REOPTIMIZATION_ANALYSIS_ID, "SUMMARY", "REALIZATION", "result"
            )
            if not reopt_summary or not reopt_summary.get("run_id"):
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "reference_realization_id": plan.reference.realization_id,
                    "sensitivity_realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "status": "UNAVAILABLE_REOPTIMIZATION_NOT_COMPLETED",
                    "reason": "COMPLETE_REOPTIMIZATION_OUTPUT_NOT_AVAILABLE",
                }
                summary_rows.append(row)
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row
                )
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.SKIPPED, payload=row,
                )
                continue

            run_id = str(reopt_summary["run_id"])
            sensitivity_optimization = self.store.get("optimization", run_id, "result")
            if sensitivity_optimization is None:
                raise RuntimeError(f"Sensitivity optimization result missing for {spec.realization_id}")
            realized = materialize_realization(reference_config, spec)
            sensitivity_cells = self._sensitivity_cells(run_id, realized)
            sensitivity_admissible = {
                key for key, row in sensitivity_cells.items()
                if admissible(row, realized["optimization"])
            }
            common = reference_admissible & sensitivity_admissible
            all_keys = sorted(set(reference_cells) | set(sensitivity_cells))

            self.persistence.put_checkpoint(
                spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING
            )
            try:
                ref_cs: list[float] = []
                sen_cs: list[float] = []
                ref_w: list[float] = []
                sen_w: list[float] = []
                local_surface_rows: list[dict[str, Any]] = []
                for key in all_keys:
                    rr = reference_cells.get(key)
                    sr = sensitivity_cells.get(key)
                    q = (rr or sr or {}).get("q")
                    lag = (rr or sr or {}).get("lag")
                    in_common = key in common
                    rcs = _finite(rr.get("wmarm_cs") if rr else None)
                    scs = _finite(sr.get("wmarm_cs") if sr else None)
                    rw = _finite(rr.get("wmarm") if rr else None)
                    sw = _finite(sr.get("wmarm") if sr else None)
                    if in_common:
                        if rcs is None or scs is None:
                            raise RuntimeError(f"Common admissible cell {key} lacks WMARMcs")
                        ref_cs.append(rcs); sen_cs.append(scs)
                        if rw is not None and sw is not None:
                            ref_w.append(rw); sen_w.append(sw)
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "analysis_id": ANALYSIS_ID,
                        "reference_realization_id": plan.reference.realization_id,
                        "sensitivity_realization_id": spec.realization_id,
                        "sensitivity_analysis_id": spec.analysis_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        "q": q,
                        "lag": lag,
                        "reference_admissible": key in reference_admissible,
                        "sensitivity_admissible": key in sensitivity_admissible,
                        "common_admissible_domain": in_common,
                        "reference_wmarm_cs": rcs,
                        "sensitivity_wmarm_cs": scs,
                        "delta_wmarm_cs": _delta(scs, rcs) if in_common else None,
                        "reference_wmarm": rw,
                        "sensitivity_wmarm": sw,
                        "delta_wmarm": _delta(sw, rw) if in_common else None,
                        "reference_bsa_coverage": _finite(rr.get("bsa_coverage") if rr else None),
                        "sensitivity_bsa_coverage": _finite(sr.get("bsa_coverage") if sr else None),
                        "reference_population_coverage": _finite(rr.get("population_coverage") if rr else None),
                        "sensitivity_population_coverage": _finite(sr.get("population_coverage") if sr else None),
                    }
                    local_surface_rows.append(payload)

                rmse_cs = _rmse(ref_cs, sen_cs)
                rho_cs, rho_cs_reason = _spearman(ref_cs, sen_cs)
                rmse_w = _rmse(ref_w, sen_w) if len(ref_w) == len(common) else None
                if len(ref_w) == len(common):
                    rho_w, rho_w_reason = _spearman(ref_w, sen_w)
                else:
                    rho_w, rho_w_reason = None, "DIAGNOSTIC_WMARM_MISSING_ON_COMMON_DOMAIN"

                ref_state = _surface_state(reference_optimization)
                sen_state = _surface_state(sensitivity_optimization)
                state_transition = f"{ref_state}_TO_{sen_state}"

                ref_region = _region_keys(reference_optimization)
                sen_region = _region_keys(sensitivity_optimization)
                if ref_region and sen_region:
                    intersection = ref_region & sen_region
                    union = ref_region | sen_region
                    jaccard = len(intersection) / len(union) if union else None
                    jaccard_reason = None
                else:
                    intersection = ref_region & sen_region
                    union = ref_region | sen_region
                    jaccard = None
                    jaccard_reason = "ROBUST_REGION_EMPTY_IN_REFERENCE_OR_SENSITIVITY"

                ref_rep = _representative(reference_optimization)
                sen_rep = _representative(sensitivity_optimization)
                if ref_rep is not None and sen_rep is not None:
                    delta_q = float(sen_rep["q"]) - float(ref_rep["q"])
                    delta_lag = int(sen_rep["lag"]) - int(ref_rep["lag"])
                    displacement_reason = None
                else:
                    delta_q = None
                    delta_lag = None
                    displacement_reason = "REPRESENTATIVE_UNAVAILABLE_IN_REFERENCE_OR_SENSITIVITY"

                ref_fp = _fingerprint_on_common_domain(reference_cells, reference_optimization, common)
                sen_fp = _fingerprint_on_common_domain(sensitivity_cells, sensitivity_optimization, common)
                local_components = _component_rows(ref_fp, sen_fp, spec=spec)

                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "reference_realization_id": plan.reference.realization_id,
                    "sensitivity_realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "comparison_mode": ComparisonMode.REOPTIMIZED.value,
                    "sensitivity_run_id": run_id,
                    "status": "COMPLETED" if common else "UNAVAILABLE_NO_COMMON_ADMISSIBLE_DOMAIN",
                    "reason": None if common else "NO_COMMON_ADMISSIBLE_THRESHOLD_LAG_CELLS",
                    "reference_surface_state": ref_state,
                    "sensitivity_surface_state": sen_state,
                    "surface_state_transition": state_transition,
                    "reference_admissible_cell_count": len(reference_admissible),
                    "sensitivity_admissible_cell_count": len(sensitivity_admissible),
                    "common_admissible_cell_count": len(common),
                    "common_fraction_of_reference_admissible": len(common) / len(reference_admissible) if reference_admissible else None,
                    "common_fraction_of_sensitivity_admissible": len(common) / len(sensitivity_admissible) if sensitivity_admissible else None,
                    "rmse_cs": rmse_cs,
                    "rho_cs": rho_cs,
                    "rho_cs_reason": rho_cs_reason,
                    "diagnostic_rmse_wmarm": rmse_w,
                    "diagnostic_rho_wmarm": rho_w,
                    "diagnostic_rho_wmarm_reason": rho_w_reason,
                    "reference_robust_region_cell_count": len(ref_region),
                    "sensitivity_robust_region_cell_count": len(sen_region),
                    "robust_region_intersection_cell_count": len(intersection),
                    "robust_region_union_cell_count": len(union),
                    "robust_region_jaccard": jaccard,
                    "robust_region_jaccard_reason": jaccard_reason,
                    "reference_representative_q": None if ref_rep is None else ref_rep.get("q"),
                    "reference_representative_lag": None if ref_rep is None else ref_rep.get("lag"),
                    "sensitivity_representative_q": None if sen_rep is None else sen_rep.get("q"),
                    "sensitivity_representative_lag": None if sen_rep is None else sen_rep.get("lag"),
                    "delta_representative_q": delta_q,
                    "delta_representative_lag": delta_lag,
                    "representative_displacement_reason": displacement_reason,
                    "fingerprint_comparable_component_count": sum(r["status"] == "COMPARABLE" for r in local_components),
                    "fingerprint_noncomparable_component_count": sum(r["status"] != "COMPARABLE" for r in local_components),
                }

                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_comparison(
                    plan.reference.realization_id, spec.realization_id, ComparisonMode.REOPTIMIZED.value, row
                )
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row
                )
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "FINGERPRINT", "REALIZATION_SIDE", "REFERENCE", ref_fp
                )
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "FINGERPRINT", "REALIZATION_SIDE", "SENSITIVITY", sen_fp
                )
                for payload in local_surface_rows:
                    key = cell_key(float(payload["q"]), int(payload["lag"]))
                    self.persistence.put_analysis_output(
                        spec.realization_id, ANALYSIS_ID, "SURFACE_CELL", "CELL", key, payload
                    )
                for payload in local_components:
                    key = f"{payload['fingerprint_domain']}/{payload['component']}"
                    self.persistence.put_analysis_output(
                        spec.realization_id, ANALYSIS_ID, "FINGERPRINT_COMPONENT", "COMPONENT", key, payload
                    )

                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.COMPLETED, payload=row,
                )
                summary_rows.append(row)
                surface_rows.extend(local_surface_rows)
                component_rows.extend(local_components)
                self.log(
                    f"RS Step 10 comparison: {spec.analysis_id}/{spec.scenario_id} step={spec.step_index}; "
                    f"common cells={len(common)}; RMSEcs={rmse_cs}; J={jaccard}"
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
            "reference_realization_id": plan.reference.realization_id,
            "requested_comparison_count": len(specs),
            "completed_comparison_count": sum(row.get("status") == "COMPLETED" for row in summary_rows),
            "summary_rows": summary_rows,
            "surface_rows": surface_rows,
            "fingerprint_component_rows": component_rows,
            "scope_note": (
                "Primary surface comparison uses WMARMcs on the intersection of the reference and sensitivity admissible domains. "
                "Standard WMARM is diagnostic. Robust-region overlap and representative displacement are only reported when defined. "
                "Fingerprint components are compared without aggregation into a single robustness score."
            ),
        }
        export_fingerprint_surface_comparison(result, self.root / "results")
        return result


def export_fingerprint_surface_comparison(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_fingerprint_surface_summary.json", {
        k: v for k, v in result.items() if k not in ("surface_rows", "fingerprint_component_rows")
    })
    _write_csv(output_dir / "robustness_surface_comparisons.csv", list(result.get("summary_rows") or []))
    _write_csv(output_dir / "robustness_surface_cells.csv", list(result.get("surface_rows") or []))
    _write_csv(output_dir / "robustness_fingerprint_components.csv", list(result.get("fingerprint_component_rows") or []))

"""RS Step 7: fixed-parameter reference-versus-sensitivity comparison.

The comparison is anchored to the representative parameter pair selected by
reference optimization.  A descriptive surface peak is never substituted.
Every completed Analysis 2/3/4/5 sensitivity realization is compared at the
same (q*, lag*) when that cell remains analytically admissible.

This module is post-processing only.  It does not rerun the estimator,
bootstrap, pooling, LOYO, null search, or optimizer.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..engine import cell_key, cell_scope, confidence_scaled_magnitude, pooled_ci_excludes_zero
from ..optimization import admissible
from ..storage import Store, atomic_json, clean, dumps
from .models import CheckpointState, ComparisonMode, RealizationSpec, RealizationState
from .planner import materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "FIXED_PARAMETER_COMPARISON"
CHECKPOINT_STAGE = "RS_FIXED_PARAMETER_COMPARISON"
SCHEMA_VERSION = "APHREH_RS_FIXED_PARAMETER_COMPARISON_V1"
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
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _delta(sensitivity: Any, reference: Any) -> float | None:
    a, b = _finite(sensitivity), _finite(reference)
    return None if a is None or b is None else a - b


def _sign(value: Any) -> int | None:
    x = _finite(value)
    if x is None:
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _describe(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_q25": None,
            f"{prefix}_q75": None,
            f"{prefix}_iqr": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    a = np.asarray(values, dtype=float)
    q25, med, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(med),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
    }


def _average_ranks(values: list[float]) -> np.ndarray:
    """Average ranks for ties, ascending, 1-based."""
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


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    if float(np.std(rx)) <= 0 or float(np.std(ry)) <= 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def _pooled_state(row: Mapping[str, Any] | None) -> tuple[str, float | None, float | None]:
    """Return CI-support state, confidence-scaled magnitude, and pooled RC.

    WMARMcs magnitude is continuously attenuated and is not thresholded by CI
    exclusion. SUPPORTED/UNSUPPORTED remains a separate robustness diagnostic
    describing whether the pooled CI excludes zero.
    """
    if not row or not row.get("eligible"):
        return "INELIGIBLE", None, None
    rc = _finite(row.get("pooled_rc"))
    pop = _finite(row.get("population"))
    s = confidence_scaled_magnitude(rc, row.get("ci_low"), row.get("ci_high"))
    ci_supported = pooled_ci_excludes_zero(row.get("ci_low"), row.get("ci_high"))
    if rc is None or pop is None or pop <= 0 or s is None or ci_supported is None:
        return "INELIGIBLE", None, None
    return ("SUPPORTED" if ci_supported else "UNSUPPORTED"), s, rc


def _metric_payload(reference: Mapping[str, Any], sensitivity: Mapping[str, Any]) -> dict[str, Any]:
    mapping = {
        "wmarm_cs": "wmarm_cs",
        "diagnostic_wmarm": "wmarm",
        "k_cs": "confidence_support_retention_ratio",
        "r_ci": "ci_excluding_zero_population_share",
        "bsa_coverage": "bsa_coverage",
        "population_coverage": "population_coverage",
        "signed_coefficient": "signed_coefficient",
        "signed_coefficient_cs": "signed_coefficient_cs",
        "directional_balance": "directional_balance",
        "directional_balance_cs": "directional_balance_cs",
    }
    out: dict[str, Any] = {}
    for label, key in mapping.items():
        rv, sv = _finite(reference.get(key)), _finite(sensitivity.get(key))
        out[f"reference_{label}"] = rv
        out[f"sensitivity_{label}"] = sv
        out[f"delta_{label}"] = _delta(sv, rv)
    return out


def _bsa_rows(
    store: Store,
    reference_run: str,
    sensitivity_run: str,
    q: float,
    lag: int,
    *,
    spec: RealizationSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref = {str(r.get("bsa_id")): r for r in store.rows("pooled", cell_scope(reference_run, q, lag))}
    sen = {str(r.get("bsa_id")): r for r in store.rows("pooled", cell_scope(sensitivity_run, q, lag))}
    ids = sorted(set(ref) | set(sen), key=lambda v: (len(v), v))
    rows: list[dict[str, Any]] = []
    common_x: list[float] = []
    common_y: list[float] = []
    delta_rc_values: list[float] = []
    delta_s_values: list[float] = []
    sign_agree = 0
    support_agree = 0
    common = 0
    reference_eligible = 0
    sensitivity_eligible = 0
    reference_supported = 0
    sensitivity_supported = 0
    common_reference_population = 0.0
    sign_agree_reference_population = 0.0

    transition_counts: dict[str, int] = {}
    for bid in ids:
        rr, sr = ref.get(bid), sen.get(bid)
        ref_state, ref_s, ref_rc = _pooled_state(rr)
        sen_state, sen_s, sen_rc = _pooled_state(sr)
        ref_ok = ref_state != "INELIGIBLE"
        sen_ok = sen_state != "INELIGIBLE"
        reference_eligible += int(ref_ok)
        sensitivity_eligible += int(sen_ok)
        reference_supported += int(ref_state == "SUPPORTED")
        sensitivity_supported += int(sen_state == "SUPPORTED")
        transition = f"{ref_state}_TO_{sen_state}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        comparable = ref_ok and sen_ok
        ref_pop = _finite(rr.get("population") if rr else None)
        sen_pop = _finite(sr.get("population") if sr else None)
        sign_same: bool | None = None
        support_same: bool | None = None
        if comparable:
            common += 1
            common_x.append(float(ref_rc))
            common_y.append(float(sen_rc))
            delta_rc_values.append(float(sen_rc - ref_rc))
            delta_s_values.append(float(sen_s - ref_s))
            sign_same = _sign(ref_rc) == _sign(sen_rc)
            support_same = (ref_state == "SUPPORTED") == (sen_state == "SUPPORTED")
            sign_agree += int(bool(sign_same))
            support_agree += int(bool(support_same))
            if ref_pop is not None and ref_pop > 0:
                common_reference_population += ref_pop
                if sign_same:
                    sign_agree_reference_population += ref_pop
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "reference_realization_id": spec.reference_realization_id,
            "sensitivity_realization_id": spec.realization_id,
            "sensitivity_analysis_id": spec.analysis_id,
            "scenario_id": spec.scenario_id,
            "step_index": spec.step_index,
            "trajectory_coordinate": spec.trajectory_coordinate,
            "q": q,
            "lag": lag,
            "bsa_id": bid,
            "reference_state": ref_state,
            "sensitivity_state": sen_state,
            "confidence_support_transition": transition,
            "reference_eligible": ref_ok,
            "sensitivity_eligible": sen_ok,
            "comparable": comparable,
            "reference_pooled_rc": ref_rc,
            "sensitivity_pooled_rc": sen_rc,
            "delta_pooled_rc": _delta(sen_rc, ref_rc) if comparable else None,
            "reference_supported_magnitude": ref_s,
            "sensitivity_supported_magnitude": sen_s,
            "delta_supported_magnitude": _delta(sen_s, ref_s) if comparable else None,
            "reference_sign": _sign(ref_rc),
            "sensitivity_sign": _sign(sen_rc),
            "sign_agreement": sign_same,
            "support_status_agreement": support_same,
            "reference_ci_low": _finite(rr.get("ci_low") if rr else None),
            "reference_ci_high": _finite(rr.get("ci_high") if rr else None),
            "sensitivity_ci_low": _finite(sr.get("ci_low") if sr else None),
            "sensitivity_ci_high": _finite(sr.get("ci_high") if sr else None),
            "reference_population": ref_pop,
            "sensitivity_population": sen_pop,
            "reference_contributing_years": rr.get("contributing_years") if rr else None,
            "sensitivity_contributing_years": sr.get("contributing_years") if sr else None,
            "reference_i_squared_percent": _finite(rr.get("i_squared_percent") if rr else None),
            "sensitivity_i_squared_percent": _finite(sr.get("i_squared_percent") if sr else None),
            "reference_reason": rr.get("reason") if rr else "MISSING_POOLED_ROW",
            "sensitivity_reason": sr.get("reason") if sr else "MISSING_POOLED_ROW",
        })

    summary = {
        "reference_eligible_bsas": reference_eligible,
        "sensitivity_eligible_bsas": sensitivity_eligible,
        "reference_supported_bsas": reference_supported,
        "sensitivity_supported_bsas": sensitivity_supported,
        "common_comparable_bsas": common,
        "bsa_sign_agreement": sign_agree / common if common else None,
        "population_weighted_sign_agreement_reference_weights": (
            sign_agree_reference_population / common_reference_population
            if common_reference_population > 0 else None
        ),
        "confidence_support_status_agreement": support_agree / common if common else None,
        "pooled_rc_spearman": _spearman(common_x, common_y),
        "common_comparable_reference_population": common_reference_population if common_reference_population > 0 else None,
        "confidence_support_transition_counts": transition_counts,
        **_describe(delta_rc_values, "delta_pooled_rc"),
        **_describe(delta_s_values, "delta_supported_magnitude"),
    }
    return rows, summary


def _reference_unavailable_status(optimization: Mapping[str, Any] | None) -> tuple[str, str]:
    if optimization is None:
        return "UNAVAILABLE_REFERENCE_OPTIMIZATION", "REFERENCE_OPTIMIZATION_NOT_AVAILABLE"
    observed = _finite(optimization.get("observed_admissible_maximum_wmarm_cs"))
    if observed is not None and observed <= 1e-15:
        return "ZERO_SUPPORTED_SURFACE", "REFERENCE_WMARM_CS_ZERO_ON_ADMISSIBLE_DOMAIN"
    return "UNAVAILABLE_NO_REFERENCE_REPRESENTATIVE", "REFERENCE_ROBUST_REGION_HAS_NO_REPRESENTATIVE"


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({k: dumps(v) if isinstance(v, (dict, list)) else clean(v) for k, v in row.items()})


@dataclass
class FixedPointComparator:
    store: Store
    persistence: RobustnessPersistence
    root: Path
    log: Any = print

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        optimization = self.store.get("optimization", "REFERENCE", "result")
        representative = optimization.get("representative") if optimization else None
        specs = [
            r for r in plan.realizations
            if r.analysis_id in SENSITIVITY_ANALYSES
            and r.comparison_mode in (ComparisonMode.FIXED_PARAMETER, ComparisonMode.BOTH)
        ]
        summary_rows: list[dict[str, Any]] = []
        bsa_rows: list[dict[str, Any]] = []

        if not representative:
            status, reason = _reference_unavailable_status(optimization)
            for spec in specs:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "reference_realization_id": plan.reference.realization_id,
                    "sensitivity_realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "q": None,
                    "lag": None,
                    "status": status,
                    "reason": reason,
                }
                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_comparison(
                    plan.reference.realization_id, spec.realization_id,
                    ComparisonMode.FIXED_PARAMETER.value, row,
                )
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row
                )
                self.persistence.put_checkpoint(
                    spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                    CheckpointState.SKIPPED, payload=row,
                )
                summary_rows.append(row)
            result = {
                "schema_version": SCHEMA_VERSION,
                "analysis_id": ANALYSIS_ID,
                "status": status,
                "reason": reason,
                "reference_realization_id": plan.reference.realization_id,
                "reference_representative": None,
                "requested_comparison_count": len(specs),
                "completed_comparison_count": 0,
                "summary_rows": summary_rows,
                "bsa_rows": [],
                "scope_note": "Fixed-parameter comparison requires a representative cell from the reference robust region; no descriptive peak fallback is used.",
            }
            export_fixed_point_comparison(result, self.root / "results")
            return result

        q = float(representative["q"])
        lag = int(representative["lag"])
        reference_cell = self.store.get("cell", "REFERENCE", cell_key(q, lag))
        if reference_cell is None:
            raise RuntimeError("Reference representative cell is missing from persisted cell outputs")

        for spec in specs:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.comparison(
                    plan.reference.realization_id, spec.realization_id, ComparisonMode.FIXED_PARAMETER.value
                )
                if stored is not None:
                    summary_rows.append(stored)
                    bsa_rows.extend(
                        r["payload"] for r in self.persistence.analysis_outputs(
                            spec.realization_id, ANALYSIS_ID, "BSA", "BSA"
                        )
                    )
                    continue

            realization_record = self.persistence.realization(spec.realization_id)
            if not realization_record or realization_record.get("state") != RealizationState.COMPLETED.value:
                row = {
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
                    "status": "UNAVAILABLE_SENSITIVITY_REALIZATION_NOT_COMPLETED",
                    "reason": "SENSITIVITY_REALIZATION_NOT_COMPLETED",
                }
                self.persistence.put_comparison(plan.reference.realization_id, spec.realization_id, ComparisonMode.FIXED_PARAMETER.value, row)
                summary_rows.append(row)
                continue

            analysis_summary = self.persistence.analysis_output(
                spec.realization_id, spec.analysis_id, "SUMMARY", "REALIZATION", "result"
            )
            if not analysis_summary or not analysis_summary.get("run_id"):
                row = {
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
                    "status": "UNAVAILABLE_SENSITIVITY_OUTPUTS",
                    "reason": "SENSITIVITY_RUN_ID_NOT_AVAILABLE",
                }
                self.persistence.put_comparison(plan.reference.realization_id, spec.realization_id, ComparisonMode.FIXED_PARAMETER.value, row)
                summary_rows.append(row)
                continue

            run_id = str(analysis_summary["run_id"])
            sensitivity_cell = self.store.get("cell", run_id, cell_key(q, lag))
            realized_config = materialize_realization(reference_config, spec)
            if sensitivity_cell is None or not admissible(sensitivity_cell, realized_config["optimization"]):
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "reference_realization_id": plan.reference.realization_id,
                    "sensitivity_realization_id": spec.realization_id,
                    "sensitivity_analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "sensitivity_run_id": run_id,
                    "q": q,
                    "lag": lag,
                    "status": "UNAVAILABLE_SENSITIVITY_CELL_INADMISSIBLE",
                    "reason": "REFERENCE_REPRESENTATIVE_NOT_ADMISSIBLE_UNDER_SENSITIVITY_REALIZATION",
                    "sensitivity_cell_valid": None if sensitivity_cell is None else bool(sensitivity_cell.get("valid")),
                    "sensitivity_bsa_coverage": None if sensitivity_cell is None else _finite(sensitivity_cell.get("bsa_coverage")),
                    "sensitivity_population_coverage": None if sensitivity_cell is None else _finite(sensitivity_cell.get("population_coverage")),
                }
                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_comparison(plan.reference.realization_id, spec.realization_id, ComparisonMode.FIXED_PARAMETER.value, row)
                self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row)
                self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.COMPLETED, payload=row)
                summary_rows.append(row)
                continue

            detail_rows, bsa_summary = _bsa_rows(
                self.store, "REFERENCE", run_id, q, lag, spec=spec
            )
            row = {
                "schema_version": SCHEMA_VERSION,
                "analysis_id": ANALYSIS_ID,
                "reference_realization_id": plan.reference.realization_id,
                "sensitivity_realization_id": spec.realization_id,
                "sensitivity_analysis_id": spec.analysis_id,
                "scenario_id": spec.scenario_id,
                "step_index": spec.step_index,
                "trajectory_coordinate": spec.trajectory_coordinate,
                "sensitivity_run_id": run_id,
                "comparison_mode": ComparisonMode.FIXED_PARAMETER.value,
                "q": q,
                "lag": lag,
                "status": "COMPLETED",
                "reason": None,
                **_metric_payload(reference_cell, sensitivity_cell),
                **bsa_summary,
            }
            self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
            self.persistence.put_comparison(plan.reference.realization_id, spec.realization_id, ComparisonMode.FIXED_PARAMETER.value, row)
            self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row)
            for detail in detail_rows:
                self.persistence.put_analysis_output(
                    spec.realization_id, ANALYSIS_ID, "BSA", "BSA", str(detail["bsa_id"]), detail
                )
                bsa_rows.append(detail)
            self.persistence.put_checkpoint(
                spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                CheckpointState.COMPLETED, payload=row,
            )
            summary_rows.append(row)
            self.log(
                f"RS fixed-point comparison: {spec.analysis_id}/{spec.scenario_id} step={spec.step_index}; "
                f"q={q:g}, lag={lag}; common BSAs={row['common_comparable_bsas']}"
            )

        completed = sum(r.get("status") == "COMPLETED" for r in summary_rows)
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if specs else "NOT_REQUESTED",
            "reference_realization_id": plan.reference.realization_id,
            "reference_representative": {"q": q, "lag": lag},
            "requested_comparison_count": len(specs),
            "completed_comparison_count": completed,
            "summary_rows": summary_rows,
            "bsa_rows": bsa_rows,
            "scope_note": "All comparisons are evaluated at the reference representative robust cell. No alternative cell or descriptive peak is substituted when the sensitivity realization is inadmissible.",
        }
        export_fixed_point_comparison(result, self.root / "results")
        return result


def export_fixed_point_comparison(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_fixed_point_summary.json", {
        k: v for k, v in result.items() if k != "bsa_rows"
    })
    _write_csv(output_dir / "robustness_fixed_point_comparisons.csv", result.get("summary_rows", []))
    _write_csv(output_dir / "robustness_fixed_point_bsa.csv", result.get("bsa_rows", []))

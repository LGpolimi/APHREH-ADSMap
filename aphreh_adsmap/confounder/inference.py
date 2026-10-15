"""CA6 fixed-parameter adjusted bootstrap inference and comparison.

The adjusted analysis remains inside the authoritative FixedParametersEngine.
The same target-hazard differential series, dependence estimates, synchronized
moving-block counts, pooling model, and WMARMcs implementation are reused.
Only the pairwise RC kernel receives a frozen held-out-year confounder context.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import (
    FixedParametersEngine,
    cell_key,
    cell_scope,
    confidence_scaled_magnitude,
    pooled_ci_excludes_zero,
)
from ..storage import Store
from .contracts import (
    ConfounderCalibrationResult,
    ConfounderCompatibilityContext,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
)
from .persistence import ConfounderCheckpointState, ConfounderPersistence


INFERENCE_STAGE = "CA6_FIXED_PARAMETER_ADJUSTED_INFERENCE"
INFERENCE_SCHEMA = "APHREH_CONFOUNDER_FIXED_INFERENCE_CA6_V1"
_NUMERICAL_TOLERANCE = 1e-12


class ConfounderInferenceError(ValueError):
    pass


def adjusted_fixed_run_id(analysis_run_id: str) -> str:
    return f"{analysis_run_id}__ADJUSTED_FIXED"


def _project_array(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfounderInferenceError("candidate percentile path escapes the project root") from exc
    if not path.is_file():
        raise ConfounderInferenceError(f"candidate percentile panel is unavailable: {path}")
    return path


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sign(value: Any) -> int | None:
    x = _finite(value)
    if x is None:
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


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


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def _describe(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_q25": None,
            f"{prefix}_q75": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    a = np.asarray(values, dtype=float)
    q25, median, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(median),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
    }


def _pooled_state(row: Mapping[str, Any] | None) -> tuple[str, float | None, float | None]:
    if not row or not row.get("eligible"):
        return "INELIGIBLE", None, None
    rc = _finite(row.get("pooled_rc"))
    pop = _finite(row.get("population"))
    magnitude = confidence_scaled_magnitude(rc, row.get("ci_low"), row.get("ci_high"))
    ci_supported = pooled_ci_excludes_zero(row.get("ci_low"), row.get("ci_high"))
    if rc is None or pop is None or pop <= 0.0 or magnitude is None or ci_supported is None:
        return "INELIGIBLE", None, None
    return ("SUPPORTED" if ci_supported else "UNSUPPORTED"), magnitude, rc


def _year_support_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    retained = [_finite(r.get("retained_support")) for r in rows]
    retained = [x for x in retained if x is not None]
    reasons = sorted({str(r.get("reason")) for r in rows if r.get("reason")})
    invalid = [_finite(r.get("invalid_proportion")) for r in rows]
    invalid = [x for x in invalid if x is not None]
    empty = [int(r.get("adjusted_bootstrap_empty_pair_realizations") or 0) for r in rows]
    return {
        "adjusted_year_rows": len(rows),
        "adjusted_eligible_year_rows": sum(bool(r.get("eligible")) for r in rows),
        "adjusted_retained_support_mean": float(np.mean(retained)) if retained else None,
        "adjusted_retained_support_min": float(np.min(retained)) if retained else None,
        "adjusted_bootstrap_invalid_proportion_mean": float(np.mean(invalid)) if invalid else None,
        "adjusted_bootstrap_empty_pair_realizations": int(sum(empty)),
        "adjusted_year_exclusion_reasons": reasons,
    }


def fixed_parameter_comparison(
    store: Store,
    *,
    analysis_run_id: str,
    adjusted_run_id: str,
    candidate_id: str,
    q: float,
    lag: int,
    bsa_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_scope = cell_scope("REFERENCE", q, lag)
    adjusted_scope = cell_scope(adjusted_run_id, q, lag)
    reference_cell = store.get("cell", "REFERENCE", cell_key(q, lag))
    adjusted_cell = store.get("cell", adjusted_run_id, cell_key(q, lag))
    if reference_cell is None:
        raise ConfounderInferenceError("reference representative cell is unavailable")
    if adjusted_cell is None:
        raise ConfounderInferenceError("adjusted representative cell is unavailable")

    ref_pooled = {r["bsa_id"]: r for r in store.rows("pooled", reference_scope)}
    adj_pooled = {r["bsa_id"]: r for r in store.rows("pooled", adjusted_scope)}
    adj_yearly: dict[str, list[Mapping[str, Any]]] = {bid: [] for bid in bsa_ids}
    for row in store.rows("yearly", adjusted_scope):
        adj_yearly.setdefault(str(row["bsa_id"]), []).append(row)

    common_ref_rc: list[float] = []
    common_adj_rc: list[float] = []
    pooled_shifts: list[float] = []
    magnitude_shifts: list[float] = []
    sign_agree = 0
    support_agree = 0
    common = 0
    common_ref_population = 0.0
    population_sign_agree = 0.0
    transitions: dict[str, int] = {}
    bsa_rows: list[dict[str, Any]] = []

    for bid in bsa_ids:
        rr, ar = ref_pooled.get(bid), adj_pooled.get(bid)
        ref_state, ref_magnitude, ref_rc = _pooled_state(rr)
        adj_state, adj_magnitude, adj_rc = _pooled_state(ar)
        transition = f"{ref_state}_TO_{adj_state}"
        transitions[transition] = transitions.get(transition, 0) + 1
        comparable = ref_state != "INELIGIBLE" and adj_state != "INELIGIBLE"
        sign_same = support_same = None
        delta_rc = delta_magnitude = None
        ref_pop = _finite(rr.get("population") if rr else None)
        adj_pop = _finite(ar.get("population") if ar else None)
        if comparable:
            common += 1
            common_ref_rc.append(float(ref_rc))
            common_adj_rc.append(float(adj_rc))
            delta_rc = float(adj_rc - ref_rc)
            delta_magnitude = float(adj_magnitude - ref_magnitude)
            pooled_shifts.append(delta_rc)
            magnitude_shifts.append(delta_magnitude)
            sign_same = _sign(ref_rc) == _sign(adj_rc)
            support_same = (ref_state == "SUPPORTED") == (adj_state == "SUPPORTED")
            sign_agree += int(bool(sign_same))
            support_agree += int(bool(support_same))
            if ref_pop is not None and ref_pop > 0.0:
                common_ref_population += ref_pop
                if sign_same:
                    population_sign_agree += ref_pop

        year_summary = _year_support_summary(adj_yearly.get(bid, []))
        row = {
            "schema_version": INFERENCE_SCHEMA,
            "analysis_run_id": analysis_run_id,
            "candidate_id": candidate_id,
            "adjusted_run_id": adjusted_run_id,
            "q": q,
            "lag": lag,
            "bsa_id": bid,
            "reference_state": ref_state,
            "adjusted_state": adj_state,
            "confidence_support_transition": transition,
            "comparable": comparable,
            "reference_pooled_rc": ref_rc,
            "adjusted_pooled_rc": adj_rc,
            "delta_pooled_rc": delta_rc,
            "reference_confidence_scaled_magnitude": ref_magnitude,
            "adjusted_confidence_scaled_magnitude": adj_magnitude,
            "delta_confidence_scaled_magnitude": delta_magnitude,
            "reference_ci_low": _finite(rr.get("ci_low") if rr else None),
            "reference_ci_high": _finite(rr.get("ci_high") if rr else None),
            "adjusted_ci_low": _finite(ar.get("ci_low") if ar else None),
            "adjusted_ci_high": _finite(ar.get("ci_high") if ar else None),
            "reference_ci_excludes_zero": None if rr is None else rr.get("ci_excludes_zero"),
            "adjusted_ci_excludes_zero": None if ar is None else ar.get("ci_excludes_zero"),
            "reference_sign": _sign(ref_rc),
            "adjusted_sign": _sign(adj_rc),
            "sign_agreement": sign_same,
            "confidence_support_status_agreement": support_same,
            "reference_population": ref_pop,
            "adjusted_population": adj_pop,
            "reference_contributing_years": None if rr is None else rr.get("contributing_years"),
            "adjusted_contributing_years": None if ar is None else ar.get("contributing_years"),
            "reference_i_squared_percent": _finite(rr.get("i_squared_percent") if rr else None),
            "adjusted_i_squared_percent": _finite(ar.get("i_squared_percent") if ar else None),
            "reference_reason": "MISSING_POOLED_ROW" if rr is None else rr.get("reason"),
            "adjusted_reason": "MISSING_POOLED_ROW" if ar is None else ar.get("reason"),
            "adjusted_confounder_exclusion_reason": None if ar is None else ar.get("confounder_pooled_exclusion_reason"),
            **year_summary,
        }
        bsa_rows.append(row)

    summary = {
        "schema_version": INFERENCE_SCHEMA,
        "status": "COMPLETED" if adjusted_cell.get("valid") else "COMPLETED_WITH_EXCLUSIONS",
        "stage": INFERENCE_STAGE,
        "analysis_run_id": analysis_run_id,
        "candidate_id": candidate_id,
        "adjusted_run_id": adjusted_run_id,
        "reference_representative": {"q": q, "lag": lag},
        "reference_metric": "WMARM_CS",
        "reference_wmarm_cs": _finite(reference_cell.get("wmarm_cs")),
        "adjusted_wmarm_cs": _finite(adjusted_cell.get("wmarm_cs")),
        "delta_wmarm_cs": None if _finite(reference_cell.get("wmarm_cs")) is None or _finite(adjusted_cell.get("wmarm_cs")) is None else float(adjusted_cell["wmarm_cs"] - reference_cell["wmarm_cs"]),
        "reference_wmarm": _finite(reference_cell.get("wmarm")),
        "adjusted_wmarm": _finite(adjusted_cell.get("wmarm")),
        "delta_wmarm": None if _finite(reference_cell.get("wmarm")) is None or _finite(adjusted_cell.get("wmarm")) is None else float(adjusted_cell["wmarm"] - reference_cell["wmarm"]),
        "reference_marm": _finite(reference_cell.get("marm")),
        "adjusted_marm": _finite(adjusted_cell.get("marm")),
        "delta_marm": None if _finite(reference_cell.get("marm")) is None or _finite(adjusted_cell.get("marm")) is None else float(adjusted_cell["marm"] - reference_cell["marm"]),
        "reference_k_cs": _finite(reference_cell.get("confidence_support_retention_ratio")),
        "adjusted_k_cs": _finite(adjusted_cell.get("confidence_support_retention_ratio")),
        "delta_k_cs": None if _finite(reference_cell.get("confidence_support_retention_ratio")) is None or _finite(adjusted_cell.get("confidence_support_retention_ratio")) is None else float(adjusted_cell["confidence_support_retention_ratio"] - reference_cell["confidence_support_retention_ratio"]),
        "reference_bsa_coverage": _finite(reference_cell.get("bsa_coverage")),
        "adjusted_bsa_coverage": _finite(adjusted_cell.get("bsa_coverage")),
        "reference_population_coverage": _finite(reference_cell.get("population_coverage")),
        "adjusted_population_coverage": _finite(adjusted_cell.get("population_coverage")),
        "reference_ci_excluding_zero_population_share": _finite(reference_cell.get("ci_excluding_zero_population_share")),
        "adjusted_ci_excluding_zero_population_share": _finite(adjusted_cell.get("ci_excluding_zero_population_share")),
        "common_comparable_bsas": common,
        "bsa_sign_agreement": sign_agree / common if common else None,
        "population_weighted_sign_agreement_reference_weights": population_sign_agree / common_ref_population if common_ref_population > 0.0 else None,
        "confidence_support_status_agreement": support_agree / common if common else None,
        "pooled_rc_spearman": _spearman(common_ref_rc, common_adj_rc),
        "confidence_support_transition_counts": transitions,
        "adjusted_cell_reason": adjusted_cell.get("reason"),
        **_describe(pooled_shifts, "delta_pooled_rc"),
        **_describe(magnitude_shifts, "delta_confidence_scaled_magnitude"),
    }
    return bsa_rows, summary


class ConfounderAdjustedInferenceRunner:
    """Run CA6 bootstrap, pooling, WMARMcs and the fixed-point comparison."""

    def __init__(
        self,
        data,
        config: dict,
        store: Store,
        project_root: str | Path,
        persistence: ConfounderPersistence,
        analysis_run_id: str,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.store = store
        self.root = Path(project_root).expanduser().resolve()
        self.persistence = persistence
        self.analysis_run_id = analysis_run_id
        self.preprocessing = preprocessing
        self.log = log
        self.engine = FixedParametersEngine(data, config, store, self.root, log)

    def _contexts(self) -> dict[int, ConfounderCompatibilityContext]:
        return {
            int(raw["held_out_year"]): ConfounderCompatibilityContext.from_dict(raw)
            for raw in self.store.rows("confounder_compatibility_context", self.analysis_run_id)
        }

    def _calibration_exclusions(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for raw in self.store.rows("confounder_calibration", self.analysis_run_id):
            calibration = ConfounderCalibrationResult.from_dict(raw)
            if calibration.exclusion_reason is not None:
                out[int(calibration.held_out_year)] = calibration.exclusion_reason.value
        return out

    def _copy_adjusted_outputs(self, adjusted_run: str, q: float, lag: int) -> tuple[int, int]:
        scope = cell_scope(adjusted_run, q, lag)
        yearly_count = pooled_count = 0
        ca5 = {
            (str(r["bsa_id"]), int(r["year"])): r
            for r in self.store.rows("confounder_adjusted_point", self.analysis_run_id)
        }
        for row in self.store.rows("yearly", scope):
            copied = {"analysis_run_id": self.analysis_run_id, **row}
            key = f"{row['bsa_id']}/{int(row['year'])}"
            point = ca5.get((str(row["bsa_id"]), int(row["year"])))
            mismatch = False
            if point is not None and point.get("adjusted_rc") is not None and row.get("rc") is not None:
                mismatch = abs(float(point["adjusted_rc"]) - float(row["rc"])) > _NUMERICAL_TOLERANCE
            copied["ca5_point_rc"] = None if point is None else point.get("adjusted_rc")
            copied["ca5_point_mismatch"] = mismatch
            if mismatch:
                raise ConfounderInferenceError(
                    f"CA6 adjusted original-sample RC diverged from CA5 for {key}"
                )
            self.store.put("confounder_adjusted_yearly", self.analysis_run_id, key, copied)
            yearly_count += 1
        for row in self.store.rows("pooled", scope):
            copied = {"analysis_run_id": self.analysis_run_id, **row}
            self.store.put("confounder_adjusted_pooled", self.analysis_run_id, str(row["bsa_id"]), copied)
            pooled_count += 1
        self.store.commit()
        return yearly_count, pooled_count

    def run(self, representative: Mapping[str, Any] | None) -> dict[str, Any]:
        if representative is None:
            return {
                "schema_version": INFERENCE_SCHEMA,
                "status": "UNAVAILABLE",
                "stage": INFERENCE_STAGE,
                "reason": ConfounderExclusionReason.REFERENCE_CONFIGURATION_UNAVAILABLE.value,
                "adjusted_estimator_enabled": True,
                "adjusted_bootstrap_enabled": True,
                "adjusted_pooling_enabled": True,
                "fixed_parameter_comparison_enabled": True,
                "adjusted_yearly_rows": 0,
                "adjusted_pooled_rows": 0,
            }

        q, lag = float(representative["q"]), int(representative["lag"])
        contexts = self._contexts()
        exclusions = self._calibration_exclusions()
        adjusted_run = adjusted_fixed_run_id(self.analysis_run_id)
        percentile_path = _project_array(self.root, self.preprocessing.percentile_panel.array_path)

        self.log(
            f"Confounder CA6: adjusted bootstrap q={q:g}; lag={lag}; "
            f"contexts={len(contexts)}/{len(self.data.years)}"
        )
        cell = self.engine.run(
            adjusted_run,
            q,
            lag,
            loyo=False,
            retain=False,
            rng_identity_run="REFERENCE",
            candidate_percentile_path=percentile_path,
            compatibility_contexts=contexts,
            compatibility_exclusion_reasons=exclusions,
        )
        yearly_count, pooled_count = self._copy_adjusted_outputs(adjusted_run, q, lag)
        bsa_rows, comparison = fixed_parameter_comparison(
            self.store,
            analysis_run_id=self.analysis_run_id,
            adjusted_run_id=adjusted_run,
            candidate_id=self.preprocessing.candidate_id,
            q=q,
            lag=lag,
            bsa_ids=list(self.data.ids),
        )
        for row in bsa_rows:
            self.store.put(
                "confounder_fixed_comparison_bsa", self.analysis_run_id, str(row["bsa_id"]), row
            )
        self.store.put("confounder_fixed_comparison", self.analysis_run_id, "result", comparison)
        summary = {
            **comparison,
            "status": comparison["status"],
            "stage": INFERENCE_STAGE,
            "adjusted_estimator_enabled": True,
            "adjusted_bootstrap_enabled": True,
            "adjusted_pooling_enabled": True,
            "fixed_parameter_comparison_enabled": True,
            "adjusted_yearly_rows": yearly_count,
            "adjusted_pooled_rows": pooled_count,
            "adjusted_cell_valid": bool(cell.get("valid")),
            "available_compatibility_contexts": len(contexts),
            "unavailable_compatibility_years": len(self.data.years) - len(contexts),
        }
        self.store.put("workflow", self.analysis_run_id, "fixed_inference_result", summary)
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            INFERENCE_STAGE,
            "fixed_parameter",
            cell_key(q, lag),
            ConfounderCheckpointState.COMPLETED,
            payload=summary,
        )
        self.log(
            f"Confounder CA6 complete: adjusted yearly={yearly_count}; pooled={pooled_count}; "
            f"WMARMcs={summary.get('adjusted_wmarm_cs')}"
        )
        return summary

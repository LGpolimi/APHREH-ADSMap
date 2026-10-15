"""CA8 adjusted optimization inference.

This module adds the realization-specific inferential layer required after the
CA7 complete adjusted surface.  It deliberately reuses the authoritative core
estimator, SurfaceRunner, WMARMcs decision rules, and fingerprint implementation.

CA8 performs three operations that the ordinary Optimizer cannot perform for a
confounder-adjusted realization:

* nested leave-one-year-out (LOYO) calibration. For outer deletion h, the
  compatibility rule applied to retained target year y is trained on Y\\{h,y};
* conditional synchronized target-hazard null search with the observed
  cross-fitted confounder calibration frozen;
* robust-region resolution, representative selection, and fingerprinting through
  the existing pure optimization functions.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import FixedParametersEngine, SurfaceRunner, cell_key, cell_scope, finite_mean
from ..optimization import (
    DIAGNOSTIC_METRIC,
    REFERENCE_METRIC,
    admissible,
    fingerprint,
    resolve_region,
)
from ..randomness import rng_for_identity
from ..storage import Store
from .calibration import (
    build_relevance_profile_from_cells,
    evaluate_calibration_curves,
    select_caliper,
)
from .configuration import ConfounderAdjustmentSettings
from .contracts import (
    ConfounderCalibrationResult,
    ConfounderCalibrationStatus,
    ConfounderCandidateDefinition,
    ConfounderCompatibilityContext,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
    NestedConfounderCalibrationKey,
)
from .execution import CrossfitOutcomeData, ConfounderProfilingData
from .inference import adjusted_fixed_run_id
from .pairwise import PairwiseSampleExtractor
from .persistence import ConfounderCheckpointState, ConfounderPersistence


OPTIMIZATION_STAGE = "CA8_ADJUSTED_OPTIMIZATION_INFERENCE"
NESTED_CALIBRATION_STAGE = "CA8_NESTED_CALIBRATION"
NESTED_LOYO_STAGE = "CA8_NESTED_LOYO"
ADJUSTED_NULL_STAGE = "CA8_ADJUSTED_NULL"
OPTIMIZATION_SCHEMA = "APHREH_CONFOUNDER_ADJUSTED_OPTIMIZATION_CA8_V1"


class ConfounderOptimizationError(ValueError):
    pass


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def _rmse(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or not x:
        return None
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(np.square(b - a))))


def _region_keys(optimization: Mapping[str, Any] | None) -> set[str]:
    return {
        cell_key(float(row["q"]), int(row["lag"]))
        for row in ((optimization or {}).get("robust_region") or [])
    }


def _surface_state(optimization: Mapping[str, Any] | None) -> str:
    if not optimization:
        return "UNAVAILABLE"
    maximum = _finite(optimization.get("observed_admissible_maximum_wmarm_cs"))
    if maximum is None:
        return "NO_ADMISSIBLE_SURFACE"
    if maximum <= 1e-15:
        return "ZERO_SUPPORTED_SURFACE"
    return "ROBUST_REGION_IDENTIFIED" if optimization.get("robust_region") else "POSITIVE_NO_ROBUST_REGION"


def _representative(optimization: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    value = (optimization or {}).get("representative")
    return value if isinstance(value, Mapping) and value.get("q") is not None and value.get("lag") is not None else None


def adjusted_optimization_run_id(analysis_run_id: str) -> str:
    """The CA8 decision layer extends the same adjusted realization as CA6/CA7."""
    return adjusted_fixed_run_id(analysis_run_id)


def _cleanup_engine_run(store: Store, run_id: str, cells: list[Mapping[str, Any]]) -> None:
    """Remove temporary generic-engine rows after compact CA8 outputs are persisted."""
    for row in cells:
        store.delete_scope(cell_scope(run_id, float(row["q"]), int(row["lag"])))
    store.delete_scope(run_id)
    store.commit()


def _nested_key(candidate_id: str, outer_year: int, target_year: int) -> str:
    key = NestedConfounderCalibrationKey(candidate_id, outer_year, target_year)
    return f"{key.outer_deleted_year}/{key.target_year}"


def _calibration_from_profile(
    *,
    data,
    config: dict,
    root: Path,
    settings: ConfounderAdjustmentSettings,
    preprocessing: ConfounderPreprocessingResult,
    candidate_id: str,
    outer_year: int,
    target_year: int,
    training_years: tuple[int, ...],
    profile,
    representative: Mapping[str, Any],
    analysis_run_id: str,
    log,
) -> ConfounderCalibrationResult:
    if profile.status is ConfounderCalibrationStatus.UNAVAILABLE:
        return ConfounderCalibrationResult(
            candidate_id=candidate_id,
            held_out_year=target_year,
            training_years=training_years,
            relevance_profile=profile,
            selected_caliper=None,
            residual_relevance_by_caliper={},
            retained_support_by_caliper={},
            status=ConfounderCalibrationStatus.UNAVAILABLE,
            exclusion_reason=ConfounderExclusionReason.RELEVANCE_PROFILE_UNAVAILABLE,
        )
    if profile.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE:
        return ConfounderCalibrationResult(
            candidate_id=candidate_id,
            held_out_year=target_year,
            training_years=training_years,
            relevance_profile=profile,
            selected_caliper=None,
            residual_relevance_by_caliper={},
            retained_support_by_caliper={},
            status=ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE,
        )

    nested_pairwise_id = f"{analysis_run_id}__CA8_NESTED_O{outer_year}"
    extractor = PairwiseSampleExtractor(data, config, root, preprocessing, log=log)
    bundle = extractor.extract(
        analysis_run_id=nested_pairwise_id,
        held_out_year=target_year,
        training_years=training_years,
        target_threshold=float(representative["q"]),
        target_lag=int(representative["lag"]),
    )
    try:
        residual_curve, support_curve, _annual_rows = evaluate_calibration_curves(
            data, root, bundle, profile, settings
        )
    finally:
        shutil.rmtree(root / "confounder_runs" / nested_pairwise_id, ignore_errors=True)

    selected_by_tolerance = {
        float(tolerance): select_caliper(
            residual_curve,
            support_curve,
            residual_tolerance=float(tolerance),
            minimum_support=settings.minimum_retained_support,
        )
        for tolerance in settings.residual_relevance_sensitivity_values
    }
    selected = select_caliper(
        residual_curve,
        support_curve,
        residual_tolerance=settings.residual_relevance_tolerance,
        minimum_support=settings.minimum_retained_support,
    )
    if selected is None:
        return ConfounderCalibrationResult(
            candidate_id=candidate_id,
            held_out_year=target_year,
            training_years=training_years,
            relevance_profile=profile,
            selected_caliper=None,
            residual_relevance_by_caliper=residual_curve,
            retained_support_by_caliper=support_curve,
            status=ConfounderCalibrationStatus.UNAVAILABLE,
            selected_caliper_by_tolerance=selected_by_tolerance,
            exclusion_reason=ConfounderExclusionReason.NO_ADMISSIBLE_CALIPER,
        )
    return ConfounderCalibrationResult(
        candidate_id=candidate_id,
        held_out_year=target_year,
        training_years=training_years,
        relevance_profile=profile,
        selected_caliper=selected,
        residual_relevance_by_caliper=residual_curve,
        retained_support_by_caliper=support_curve,
        status=ConfounderCalibrationStatus.AVAILABLE,
        selected_caliper_by_tolerance=selected_by_tolerance,
    )


class ConfounderAdjustedOptimizationRunner:
    """CA8 nested LOYO, conditional null search, and adjusted optimization."""

    def __init__(
        self,
        data,
        config: dict,
        store: Store,
        project_root: str | Path,
        persistence: ConfounderPersistence,
        analysis_run_id: str,
        candidate: ConfounderCandidateDefinition,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.store = store
        self.root = Path(project_root).expanduser().resolve()
        self.persistence = persistence
        self.analysis_run_id = analysis_run_id
        self.candidate = candidate
        self.preprocessing = preprocessing
        self.log = log
        self.settings = ConfounderAdjustmentSettings.from_dict(config["confounder"])
        self.adjusted_run_id = adjusted_optimization_run_id(analysis_run_id)
        self.percentile_path = (self.root / preprocessing.percentile_panel.array_path).resolve()
        try:
            self.percentile_path.relative_to(self.root)
        except ValueError as exc:
            raise ConfounderOptimizationError("candidate percentile panel escapes project root") from exc
        if not self.percentile_path.is_file():
            raise ConfounderOptimizationError("candidate percentile panel is unavailable")

    def _observed_cells(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.rows("confounder_adjusted_surface", self.analysis_run_id)]

    def _primary_contexts(self) -> tuple[dict[int, ConfounderCompatibilityContext], dict[int, str]]:
        contexts = {
            int(raw["held_out_year"]): ConfounderCompatibilityContext.from_dict(raw)
            for raw in self.store.rows("confounder_compatibility_context", self.analysis_run_id)
        }
        exclusions: dict[int, str] = {}
        for raw in self.store.rows("confounder_calibration", self.analysis_run_id):
            if raw.get("exclusion_reason"):
                exclusions[int(raw["held_out_year"])] = str(raw["exclusion_reason"])
        return contexts, exclusions

    def _nested_context(
        self,
        *,
        outer_year: int,
        target_year: int,
        representative: Mapping[str, Any],
    ) -> tuple[ConfounderCompatibilityContext | None, str | None]:
        key = _nested_key(self.candidate.candidate_id, outer_year, target_year)
        prior = self.store.get("confounder_nested_context", self.analysis_run_id, key)
        if prior is not None:
            return ConfounderCompatibilityContext.from_dict(prior), None
        prior_cal = self.store.get("confounder_nested_calibration", self.analysis_run_id, key)
        if prior_cal is not None and prior_cal.get("status") == ConfounderCalibrationStatus.UNAVAILABLE.value:
            return None, str(prior_cal.get("exclusion_reason") or ConfounderExclusionReason.NESTED_CALIBRATION_UNAVAILABLE.value)

        years = tuple(int(y) for y in self.data.years)
        training = tuple(y for y in years if y not in (int(outer_year), int(target_year)))
        checkpoint = self.persistence.checkpoint(
            self.analysis_run_id, NESTED_CALIBRATION_STAGE, "outer_target", key
        )
        if checkpoint is None or checkpoint["state"] != ConfounderCheckpointState.COMPLETED.value:
            self.persistence.put_checkpoint(
                self.analysis_run_id,
                NESTED_CALIBRATION_STAGE,
                "outer_target",
                key,
                ConfounderCheckpointState.RUNNING,
                payload={"outer_deleted_year": outer_year, "target_year": target_year, "training_years": list(training)},
            )

        profile_data = ConfounderProfilingData(
            self.data,
            project_root=self.root,
            preprocessing=self.preprocessing,
            training_years=training,
        )
        nested_profile_run = f"{self.analysis_run_id}__CA8_PROFILE_O{outer_year}_T{target_year}"
        engine = FixedParametersEngine(profile_data, self.c, self.store, self.root, self.log)
        cells = SurfaceRunner(engine).run(
            nested_profile_run,
            loyo=False,
            retain=False,
            rng_identity_run=nested_profile_run,
        )
        try:
            profile = build_relevance_profile_from_cells(
                candidate_id=self.candidate.candidate_id,
                held_out_year=target_year,
                training_years=training,
                thresholds=tuple(float(q) for q in self.c["thresholds"]),
                cells=cells,
            )
            calibration = _calibration_from_profile(
                data=self.data,
                config=self.c,
                root=self.root,
                settings=self.settings,
                preprocessing=self.preprocessing,
                candidate_id=self.candidate.candidate_id,
                outer_year=outer_year,
                target_year=target_year,
                training_years=training,
                profile=profile,
                representative=representative,
                analysis_run_id=self.analysis_run_id,
                log=self.log,
            )
        finally:
            _cleanup_engine_run(self.store, nested_profile_run, cells)

        cal_payload = {
            "schema_version": OPTIMIZATION_SCHEMA,
            "analysis_run_id": self.analysis_run_id,
            "outer_deleted_year": int(outer_year),
            "target_year": int(target_year),
            **calibration.to_dict(),
        }
        self.store.put("confounder_nested_calibration", self.analysis_run_id, key, cal_payload)
        context = None
        exclusion = None
        if calibration.status is not ConfounderCalibrationStatus.UNAVAILABLE:
            context = ConfounderCompatibilityContext.from_calibration(
                calibration, self.settings, outer_deleted_year=int(outer_year)
            )
            self.store.put("confounder_nested_context", self.analysis_run_id, key, context.to_dict())
        else:
            exclusion = (
                calibration.exclusion_reason.value
                if calibration.exclusion_reason is not None
                else ConfounderExclusionReason.NESTED_CALIBRATION_UNAVAILABLE.value
            )
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            NESTED_CALIBRATION_STAGE,
            "outer_target",
            key,
            ConfounderCheckpointState.COMPLETED,
            payload={"context_available": context is not None, "exclusion_reason": exclusion},
        )
        return context, exclusion

    def _loyo_outer(
        self,
        outer_year: int,
        representative: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        expected = len(self.c["thresholds"]) * len(self.c["lags"])
        existing = [
            dict(row) for row in self.store.rows("confounder_adjusted_loyo", self.analysis_run_id)
            if int(row.get("excluded_year")) == int(outer_year)
        ]
        cp = self.persistence.checkpoint(
            self.analysis_run_id, NESTED_LOYO_STAGE, "outer_year", str(int(outer_year))
        )
        if cp is not None and cp["state"] == ConfounderCheckpointState.COMPLETED.value and len(existing) == expected:
            return existing

        years = tuple(int(y) for y in self.data.years)
        contexts: dict[int, ConfounderCompatibilityContext] = {}
        exclusions: dict[int, str] = {
            int(outer_year): ConfounderExclusionReason.NESTED_CALIBRATION_UNAVAILABLE.value
        }
        for target_year in years:
            if target_year == int(outer_year):
                continue
            context, reason = self._nested_context(
                outer_year=int(outer_year), target_year=target_year, representative=representative
            )
            if context is not None:
                contexts[target_year] = context
            else:
                exclusions[target_year] = reason or ConfounderExclusionReason.NESTED_CALIBRATION_UNAVAILABLE.value

        self.persistence.put_checkpoint(
            self.analysis_run_id,
            NESTED_LOYO_STAGE,
            "outer_year",
            str(int(outer_year)),
            ConfounderCheckpointState.RUNNING,
            payload={"context_count": len(contexts)},
        )
        filtered = CrossfitOutcomeData(
            self.data, training_years=tuple(y for y in years if y != int(outer_year))
        )
        nested_run = f"{self.adjusted_run_id}__CA8_LOYO_{int(outer_year)}"
        engine = FixedParametersEngine(filtered, self.c, self.store, self.root, self.log)
        cells = SurfaceRunner(engine).run(
            nested_run,
            loyo=False,
            retain=False,
            rng_identity_run="REFERENCE",
            candidate_percentile_path=self.percentile_path,
            compatibility_contexts=contexts,
            compatibility_exclusion_reasons=exclusions,
        )
        rows: list[dict[str, Any]] = []
        try:
            for cell in cells:
                row = {
                    **dict(cell),
                    "schema_version": OPTIMIZATION_SCHEMA,
                    "analysis_run_id": self.analysis_run_id,
                    "run": self.adjusted_run_id,
                    "excluded_year": int(outer_year),
                    "nested_crossfit": True,
                    "nested_context_count": len(contexts),
                    "nested_context_missing_years": sorted(y for y in years if y != outer_year and y not in contexts),
                }
                key = f"{cell_key(float(cell['q']), int(cell['lag']))}/{int(outer_year)}"
                self.store.put("confounder_adjusted_loyo", self.analysis_run_id, key, row)
                rows.append(row)
            self.store.commit()
        finally:
            _cleanup_engine_run(self.store, nested_run, cells)

        self.persistence.put_checkpoint(
            self.analysis_run_id,
            NESTED_LOYO_STAGE,
            "outer_year",
            str(int(outer_year)),
            ConfounderCheckpointState.COMPLETED,
            payload={"surface_cell_count": len(rows), "context_count": len(contexts)},
        )
        return rows

    def _paired_null_permutation(self, replicate: int) -> tuple[np.ndarray, dict[str, int]]:
        prior = self.store.get("null_maximum", "REFERENCE", f"{int(replicate):05d}")
        permutation = np.arange(self.data.n)
        shifts: dict[str, int] = {}
        if prior is not None and isinstance(prior.get("shifts"), Mapping):
            for iy, year in enumerate(self.data.years):
                key = str(year)
                if key not in prior["shifts"]:
                    raise ConfounderOptimizationError(
                        f"reference null replicate {replicate} does not contain shift for year {year}"
                    )
                shift = int(prior["shifts"][key])
                ix = np.flatnonzero(self.data.year_index == iy)
                permutation[ix] = np.roll(ix, shift)
                shifts[key] = shift
            return permutation, shifts

        shiftmin = int(self.c["optimization"]["null_randomization_rule"]["parameters"]["minimum_shift_days"])
        rng = rng_for_identity(
            self.c["seed"],
            namespace="DISK_DEV_NULL_V1",
            identity={"run": "REFERENCE", "replicate": int(replicate)},
        )
        for iy, year in enumerate(self.data.years):
            ix = np.flatnonzero(self.data.year_index == iy)
            if len(ix) < 2 * shiftmin:
                raise ConfounderOptimizationError(
                    f"Year {year}: record too short for the specified minimum circular shift"
                )
            shift = rng.randint(shiftmin, len(ix) - shiftmin)
            permutation[ix] = np.roll(ix, shift)
            shifts[str(year)] = int(shift)
        return permutation, shifts

    def _null_maxima(self) -> list[float | None]:
        contexts, exclusions = self._primary_contexts()
        R = int(self.c["optimization"]["null_realizations_R"])
        maxima: list[float | None] = []
        for replicate in range(R):
            key = f"{replicate:05d}"
            prior = self.store.get("confounder_adjusted_null_maximum", self.analysis_run_id, key)
            if prior is not None:
                maxima.append(prior.get("maximum_wmarm_cs"))
                continue
            self.persistence.put_checkpoint(
                self.analysis_run_id,
                ADJUSTED_NULL_STAGE,
                "replicate",
                key,
                ConfounderCheckpointState.RUNNING,
            )
            permutation, shifts = self._paired_null_permutation(replicate)
            nullrun = f"{self.adjusted_run_id}__CA8_NULL_{key}"
            engine = FixedParametersEngine(self.data, self.c, self.store, self.root, self.log)
            nullcells = SurfaceRunner(engine).run(
                nullrun,
                permutation=permutation,
                loyo=False,
                retain=False,
                rng_identity_run=f"REFERENCE__NULL_{key}",
                candidate_percentile_path=self.percentile_path,
                compatibility_contexts=contexts,
                compatibility_exclusion_reasons=exclusions,
            )
            try:
                eligible = [row for row in nullcells if admissible(row, self.c["optimization"])]
                maximum_cs = max((row[REFERENCE_METRIC] for row in eligible), default=None)
                maximum_point = max(
                    (row[DIAGNOSTIC_METRIC] for row in eligible if row.get(DIAGNOSTIC_METRIC) is not None),
                    default=None,
                )
                payload = {
                    "schema_version": OPTIMIZATION_SCHEMA,
                    "analysis_run_id": self.analysis_run_id,
                    "adjusted_run_id": self.adjusted_run_id,
                    "replicate": replicate,
                    "reference_metric": "WMARM_CS",
                    "maximum_wmarm_cs": maximum_cs,
                    "maximum_wmarm": maximum_point,
                    "shifts": shifts,
                    "paired_reference_null_permutation": True,
                    "frozen_observed_confounder_calibration": True,
                    "valid_cells": len(eligible),
                    "complete_grid_cells": len(nullcells),
                    "reason": None if maximum_cs is not None else ConfounderExclusionReason.ADJUSTED_REFERENCE_REGION_UNAVAILABLE.value,
                }
                self.store.put("confounder_adjusted_null_maximum", self.analysis_run_id, key, payload)
                for row in nullcells:
                    ckey = f"{key}/{cell_key(float(row['q']), int(row['lag']))}"
                    self.store.put(
                        "confounder_adjusted_null_cell",
                        self.analysis_run_id,
                        ckey,
                        {
                            "schema_version": OPTIMIZATION_SCHEMA,
                            "analysis_run_id": self.analysis_run_id,
                            "replicate": replicate,
                            **{name: row.get(name) for name in (
                                "q", "lag", "wmarm_cs", "wmarm", "marm",
                                "confidence_support_retention_ratio", "bsa_coverage",
                                "population_coverage", "eligible_bsas", "valid",
                            )},
                        },
                    )
                self.store.commit()
            finally:
                _cleanup_engine_run(self.store, nullrun, nullcells)
            maxima.append(maximum_cs)
            self.persistence.put_checkpoint(
                self.analysis_run_id,
                ADJUSTED_NULL_STAGE,
                "replicate",
                key,
                ConfounderCheckpointState.COMPLETED,
                payload={"maximum_wmarm_cs": maximum_cs, "valid_cells": len(eligible)},
            )
            if replicate == 0 or (replicate + 1) % 5 == 0 or replicate + 1 == R:
                self.log(f"Confounder CA8: adjusted null {replicate + 1}/{R} complete")
        return maxima

    def _surface_comparison(
        self,
        adjusted_cells: list[Mapping[str, Any]],
        adjusted_optimization: Mapping[str, Any],
    ) -> dict[str, Any]:
        reference_optimization = self.store.get("optimization", "REFERENCE", "result")
        if reference_optimization is None:
            raise ConfounderOptimizationError("reference optimization is unavailable")
        reference_cells = {
            cell_key(float(row["q"]), int(row["lag"])): row
            for row in self.store.rows("cell", "REFERENCE")
        }
        adjusted = {
            cell_key(float(row["q"]), int(row["lag"])): row for row in adjusted_cells
        }
        common = sorted(
            key for key in set(reference_cells) & set(adjusted)
            if admissible(reference_cells[key], self.c["optimization"])
            and admissible(adjusted[key], self.c["optimization"])
        )
        ref_cs = [float(reference_cells[key][REFERENCE_METRIC]) for key in common]
        adj_cs = [float(adjusted[key][REFERENCE_METRIC]) for key in common]
        diagnostic_keys = [
            key for key in common
            if _finite(reference_cells[key].get(DIAGNOSTIC_METRIC)) is not None
            and _finite(adjusted[key].get(DIAGNOSTIC_METRIC)) is not None
        ]
        ref_w = [float(reference_cells[key][DIAGNOSTIC_METRIC]) for key in diagnostic_keys]
        adj_w = [float(adjusted[key][DIAGNOSTIC_METRIC]) for key in diagnostic_keys]
        ref_region = _region_keys(reference_optimization)
        adj_region = _region_keys(adjusted_optimization)
        union = ref_region | adj_region
        intersection = ref_region & adj_region
        ref_rep = _representative(reference_optimization)
        adj_rep = _representative(adjusted_optimization)
        return {
            "schema_version": OPTIMIZATION_SCHEMA,
            "analysis_run_id": self.analysis_run_id,
            "reference_metric": "WMARM_CS",
            "common_admissible_cell_count": len(common),
            "wmarm_cs_surface_rmse": _rmse(ref_cs, adj_cs),
            "wmarm_cs_surface_spearman": _spearman(ref_cs, adj_cs),
            "diagnostic_wmarm_common_cell_count": len(diagnostic_keys),
            "diagnostic_wmarm_surface_rmse": _rmse(ref_w, adj_w),
            "diagnostic_wmarm_surface_spearman": _spearman(ref_w, adj_w),
            "reference_surface_state": _surface_state(reference_optimization),
            "adjusted_surface_state": _surface_state(adjusted_optimization),
            "reference_robust_region_cell_count": len(ref_region),
            "adjusted_robust_region_cell_count": len(adj_region),
            "robust_region_intersection_cell_count": len(intersection),
            "robust_region_union_cell_count": len(union),
            "robust_region_jaccard": len(intersection) / len(union) if union else None,
            "reference_representative": None if ref_rep is None else {"q": float(ref_rep["q"]), "lag": int(ref_rep["lag"])},
            "adjusted_representative": None if adj_rep is None else {"q": float(adj_rep["q"]), "lag": int(adj_rep["lag"])},
            "representative_delta_q": None if ref_rep is None or adj_rep is None else float(adj_rep["q"]) - float(ref_rep["q"]),
            "representative_delta_lag": None if ref_rep is None or adj_rep is None else int(adj_rep["lag"]) - int(ref_rep["lag"]),
        }

    def run(
        self,
        *,
        requested: bool,
        representative: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not requested:
            return {
                "schema_version": OPTIMIZATION_SCHEMA,
                "status": "NOT_REQUESTED",
                "stage": OPTIMIZATION_STAGE,
                "adjusted_reoptimization_inference_enabled": False,
                "loyo_executed": False,
                "null_search_executed": False,
                "robust_region_resolved": False,
                "representative_selected": False,
            }
        if representative is None:
            return {
                "schema_version": OPTIMIZATION_SCHEMA,
                "status": "UNAVAILABLE",
                "stage": OPTIMIZATION_STAGE,
                "reason": ConfounderExclusionReason.REFERENCE_CONFIGURATION_UNAVAILABLE.value,
                "adjusted_reoptimization_inference_enabled": False,
                "loyo_executed": False,
                "null_search_executed": False,
                "robust_region_resolved": False,
                "representative_selected": False,
            }
        years = [int(y) for y in self.data.years]
        if not self.settings.supports_standard_reoptimization(len(years)):
            return {
                "schema_version": OPTIMIZATION_SCHEMA,
                "status": "UNAVAILABLE",
                "stage": OPTIMIZATION_STAGE,
                "reason": ConfounderExclusionReason.INSUFFICIENT_YEARS_REOPTIMIZATION.value,
                "available_years": years,
                "minimum_years_reoptimization": self.settings.minimum_years_reoptimization,
                "adjusted_reoptimization_inference_enabled": False,
                "loyo_executed": False,
                "null_search_executed": False,
                "robust_region_resolved": False,
                "representative_selected": False,
            }
        cells = self._observed_cells()
        expected = len(self.c["thresholds"]) * len(self.c["lags"])
        if len(cells) != expected:
            return {
                "schema_version": OPTIMIZATION_SCHEMA,
                "status": "UNAVAILABLE",
                "stage": OPTIMIZATION_STAGE,
                "reason": ConfounderExclusionReason.ADJUSTED_REFERENCE_REGION_UNAVAILABLE.value,
                "adjusted_surface_cell_count": len(cells),
                "expected_surface_cell_count": expected,
                "adjusted_reoptimization_inference_enabled": False,
                "loyo_executed": False,
                "null_search_executed": False,
                "robust_region_resolved": False,
                "representative_selected": False,
            }

        self.log(
            f"Confounder CA8: nested LOYO years={len(years)}; cells={expected}; "
            f"null R={self.c['optimization']['null_realizations_R']}"
        )
        loyo: list[dict[str, Any]] = []
        for outer_year in years:
            loyo.extend(self._loyo_outer(outer_year, representative))
        maxima = self._null_maxima()
        result = resolve_region(cells, loyo, maxima, years, self.c)
        absdev: list[float] = []
        for diag in result["diagnostics"]:
            if diag["admissible_support"] and diag["loyo_complete"]:
                for row in loyo:
                    if row["q"] == diag["q"] and row["lag"] == diag["lag"]:
                        absdev.append(abs(float(row[REFERENCE_METRIC]) - float(diag[REFERENCE_METRIC])))
        result["surface_coherence"] = 1 - float(np.mean(absdev)) if absdev else None
        result["run"] = self.adjusted_run_id
        result["schema_version"] = OPTIMIZATION_SCHEMA
        result["analysis_run_id"] = self.analysis_run_id
        result["nested_crossfit"] = True
        result["conditional_adjusted_null"] = True
        result["frozen_observed_confounder_calibration_in_null"] = True
        result["paired_reference_null_permutations"] = True

        fp = fingerprint(cells, result, self.c)
        comparison = self._surface_comparison(cells, result)
        self.store.put("confounder_adjusted_optimization", self.analysis_run_id, "result", result)
        for diag in result["diagnostics"]:
            self.store.put(
                "confounder_adjusted_optimization_cell",
                self.analysis_run_id,
                cell_key(float(diag["q"]), int(diag["lag"])),
                {"schema_version": OPTIMIZATION_SCHEMA, "analysis_run_id": self.analysis_run_id, **diag},
            )
        self.store.put("confounder_adjusted_fingerprint", self.analysis_run_id, "result", fp)
        self.store.put("confounder_surface_comparison", self.analysis_run_id, "result", comparison)
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            OPTIMIZATION_STAGE,
            "optimization",
            "result",
            ConfounderCheckpointState.COMPLETED,
            payload={
                "status": result["status"],
                "representative": result.get("representative"),
                "robust_region_cell_count": len(result.get("robust_region") or []),
            },
        )
        summary = {
            "schema_version": OPTIMIZATION_SCHEMA,
            "status": result["status"],
            "stage": OPTIMIZATION_STAGE,
            "reason": None,
            "candidate_id": self.preprocessing.candidate_id,
            "analysis_run_id": self.analysis_run_id,
            "adjusted_run_id": self.adjusted_run_id,
            "reference_metric": "WMARM_CS",
            "adjusted_reoptimization_inference_enabled": True,
            "loyo_executed": True,
            "nested_loyo": True,
            "nested_loyo_row_count": len(loyo),
            "nested_calibration_count": len(list(self.store.rows("confounder_nested_calibration", self.analysis_run_id))),
            "nested_context_count": len(list(self.store.rows("confounder_nested_context", self.analysis_run_id))),
            "null_search_executed": True,
            "conditional_adjusted_null": True,
            "adjusted_null_realizations_requested": int(self.c["optimization"]["null_realizations_R"]),
            "adjusted_null_realizations_valid": sum(value is not None and math.isfinite(float(value)) for value in maxima),
            "adjusted_null_complete": bool(result.get("null_complete")),
            "robust_region_resolved": True,
            "robust_region_status": result["status"],
            "robust_region_cell_count": len(result.get("robust_region") or []),
            "representative_selected": result.get("representative") is not None,
            "adjusted_representative": result.get("representative"),
            "adjusted_p_max": result.get("p_max"),
            "adjusted_null_cutoff_wmarm_cs": result.get("null_cutoff_wmarm_cs"),
            "adjusted_surface_coherence": result.get("surface_coherence"),
            "adjusted_fingerprint_available": True,
            "surface_comparison_available": True,
            **comparison,
        }
        self.store.put("workflow", self.analysis_run_id, "adjusted_optimization_result", summary)
        self.store.commit()
        self.log(
            f"Confounder CA8 complete: {result['status']}; "
            f"region={len(result.get('robust_region') or [])}; representative={result.get('representative')}"
        )
        return summary

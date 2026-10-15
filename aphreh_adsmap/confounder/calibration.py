"""CA4 WMARMcs relevance profiling and cross-fitted caliper calibration."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data import Data
from ..engine import Cancelled
from ..storage import Store, mapped
from .configuration import ConfounderAdjustmentSettings
from .contracts import (
    ConfounderCalibrationResult,
    ConfounderCalibrationStatus,
    ConfounderCandidateDefinition,
    ConfounderCompatibilityContext,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
    ConfounderProfilingResult,
    ConfounderProfilingStatus,
    ConfounderRelevanceProfile,
    PairwiseSampleBundle,
)
from .pairwise import PairwiseSampleExtractor
from .persistence import (
    ConfounderCheckpointState,
    ConfounderPersistence,
    ConfounderRunState,
)


RELEVANCE_STAGE = "RELEVANCE_PROFILE"
PAIRWISE_STAGE = "PAIRWISE_SAMPLE_EXTRACTION"
CALIBRATION_STAGE = "CALIPER_CALIBRATION"


class ConfounderCalibrationError(ValueError):
    pass


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def build_relevance_profile_from_cells(
    *,
    candidate_id: str,
    held_out_year: int,
    training_years: tuple[int, ...],
    thresholds: tuple[float, ...],
    cells: list[Mapping[str, Any]],
) -> ConfounderRelevanceProfile:
    """CA6: population-support-weighted threshold relevance from WMARMcs cells.

    This pure helper is shared by ordinary CA4 cross-fitting and CA8 nested
    cross-fitting. Standard WMARM is retained only as a diagnostic surface.
    """
    relevance: dict[float, float] = {}
    diagnostic: dict[float, float] = {}
    coverage: dict[float, float] = {}
    for q in thresholds:
        q = float(q)
        qcells = [
            row for row in cells
            if float(row.get("q")) == q
            and bool(row.get("valid"))
            and _finite(row.get("wmarm_cs"))
            and _finite(row.get("population_coverage"))
            and float(row["population_coverage"]) > 0.0
        ]
        denominator = sum(float(row["population_coverage"]) for row in qcells)
        if denominator <= 0.0:
            continue
        relevance[q] = sum(
            float(row["population_coverage"]) * float(row["wmarm_cs"])
            for row in qcells
        ) / denominator
        diagnostic_cells = [row for row in qcells if _finite(row.get("wmarm")) or _finite(row.get("diagnostic_wmarm"))]
        diagnostic_denominator = sum(
            float(row["population_coverage"]) for row in diagnostic_cells
        )
        if diagnostic_denominator > 0.0:
            diagnostic[q] = sum(
                float(row["population_coverage"])
                * float(
                    row.get("diagnostic_wmarm")
                    if _finite(row.get("diagnostic_wmarm"))
                    else row.get("wmarm")
                )
                for row in diagnostic_cells
            ) / diagnostic_denominator
        coverage[q] = float(np.mean([float(row["population_coverage"]) for row in qcells]))

    if not relevance:
        return ConfounderRelevanceProfile(
            candidate_id=candidate_id,
            held_out_year=held_out_year,
            training_years=training_years,
            threshold_relevance={},
            normalized_relevance={},
            relevance_amplitude=0.0,
            status=ConfounderCalibrationStatus.UNAVAILABLE,
            diagnostic_wmarm_by_threshold=diagnostic,
            population_coverage_by_threshold=coverage,
        )

    total = sum(float(v) for v in relevance.values())
    if total == 0.0:
        return ConfounderRelevanceProfile(
            candidate_id=candidate_id,
            held_out_year=held_out_year,
            training_years=training_years,
            threshold_relevance=relevance,
            normalized_relevance={},
            relevance_amplitude=0.0,
            status=ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE,
            diagnostic_wmarm_by_threshold=diagnostic,
            population_coverage_by_threshold=coverage,
        )

    normalized = {q: float(value) / total for q, value in relevance.items()}
    return ConfounderRelevanceProfile(
        candidate_id=candidate_id,
        held_out_year=held_out_year,
        training_years=training_years,
        threshold_relevance=relevance,
        normalized_relevance=normalized,
        relevance_amplitude=max(relevance.values()),
        status=ConfounderCalibrationStatus.AVAILABLE,
        diagnostic_wmarm_by_threshold=diagnostic,
        population_coverage_by_threshold=coverage,
    )


def build_relevance_profile(
    store: Store,
    analysis_run_id: str,
    profiling: ConfounderProfilingResult,
) -> ConfounderRelevanceProfile:
    """CA6: build the ordinary held-out relevance profile from persisted CA3 cells."""
    if profiling.status is not ConfounderProfilingStatus.AVAILABLE:
        return ConfounderRelevanceProfile(
            candidate_id=profiling.candidate_id,
            held_out_year=profiling.held_out_year,
            training_years=profiling.training_years,
            threshold_relevance={},
            normalized_relevance={},
            relevance_amplitude=0.0,
            status=ConfounderCalibrationStatus.UNAVAILABLE,
        )
    rows = [
        row for row in store.rows("confounder_profile_cell", analysis_run_id)
        if int(row.get("held_out_year")) == profiling.held_out_year
    ]
    return build_relevance_profile_from_cells(
        candidate_id=profiling.candidate_id,
        held_out_year=profiling.held_out_year,
        training_years=profiling.training_years,
        thresholds=profiling.thresholds,
        cells=rows,
    )


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfounderCalibrationError("calibration array path escapes project root") from exc
    if not path.is_file():
        raise ConfounderCalibrationError(f"calibration array is unavailable: {path}")
    return path


def _distance_matrices(
    exposed_percentile: np.ndarray,
    nonexposed_percentile: np.ndarray,
    profile: ConfounderRelevanceProfile,
) -> tuple[np.ndarray, np.ndarray]:
    d_rel = np.zeros((len(exposed_percentile), len(nonexposed_percentile)), dtype=float)
    for q, rho in sorted(profile.normalized_relevance.items()):
        e_state = exposed_percentile >= float(q) / 100.0
        n_state = nonexposed_percentile >= float(q) / 100.0
        d_rel += float(rho) * (e_state[:, None] != n_state[None, :])
    d_rel *= float(profile.relevance_amplitude)
    d_cont = np.abs(exposed_percentile[:, None] - nonexposed_percentile[None, :])
    return d_rel, d_cont


def evaluate_calibration_curves(
    data: Data,
    project_root: str | Path,
    bundle: PairwiseSampleBundle,
    profile: ConfounderRelevanceProfile,
    settings: ConfounderAdjustmentSettings,
) -> tuple[dict[float, float], dict[float, float], list[dict[str, Any]]]:
    """CA9.2-CA9.5: evaluate R_C and S_C over the fixed caliper grid.

    Original target-hazard weights are normalized before candidate missingness is
    applied. Consequently positive target-hazard weight attached to an invalid
    candidate observation is correctly represented as lost retained support.
    """
    root = Path(project_root).expanduser().resolve()
    differential_path = _project_path(root, bundle.differential_array_path)
    raw_weight_path = _project_path(root, bundle.raw_weight_array_path)
    label_path = _project_path(root, bundle.exposure_label_array_path)
    percentile_path = _project_path(root, bundle.candidate_percentile_array_path)
    grid = tuple(float(v) for v in settings.relevance_caliper_grid)
    annual_by_caliper: dict[float, dict[int, tuple[float, float]]] = {
        k: {} for k in grid
    }
    annual_rows: list[dict[str, Any]] = []

    with mapped(differential_path) as differentials, \
         mapped(raw_weight_path) as raw_weights, \
         mapped(label_path) as labels, \
         mapped(percentile_path) as percentiles:
        expected = (data.B, data.n)
        for array, name in (
            (differentials, "differential"), (raw_weights, "raw_weight"),
            (labels, "exposure_label"), (percentiles, "candidate_percentile"),
        ):
            if array.shape != expected:
                raise ConfounderCalibrationError(f"{name} panel shape is inconsistent")

        for year in bundle.training_years:
            iy = data.years.index(int(year))
            date_mask = data.year_index == iy
            accum = {k: [0.0, 0.0, 0.0, 0] for k in grid}
            for b, _bid in enumerate(data.ids):
                population = float(data.population[b, iy])
                if not math.isfinite(population) or population <= 0.0:
                    continue
                x = np.asarray(differentials[b], dtype=float)
                w = np.asarray(raw_weights[b], dtype=float)
                lab = np.asarray(labels[b], dtype=np.int8)
                u = np.asarray(percentiles[b], dtype=float)
                exposed_all = date_mask & np.isfinite(x) & (lab == 1)
                nonexposed_all = date_mask & np.isfinite(x) & (lab == 0)
                sum_e = float(np.sum(w[exposed_all]))
                sum_n = float(np.sum(w[nonexposed_all]))
                if not (sum_e > 0.0 and sum_n > 0.0):
                    continue

                e = exposed_all & np.isfinite(u) & (w > 0.0)
                n = nonexposed_all & np.isfinite(u) & (w > 0.0)
                if not np.any(e) or not np.any(n):
                    continue
                w_e = w[e] / sum_e
                w_n = w[n] / sum_n
                u_e = u[e]
                u_n = u[n]
                d_rel, d_cont = _distance_matrices(u_e, u_n, profile)
                original_pair_weight = w_e[:, None] * w_n[None, :]
                safeguard = d_cont <= float(settings.continuous_percentile_safeguard) + 1e-15

                for kappa in grid:
                    compatible = safeguard & (d_rel <= kappa + 1e-15)
                    retained_pair_weight = original_pair_weight * compatible
                    denominator = float(np.sum(retained_pair_weight))
                    if denominator <= 0.0:
                        continue
                    residual = float(np.sum(retained_pair_weight * d_rel) / denominator)
                    exposed_match = np.sum(w_n[None, :] * compatible, axis=1) > 0.0
                    nonexposed_match = np.sum(w_e[:, None] * compatible, axis=0) > 0.0
                    support_e = float(np.sum(w_e[exposed_match]))
                    support_n = float(np.sum(w_n[nonexposed_match]))
                    retained_support = min(support_e, support_n)
                    slot = accum[kappa]
                    slot[0] += population * residual
                    slot[1] += population * retained_support
                    slot[2] += population
                    slot[3] += 1

            for kappa in grid:
                residual_num, support_num, pop_den, bsa_count = accum[kappa]
                if pop_den > 0.0:
                    annual_r = residual_num / pop_den
                    annual_s = support_num / pop_den
                    annual_by_caliper[kappa][int(year)] = (annual_r, annual_s)
                    annual_rows.append({
                        "held_out_year": bundle.held_out_year,
                        "training_year": int(year),
                        "caliper": kappa,
                        "residual_relevance": annual_r,
                        "retained_support": annual_s,
                        "valid_bsas": bsa_count,
                        "represented_population": pop_den,
                        "valid": True,
                        "reason": None,
                    })
                else:
                    annual_rows.append({
                        "held_out_year": bundle.held_out_year,
                        "training_year": int(year),
                        "caliper": kappa,
                        "residual_relevance": None,
                        "retained_support": None,
                        "valid_bsas": 0,
                        "represented_population": 0.0,
                        "valid": False,
                        "reason": "NO_VALID_BSA_CALIBRATION_SUMMARY",
                    })

    residual_curve: dict[float, float] = {}
    support_curve: dict[float, float] = {}
    required_years = set(int(y) for y in bundle.training_years)
    for kappa in grid:
        annual = annual_by_caliper[kappa]
        if set(annual) != required_years:
            continue
        residual_curve[kappa] = float(np.mean([annual[y][0] for y in bundle.training_years]))
        support_curve[kappa] = float(np.mean([annual[y][1] for y in bundle.training_years]))
    return residual_curve, support_curve, annual_rows


def select_caliper(
    residual_curve: Mapping[float, float],
    support_curve: Mapping[float, float],
    *,
    residual_tolerance: float,
    minimum_support: float,
) -> float | None:
    admissible = [
        float(kappa)
        for kappa, residual in residual_curve.items()
        if kappa in support_curve
        and float(residual) <= float(residual_tolerance) + 1e-15
        and float(support_curve[kappa]) >= float(minimum_support) - 1e-15
    ]
    return max(admissible) if admissible else None


class ConfounderCalibrationRunner:
    """CA4 orchestrator for relevance profiles, pair bundles and calipers."""

    def __init__(
        self,
        data: Data,
        config: dict,
        store: Store,
        root: str | Path,
        persistence: ConfounderPersistence,
        analysis_run_id: str,
        candidate: ConfounderCandidateDefinition,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.store = store
        self.root = Path(root).expanduser().resolve()
        self.persistence = persistence
        self.analysis_run_id = analysis_run_id
        self.candidate = candidate
        self.preprocessing = preprocessing
        self.log = log
        self.settings = ConfounderAdjustmentSettings.from_dict(config["confounder"])
        self.extractor = PairwiseSampleExtractor(
            data, config, self.root, preprocessing, log=log
        )

    def _profiles(self) -> list[ConfounderProfilingResult]:
        values = [
            ConfounderProfilingResult.from_dict(row)
            for row in self.store.rows("confounder_profile", self.analysis_run_id)
        ]
        return sorted(values, key=lambda value: value.held_out_year)

    def _store_profile(self, profile: ConfounderRelevanceProfile) -> None:
        self.store.put(
            "confounder_relevance_profile",
            self.analysis_run_id,
            str(profile.held_out_year),
            profile.to_dict(),
        )

    def _bundle_from_store(self, held_out_year: int) -> PairwiseSampleBundle | None:
        raw = self.store.get(
            "confounder_pairwise_bundle", self.analysis_run_id, str(held_out_year)
        )
        return None if raw is None else PairwiseSampleBundle.from_dict(raw)

    def _calibration_from_store(self, held_out_year: int) -> ConfounderCalibrationResult | None:
        raw = self.store.get(
            "confounder_calibration", self.analysis_run_id, str(held_out_year)
        )
        return None if raw is None else ConfounderCalibrationResult.from_dict(raw)

    def _unavailable(
        self,
        profiling: ConfounderProfilingResult,
        profile: ConfounderRelevanceProfile | None,
        reason: ConfounderExclusionReason,
    ) -> ConfounderCalibrationResult:
        return ConfounderCalibrationResult(
            candidate_id=self.candidate.candidate_id,
            held_out_year=profiling.held_out_year,
            training_years=profiling.training_years,
            relevance_profile=profile,
            selected_caliper=None,
            residual_relevance_by_caliper={},
            retained_support_by_caliper={},
            status=ConfounderCalibrationStatus.UNAVAILABLE,
            exclusion_reason=reason,
        )

    def _calibrate_one(
        self,
        profiling: ConfounderProfilingResult,
        representative: Mapping[str, Any] | None,
    ) -> ConfounderCalibrationResult:
        key = str(profiling.held_out_year)
        existing_cp = self.persistence.checkpoint(
            self.analysis_run_id, CALIBRATION_STAGE, "held_out_year", key
        )
        existing = self._calibration_from_store(profiling.held_out_year)
        if (
            existing_cp is not None
            and existing_cp["state"] == ConfounderCheckpointState.COMPLETED.value
            and existing is not None
        ):
            self.log(
                f"Confounder CA4: reuse held-out {profiling.held_out_year} calibration; "
                f"status={existing.status.value}"
            )
            return existing

        self.persistence.put_checkpoint(
            self.analysis_run_id, RELEVANCE_STAGE, "held_out_year", key,
            ConfounderCheckpointState.RUNNING,
        )
        profile = build_relevance_profile(self.store, self.analysis_run_id, profiling)
        self._store_profile(profile)
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id, RELEVANCE_STAGE, "held_out_year", key,
            ConfounderCheckpointState.COMPLETED, payload=profile.to_dict(),
        )

        if profiling.status is not ConfounderProfilingStatus.AVAILABLE:
            result = self._unavailable(
                profiling, profile,
                ConfounderExclusionReason.PROFILING_WMARM_CS_SURFACE_UNAVAILABLE,
            )
        elif profile.status is ConfounderCalibrationStatus.UNAVAILABLE:
            result = self._unavailable(
                profiling, profile, ConfounderExclusionReason.RELEVANCE_PROFILE_UNAVAILABLE
            )
        elif profile.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE:
            result = ConfounderCalibrationResult(
                candidate_id=self.candidate.candidate_id,
                held_out_year=profiling.held_out_year,
                training_years=profiling.training_years,
                relevance_profile=profile,
                selected_caliper=None,
                residual_relevance_by_caliper={},
                retained_support_by_caliper={},
                status=ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE,
            )
        elif representative is None:
            result = self._unavailable(
                profiling, profile,
                ConfounderExclusionReason.REFERENCE_CONFIGURATION_UNAVAILABLE,
            )
        else:
            q = float(representative["q"])
            lag = int(representative["lag"])
            bundle_cp = self.persistence.checkpoint(
                self.analysis_run_id, PAIRWISE_STAGE, "held_out_year", key
            )
            bundle = self._bundle_from_store(profiling.held_out_year)
            if not (
                bundle_cp is not None
                and bundle_cp["state"] == ConfounderCheckpointState.COMPLETED.value
                and bundle is not None
                and bundle.target_threshold == q
                and bundle.target_lag == lag
                and bundle.training_years == profiling.training_years
            ):
                self.persistence.put_checkpoint(
                    self.analysis_run_id, PAIRWISE_STAGE, "held_out_year", key,
                    ConfounderCheckpointState.RUNNING,
                    payload={"q": q, "lag": lag},
                )
                bundle = self.extractor.extract(
                    analysis_run_id=self.analysis_run_id,
                    held_out_year=profiling.held_out_year,
                    training_years=profiling.training_years,
                    target_threshold=q,
                    target_lag=lag,
                )
                self.store.put(
                    "confounder_pairwise_bundle", self.analysis_run_id, key,
                    bundle.to_dict(),
                )
                self.store.commit()
                self.persistence.put_checkpoint(
                    self.analysis_run_id, PAIRWISE_STAGE, "held_out_year", key,
                    ConfounderCheckpointState.COMPLETED, payload=bundle.to_dict(),
                )

            residual_curve, support_curve, annual_rows = evaluate_calibration_curves(
                self.data, self.root, bundle, profile, self.settings
            )
            for row in annual_rows:
                self.store.put(
                    "confounder_calibration_annual", self.analysis_run_id,
                    f"{profiling.held_out_year}/{int(row['training_year'])}/{float(row['caliper']):.3f}",
                    {"analysis_run_id": self.analysis_run_id,
                     "candidate_id": self.candidate.candidate_id, **row},
                )
            selected_by_tolerance = {
                float(tolerance): select_caliper(
                    residual_curve, support_curve,
                    residual_tolerance=float(tolerance),
                    minimum_support=self.settings.minimum_retained_support,
                )
                for tolerance in self.settings.residual_relevance_sensitivity_values
            }
            selected = select_caliper(
                residual_curve, support_curve,
                residual_tolerance=self.settings.residual_relevance_tolerance,
                minimum_support=self.settings.minimum_retained_support,
            )
            for kappa in self.settings.relevance_caliper_grid:
                kappa = float(kappa)
                self.store.put(
                    "confounder_calibration_curve", self.analysis_run_id,
                    f"{profiling.held_out_year}/{kappa:.3f}",
                    {
                        "analysis_run_id": self.analysis_run_id,
                        "candidate_id": self.candidate.candidate_id,
                        "held_out_year": profiling.held_out_year,
                        "caliper": kappa,
                        "residual_relevance": residual_curve.get(kappa),
                        "retained_support": support_curve.get(kappa),
                        "complete_training_year_support": kappa in residual_curve and kappa in support_curve,
                        "selected_default": selected == kappa,
                        "selected_tolerances": [
                            tolerance for tolerance, value in selected_by_tolerance.items()
                            if value == kappa
                        ],
                    },
                )
            if selected is None:
                result = ConfounderCalibrationResult(
                    candidate_id=self.candidate.candidate_id,
                    held_out_year=profiling.held_out_year,
                    training_years=profiling.training_years,
                    relevance_profile=profile,
                    selected_caliper=None,
                    residual_relevance_by_caliper=residual_curve,
                    retained_support_by_caliper=support_curve,
                    status=ConfounderCalibrationStatus.UNAVAILABLE,
                    selected_caliper_by_tolerance=selected_by_tolerance,
                    exclusion_reason=ConfounderExclusionReason.NO_ADMISSIBLE_CALIPER,
                )
            else:
                result = ConfounderCalibrationResult(
                    candidate_id=self.candidate.candidate_id,
                    held_out_year=profiling.held_out_year,
                    training_years=profiling.training_years,
                    relevance_profile=profile,
                    selected_caliper=selected,
                    residual_relevance_by_caliper=residual_curve,
                    retained_support_by_caliper=support_curve,
                    status=ConfounderCalibrationStatus.AVAILABLE,
                    selected_caliper_by_tolerance=selected_by_tolerance,
                )

        self.store.put("confounder_calibration", self.analysis_run_id, key, result.to_dict())
        if result.status is not ConfounderCalibrationStatus.UNAVAILABLE:
            context = ConfounderCompatibilityContext.from_calibration(
                result, self.settings
            )
            self.store.put(
                "confounder_compatibility_context", self.analysis_run_id, key,
                context.to_dict(),
            )
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id, CALIBRATION_STAGE, "held_out_year", key,
            ConfounderCheckpointState.COMPLETED, payload=result.to_dict(),
        )
        return result

    def run(self, reference_optimization: Mapping[str, Any] | None) -> dict[str, Any]:
        profiles = self._profiles()
        representative = (
            reference_optimization.get("representative")
            if isinstance(reference_optimization, Mapping) else None
        )
        self.persistence.set_run_state(self.analysis_run_id, ConfounderRunState.RUNNING)
        try:
            results = [self._calibrate_one(profile, representative) for profile in profiles]
            available = [r for r in results if r.status is ConfounderCalibrationStatus.AVAILABLE]
            neutral = [r for r in results if r.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE]
            unavailable = [r for r in results if r.status is ConfounderCalibrationStatus.UNAVAILABLE]
            summary = {
                "status": "COMPLETED" if not unavailable else "COMPLETED_WITH_EXCLUSIONS",
                "stage": "CA4_PAIRWISE_SAMPLE_EXTRACTION_AND_CALIBRATION",
                "candidate_id": self.candidate.candidate_id,
                "analysis_run_id": self.analysis_run_id,
                "reference_metric": "WMARM_CS",
                "reference_representative": None if representative is None else {
                    "q": float(representative["q"]), "lag": int(representative["lag"])
                },
                "calibration_count": len(results),
                "available_calibration_count": len(available),
                "neutral_calibration_count": len(neutral),
                "unavailable_calibration_count": len(unavailable),
                "adjusted_estimator_enabled": False,
                "calibrations": [r.to_dict() for r in results],
            }
            self.store.put("workflow", self.analysis_run_id, "calibration_result", summary)
            self.store.commit()
            self.persistence.set_run_state(self.analysis_run_id, ConfounderRunState.COMPLETED)
            return summary
        except BaseException as exc:
            self.persistence.set_run_state(
                self.analysis_run_id,
                ConfounderRunState.CANCELLED
                if isinstance(exc, (Cancelled, KeyboardInterrupt))
                else ConfounderRunState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

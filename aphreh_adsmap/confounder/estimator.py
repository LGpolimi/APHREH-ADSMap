"""CA5 compatibility-aware pairwise estimator integration.

This module implements the candidate-confounder restricted weighted Cliff's
Delta while preserving the bounded-memory architecture. Compatibility is built
once for one BSA-year. Bootstrap batches can later reuse the same pair matrix
without materializing a batch x exposed x non-exposed tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

import numpy as np

from .. import numerics as nm
from ..engine import FixedParametersEngine, cell_scope
from ..storage import Store, mapped
from ..support import active_support_policy, apply_baseline_support_filter
from .contracts import (
    ConfounderCompatibilityContext,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
)
from .persistence import (
    ConfounderCheckpointState,
    ConfounderPersistence,
)


POINT_ESTIMATION_STAGE = "CA5_PAIRWISE_ESTIMATOR"
_NUMERICAL_TOLERANCE = 1e-15


class ConfounderEstimatorError(ValueError):
    pass


@dataclass(frozen=True)
class PairwiseSupportDiagnostics:
    exposed_support: float
    nonexposed_support: float
    minimum_support: float
    normalized_pair_weight: float
    compatible_pair_count: int
    target_exposed_count: int
    target_nonexposed_count: int
    candidate_valid_exposed_count: int
    candidate_valid_nonexposed_count: int

    def to_dict(self) -> dict:
        return {
            "retained_support_exposed": self.exposed_support,
            "retained_support_nonexposed": self.nonexposed_support,
            "retained_support": self.minimum_support,
            "normalized_compatible_pair_weight": self.normalized_pair_weight,
            "compatible_pair_count": self.compatible_pair_count,
            "target_exposed_positive_weight_count": self.target_exposed_count,
            "target_nonexposed_positive_weight_count": self.target_nonexposed_count,
            "candidate_valid_exposed_positive_weight_count": self.candidate_valid_exposed_count,
            "candidate_valid_nonexposed_positive_weight_count": self.candidate_valid_nonexposed_count,
        }


@dataclass(frozen=True)
class AdjustedPairwisePointResult:
    rc: float | None
    eligible: bool
    exclusion_reason: str | None
    support: PairwiseSupportDiagnostics

    def to_dict(self) -> dict:
        return {
            "adjusted_rc": self.rc,
            "adjusted_point_eligible": self.eligible,
            "adjusted_point_exclusion_reason": self.exclusion_reason,
            **self.support.to_dict(),
        }


def _validate_inputs(
    values: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    percentiles: np.ndarray,
    counts: np.ndarray | None = None,
) -> None:
    n = len(values)
    if values.ndim != 1 or raw.ndim != 1 or labels.ndim != 1 or percentiles.ndim != 1:
        raise ConfounderEstimatorError("pairwise estimator inputs must be one-dimensional")
    if not (len(raw) == len(labels) == len(percentiles) == n):
        raise ConfounderEstimatorError("pairwise estimator input lengths differ")
    if counts is not None and (counts.ndim != 2 or counts.shape[1] != n):
        raise ConfounderEstimatorError("pairwise count matrix must have shape (batch, observations)")


def compatibility_matrix(
    exposed_percentiles: np.ndarray,
    nonexposed_percentiles: np.ndarray,
    context: ConfounderCompatibilityContext,
) -> np.ndarray:
    """CA7-CA8/CA10: exact frozen compatibility for one exposed/non-exposed grid."""
    pe = np.asarray(exposed_percentiles, dtype=float)
    pn = np.asarray(nonexposed_percentiles, dtype=float)
    if pe.ndim != 1 or pn.ndim != 1:
        raise ConfounderEstimatorError("candidate percentile vectors must be one-dimensional")
    if context.neutral_compatibility:
        return np.ones((len(pe), len(pn)), dtype=bool)
    if context.selected_caliper is None:
        raise ConfounderEstimatorError("available compatibility context has no selected caliper")
    if not np.all(np.isfinite(pe)) or not np.all(np.isfinite(pn)):
        raise ConfounderEstimatorError("compatibility_matrix requires finite candidate percentiles")

    d_rel = np.zeros((len(pe), len(pn)), dtype=float)
    for threshold, rho in sorted(context.normalized_relevance.items()):
        e_state = pe >= float(threshold) / 100.0
        n_state = pn >= float(threshold) / 100.0
        d_rel += float(rho) * (e_state[:, None] != n_state[None, :])
    d_rel *= float(context.relevance_amplitude)
    d_cont = np.abs(pe[:, None] - pn[None, :])
    return (
        (d_rel <= float(context.selected_caliper) + _NUMERICAL_TOLERANCE)
        & (d_cont <= float(context.continuous_percentile_safeguard) + _NUMERICAL_TOLERANCE)
    )


def _target_indices(values: np.ndarray, raw: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(values) & np.isfinite(raw) & (raw > 0) & (labels >= 0)
    return np.flatnonzero(keep & (labels == 1)), np.flatnonzero(keep & (labels == 0))


def pairwise_support_diagnostics(
    values: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    percentiles: np.ndarray,
    context: ConfounderCompatibilityContext,
) -> PairwiseSupportDiagnostics:
    """CA9.4/CA10: original-weight retained support for the held-out sample."""
    values = np.asarray(values, dtype=float)
    raw = np.asarray(raw, dtype=float)
    labels = np.asarray(labels)
    percentiles = np.asarray(percentiles, dtype=float)
    _validate_inputs(values, raw, labels, percentiles)
    e_all, n_all = _target_indices(values, raw, labels)
    e_total = float(np.sum(raw[e_all])) if len(e_all) else 0.0
    n_total = float(np.sum(raw[n_all])) if len(n_all) else 0.0
    if e_total <= 0.0 or n_total <= 0.0:
        return PairwiseSupportDiagnostics(0.0, 0.0, 0.0, 0.0, 0, len(e_all), len(n_all), 0, 0)

    if context.neutral_compatibility:
        return PairwiseSupportDiagnostics(
            1.0, 1.0, 1.0, 1.0, int(len(e_all) * len(n_all)),
            len(e_all), len(n_all), len(e_all), len(n_all),
        )

    e = e_all[np.isfinite(percentiles[e_all])]
    n = n_all[np.isfinite(percentiles[n_all])]
    if not len(e) or not len(n):
        return PairwiseSupportDiagnostics(
            0.0, 0.0, 0.0, 0.0, 0, len(e_all), len(n_all), len(e), len(n)
        )
    compatible = compatibility_matrix(percentiles[e], percentiles[n], context)
    e_has = np.any(compatible, axis=1)
    n_has = np.any(compatible, axis=0)
    exposed_support = float(np.sum(raw[e][e_has]) / e_total)
    nonexposed_support = float(np.sum(raw[n][n_has]) / n_total)
    pair_weight = float(np.dot(raw[e], compatible.astype(float) @ raw[n]) / (e_total * n_total))
    return PairwiseSupportDiagnostics(
        exposed_support=exposed_support,
        nonexposed_support=nonexposed_support,
        minimum_support=min(exposed_support, nonexposed_support),
        normalized_pair_weight=pair_weight,
        compatible_pair_count=int(np.sum(compatible)),
        target_exposed_count=len(e_all),
        target_nonexposed_count=len(n_all),
        candidate_valid_exposed_count=len(e),
        candidate_valid_nonexposed_count=len(n),
    )


def prepare_adjusted_delta(values: np.ndarray, raw: np.ndarray, labels: np.ndarray,
                           percentiles: np.ndarray, context: ConfounderCompatibilityContext):
    """CA10-CA11: freeze one BSA/year's compatibility and sign matrices."""
    values = np.asarray(values, dtype=float)
    raw = np.asarray(raw, dtype=float)
    labels = np.asarray(labels)
    percentiles = np.asarray(percentiles, dtype=float)
    _validate_inputs(values, raw, labels, percentiles)
    if context.neutral_compatibility:
        return ("neutral", nm.prepare_weighted_delta(values, raw, labels))
    e_all, n_all = _target_indices(values, raw, labels)
    e = e_all[np.isfinite(percentiles[e_all])]
    n = n_all[np.isfinite(percentiles[n_all])]
    if not len(e) or not len(n):
        return None
    compatible = compatibility_matrix(percentiles[e], percentiles[n], context)
    if not np.any(compatible):
        return None

    pair_keep = compatible.astype(float)
    pair_sign = np.sign(values[e, None] - values[n][None, :]) * pair_keep
    return ("adjusted", e, n, raw[e], raw[n], pair_keep, pair_sign)


def adjusted_weighted_delta_batch(
    values: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    percentiles: np.ndarray,
    counts: np.ndarray,
    context: ConfounderCompatibilityContext,
    *, _prepared=None,
) -> np.ndarray:
    """CA11: unchanged compatibility-restricted weighted Cliff's Delta.

    Only invariant matrix construction is reused between bootstrap batches;
    matrix multiplication and reduction order remain unchanged.
    """
    values = np.asarray(values, dtype=float)
    raw = np.asarray(raw, dtype=float)
    labels = np.asarray(labels)
    percentiles = np.asarray(percentiles, dtype=float)
    counts = np.asarray(counts)
    _validate_inputs(values, raw, labels, percentiles, counts)
    plan = (prepare_adjusted_delta(values, raw, labels, percentiles, context)
            if _prepared is None else _prepared[0])
    if plan is None:
        return np.full(len(counts), np.nan)
    if plan[0] == "neutral":
        return nm.weighted_delta_prepared(plan[1], counts)
    _, e, n, e_raw, n_raw, pair_keep, pair_sign = plan
    e_weight = counts[:, e] * e_raw[None, :]
    n_weight = counts[:, n] * n_raw[None, :]
    denominator = np.sum((e_weight @ pair_keep) * n_weight, axis=1)
    numerator = np.sum((e_weight @ pair_sign) * n_weight, axis=1)
    result = np.full(len(counts), np.nan)
    ok = denominator > 0.0
    result[ok] = np.clip(numerator[ok] / denominator[ok], -1.0, 1.0)
    return result


def adjusted_point_result(
    values: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    percentiles: np.ndarray,
    context: ConfounderCompatibilityContext,
) -> AdjustedPairwisePointResult:
    support = pairwise_support_diagnostics(values, raw, labels, percentiles, context)
    counts = np.ones((1, len(values)), dtype=np.uint16)
    rc = float(adjusted_weighted_delta_batch(values, raw, labels, percentiles, counts, context)[0])
    rc_value = rc if isfinite(rc) else None
    if support.normalized_pair_weight <= 0.0 or rc_value is None:
        reason = ConfounderExclusionReason.NO_COMPATIBLE_PAIR.value
        eligible = False
    elif support.minimum_support + _NUMERICAL_TOLERANCE < float(context.minimum_retained_support):
        reason = ConfounderExclusionReason.HELD_OUT_SUPPORT_BELOW_MINIMUM.value
        eligible = False
    else:
        reason = None
        eligible = True
    return AdjustedPairwisePointResult(rc_value, eligible, reason, support)


def _project_array(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfounderEstimatorError("candidate percentile path escapes project root") from exc
    if not path.is_file():
        raise ConfounderEstimatorError(f"candidate percentile array is unavailable: {path}")
    return path


def _prepare_bsa_target_samples(data, config: dict, b: int, q: float, lag: int,
                                common_labels: np.ndarray | None):
    """Reconstruct the ordinary target-hazard original sample for one BSA.

    CA6 closes the CA5 compatibility gap for BSA_SPECIFIC exposure mode by
    deriving the local exposure calendar when the reference realization uses
    BSA-specific classification.
    """
    support_policy = active_support_policy(config)
    source = data.series("outcome", b)
    incidence = nm.cumulative_incidence(
        source, data.daily_population(b), int(lag),
        config["core"]["incidence"]["scaling_multiplier"], "EVENT_COUNTS", data.segment_index,
    )
    local = data.series("hazard", b)
    exposure_mode = config["core"]["exposure"].get("mode", "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE")
    if exposure_mode == "BSA_SPECIFIC":
        labels_b, _ = nm.exposure(local, data.year_index, float(q))
    elif exposure_mode == "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE":
        if common_labels is None:
            raise ConfounderEstimatorError("territory-wide target exposure requires common labels")
        labels_b = common_labels
    else:
        raise ConfounderEstimatorError(f"unsupported exposure mode: {exposure_mode}")
    baseline, _, donors = nm.baseline(
        incidence, labels_b, config["core"]["baseline"], data.segment_index,
        data.weekday_index, data.holiday_state,
    )
    differential, _ = apply_baseline_support_filter(
        incidence - baseline, donors, support_policy.get("min_baseline_support")
    )
    weight, _ = nm.relevance(local, labels_b, data.year_index, float(q))
    return differential, weight, labels_b


class ConfounderPointEstimator:
    """Apply frozen CA4 contexts to held-out years at the reference q/lag.

    CA5 intentionally stops at original-sample adjusted RCs. Bootstrap
    reconstruction, variance, pooling and WMARMcs are added in CA6.
    """

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

    def run(self, representative: Mapping | None) -> dict:
        if representative is None:
            return {
                "status": "UNAVAILABLE",
                "stage": POINT_ESTIMATION_STAGE,
                "reason": ConfounderExclusionReason.REFERENCE_CONFIGURATION_UNAVAILABLE.value,
                "adjusted_estimator_enabled": True,
                "adjusted_bootstrap_enabled": False,
                "adjusted_pooling_enabled": False,
                "point_rows": 0,
                "eligible_point_rows": 0,
            }
        q, lag = float(representative["q"]), int(representative["lag"])
        contexts = self._contexts()
        if not contexts:
            return {
                "status": "COMPLETED_WITH_EXCLUSIONS",
                "stage": POINT_ESTIMATION_STAGE,
                "reason": "NO_AVAILABLE_COMPATIBILITY_CONTEXTS",
                "reference_representative": {"q": q, "lag": lag},
                "adjusted_estimator_enabled": True,
                "adjusted_bootstrap_enabled": False,
                "adjusted_pooling_enabled": False,
                "point_rows": 0,
                "eligible_point_rows": 0,
            }

        percentile_path = _project_array(self.root, self.preprocessing.percentile_panel.array_path)
        exposure_mode = self.c["core"]["exposure"].get("mode", "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE")
        if exposure_mode == "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE":
            territory = self.data.series("territory")
            labels, _ = nm.exposure(territory, self.data.year_index, q)
        elif exposure_mode == "BSA_SPECIFIC":
            labels = None
        else:
            raise ConfounderEstimatorError(f"unsupported exposure mode: {exposure_mode}")
        total_rows = eligible_rows = mismatch_count = 0
        with mapped(percentile_path) as percentiles:
            if percentiles.shape != (self.data.B, self.data.n):
                raise ConfounderEstimatorError("candidate percentile panel shape is inconsistent")
            for b, bid in enumerate(self.data.ids):
                differential, raw, labels_b = _prepare_bsa_target_samples(
                    self.data, self.c, b, q, lag, labels
                )
                for year in self.data.years:
                    context = contexts.get(int(year))
                    if context is None:
                        continue
                    cp_key = f"{int(year)}/{bid}"
                    existing = self.persistence.checkpoint(
                        self.analysis_run_id, POINT_ESTIMATION_STAGE, "bsa_year", cp_key
                    )
                    if existing is not None and existing["state"] == ConfounderCheckpointState.COMPLETED.value:
                        total_rows += 1
                        payload = existing.get("payload") or {}
                        eligible_rows += int(bool(payload.get("adjusted_point_eligible")))
                        mismatch_count += int(bool(payload.get("reference_point_mismatch")))
                        continue
                    iy = self.data.years.index(int(year))
                    ix = self.data.year_index == iy
                    x = np.asarray(differential[ix], dtype=float)
                    w = np.asarray(raw[ix], dtype=float)
                    e = np.asarray(labels_b[ix], dtype=np.int8)
                    u = np.asarray(percentiles[b, ix], dtype=float)
                    ones = np.ones((1, len(x)), dtype=np.uint16)
                    unadjusted = float(self.engine.pairwise_delta_batch(x, w, e, ones)[0])
                    result = self.engine.adjusted_pairwise_point(x, w, e, u, context)
                    reference = self.store.get("yearly", cell_scope("REFERENCE", q, lag), f"{bid}/{int(year)}")
                    reference_rc = None if reference is None else reference.get("rc")
                    mismatch = False
                    if reference_rc is not None and np.isfinite(unadjusted):
                        mismatch = abs(float(reference_rc) - unadjusted) > 1e-12
                    row = {
                        "analysis_run_id": self.analysis_run_id,
                        "candidate_id": context.candidate_id,
                        "bsa_id": bid,
                        "year": int(year),
                        "q": q,
                        "lag": lag,
                        "context_status": context.calibration_status.value,
                        "neutral_compatibility": context.neutral_compatibility,
                        "selected_caliper": context.selected_caliper,
                        "relevance_amplitude": context.relevance_amplitude,
                        "minimum_retained_support": context.minimum_retained_support,
                        "unadjusted_point_rc": unadjusted if np.isfinite(unadjusted) else None,
                        "reference_yearly_rc": reference_rc,
                        "reference_point_mismatch": mismatch,
                        **result.to_dict(),
                    }
                    row["adjustment_shift_point"] = (
                        None if row["adjusted_rc"] is None or row["unadjusted_point_rc"] is None
                        else float(row["adjusted_rc"] - row["unadjusted_point_rc"])
                    )
                    self.store.put("confounder_adjusted_point", self.analysis_run_id, cp_key, row)
                    self.store.commit()
                    self.persistence.put_checkpoint(
                        self.analysis_run_id, POINT_ESTIMATION_STAGE, "bsa_year", cp_key,
                        ConfounderCheckpointState.COMPLETED, payload=row,
                    )
                    total_rows += 1
                    eligible_rows += int(result.eligible)
                    mismatch_count += int(mismatch)

        if mismatch_count:
            raise ConfounderEstimatorError(
                "CA5 standard-kernel integration diverged from persisted reference yearly RCs"
            )
        summary = {
            "status": "COMPLETED" if eligible_rows == total_rows else "COMPLETED_WITH_EXCLUSIONS",
            "stage": POINT_ESTIMATION_STAGE,
            "reference_representative": {"q": q, "lag": lag},
            "adjusted_estimator_enabled": True,
            "adjusted_bootstrap_enabled": False,
            "adjusted_pooling_enabled": False,
            "point_rows": total_rows,
            "eligible_point_rows": eligible_rows,
            "excluded_point_rows": total_rows - eligible_rows,
            "reference_point_mismatch_count": mismatch_count,
        }
        self.store.put("workflow", self.analysis_run_id, "point_estimation_result", summary)
        self.store.commit()
        self.log(
            f"Confounder CA5: adjusted point estimator q={q:g}; lag={lag}; "
            f"eligible={eligible_rows}/{total_rows}"
        )
        return summary

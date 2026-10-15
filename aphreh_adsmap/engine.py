"""One complete fixed-parameter estimator; the surface runner only repeats it.

The resident numerical working set uses explicitly budgeted arrays and a bounded
bootstrap batch. Disposable differential/weight panels and bootstrap counts use
RAM when admitted by the shared budget, otherwise explicitly closed memmaps.
Persistent artifacts retain the original formats and checkpoint semantics.
"""
from __future__ import annotations
import hashlib
import math
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from .randomness import rng_for_identity
from .data import Data
from .storage import Store, mapped
from .performance import memory_budget, working_array, file_signature
from . import numerics as nm
from .support import (
    active_support_policy, apply_baseline_support_filter, evaluate_prebootstrap_support,
    evaluate_postbootstrap_support, enforce_pooled_support,
)


class Cancelled(RuntimeError):
    pass


def cell_key(q: float, lag: int) -> str:
    return f"q{q:g}_lag{lag:04d}"


def cell_scope(run: str, q: float, lag: int) -> str:
    return f"{run}/{cell_key(q,lag)}"


def finite_mean(values) -> float | None:
    a = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(a)) if a else None


def weighted_mean(values: list[float | None], weights: list[float]) -> float | None:
    pairs = [(v,w) for v,w in zip(values,weights) if v is not None and math.isfinite(v) and math.isfinite(w) and w>0]
    return float(sum(v*w for v,w in pairs)/sum(w for _,w in pairs)) if pairs else None


_COVERAGE_ROUNDOFF_TOL = 1e-12


def bounded_coverage_ratio(numerator: float, denominator: float) -> float | None:
    """Return a coverage ratio with machine-roundoff correction at 0 and 1.

    This is a numerical guard only. Values outside [0, 1] by more than the
    tolerance are deliberately left unchanged so downstream validation still
    detects genuine coverage/accounting errors.
    """
    if denominator <= 0:
        return None
    value = float(numerator / denominator)
    if -_COVERAGE_ROUNDOFF_TOL <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + _COVERAGE_ROUNDOFF_TOL:
        return 1.0
    return value


def pooled_ci_half_width(ci_low: float | None, ci_high: float | None) -> float | None:
    """MF4.5: half-width of a valid pooled confidence interval."""
    if ci_low is None or ci_high is None:
        return None
    lo, hi = float(ci_low), float(ci_high)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        return None
    return (hi - lo) / 2.0


def pooled_ci_excludes_zero(ci_low: float | None, ci_high: float | None) -> bool | None:
    """MF6.4.1: separate CI-support diagnostic, independent of WMARMcs scaling."""
    if ci_low is None or ci_high is None:
        return None
    lo, hi = float(ci_low), float(ci_high)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        return None
    return bool(lo > 0.0 or hi < 0.0)


def uncertainty_attenuation_factor(
    pooled_rc: float | None, ci_low: float | None, ci_high: float | None
) -> float | None:
    """MF4.5: continuous relative-uncertainty confidence scaling factor C_b.

    C_b = RC_b^2 / (RC_b^2 + H_b^2), where H_b is the pooled confidence
    interval half-width. If RC_b = H_b = 0, C_b is zero by convention.
    """
    if pooled_rc is None:
        return None
    rc = float(pooled_rc)
    if not math.isfinite(rc):
        return None
    half = pooled_ci_half_width(ci_low, ci_high)
    if half is None:
        return None
    rc2, h2 = rc * rc, half * half
    denominator = rc2 + h2
    if denominator == 0.0:
        return 0.0
    return min(1.0, max(0.0, rc2 / denominator))


def confidence_scaled_magnitude(
    pooled_rc: float | None, ci_low: float | None, ci_high: float | None
) -> float | None:
    """MF4.5/MF5.9: confidence-scaled unsigned pooled effect magnitude M_b,CS.

    The contribution is |RC_b| multiplied by the continuous relative-uncertainty
    attenuation factor. A confidence interval crossing zero does not force the
    contribution to zero. Missing or invalid pooled estimates/intervals remain
    ineligible.
    """
    factor = uncertainty_attenuation_factor(pooled_rc, ci_low, ci_high)
    if factor is None:
        return None
    rc = float(pooled_rc)
    return abs(rc) * factor


def confidence_supported_magnitude(
    pooled_rc: float | None, ci_low: float | None, ci_high: float | None
) -> float | None:
    """Backward-compatible alias for the revised confidence-scaled magnitude."""
    return confidence_scaled_magnitude(pooled_rc, ci_low, ci_high)


class FixedParametersEngine:
    def __init__(self, data: Data, config: dict, store: Store, root: Path, log=print,
                 cancellation_root: Path | None = None):
        self.data,self.c,self.store,self.root,self.log = data,config,store,root,log
        # Worker sandboxes may live outside the project directory.  Cancellation
        # remains project-scoped so one GUI CANCEL marker stops every process.
        self.cancellation_root = Path(cancellation_root) if cancellation_root is not None else root
        self.scratch = root/"scratch"
        self.scratch.mkdir(parents=True,exist_ok=True)
        self._performance_token = object()

    def check(self) -> None:
        if (self.cancellation_root/"CANCEL").exists():
            raise Cancelled("Cancellation requested; committed cells are resumable")
        memory_budget().check()
        free = shutil.disk_usage(self.root).free
        if free < self.c["runtime"]["minimum_free_disk_mb"]*1024**2:
            raise OSError("Free disk space below the configured safety threshold")

    def pairwise_delta_batch(self, values: np.ndarray, raw: np.ndarray, labels: np.ndarray,
                             counts: np.ndarray, *, candidate_percentiles: np.ndarray | None = None,
                             compatibility_context=None, _prepared=None) -> np.ndarray:
        """MF3.1/CA11: one authoritative pairwise-estimator dispatch point.

        The ordinary reference path remains byte-for-byte equivalent at the
        numerical-kernel level. A frozen confounder context activates the CA5
        compatibility-restricted kernel. Bootstrap wiring for that adjusted
        branch is intentionally deferred to CA6.
        """
        if compatibility_context is None:
            if candidate_percentiles is not None:
                raise ValueError("candidate_percentiles require a compatibility_context")
            if _prepared is not None:
                return nm.weighted_delta_prepared(_prepared[0], counts)
            return nm.weighted_delta_batch(values, raw, labels, counts)
        if candidate_percentiles is None:
            raise ValueError("compatibility_context requires candidate_percentiles")
        from .confounder.estimator import adjusted_weighted_delta_batch
        return adjusted_weighted_delta_batch(
            values, raw, labels, candidate_percentiles, counts, compatibility_context,
            _prepared=_prepared,
        )

    def adjusted_pairwise_point(self, values: np.ndarray, raw: np.ndarray, labels: np.ndarray,
                                candidate_percentiles: np.ndarray, compatibility_context):
        """CA10-CA11: held-out original-sample adjusted RC and support."""
        from .confounder.estimator import adjusted_point_result
        return adjusted_point_result(
            values, raw, labels, candidate_percentiles, compatibility_context
        )

    def run(self, run: str, q: float, lag: int, *, outcome_path: Path | None = None,
            outcome_semantics: str = "EVENT_COUNTS", permutation: np.ndarray | None = None,
            loyo: bool = True, retain: bool = True, rng_identity_run: str | None = None,
            prebaseline_cache: Path | None = None,
            bsa_year_eligibility_path: Path | None = None,
            candidate_percentile_path: Path | None = None,
            compatibility_contexts: dict[int, object] | None = None,
            compatibility_exclusion_reasons: dict[int, str] | None = None) -> dict:
        """Execute MF1-5 once. Every workflow enters through this same method.

        CA6 optionally activates candidate-confounder-adjusted inference through a
        disk-backed candidate-percentile panel and frozen year-specific
        compatibility contexts. The ordinary reference path is unchanged when
        those arguments are absent.
        """
        previous = self.store.get("cell",run,cell_key(q,lag))
        if previous is not None:
            return previous
        self.check()
        start = time.perf_counter()
        d,c = self.data,self.c
        support_policy = active_support_policy(c)
        adjusted_confounder = compatibility_contexts is not None or candidate_percentile_path is not None
        if adjusted_confounder:
            if compatibility_contexts is None or candidate_percentile_path is None:
                raise ValueError("adjusted confounder inference requires both candidate_percentile_path and compatibility_contexts")
            compatibility_exclusion_reasons = compatibility_exclusion_reasons or {}
        elif compatibility_exclusion_reasons:
            raise ValueError("compatibility_exclusion_reasons require adjusted confounder inference")
        exposure_mode = c["core"]["exposure"].get("mode", "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE")
        local_exposure = exposure_mode == "BSA_SPECIFIC"
        if exposure_mode not in ("TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE", "BSA_SPECIFIC"):
            raise ValueError(f"Unsupported exposure mode: {exposure_mode}")
        if local_exposure and support_policy.get("min_local_concordance") is not None:
            raise ValueError("min_local_concordance is not applicable under BSA_SPECIFIC exposure classification")
        if local_exposure and (outcome_path is not None or outcome_semantics != "EVENT_COUNTS" or prebaseline_cache is not None):
            raise ValueError("BSA_SPECIFIC exposure supports observed EVENT_COUNTS analyses and synchronized hazard permutations only")
        # Only hazard-derived terms are reused across outcomes/lags. A changed
        # permutation or canonical source file gets a distinct cache identity.
        budget = memory_budget()
        hazard_identity = (
            self._performance_token, float(q), exposure_mode,
            file_signature(d.root / "hazard.npy"),
            file_signature(d.root / "territory.npy"),
            None if permutation is None else hashlib.sha256(np.asarray(permutation).tobytes()).digest(),
        )
        synthetic_bsa_year_eligibility = None
        if bsa_year_eligibility_path is not None:
            if outcome_path is None or outcome_semantics != "DAILY_INCIDENCE":
                raise ValueError("BSA-year eligibility masks are supported only for synthetic DAILY_INCIDENCE outcomes")
            synthetic_bsa_year_eligibility = np.load(Path(bsa_year_eligibility_path), allow_pickle=False)
            if synthetic_bsa_year_eligibility.shape != (d.B, len(d.years)):
                raise ValueError(
                    "BSA-year eligibility mask shape does not match canonical BSA/year dimensions: "
                    f"{synthetic_bsa_year_eligibility.shape} != {(d.B, len(d.years))}"
                )
            synthetic_bsa_year_eligibility = np.asarray(synthetic_bsa_year_eligibility, dtype=bool)
        scope = cell_scope(run,q,lag)
        folder = self.scratch/hashlib.sha256(scope.encode()).hexdigest()[:24]
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        self.store.delete_scope(scope)  # stale, unmarked partial work is replaced
        if local_exposure:
            labels = None
            territory_threshold = np.full(len(d.years), np.nan)
        elif prebaseline_cache is None:
            cached_exposure = budget.get(("territory_exposure", hazard_identity))
            if cached_exposure is None:
                territory = d.series("territory")
                if permutation is not None:
                    territory = territory[permutation]
                labels, territory_threshold = nm.exposure(territory,d.year_index,q)
                budget.put(("territory_exposure", hazard_identity), (labels, territory_threshold))
                del territory
            else:
                labels, territory_threshold = cached_exposure
        else:
            if permutation is not None or outcome_path is not None or outcome_semantics != "EVENT_COUNTS":
                raise ValueError("prebaseline_cache is available only for unpermuted observed EVENT_COUNTS analyses")
            labels = np.load(prebaseline_cache/"labels.npy", allow_pickle=False)
            territory_threshold = np.load(prebaseline_cache/"territory_threshold.npy", allow_pickle=False)
        K = c["core"]["dependence"]["maximum_acf_lag_days"]
        hist = np.zeros((len(d.years),K+1),dtype=np.int64)
        territory_diff = np.zeros(d.n)
        territory_complete = np.ones(d.n,dtype=bool)
        territory_contributors = np.zeros(d.n,dtype=np.int32)
        R = c["core"]["bootstrap"]["iterations"]
        try:
            with ExitStack() as stack:
                diffs = stack.enter_context(working_array(folder/"differentials.npy",shape=(d.B,d.n)))
                raw = stack.enter_context(working_array(folder/"raw_weights.npy",shape=(d.B,d.n)))
                local_labels = stack.enter_context(working_array(folder/"local_labels.npy",shape=(d.B,d.n),dtype="int8")) if local_exposure else None
                candidate_percentiles = stack.enter_context(mapped(candidate_percentile_path)) if adjusted_confounder else None
                if adjusted_confounder and candidate_percentiles.shape != (d.B, d.n):
                    raise ValueError("candidate percentile panel shape does not match canonical BSA/date dimensions")
                cached_incidence = cached_weights = cached_event_windows = cached_thresholds = None
                if prebaseline_cache is not None:
                    cached_incidence = stack.enter_context(mapped(prebaseline_cache/"incidence.npy"))
                    cached_weights = stack.enter_context(mapped(prebaseline_cache/"raw_weights.npy"))
                    cached_event_windows = stack.enter_context(mapped(prebaseline_cache/"event_windows.npy"))
                    cached_thresholds = stack.enter_context(mapped(prebaseline_cache/"local_thresholds.npy"))
                # Small BSA x year threshold table, rather than a date-wise object graph.
                local_thresholds = np.full((d.B,len(d.years)),np.nan)
                for b,bid in enumerate(d.ids):
                    self.check()
                    if prebaseline_cache is None:
                        source = d.series("outcome",b,path=outcome_path)
                        inc = nm.cumulative_incidence(source,d.daily_population(b),lag,
                                                      c["core"]["incidence"]["scaling_multiplier"],outcome_semantics,
                                                      d.segment_index)
                        event_windows = (
                            nm.cumulative_incidence(source,np.ones(d.n,dtype=float),lag,1.0,"EVENT_COUNTS",d.segment_index)
                            if outcome_semantics == "EVENT_COUNTS" else None
                        )
                    else:
                        inc = np.array(cached_incidence[b])
                        event_windows = np.array(cached_event_windows[b])
                    hazard_key = ("local_hazard_terms", hazard_identity, b)
                    hazard_terms = budget.get(hazard_key) if prebaseline_cache is None else None
                    if hazard_terms is None:
                        local = d.series("hazard",b)
                        if permutation is not None:
                            local = local[permutation]
                        if local_exposure:
                            labels_b, thresholds = nm.exposure(local,d.year_index,q)
                        else:
                            labels_b = labels
                            thresholds = None
                    else:
                        local, w, thresholds, cached_labels = hazard_terms
                        labels_b = cached_labels if local_exposure else labels
                    if local_exposure:
                        local_labels[b] = labels_b
                    base,widths,donors = nm.baseline(inc,labels_b,c["core"]["baseline"],d.segment_index,d.weekday_index,d.holiday_state)
                    x_unfiltered = inc-base
                    baseline_min = support_policy.get("min_baseline_support")
                    baseline_support_excluded = (
                        np.isfinite(x_unfiltered) & (donors < int(baseline_min))
                        if baseline_min is not None else np.zeros(d.n,dtype=bool)
                    )
                    x, _ = apply_baseline_support_filter(x_unfiltered,donors,baseline_min)
                    if support_policy.get("min_event_contribution") is not None and event_windows is None:
                        raise ValueError("min_event_contribution support is defined only for observed EVENT_COUNTS outcomes")
                    if hazard_terms is not None:
                        pass
                    elif local_exposure:
                        w, thresholds_check = nm.relevance(local,labels_b,d.year_index,q)
                        if not np.allclose(thresholds, thresholds_check, equal_nan=True):
                            raise ArithmeticError("Local exposure and relevance thresholds diverged")
                    elif prebaseline_cache is None:
                        w,thresholds = nm.relevance(local,labels_b,d.year_index,q)
                    else:
                        w = np.array(cached_weights[b])
                        thresholds = np.array(cached_thresholds[b])
                    if hazard_terms is None and prebaseline_cache is None:
                        budget.put(hazard_key, (local, w, thresholds, labels_b if local_exposure else None))
                    diffs[b],raw[b],local_thresholds[b] = x,w,thresholds
                    # CM5.2 / CM24: a BSA-year whose percentage alteration is
                    # undefined is excluded from synthetic calibration rather than
                    # poisoning the whole-territory dependence diagnostic. For all
                    # participating BSA-years the original complete-row rule is
                    # retained exactly, so ordinary observed analyses are unchanged.
                    if synthetic_bsa_year_eligibility is None:
                        territory_active = np.ones(d.n,dtype=bool)
                    else:
                        territory_active = synthetic_bsa_year_eligibility[b,d.year_index]
                    finite_x = np.isfinite(x)
                    territory_complete[territory_active] &= finite_x[territory_active]
                    territory_contributors += territory_active.astype(np.int32)
                    territory_diff += np.where(territory_active & finite_x,x,0.0)
                    radius = nm.dependence(x,c["core"]["dependence"],d.segment_index)
                    for iy,year in enumerate(d.years):
                        ix = d.year_index==iy
                        r = radius[ix]
                        r = r[np.isfinite(r)].astype(int)
                        hist[iy] += np.bincount(r,minlength=K+1)
                        ex,non = ix&(labels_b==1),ix&(labels_b==0)
                        xe,xn = ex&np.isfinite(x),non&np.isfinite(x)
                        def ess(mask):
                            ww = w[mask]
                            return float(ww.sum()**2/np.dot(ww,ww)) if np.dot(ww,ww)>0 else None
                        valid_target = ix & np.isfinite(x) & (labels_b>=0)
                        event_contribution = (
                            float(np.sum(event_windows[valid_target]))
                            if event_windows is not None and np.any(valid_target) else
                            (0.0 if event_windows is not None else None)
                        )
                        info = {"run":run,"q":q,"lag":lag,"bsa_id":bid,"year":year,
                                "population":float(d.population[b,iy]),"population_source_year":int(d.population_source_year[b,iy]) if d.population_source_year[b,iy]>=0 else None,"threshold_local":float(thresholds[iy]),
                                "threshold_territory":None if local_exposure else float(territory_threshold[iy]),
                                "exposure_mode":exposure_mode,
                                "exposure_calendar_hash":hashlib.sha256(np.asarray(labels_b,dtype=np.int8).tobytes()).hexdigest()[:16],
                                "n_exposed":int(np.sum(xe)),"n_nonexposed":int(np.sum(xn)),
                                "n_exposed_positive_weight":int(np.sum(xe&(w>0))),
                                "n_nonexposed_positive_weight":int(np.sum(xn&(w>0))),
                                "effective_exposed":ess(xe),"effective_nonexposed":ess(xn),
                                "event_contribution":event_contribution,
                                "concordance_exposed":None if local_exposure else (float(np.mean(local[ex]>=thresholds[iy])) if np.any(ex) and np.isfinite(thresholds[iy]) else None),
                                "concordance_nonexposed":None if local_exposure else (float(np.mean(local[non]<thresholds[iy])) if np.any(non) and np.isfinite(thresholds[iy]) else None),
                                "missing_incidence_days":int(np.sum(ix&~np.isfinite(inc))),
                                "unavailable_baseline_days":int(np.sum(ix&np.isfinite(inc)&~np.isfinite(base))),
                                "baseline_support_excluded_days":int(np.sum(ix&baseline_support_excluded)),
                                "missing_local_hazard_days":int(np.sum(ix&~np.isfinite(local))),
                                "zero_weight_valid_days":int(np.sum(ix&np.isfinite(x)&(w==0))),
                                "baseline_donor_count_mean":finite_mean(donors[ix&np.isfinite(x)]),
                                "baseline_half_width_mean":finite_mean(widths[ix&np.isfinite(x)]),
                                "baseline_half_width_max":int(np.max(widths[ix])) if np.any(ix&np.isfinite(x)) else None,
                                "baseline_policy":dict(c["core"]["baseline"]),
                                "support_policy":support_policy}
                        support_failures_prebootstrap = evaluate_prebootstrap_support(info,support_policy)
                        if adjusted_confounder:
                            context = compatibility_contexts.get(int(year))
                            info["confounder_adjusted"] = True
                            info["confounder_context_available"] = context is not None
                            if context is None:
                                reason = compatibility_exclusion_reasons.get(int(year), "CA17_09_NO_ADMISSIBLE_CALIPER")
                                info.update({
                                    "adjusted_point_rc": None,
                                    "adjusted_point_eligible": False,
                                    "adjusted_point_exclusion_reason": reason,
                                    "retained_support_exposed": None,
                                    "retained_support_nonexposed": None,
                                    "retained_support": None,
                                    "normalized_compatible_pair_weight": None,
                                    "compatible_pair_count": 0,
                                })
                                support_failures_prebootstrap.append(reason)
                            else:
                                u = np.asarray(candidate_percentiles[b, ix], dtype=float)
                                adjusted_point = self.adjusted_pairwise_point(
                                    np.asarray(x[ix], dtype=float), np.asarray(w[ix], dtype=float),
                                    np.asarray(labels_b[ix], dtype=np.int8), u, context
                                )
                                info.update(adjusted_point.to_dict())
                                info["confounder_calibration_status"] = context.calibration_status.value
                                info["confounder_selected_caliper"] = context.selected_caliper
                                info["confounder_relevance_amplitude"] = context.relevance_amplitude
                                info["confounder_neutral_compatibility"] = context.neutral_compatibility
                                if not adjusted_point.eligible and adjusted_point.exclusion_reason:
                                    support_failures_prebootstrap.append(adjusted_point.exclusion_reason)
                        info["support_failures_prebootstrap"] = support_failures_prebootstrap
                        self.store.put("support",scope,f"{bid}/{year}",info)
                    if retain and c["runtime"]["retain_daily_intermediates"]:
                        target = self.root/"intermediates"/hashlib.sha256(scope.encode()).hexdigest()[:24]
                        target.mkdir(parents=True,exist_ok=True)
                        # One BSA per compressed archive; never a whole panel dictionary.
                        np.savez_compressed(target/f"bsa_{bid}.npz",incidence=inc,baseline=base,differential=x,
                                            raw_weight=w,labels=labels_b,half_width=widths,donors=donors,radius=radius)
                territory_complete &= territory_contributors > 0
                territory_diff[~territory_complete] = np.nan
                territory_radius = nm.dependence(territory_diff,c["core"]["dependence"],d.segment_index)
                for iy,year in enumerate(d.years):
                    self.check()
                    indices = np.flatnonzero(d.year_index==iy)
                    annual_segments = d.segment_index[indices]
                    Dloc = nm.histogram_quantile(hist[iy],c["core"]["dependence"]["annual_dependence_quantile"])
                    Dtot = nm.finite_quantile(territory_radius[indices],c["core"]["dependence"]["annual_dependence_quantile"])
                    if local_exposure:
                        eligible_local_labels = []
                        for b,bid in enumerate(d.ids):
                            labels_b_year = np.array(local_labels[b,indices],dtype=np.int8)
                            x0,w0 = np.array(diffs[b,indices]),np.array(raw[b,indices])
                            original0 = float(self.pairwise_delta_batch(x0,w0,labels_b_year,np.ones((1,len(indices)),dtype=np.uint16))[0])
                            support0 = self.store.get("support",scope,f"{bid}/{year}") or {}
                            if np.isfinite(original0) and not list(support0.get("support_failures_prebootstrap") or []):
                                eligible_local_labels.append(labels_b_year)
                        Bexp = nm.pooled_episode_requirement(eligible_local_labels,c["core"]["dependence"]["exposed_episode_quantile"],annual_segments)
                        annual_labels = None
                    else:
                        annual_labels = labels[indices]
                        Bexp = nm.episode_requirement(annual_labels,c["core"]["dependence"]["exposed_episode_quantile"],annual_segments)
                    raw_block, block = nm.block_length_requirement(
                        lag, Dloc, Dtot, Bexp, c["core"]["dependence"].get("final_block_length_multiplier", 1.0)
                    )
                    block_reason = None
                    if block is None:
                        block_reason = "DEPENDENCE_UNAVAILABLE"
                    elif len(indices)<=block:
                        block_reason = "RECORD_NOT_LONGER_THAN_BLOCK"
                    elif nm.max_contiguous_length(annual_segments)<=block:
                        block_reason = "CONTIGUOUS_RECORD_NOT_LONGER_THAN_BLOCK"
                    elif not local_exposure and (not np.any(annual_labels==1) or not np.any(annual_labels==0)):
                        block_reason = "ANNUAL_EXPOSURE_CLASS_ABSENT"
                    bootfile = folder/f"draw_counts_{year}.npy"
                    global_redraws = 0
                    with ExitStack() as yearly_stack:
                        counts = None
                        if block_reason is None:
                            counts = yearly_stack.enter_context(working_array(bootfile,shape=(R,len(indices)),dtype="uint16"))
                            rng = rng_for_identity(c["seed"],namespace="DISK_DEV_BOOTSTRAP_V1",
                                                   identity={"run":rng_identity_run or run,"q":float(q),"lag":lag,"year":year})
                            global_redraws = (
                                nm.draw_synchronized_counts(len(indices),block,rng,counts,annual_segments)
                                if local_exposure else nm.draw_counts(annual_labels,block,rng,counts,annual_segments)
                            )
                        for b,bid in enumerate(d.ids):
                            self.check()
                            labels_b_year = np.array(local_labels[b,indices],dtype=np.int8) if local_exposure else annual_labels
                            x,w = np.array(diffs[b,indices]),np.array(raw[b,indices])
                            context = compatibility_contexts.get(int(year)) if adjusted_confounder else None
                            candidate_year = (
                                np.asarray(candidate_percentiles[b, indices], dtype=float)
                                if adjusted_confounder and context is not None else None
                            )
                            prepared_pairwise = None
                            if not adjusted_confounder:
                                prepared_pairwise = (nm.prepare_weighted_delta(x, w, labels_b_year),)
                            elif context is not None:
                                from .confounder.estimator import prepare_adjusted_delta
                                prepared_pairwise = (prepare_adjusted_delta(x, w, labels_b_year, candidate_year, context),)
                            if adjusted_confounder and context is None:
                                original = float("nan")
                            else:
                                original = float(self.pairwise_delta_batch(
                                    x, w, labels_b_year, np.ones((1,len(indices)),dtype=np.uint16),
                                    candidate_percentiles=candidate_year, compatibility_context=context,
                                    _prepared=prepared_pairwise,
                                )[0])
                            drawfile = folder/f"rc_{year}_{b}.npy"
                            with working_array(drawfile,shape=(R,)) as draws:
                                draws[:] = np.nan
                                if counts is not None and np.isfinite(original):
                                    batch = c["runtime"]["bootstrap_batch_size"]
                                    for r in range(0,R,batch):
                                        self.check()
                                        draws[r:r+batch] = self.pairwise_delta_batch(
                                            x, w, labels_b_year, counts[r:r+batch],
                                            candidate_percentiles=candidate_year, compatibility_context=context,
                                            _prepared=prepared_pairwise,
                                        )
                                summary = nm.bootstrap_summary(original,draws,R,c["core"]["bootstrap"]["maximum_invalid_realization_proportion"])
                                if adjusted_confounder:
                                    invalid_adjusted = int(np.sum(~np.isfinite(draws)))
                                    summary["adjusted_bootstrap_empty_pair_realizations"] = invalid_adjusted
                                    summary["adjusted_bootstrap_invalid_reason"] = (
                                        "CA17_12_ADJUSTED_BOOTSTRAP_PAIR_SUPPORT_EMPTY" if invalid_adjusted else None
                                    )
                                    if summary.get("reason") == "EXCESSIVE_INVALID_BOOTSTRAP":
                                        summary["reason"] = "CA17_13_EXCESSIVE_INVALID_ADJUSTED_BOOTSTRAPS"
                            support = self.store.get("support",scope,f"{bid}/{year}")
                            support_failures = list(support.get("support_failures_prebootstrap") or [])
                            support_failures.extend(evaluate_postbootstrap_support(summary,support_policy))
                            if adjusted_confounder and support.get("adjusted_point_exclusion_reason"):
                                summary.update(eligible=False, reason=support["adjusted_point_exclusion_reason"])
                            elif block_reason is not None:
                                summary.update(eligible=False,reason=block_reason)
                            elif local_exposure and not (np.any(labels_b_year==1) and np.any(labels_b_year==0)):
                                summary.update(eligible=False,reason="LOCAL_EXPOSURE_CLASS_ABSENT")
                            elif summary.get("eligible") and support_failures:
                                summary.update(eligible=False,reason=support_failures[0])
                            summary["support_failures"] = support_failures
                            summary["support_eligible"] = not support_failures
                            row = {**support,**summary,"D_local":Dloc,"D_territory":Dtot,"exposure_block_requirement":Bexp,
                                   "raw_block_length":raw_block,"block_length_multiplier":c["core"]["dependence"].get("final_block_length_multiplier",1.0),
                                   "block_length":block,"global_redraws":global_redraws,"block_reason":block_reason,
                                   "outcome_semantics":outcome_semantics,"exposure_mode":exposure_mode,
                                   "confounder_adjusted":adjusted_confounder}
                            self.store.put("yearly",scope,f"{bid}/{year}",row)
                            if retain and c["runtime"]["retain_bootstrap_arrays"]:
                                target = self.root/"bootstrap"/hashlib.sha256(scope.encode()).hexdigest()[:24]
                                target.mkdir(parents=True,exist_ok=True)
                                destination = target/f"{year}_{bid}.npy"
                                if isinstance(draws, np.memmap):
                                    # working_array has already flushed/closed this map.
                                    shutil.copyfile(drawfile, destination)
                                else:
                                    np.save(destination, draws, allow_pickle=False)
                            drawfile.unlink(missing_ok=True)
                            del draws
                            # A prepared compatibility matrix lives for one BSA/year.
                            prepared_pairwise = None
                    if bootfile.exists():
                        bootfile.unlink()
            summary = self.pool_cell(
                run,q,lag,excluded_year=None,save=True, adjusted_confounder=adjusted_confounder,
                synthetic_bsa_year_eligibility=synthetic_bsa_year_eligibility,
            )
            if loyo:
                for year in d.years:
                    loo = self.pool_cell(
                        run,q,lag,excluded_year=year,save=False, adjusted_confounder=adjusted_confounder,
                        synthetic_bsa_year_eligibility=synthetic_bsa_year_eligibility,
                    )
                    self.store.put("loyo",run,f"{cell_key(q,lag)}/{year}",loo)
            summary.update(run=run,q=q,lag=lag,outcome_semantics=outcome_semantics,
                           elapsed_seconds=time.perf_counter()-start)
            self.store.put("cell",run,cell_key(q,lag),summary)
            self.store.commit()  # checkpoint only after the COMPLETE cell succeeds
            return summary
        except BaseException:
            self.store.db.rollback()
            raise
        finally:
            # ExitStack has already closed every mapping, including on exceptions.
            if folder.exists():
                shutil.rmtree(folder)

    def pool_cell(self, run: str, q: float, lag: int, *, excluded_year: int | None, save: bool,
                  adjusted_confounder: bool = False,
                  synthetic_bsa_year_eligibility: np.ndarray | None = None) -> dict:
        """PO4.1: leave a year out of pooling, not out of the upstream baseline.

        CA6 reuses the same pooling implementation for adjusted yearly RCs.
        """
        d = self.data
        scope = cell_scope(run,q,lag)
        populations, coefficients, supported, cis, precisions, het, ds, signs, counts = [],[],[],[],[],[],[],[],[]
        vs, vw, ces, cns, cw = [],[],[],[],[]
        all_pop = 0.0

        # CM5.2 / CM24 calibration-conditional support universe. Structural
        # synthetic exclusions (for example a zero median reference incidence)
        # are outside the calibration universe and therefore must not count
        # against the optimization coverage criterion. The ordinary full-
        # territory coverage is retained unchanged as a diagnostic. A BSA can
        # enter the conditional denominator only if enough structurally eligible
        # years remain to make pooled inference mathematically possible.
        calibration_conditioned = synthetic_bsa_year_eligibility is not None
        support_policy = active_support_policy(self.c)
        minimum_pooled_years = max(2, int(support_policy.get("min_contributing_years") or 0))
        calibration_eligible_bsas = 0
        calibration_represented_bsas = 0
        calibration_reference_population = 0.0
        calibration_represented_population = 0.0

        for b,bid in enumerate(d.ids):
            years = [y for y in d.years if y!=excluded_year]
            year_indices = [d.years.index(y) for y in years]
            calibration_bsa_eligible = True
            calibration_bsa_reference_population = None
            if calibration_conditioned:
                structural_indices = [
                    iy for iy in year_indices
                    if bool(synthetic_bsa_year_eligibility[b,iy])
                ]
                calibration_bsa_eligible = len(structural_indices) >= minimum_pooled_years
                if calibration_bsa_eligible:
                    calibration_eligible_bsas += 1
                    calibration_bsa_reference_population = finite_mean(d.population[b,structural_indices])
                    if calibration_bsa_reference_population is not None:
                        calibration_reference_population += calibration_bsa_reference_population

            rows = [self.store.get("yearly",scope,f"{bid}/{year}") for year in years]
            rows = [r for r in rows if r is not None]
            pooled = enforce_pooled_support(nm.pool(rows), support_policy)
            if adjusted_confounder and not pooled.get("eligible") and pooled.get("reason") == "INSUFFICIENT_VALID_YEARS":
                pooled["reason"] = "CA17_14_INSUFFICIENT_ADJUSTED_YEARS_FOR_POOLING"
            valid_rows = [r for r in rows if r["eligible"]]
            pref = finite_mean(d.population[b,[d.years.index(y) for y in years]])
            p = finite_mean(r["population"] for r in valid_rows) if pooled["eligible"] else pref
            if p is not None:
                all_pop += p
            rc = pooled.get("pooled_rc")
            half = pooled_ci_half_width(pooled.get("ci_low"), pooled.get("ci_high"))
            attenuation = uncertainty_attenuation_factor(rc, pooled.get("ci_low"), pooled.get("ci_high"))
            support_magnitude = confidence_scaled_magnitude(rc, pooled.get("ci_low"), pooled.get("ci_high"))
            ci_supported = pooled_ci_excludes_zero(pooled.get("ci_low"), pooled.get("ci_high"))
            if save:
                pooled_metric_reason = None
                if adjusted_confounder:
                    if not pooled.get("eligible"):
                        pooled_metric_reason = pooled.get("reason")
                    elif support_magnitude is None:
                        pooled_metric_reason = "CA17_15_ADJUSTED_POOLED_CI_UNAVAILABLE"
                self.store.put("pooled",scope,bid,{"run":run,"q":q,"lag":lag,"bsa_id":bid,
                                                   "population":p,**pooled,
                                                   "ci_half_width":half,
                                                   "uncertainty_attenuation_factor":attenuation,
                                                   "confidence_scaled_magnitude":support_magnitude,
                                                   "ci_excludes_zero":ci_supported,
                                                   "confounder_adjusted":adjusted_confounder,
                                                   "confounder_pooled_exclusion_reason":pooled_metric_reason})
            for r in rows:
                if r["rc"] is not None and r["block_reason"] is None and p is not None:
                    vs.append(1-r["invalid_proportion"]);vw.append(p)
                if r["eligible"] and p is not None:
                    ces.append(r["concordance_exposed"]);cns.append(r["concordance_nonexposed"]);cw.append(p)
            if not pooled["eligible"] or p is None:
                continue
            if support_magnitude is None:
                # WMARMcs and diagnostic WMARM intentionally use the same eligible
                # BSA set. An unavailable pooled RC/CI is therefore an eligibility
                # failure, never an artificial zero contribution.
                continue
            populations.append(p);coefficients.append(rc);supported.append(support_magnitude);counts.append(pooled["contributing_years"])
            if calibration_conditioned and calibration_bsa_eligible:
                calibration_represented_bsas += 1
                if calibration_bsa_reference_population is not None:
                    calibration_represented_population += calibration_bsa_reference_population
            cis.append(float(ci_supported) if ci_supported is not None else None)
            precisions.append(abs(rc)/(abs(rc)+half) if half is not None and abs(rc)+half>0 else None)
            het.append(pooled["i_squared_percent"])
            ds.append(finite_mean(r["directional_stability"] for r in valid_rows))
            signs.append(finite_mean(float(np.sign(r["rc"])==np.sign(rc)) for r in valid_rows) if rc!=0 else None)
        pop = np.array(populations,dtype=float)
        rc = np.array(coefficients,dtype=float)
        cs = np.array(supported,dtype=float)
        valid = len(rc)>0 and pop.sum()>0
        wmarm = float(np.dot(pop,np.abs(rc))/pop.sum()) if valid else None
        wmarm_cs = float(np.dot(pop,cs)/pop.sum()) if valid else None
        signed = float(np.dot(pop,rc)/pop.sum()) if valid else None
        signed_cs = float(np.dot(pop,np.sign(rc)*cs)/pop.sum()) if valid else None
        k_cs = wmarm_cs/wmarm if valid and wmarm is not None and wmarm>0 else None
        result = {"run":run,"q":q,"lag":lag,"excluded_year":excluded_year,"valid":valid,
                "reason":None if valid else ("CA17_16_NO_ELIGIBLE_ADJUSTED_POOLED_BSAS" if adjusted_confounder else "NO_ELIGIBLE_POOLED_BSAS_WITH_VALID_RC_AND_CI"),
                "confounder_adjusted":adjusted_confounder,
                "reference_metric":"WMARM_CS",
                "wmarm_cs":wmarm_cs,"wmarm":wmarm,"marm":float(np.mean(np.abs(rc))) if valid else None,
                "confidence_support_retention_ratio":k_cs,
                "eligible_bsas":len(rc),"total_bsas":d.B,"bsa_coverage":len(rc)/d.B,
                "population_coverage":bounded_coverage_ratio(float(pop.sum()), all_pop),
                "represented_population":float(pop.sum()),"total_reference_population":all_pop,
                "contributing_years_min":min(counts) if counts else None,
                "contributing_years_max":max(counts) if counts else None,
                "contributing_years_mean":finite_mean(counts),"signed_coefficient":signed,
                "signed_coefficient_cs":signed_cs,
                "directional_balance_cs":signed_cs/wmarm_cs if valid and wmarm_cs is not None and wmarm_cs>0 else None,
                "directional_balance":signed/wmarm if valid and wmarm>0 else None,
                "positive_population_share":float(pop[rc>0].sum()/pop.sum()) if valid else None,
                "negative_population_share":float(pop[rc<0].sum()/pop.sum()) if valid else None,
                "ci_excluding_zero_population_share":weighted_mean(cis,populations),
                "directional_stability":weighted_mean(ds,populations),
                "relative_precision":weighted_mean(precisions,populations),
                "mean_i_squared_percent":weighted_mean(het,populations),
                "year_pooled_sign_concordance":weighted_mean(signs,populations),
                "bootstrap_validity":weighted_mean(vs,vw),
                "local_concordance_exposed":weighted_mean(ces,cw),
                "local_concordance_nonexposed":weighted_mean(cns,cw),
                "spatial_directional_coherence_cs":abs(signed_cs)/wmarm_cs if valid and wmarm_cs is not None and wmarm_cs>0 else None,
                "spatial_magnitude_dispersion_cs":float(np.sqrt(np.dot(pop,(cs-wmarm_cs)**2)/pop.sum())) if valid else None,
                "spatial_directional_coherence":abs(signed)/wmarm if valid and wmarm>0 else None,
                "spatial_magnitude_dispersion":float(np.sqrt(np.dot(pop,(np.abs(rc)-wmarm)**2)/pop.sum())) if valid else None}
        if calibration_conditioned:
            calibration_bsa_coverage = (
                calibration_represented_bsas/calibration_eligible_bsas if calibration_eligible_bsas>0 else None
            )
            calibration_population_coverage = bounded_coverage_ratio(
                calibration_represented_population, calibration_reference_population
            )
            result.update({
                "calibration_support_conditioned": True,
                "calibration_eligible_bsas": calibration_eligible_bsas,
                "calibration_represented_bsas": calibration_represented_bsas,
                "calibration_bsa_coverage": calibration_bsa_coverage,
                "calibration_represented_reference_population": calibration_represented_population,
                "calibration_total_reference_population": calibration_reference_population,
                "calibration_population_coverage": calibration_population_coverage,
                "calibration_minimum_structurally_eligible_years": minimum_pooled_years,
                "admissibility_coverage_basis": "CALIBRATION_CONDITIONAL_STRUCTURAL_ELIGIBILITY",
                "admissibility_bsa_coverage": calibration_bsa_coverage,
                "admissibility_population_coverage": calibration_population_coverage,
            })
        return result


class SurfaceRunner:
    """Every candidate is delegated to the same fixed estimator, in q/lag order."""
    def __init__(self, engine: FixedParametersEngine):
        self.engine = engine

    def run(self, run: str, *, outcome_path: Path | None = None, outcome_semantics: str = "EVENT_COUNTS",
            permutation: np.ndarray | None = None, loyo: bool = True, retain: bool = True,
            rng_identity_run: str | None = None, prebaseline_cache: Path | None = None,
            bsa_year_eligibility_path: Path | None = None,
            candidate_percentile_path: Path | None = None,
            compatibility_contexts: dict[int, object] | None = None,
            compatibility_exclusion_reasons: dict[int, str] | None = None) -> list[dict]:
        """Repeat the authoritative fixed estimator over the configured domain.

        CA7 forwards the same optional frozen confounder context accepted by
        :meth:`FixedParametersEngine.run`. The ordinary reference, calibration,
        robustness, and null paths are unchanged when those arguments are absent.
        """
        cells = []
        c = self.engine.c
        for q in c["thresholds"]:
            for lag in c["lags"]:
                cell = self.engine.run(
                    run,q,lag,outcome_path=outcome_path,outcome_semantics=outcome_semantics,
                    permutation=permutation,loyo=loyo,retain=retain,rng_identity_run=rng_identity_run,
                    prebaseline_cache=prebaseline_cache,
                    bsa_year_eligibility_path=bsa_year_eligibility_path,
                    candidate_percentile_path=candidate_percentile_path,
                    compatibility_contexts=compatibility_contexts,
                    compatibility_exclusion_reasons=compatibility_exclusion_reasons,
                )
                cells.append(cell)  # tiny summaries only, never panels or bootstrap arrays
        return cells

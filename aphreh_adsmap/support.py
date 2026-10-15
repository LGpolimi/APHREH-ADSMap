"""Shared support-eligibility rules for reference and RS Analysis 2 realizations."""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np

SUPPORT_FIELDS = (
    "min_baseline_support",
    "min_bootstrap_validity_rate",
    "min_contributing_years",
    "min_effective_weighted_sample_size",
    "min_event_contribution",
    "min_exposed_observations",
    "min_local_concordance",
    "min_nonexposed_observations",
)

INTEGER_SUPPORT_FIELDS = {
    "min_baseline_support", "min_contributing_years", "min_event_contribution",
    "min_exposed_observations", "min_nonexposed_observations",
}

SUPPORT_FAILURE_CODES = {
    "min_exposed_observations": "RS_MIN_EXPOSED_OBSERVATIONS",
    "min_nonexposed_observations": "RS_MIN_NONEXPOSED_OBSERVATIONS",
    "min_event_contribution": "RS_MIN_EVENT_CONTRIBUTION",
    "min_effective_weighted_sample_size": "RS_MIN_EFFECTIVE_WEIGHTED_SAMPLE_SIZE",
    "min_bootstrap_validity_rate": "RS_MIN_BOOTSTRAP_VALIDITY_RATE",
    "min_local_concordance": "RS_MIN_LOCAL_CONCORDANCE",
}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def active_support_policy(config: Mapping[str, Any]) -> dict[str, float | int | None]:
    raw = dict(config.get("core", {}).get("support", {}))
    out: dict[str, float | int | None] = {}
    for name in SUPPORT_FIELDS:
        value = raw.get(name)
        if value is None:
            out[name] = None
        elif name in INTEGER_SUPPORT_FIELDS:
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def validate_support_policy(policy: Mapping[str, Any]) -> None:
    bounds = {
        "min_baseline_support": (1, None),
        "min_bootstrap_validity_rate": (0, 1),
        "min_contributing_years": (2, None),
        "min_effective_weighted_sample_size": (0, None),
        "min_event_contribution": (0, None),
        "min_exposed_observations": (1, None),
        "min_local_concordance": (0, 1),
        "min_nonexposed_observations": (1, None),
    }
    unknown = set(policy) - set(SUPPORT_FIELDS)
    if unknown:
        raise ValueError("Unknown core.support fields: " + ", ".join(sorted(unknown)))
    for name in SUPPORT_FIELDS:
        value = policy.get(name)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"core.support.{name} must be null or finite")
        if name in INTEGER_SUPPORT_FIELDS and int(value) != float(value):
            raise ValueError(f"core.support.{name} must be integer-valued")
        low, high = bounds[name]
        if low is not None and float(value) < low:
            raise ValueError(f"core.support.{name} must be >= {low}")
        if high is not None and float(value) > high:
            raise ValueError(f"core.support.{name} must be <= {high}")


def apply_baseline_support_filter(differential: np.ndarray, donor_counts: np.ndarray,
                                  minimum: int | None) -> tuple[np.ndarray, int]:
    """RS6.4: target-day baseline-support eligibility."""
    x = np.asarray(differential, dtype=float).copy()
    if minimum is None:
        return x, 0
    remove = np.isfinite(x) & (np.asarray(donor_counts) < int(minimum))
    x[remove] = np.nan
    return x, int(np.sum(remove))


def evaluate_prebootstrap_support(info: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    """RS6.2, RS6.3, RS6.5, RS6.8, RS6.9 conjunctive BSA-year support."""
    failures: list[str] = []
    for name, field in (
        ("min_exposed_observations", "n_exposed"),
        ("min_nonexposed_observations", "n_nonexposed"),
        ("min_event_contribution", "event_contribution"),
    ):
        required = policy.get(name)
        if required is None:
            continue
        actual = _finite(info.get(field))
        if actual is None or actual < float(required):
            failures.append(SUPPORT_FAILURE_CODES[name])
    required = policy.get("min_effective_weighted_sample_size")
    if required is not None:
        a, b = _finite(info.get("effective_exposed")), _finite(info.get("effective_nonexposed"))
        if a is None or b is None or min(a, b) < float(required):
            failures.append(SUPPORT_FAILURE_CODES["min_effective_weighted_sample_size"])
    required = policy.get("min_local_concordance")
    if required is not None:
        a, b = _finite(info.get("concordance_exposed")), _finite(info.get("concordance_nonexposed"))
        if a is None or b is None or min(a, b) < float(required):
            failures.append(SUPPORT_FAILURE_CODES["min_local_concordance"])
    return failures


def evaluate_postbootstrap_support(summary: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    """RS6.6: optional additional bootstrap-validity support threshold."""
    required = policy.get("min_bootstrap_validity_rate")
    if required is None:
        return []
    valid, requested = _finite(summary.get("bootstrap_valid")), _finite(summary.get("bootstrap_requested"))
    rate = valid / requested if valid is not None and requested not in (None, 0) else None
    return [] if rate is not None and rate >= float(required) - 1e-12 else [SUPPORT_FAILURE_CODES["min_bootstrap_validity_rate"]]


def enforce_pooled_support(pooled: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """RS6.7: minimum contributing-year support for pooled inference."""
    out = dict(pooled)
    required = policy.get("min_contributing_years")
    if required is None:
        return out
    k = int(out.get("contributing_years") or 0)
    out["support_min_contributing_years"] = int(required)
    if k >= int(required):
        return out
    out.update(
        eligible=False,
        reason="RS_MIN_CONTRIBUTING_YEARS",
        pooled_rc=None, ci_low=None, ci_high=None, tau_squared=None,
        i_squared_percent=None, prediction_low=None, prediction_high=None,
    )
    return out

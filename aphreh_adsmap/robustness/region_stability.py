"""RS Analysis 1: stability across the complete reference robust region.

This analysis is deliberately post-processing only.  It consumes the reference
optimization result and the pooled BSA rows already retained for every
threshold-lag cell.  It never reruns exposure classification, incidence,
baseline, RC estimation, bootstrap, pooling, LOYO, or null calibration.

The implementation follows the revised RS formulation:
* pooled signed RC stability remains a point-estimate spatial diagnostic;
* confidence-scaled magnitude S_b is summarized in parallel;
* confidence-support persistence remains a separate diagnostic recording the
  proportion of valid robust-region cells for which the pooled CI excludes zero;
* exact-zero RC values stay in the sign-consistency denominator;
* disconnected robust-region components are summarized separately;
* no robust region (including an degenerate all-zero WMARMcs reference-relevance surface) is an explicit,
  non-error scientific outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr

from ..engine import confidence_scaled_magnitude, pooled_ci_excludes_zero
from ..storage import Store, atomic_json, clean
from .models import CheckpointState
from .persistence import RobustnessPersistence

ANALYSIS_ID = "RS_ANALYSIS_1_ROBUST_REGION_STABILITY"
CHECKPOINT_STAGE = "RS_ANALYSIS_1"
SCHEMA_VERSION = "APHREH_RS_ANALYSIS_1_V1"

BSA_FIELDS = (
    "analysis_id", "reference_realization_id", "region_scope", "component_index",
    "bsa_id", "region_cell_count", "valid_cell_count", "valid_cell_proportion",
    "minimum_valid_cell_proportion", "stability_interpretable", "support_reason",
    "median_pooled_rc", "pooled_rc_q25", "pooled_rc_q75", "pooled_rc_iqr",
    "pooled_rc_min", "pooled_rc_max", "positive_cell_count", "negative_cell_count",
    "zero_cell_count", "zero_cell_proportion", "dominant_direction", "sign_consistency",
    "median_supported_magnitude", "supported_magnitude_q25", "supported_magnitude_q75",
    "supported_magnitude_iqr", "supported_magnitude_min", "supported_magnitude_max",
    "confidence_supported_cell_count", "confidence_support_persistence",
    "rank_median", "rank_q25", "rank_q75", "rank_iqr",
)

TERRITORY_FIELDS = (
    "analysis_id", "reference_realization_id", "region_scope", "component_index",
    "region_cell_count", "q_min", "q_max", "lag_min", "lag_max",
    "bsa_rows", "bsa_interpretable_count", "bsa_complete_region_support_count",
    "median_valid_cell_proportion", "median_sign_consistency",
    "median_confidence_support_persistence", "median_rank_iqr",
    "pairwise_spearman_pairs_possible", "pairwise_spearman_pairs_valid",
    "pairwise_spearman_median", "pairwise_spearman_q25", "pairwise_spearman_q75",
    "pairwise_spearman_iqr",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summary(values: Iterable[float | None]) -> dict[str, float | None]:
    arr = np.asarray([v for v in (_finite(x) for x in values) if v is not None], dtype=float)
    if arr.size == 0:
        return {"median": None, "q25": None, "q75": None, "iqr": None, "min": None, "max": None}
    q25, med, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "median": float(med), "q25": float(q25), "q75": float(q75),
        "iqr": float(q75 - q25), "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def _cell_id(q: Any, lag: Any) -> tuple[float, int]:
    return (float(q), int(lag))


def _valid_pooled_row(row: dict[str, Any]) -> tuple[float, float, bool] | None:
    """Return pooled RC, confidence-scaled magnitude, and CI-exclusion state."""
    if not bool(row.get("eligible")):
        return None
    rc = _finite(row.get("pooled_rc"))
    pop = _finite(row.get("population"))
    if rc is None or pop is None:
        return None
    scaled = confidence_scaled_magnitude(rc, _finite(row.get("ci_low")), _finite(row.get("ci_high")))
    ci_supported = pooled_ci_excludes_zero(_finite(row.get("ci_low")), _finite(row.get("ci_high")))
    if scaled is None or ci_supported is None:
        return None
    return rc, float(scaled), bool(ci_supported)


def _region_bounds(cells: list[tuple[float, int]]) -> dict[str, Any]:
    return {
        "q_min": min((q for q, _ in cells), default=None),
        "q_max": max((q for q, _ in cells), default=None),
        "lag_min": min((lag for _, lag in cells), default=None),
        "lag_max": max((lag for _, lag in cells), default=None),
    }


def _region_components(optimization: dict[str, Any], robust_cells: list[tuple[float, int]]) -> list[tuple[str, int | None, list[tuple[float, int]]]]:
    result: list[tuple[str, int | None, list[tuple[float, int]]]] = [("ALL", None, robust_cells)]
    components = optimization.get("components") or []
    # The methodology asks for component-specific reporting when the region is
    # disconnected. Avoid duplicating ALL for a single connected component.
    if len(components) > 1:
        for index, component in enumerate(components, 1):
            cells = [_cell_id(c["q"], c["lag"]) for c in component.get("cells", [])]
            result.append((f"COMPONENT_{index:03d}", index, cells))
    return result


def _ranks_by_cell(rows_by_cell: dict[tuple[float, int], dict[str, dict[str, Any]]], cells: list[tuple[float, int]]) -> dict[tuple[float, int], dict[str, float]]:
    result: dict[tuple[float, int], dict[str, float]] = {}
    for cell in cells:
        valid: list[tuple[str, float]] = []
        for bid, row in rows_by_cell.get(cell, {}).items():
            parsed = _valid_pooled_row(row)
            if parsed is not None:
                valid.append((bid, parsed[0]))
        if not valid:
            result[cell] = {}
            continue
        # Average ranks for ties, exactly as specified by RS Analysis 1.
        values = np.asarray([rc for _, rc in valid], dtype=float)
        ranks = rankdata(values, method="average")
        result[cell] = {valid[i][0]: float(ranks[i]) for i in range(len(valid))}
    return result


def _pairwise_rank_summary(ranks: dict[tuple[float, int], dict[str, float]], cells: list[tuple[float, int]]) -> dict[str, Any]:
    correlations: list[float] = []
    possible = 0
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            possible += 1
            common = sorted(set(ranks.get(a, {})) & set(ranks.get(b, {})))
            if len(common) < 2:
                continue
            va = np.asarray([ranks[a][bid] for bid in common], dtype=float)
            vb = np.asarray([ranks[b][bid] for bid in common], dtype=float)
            # Constant vectors have undefined Spearman correlation.
            if np.allclose(va, va[0]) or np.allclose(vb, vb[0]):
                continue
            rho = spearmanr(va, vb).statistic
            if rho is not None and math.isfinite(float(rho)):
                correlations.append(float(rho))
    stats = _summary(correlations)
    return {
        "pairwise_spearman_pairs_possible": possible,
        "pairwise_spearman_pairs_valid": len(correlations),
        "pairwise_spearman_median": stats["median"],
        "pairwise_spearman_q25": stats["q25"],
        "pairwise_spearman_q75": stats["q75"],
        "pairwise_spearman_iqr": stats["iqr"],
    }


def compute_region_stability(
    store: Store,
    optimization: dict[str, Any] | None,
    *,
    reference_realization_id: str,
    minimum_valid_cell_proportion: float | None = None,
) -> dict[str, Any]:
    """Compute RS Analysis 1 entirely from retained reference outputs."""
    if minimum_valid_cell_proportion is not None and not 0 <= minimum_valid_cell_proportion <= 1:
        raise ValueError("minimum_valid_cell_proportion must be null or in [0, 1]")
    if optimization is None:
        return {
            "schema_version": SCHEMA_VERSION, "analysis_id": ANALYSIS_ID,
            "reference_realization_id": reference_realization_id,
            "status": "NO_REFERENCE_OPTIMIZATION", "reason": "REFERENCE_OPTIMIZATION_UNAVAILABLE",
            "minimum_valid_cell_proportion": minimum_valid_cell_proportion,
            "robust_region_cell_count": 0, "component_count": 0,
            "bsa_rows": [], "territory_rows": [],
        }

    observed = _finite(optimization.get("observed_admissible_maximum_wmarm_cs", optimization.get("observed_admissible_maximum")))
    robust = optimization.get("robust_region") or []
    robust_cells = [_cell_id(r["q"], r["lag"]) for r in robust]
    if not robust_cells:
        zero_surface = observed is not None and abs(observed) <= 1e-15
        return {
            "schema_version": SCHEMA_VERSION, "analysis_id": ANALYSIS_ID,
            "reference_realization_id": reference_realization_id,
            "status": "ZERO_SUPPORTED_SURFACE" if zero_surface else "NO_ROBUST_REGION",
            "reason": "WMARM_CS_ZERO_ON_ADMISSIBLE_DOMAIN" if zero_surface else "REFERENCE_ROBUST_REGION_EMPTY",
            "minimum_valid_cell_proportion": minimum_valid_cell_proportion,
            "robust_region_cell_count": 0, "component_count": 0,
            "bsa_rows": [], "territory_rows": [],
        }

    robust_set = set(robust_cells)
    rows_by_cell: dict[tuple[float, int], dict[str, dict[str, Any]]] = {cell: {} for cell in robust_cells}
    all_bsa_ids: set[str] = set()
    for row in store.rows("pooled"):
        if row.get("run") != "REFERENCE":
            continue
        try:
            cell = _cell_id(row.get("q"), row.get("lag"))
        except (TypeError, ValueError):
            continue
        if cell not in robust_set:
            continue
        bid = str(row.get("bsa_id"))
        rows_by_cell[cell][bid] = row
        all_bsa_ids.add(bid)

    bsa_rows: list[dict[str, Any]] = []
    territory_rows: list[dict[str, Any]] = []
    scopes = _region_components(optimization, robust_cells)
    for region_scope, component_index, cells in scopes:
        ranks = _ranks_by_cell(rows_by_cell, cells)
        region_bsas = sorted({bid for cell in cells for bid in rows_by_cell.get(cell, {})})
        scope_bsa_rows: list[dict[str, Any]] = []
        for bid in region_bsas:
            rcs: list[float] = []
            supports: list[float] = []
            rank_values: list[float] = []
            pos = neg = zero = supported_count = 0
            for cell in cells:
                row = rows_by_cell.get(cell, {}).get(bid)
                if row is None:
                    continue
                parsed = _valid_pooled_row(row)
                if parsed is None:
                    continue
                rc, support, ci_supported = parsed
                rcs.append(rc); supports.append(support)
                if rc > 0: pos += 1
                elif rc < 0: neg += 1
                else: zero += 1
                if ci_supported: supported_count += 1
                if bid in ranks.get(cell, {}):
                    rank_values.append(ranks[cell][bid])
            valid_count = len(rcs)
            proportion = valid_count / len(cells) if cells else None
            interpretable = valid_count > 0 and (
                minimum_valid_cell_proportion is None or proportion >= minimum_valid_cell_proportion
            )
            reason = None if interpretable else (
                "NO_VALID_ROBUST_REGION_CELLS" if valid_count == 0 else "INSUFFICIENT_VALID_REGION_SUPPORT"
            )
            rc_stats = _summary(rcs)
            support_stats = _summary(supports)
            rank_stats = _summary(rank_values)
            if valid_count:
                sign_consistency = max(pos, neg) / valid_count
                zero_proportion = zero / valid_count
                if pos > neg: dominant = "POSITIVE"
                elif neg > pos: dominant = "NEGATIVE"
                elif pos == 0 and neg == 0: dominant = "NONE"
                else: dominant = "TIED"
                cs_persistence = supported_count / valid_count
            else:
                sign_consistency = zero_proportion = cs_persistence = None
                dominant = "UNAVAILABLE"
            out = {
                "analysis_id": ANALYSIS_ID,
                "reference_realization_id": reference_realization_id,
                "region_scope": region_scope,
                "component_index": component_index,
                "bsa_id": bid,
                "region_cell_count": len(cells),
                "valid_cell_count": valid_count,
                "valid_cell_proportion": proportion,
                "minimum_valid_cell_proportion": minimum_valid_cell_proportion,
                "stability_interpretable": interpretable,
                "support_reason": reason,
                "median_pooled_rc": rc_stats["median"],
                "pooled_rc_q25": rc_stats["q25"], "pooled_rc_q75": rc_stats["q75"],
                "pooled_rc_iqr": rc_stats["iqr"], "pooled_rc_min": rc_stats["min"], "pooled_rc_max": rc_stats["max"],
                "positive_cell_count": pos, "negative_cell_count": neg, "zero_cell_count": zero,
                "zero_cell_proportion": zero_proportion, "dominant_direction": dominant,
                "sign_consistency": sign_consistency,
                "median_supported_magnitude": support_stats["median"],
                "supported_magnitude_q25": support_stats["q25"], "supported_magnitude_q75": support_stats["q75"],
                "supported_magnitude_iqr": support_stats["iqr"],
                "supported_magnitude_min": support_stats["min"], "supported_magnitude_max": support_stats["max"],
                "confidence_supported_cell_count": supported_count,
                "confidence_support_persistence": cs_persistence,
                "rank_median": rank_stats["median"], "rank_q25": rank_stats["q25"], "rank_q75": rank_stats["q75"],
                "rank_iqr": rank_stats["iqr"],
            }
            scope_bsa_rows.append(out); bsa_rows.append(out)

        bounds = _region_bounds(cells)
        rank_summary = _pairwise_rank_summary(ranks, cells)
        interpretable_rows = [r for r in scope_bsa_rows if r["stability_interpretable"]]
        territory = {
            "analysis_id": ANALYSIS_ID, "reference_realization_id": reference_realization_id,
            "region_scope": region_scope, "component_index": component_index,
            "region_cell_count": len(cells), **bounds,
            "bsa_rows": len(scope_bsa_rows),
            "bsa_interpretable_count": len(interpretable_rows),
            "bsa_complete_region_support_count": sum(r["valid_cell_count"] == len(cells) for r in scope_bsa_rows),
            "median_valid_cell_proportion": _summary(r["valid_cell_proportion"] for r in scope_bsa_rows)["median"],
            "median_sign_consistency": _summary(r["sign_consistency"] for r in interpretable_rows)["median"],
            "median_confidence_support_persistence": _summary(r["confidence_support_persistence"] for r in interpretable_rows)["median"],
            "median_rank_iqr": _summary(r["rank_iqr"] for r in interpretable_rows)["median"],
            **rank_summary,
        }
        territory_rows.append(territory)

    all_row = next((r for r in territory_rows if r["region_scope"] == "ALL"), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "reference_realization_id": reference_realization_id,
        "status": "COMPLETED",
        "reason": None,
        "minimum_valid_cell_proportion": minimum_valid_cell_proportion,
        "robust_region_cell_count": len(robust_cells),
        "component_count": len(optimization.get("components") or []) or 1,
        "bsa_count_observed_in_region": len(all_bsa_ids),
        "territory_summary": all_row,
        "bsa_rows": bsa_rows,
        "territory_rows": territory_rows,
        "method_notes": {
            "valid_bsa_cell_rule": "pooled row eligible + finite pooled RC + finite population + valid pooled CI; CI crossing zero remains valid and retains a continuously attenuated S_b",
            "sign_consistency": "max(positive_count, negative_count) / valid_cell_count; exact zero remains in denominator",
            "confidence_support_persistence": "proportion of valid robust-region cells whose pooled CI excludes zero; separate from continuous S_b attenuation",
            "rank_rule": "signed pooled RC; average ranks for ties",
            "territory_rank_concordance": "pairwise Spearman correlations on BSAs valid in both compared cells",
        },
    }


@dataclass
class RegionStabilityAnalyzer:
    store: Store
    persistence: RobustnessPersistence
    reference_realization_id: str
    minimum_valid_cell_proportion: float | None = None

    def run(self, optimization: dict[str, Any] | None) -> dict[str, Any]:
        checkpoint = self.persistence.checkpoint(
            self.reference_realization_id, CHECKPOINT_STAGE, "REFERENCE", "result"
        )
        if checkpoint is not None and checkpoint["state"] == CheckpointState.COMPLETED.value:
            cached = self.persistence.analysis_output(
                self.reference_realization_id, ANALYSIS_ID, "SUMMARY", "REFERENCE", "result"
            )
            if cached is not None:
                cached = dict(cached)
                cached["bsa_rows"] = [
                    item["payload"] for item in self.persistence.analysis_outputs(
                        self.reference_realization_id, ANALYSIS_ID, "BSA"
                    )
                ]
                cached["territory_rows"] = [
                    item["payload"] for item in self.persistence.analysis_outputs(
                        self.reference_realization_id, ANALYSIS_ID, "TERRITORY"
                    )
                ]
                return cached
        self.persistence.put_checkpoint(
            self.reference_realization_id, CHECKPOINT_STAGE, "REFERENCE", "result",
            CheckpointState.RUNNING,
            payload={"analysis_id": ANALYSIS_ID},
        )
        try:
            self.persistence.clear_analysis_outputs(self.reference_realization_id, ANALYSIS_ID)
            result = compute_region_stability(
                self.store, optimization,
                reference_realization_id=self.reference_realization_id,
                minimum_valid_cell_proportion=self.minimum_valid_cell_proportion,
            )
            summary = {k: v for k, v in result.items() if k not in ("bsa_rows", "territory_rows")}
            self.persistence.put_analysis_output(
                self.reference_realization_id, ANALYSIS_ID, "SUMMARY", "REFERENCE", "result", summary
            )
            for row in result["bsa_rows"]:
                self.persistence.put_analysis_output(
                    self.reference_realization_id, ANALYSIS_ID, "BSA", row["region_scope"], str(row["bsa_id"]), row
                )
            for row in result["territory_rows"]:
                self.persistence.put_analysis_output(
                    self.reference_realization_id, ANALYSIS_ID, "TERRITORY", row["region_scope"], "summary", row
                )
            self.store.commit()
            self.persistence.put_checkpoint(
                self.reference_realization_id, CHECKPOINT_STAGE, "REFERENCE", "result",
                CheckpointState.COMPLETED,
                payload={
                    "analysis_id": ANALYSIS_ID, "status": result["status"],
                    "bsa_rows": len(result["bsa_rows"]), "territory_rows": len(result["territory_rows"]),
                },
            )
            return result
        except BaseException as exc:
            self.persistence.put_checkpoint(
                self.reference_realization_id, CHECKPOINT_STAGE, "REFERENCE", "result",
                CheckpointState.FAILED, error=f"{type(exc).__name__}: {exc}",
            )
            raise


def export_region_stability(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k not in ("bsa_rows", "territory_rows")}
    atomic_json(out / "reference_region_stability.json", summary)
    _write_csv(out / "reference_region_stability_bsa.csv", BSA_FIELDS, result.get("bsa_rows", []))
    _write_csv(out / "reference_region_stability_territory.csv", TERRITORY_FIELDS, result.get("territory_rows", []))


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(row.get(k)) for k in fields})
    tmp.replace(path)

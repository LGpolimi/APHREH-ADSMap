"""RS Analysis 3: regional versus BSA-specific exposure classification.

The alternative realization changes exposure classification at the EXPOSURE
stage and then reuses the production FixedParametersEngine for baseline,
yearly RC, synchronized bootstrap, pooling, and WMARMcs.  Step 6 executes the
alternative surface through the production estimator; Steps 7 and 8 subsequently provide fixed-point comparison and complete reoptimization when requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import FixedParametersEngine
from ..storage import Store, atomic_json, clean, dumps, sha256_file
from .models import (
    CheckpointState, ComparisonMode, FactorKind, InvalidationStage,
    RealizationSpec, RealizationState,
)
from .planner import FactorSpec, ScenarioSpec, materialize_realization
from .persistence import RobustnessPersistence
from .. import numerics as nm

ANALYSIS_ID = "ANALYSIS_3_BSA_SPECIFIC_EXPOSURE"
CHECKPOINT_STAGE = "RS_ANALYSIS_3"
SCHEMA_VERSION = "APHREH_RS_ANALYSIS_3_V2"
REFERENCE_MODE = "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE"
ALTERNATIVE_MODE = "BSA_SPECIFIC"


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def exposure_scenarios_from_config(config: Mapping[str, Any]) -> tuple[ScenarioSpec, ...]:
    section = config.get("robustness", {}).get("analysis_3_bsa_specific_exposure", {})
    if not section.get("enabled", False):
        return ()
    reference_mode = config["core"]["exposure"]["mode"]
    factor = FactorSpec(
        name="exposure_classification_mode",
        config_path="core.exposure.mode",
        analysis_id=ANALYSIS_ID,
        kind=FactorKind.CATEGORICAL,
        invalidates_from=InvalidationStage.EXPOSURE,
        reference_value=reference_mode,
        alternative_value=ALTERNATIVE_MODE,
    )
    return (ScenarioSpec(
        scenario_id="bsa_specific_exposure",
        analysis_id=ANALYSIS_ID,
        factors=(factor,),
        numerical_steps=1,
        comparison_mode=ComparisonMode(section.get("comparison_mode", "FIXED_PARAMETER")),
        label=section.get("label", "BSA-specific exposure classification"),
    ),)


def realization_run_id(spec: RealizationSpec) -> str:
    return f"RS_A3_{spec.realization_id.split('.')[-1]}"


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
    rows = []
    fields = (
        "run", "q", "lag", "bsa_id", "year", "exposure_mode", "exposure_calendar_hash", "eligible", "reason",
        "rc", "variance", "ci_low", "ci_high", "population", "population_source_year",
        "threshold_local", "threshold_territory", "n_exposed", "n_nonexposed",
        "n_exposed_positive_weight", "n_nonexposed_positive_weight",
        "effective_exposed", "effective_nonexposed", "event_contribution",
        "concordance_exposed", "concordance_nonexposed", "baseline_donor_count_mean",
        "baseline_half_width_mean", "baseline_half_width_max", "unavailable_baseline_days",
        "missing_local_hazard_days", "zero_weight_valid_days", "D_local", "D_territory",
        "exposure_block_requirement", "raw_block_length", "block_length", "global_redraws",
        "bootstrap_valid", "bootstrap_requested", "invalid_proportion", "directional_stability",
    )
    for row in store.rows("yearly"):
        if row.get("run") != run:
            continue
        rows.append({k: row.get(k) for k in fields})
    return rows


def _exclusion_summary(store: Store, run: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for kind, level in (("yearly", "YEARLY"), ("pooled", "POOLED")):
        for row in store.rows(kind):
            if row.get("run") != run:
                continue
            reason = "ELIGIBLE" if row.get("eligible") else str(row.get("reason") or "INELIGIBLE")
            counts[(level, reason)] = counts.get((level, reason), 0) + 1
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


def export_exposure_sensitivity(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_analysis3_summary.json", {
        k: v for k, v in result.items() if k not in (
            "surface_rows", "yearly_rows", "exclusion_rows", "calendar_index_rows", "episode_rows"
        )
    })
    _write_csv(output_dir / "robustness_analysis3_realizations.csv", list(result.get("summary_rows", [])))
    _write_csv(output_dir / "robustness_analysis3_surfaces.csv", list(result.get("surface_rows", [])))
    _write_csv(output_dir / "robustness_analysis3_yearly.csv", list(result.get("yearly_rows", [])))
    _write_csv(output_dir / "robustness_analysis3_exclusions.csv", list(result.get("exclusion_rows", [])))
    _write_csv(output_dir / "robustness_analysis3_calendars.csv", list(result.get("calendar_index_rows", [])))
    _write_csv(output_dir / "robustness_analysis3_episode_summaries.csv", list(result.get("episode_rows", [])))


def _q_token(q: float) -> str:
    return f"{float(q):g}".replace("-", "m").replace(".", "p")


def _local_exposure_outputs(
    data: Any, spec: RealizationSpec, realized: Mapping[str, Any], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain RS7 local exposure calendars and BSA-year episode summaries.

    Calendars can be large, so the actual date-level classifications are stored
    as compressed NumPy archives rather than expanded into a very large CSV.
    Each archive is self-describing and loadable with ``allow_pickle=False``.
    A compact CSV index records its identity and SHA-256 digest.
    """
    calendar_dir = output_dir / "robustness_analysis3_calendars"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    calendar_index_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    episode_quantile = float(realized["core"]["dependence"]["exposed_episode_quantile"])
    bsa_ids = np.asarray([str(v) for v in data.ids], dtype="U")
    dates = np.asarray(data.meta["dates"], dtype="U10")
    years = np.asarray(data.years, dtype=np.int32)
    segment_index = np.asarray(data.segment_index, dtype=np.int32)

    for q in realized["thresholds"]:
        q = float(q)
        labels_matrix = np.full((data.B, data.n), -1, dtype=np.int8)
        thresholds_matrix = np.full((data.B, len(data.years)), np.nan, dtype=np.float64)
        for b, bid in enumerate(data.ids):
            local = data.series("hazard", b)
            labels, thresholds = nm.exposure(local, data.year_index, q)
            labels_matrix[b] = labels
            thresholds_matrix[b] = thresholds
            for iy, year in enumerate(data.years):
                ix = data.year_index == iy
                annual_labels = np.asarray(labels[ix], dtype=np.int8)
                annual_segments = np.asarray(data.segment_index[ix], dtype=np.int32)
                lengths = nm.episode_lengths(annual_labels, annual_segments)
                q_value = nm.finite_quantile(lengths, episode_quantile) if len(lengths) else float("nan")
                episode_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "q": q,
                    "bsa_id": bid,
                    "year": int(year),
                    "threshold_local": _finite(thresholds[iy]),
                    "calendar_hash": hashlib.sha256(annual_labels.tobytes()).hexdigest()[:16],
                    "exposed_days": int(np.sum(annual_labels == 1)),
                    "nonexposed_days": int(np.sum(annual_labels == 0)),
                    "unavailable_days": int(np.sum(annual_labels < 0)),
                    "episode_count": int(len(lengths)),
                    "episode_length_min": float(np.min(lengths)) if len(lengths) else None,
                    "episode_length_mean": float(np.mean(lengths)) if len(lengths) else None,
                    "episode_length_median": float(np.median(lengths)) if len(lengths) else None,
                    "episode_length_max": float(np.max(lengths)) if len(lengths) else None,
                    "episode_quantile": episode_quantile,
                    "episode_length_quantile": _finite(q_value),
                    "local_episode_requirement": int(math.ceil(q_value)) if math.isfinite(q_value) else 1,
                })

        filename = f"{spec.realization_id.replace('.', '_')}_q{_q_token(q)}.npz"
        target = calendar_dir / filename
        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(
                f,
                schema_version=np.asarray([SCHEMA_VERSION], dtype="U"),
                realization_id=np.asarray([spec.realization_id], dtype="U"),
                q=np.asarray([q], dtype=np.float64),
                bsa_ids=bsa_ids,
                dates=dates,
                years=years,
                year_index=np.asarray(data.year_index, dtype=np.int32),
                segment_index=segment_index,
                labels=labels_matrix,
                thresholds=thresholds_matrix,
            )
        temp.replace(target)
        calendar_index_rows.append({
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "realization_id": spec.realization_id,
            "scenario_id": spec.scenario_id,
            "step_index": spec.step_index,
            "trajectory_coordinate": spec.trajectory_coordinate,
            "q": q,
            "archive": (Path("robustness_analysis3_calendars") / filename).as_posix(),
            "archive_sha256": sha256_file(target),
            "bsa_count": int(data.B),
            "date_count": int(data.n),
            "year_count": int(len(data.years)),
            "label_encoding": "-1=UNAVAILABLE;0=NONEXPOSED;1=EXPOSED",
            "archive_arrays": "bsa_ids,dates,years,year_index,segment_index,labels,thresholds",
        })
    return calendar_index_rows, episode_rows


def _persist_exposure_traceability(
    persistence: RobustnessPersistence, spec: RealizationSpec,
    calendar_index_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]
) -> None:
    records: list[tuple[str, str, str, Any]] = []
    records.extend(
        ("CALENDAR_INDEX", "THRESHOLD", f"q{float(row['q']):g}", row)
        for row in calendar_index_rows
    )
    records.extend(
        ("EPISODE_SUMMARY", "BSA_YEAR", f"{float(row['q']):g}/{row['bsa_id']}/{int(row['year'])}", row)
        for row in episode_rows
    )
    persistence.put_analysis_outputs(spec.realization_id, ANALYSIS_ID, records)


@dataclass
class ExposureSensitivityRunner:
    data: Any
    store: Store
    persistence: RobustnessPersistence
    root: Path
    log: Any = print

    def run(self, reference_config: dict[str, Any], plan) -> dict[str, Any]:
        specs = [r for r in plan.realizations if r.analysis_id == ANALYSIS_ID]
        summaries: list[dict[str, Any]] = []
        surfaces: list[dict[str, Any]] = []
        yearly: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        calendar_index: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        for spec in specs:
            checkpoint = self.persistence.checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result")
            if checkpoint and checkpoint.get("state") == CheckpointState.COMPLETED.value:
                stored = self.persistence.analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result")
                if stored is not None:
                    # 2.3.4 output completion: old 2.3.3 checkpoints can be
                    # upgraded without rerunning the expensive sensitivity
                    # surface.  Local exposure calendars and episode summaries
                    # are deterministic functions of canonical hazard data,
                    # q, and the realized exposure definition.
                    realized = materialize_realization(reference_config, spec)
                    cal_rows, episode_rows = _local_exposure_outputs(
                        self.data, spec, realized, self.root / "results"
                    )
                    _persist_exposure_traceability(self.persistence, spec, cal_rows, episode_rows)
                    stored = {
                        **stored,
                        "schema_version": SCHEMA_VERSION,
                        "calendar_archive_count": len(cal_rows),
                        "local_episode_summary_count": len(episode_rows),
                        "calendar_archive_format": "NPZ_COMPRESSED_V1",
                    }
                    self.persistence.put_analysis_output(
                        spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", stored
                    )
                    self.persistence.put_checkpoint(
                        spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result",
                        CheckpointState.COMPLETED, payload=stored,
                    )
                    summaries.append(stored)
                    surfaces.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "SURFACE", "CELL"))
                    yearly.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "YEARLY", "BSA_YEAR"))
                    exclusions.extend(r["payload"] for r in self.persistence.analysis_outputs(spec.realization_id, ANALYSIS_ID, "EXCLUSION", "COUNT"))
                    calendar_index.extend(cal_rows)
                    episodes.extend(episode_rows)
                    continue
            self.persistence.set_realization_state(spec.realization_id, RealizationState.RUNNING)
            self.persistence.put_checkpoint(spec.realization_id, CHECKPOINT_STAGE, "REALIZATION", "result", CheckpointState.RUNNING)
            try:
                realized = materialize_realization(reference_config, spec)
                if realized["core"]["exposure"]["mode"] != ALTERNATIVE_MODE:
                    raise ValueError("Analysis 3 realization did not materialize BSA_SPECIFIC exposure mode")
                if realized["core"].get("support", {}).get("min_local_concordance") is not None:
                    raise ValueError("Analysis 3 does not apply the reference local-concordance support criterion under BSA-specific exposure")
                run_id = realization_run_id(spec)
                self.log(f"RS Analysis 3: BSA-specific exposure classification; run={run_id}")
                engine = FixedParametersEngine(self.data, realized, self.store, self.root, self.log)
                cells = []
                for q in realized["thresholds"]:
                    for lag in realized["lags"]:
                        cells.append(engine.run(
                            run_id, float(q), int(lag), loyo=False, retain=False,
                            rng_identity_run="REFERENCE",
                        ))
                yr = _yearly_rows(self.store, run_id)
                local_thresholds = [_finite(r.get("threshold_local")) for r in yr]
                local_thresholds = [x for x in local_thresholds if x is not None]
                exposed = [int(r.get("n_exposed") or 0) for r in yr]
                nonexposed = [int(r.get("n_nonexposed") or 0) for r in yr]
                episode_req = [_finite(r.get("exposure_block_requirement")) for r in yr]
                episode_req = [x for x in episode_req if x is not None]
                cal_rows, episode_rows = _local_exposure_outputs(
                    self.data, spec, realized, self.root / "results"
                )
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "analysis_id": ANALYSIS_ID,
                    "realization_id": spec.realization_id,
                    "reference_realization_id": spec.reference_realization_id,
                    "scenario_id": spec.scenario_id,
                    "step_index": spec.step_index,
                    "trajectory_coordinate": spec.trajectory_coordinate,
                    "run_id": run_id,
                    "exposure_mode": ALTERNATIVE_MODE,
                    "invalidates_from": spec.invalidates_from.name,
                    "mean_local_threshold": float(np.mean(local_thresholds)) if local_thresholds else None,
                    "mean_exposed_observations": float(np.mean(exposed)) if exposed else None,
                    "mean_nonexposed_observations": float(np.mean(nonexposed)) if nonexposed else None,
                    "mean_pooled_local_episode_requirement": float(np.mean(episode_req)) if episode_req else None,
                    "local_concordance_comparison": "NOT_APPLICABLE",
                    "calendar_archive_count": len(cal_rows),
                    "local_episode_summary_count": len(episode_rows),
                    "calendar_archive_format": "NPZ_COMPRESSED_V1",
                    **_surface_summary(cells),
                }
                self.persistence.clear_analysis_outputs(spec.realization_id, ANALYSIS_ID)
                self.persistence.put_analysis_output(spec.realization_id, ANALYSIS_ID, "SUMMARY", "REALIZATION", "result", row)
                _persist_exposure_traceability(self.persistence, spec, cal_rows, episode_rows)
                calendar_index.extend(cal_rows)
                episodes.extend(episode_rows)
                for cell in cells:
                    payload = {
                        "analysis_id": ANALYSIS_ID,
                        "realization_id": spec.realization_id,
                        "scenario_id": spec.scenario_id,
                        "step_index": spec.step_index,
                        "trajectory_coordinate": spec.trajectory_coordinate,
                        "exposure_mode": ALTERNATIVE_MODE,
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
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": ANALYSIS_ID,
            "status": "COMPLETED" if specs else "NOT_REQUESTED",
            "realization_count": len(specs),
            "reference_exposure_mode": reference_config["core"]["exposure"]["mode"],
            "alternative_exposure_mode": ALTERNATIVE_MODE,
            "summary_rows": summaries,
            "surface_rows": surfaces,
            "yearly_rows": yearly,
            "exclusion_rows": exclusions,
            "calendar_index_rows": calendar_index,
            "episode_rows": episodes,
            "calendar_output": {
                "index_file": "results/robustness_analysis3_calendars.csv",
                "archive_directory": "results/robustness_analysis3_calendars",
                "archive_format": "NPZ_COMPRESSED_V1",
                "label_encoding": "-1=UNAVAILABLE;0=NONEXPOSED;1=EXPOSED",
            },
            "episode_summary_file": "results/robustness_analysis3_episode_summaries.csv",
            "scope_note": "Step 6 executes the BSA-specific exposure sensitivity surface; Step 7 provides fixed-parameter comparison and Step 8 optionally performs complete realization-specific reoptimization.",
        }
        export_exposure_sensitivity(result, self.root / "results")
        return result

"""Candidate-confounder preprocessing using the standard APHREH hazard-field model.

Each candidate is supplied exactly like the ordinary environmental hazard: a
wide CSV with one date column and one column per source grid cell, plus a
standard APHREH BSA-grid intersection crosswalk. The authoritative
``data.project_hazard_field`` routine performs both BSA and territory projection,
so candidate and primary-hazard spatial semantics cannot diverge.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..data import SourceError, parse_date, project_hazard_field, source_path
from ..storage import atomic_json, mapped, sha256_file
from .contracts import (
    CONFOUNDER_HAZARD_FIELD_FORMAT,
    ConfounderCandidateDefinition,
    ConfounderExclusionReason,
    ConfounderPercentilePanel,
    ConfounderPreprocessingExclusion,
    ConfounderPreprocessingResult,
    ConfounderSpatialRepresentation,
)


# Retained as a compatibility export. There is now one authoritative projection
# rule and it is shared with the main hazard backend.
SUPPORTED_HARMONIZATION_RULES = {"APHREH_STANDARD_HAZARD_FIELD_PROJECTION_V1"}
PERCENTILE_DEFINITION = "EMPIRICAL_CDF_RIGHT"
THRESHOLD_EQUALITY_CONVENTION = "GREATER_THAN_OR_EQUAL"


class ConfounderPreprocessingError(ValueError):
    """Candidate hazard-field or projection failure detected before CA3."""


def _resolve_reference(reference: str, *, data_dir: Path, project_root: Path) -> Path:
    """Resolve local candidate references using the same project/data semantics."""
    reference = str(reference).strip()
    if reference.startswith("data://"):
        relative = reference[len("data://"):].lstrip("/")
        base = data_dir.resolve()
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ConfounderPreprocessingError("data:// source escapes data_dir") from exc
        return path
    if reference.startswith("project://"):
        relative = reference[len("project://"):].lstrip("/")
        base = project_root.resolve()
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ConfounderPreprocessingError("project:// source escapes project root") from exc
        return path
    if "://" in reference:
        raise ConfounderPreprocessingError(
            f"Unsupported candidate source reference scheme: {reference!r}"
        )
    return source_path(data_dir, reference)


def _require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise ConfounderPreprocessingError(f"{role} file does not exist: {path}")


def _canonical_context(canonical_root: Path) -> tuple[dict, list[str], list[int], list, dict]:
    meta_path = canonical_root / "metadata.json"
    if not meta_path.is_file():
        raise ConfounderPreprocessingError(f"Canonical metadata is unavailable: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = [str(v) for v in meta.get("bsa_ids", [])]
    years = [int(v) for v in meta.get("years", [])]
    dates = [parse_date(str(v)) for v in meta.get("dates", [])]
    if not ids or not years or not dates:
        raise ConfounderPreprocessingError("Canonical metadata lacks BSA, year or date support")
    if len(set(ids)) != len(ids) or tuple(sorted(set(years))) != tuple(years):
        raise ConfounderPreprocessingError("Canonical BSA/year metadata is not unique and sorted")
    date_index = {d: i for i, d in enumerate(dates)}
    if len(date_index) != len(dates):
        raise ConfounderPreprocessingError("Canonical date axis contains duplicates")
    return meta, ids, years, dates, date_index


def empirical_percentiles_right(values: np.ndarray) -> np.ndarray:
    """Within-BSA/year empirical CDF with right-continuous tie handling."""
    x = np.asarray(values, dtype=float)
    result = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return result
    sorted_values = np.sort(x[valid])
    result[valid] = np.searchsorted(sorted_values, x[valid], side="right") / float(sorted_values.size)
    return result


def _write_percentiles(
    local_path: Path,
    percentile_path: Path,
    valid_counts_path: Path,
    unique_counts_path: Path,
    *,
    bsa_ids: list[str],
    years: list[int],
    dates: list,
) -> tuple[int, tuple[ConfounderPreprocessingExclusion, ...]]:
    year_positions = {
        year: np.array([i for i, d in enumerate(dates) if d.year == year], dtype=int)
        for year in years
    }
    exclusions: list[ConfounderPreprocessingExclusion] = []
    valid_total = 0
    B, n, Y = len(bsa_ids), len(dates), len(years)
    with mapped(local_path) as local, \
         mapped(percentile_path, shape=(B, n)) as percentiles, \
         mapped(valid_counts_path, shape=(B, Y), dtype="int32") as valid_counts, \
         mapped(unique_counts_path, shape=(B, Y), dtype="int32") as unique_counts:
        percentiles[:] = np.nan
        valid_counts[:] = 0
        unique_counts[:] = 0
        for ib, bsa_id in enumerate(bsa_ids):
            for iy, year in enumerate(years):
                positions = year_positions[year]
                values = np.asarray(local[ib, positions], dtype=float)
                finite = np.isfinite(values)
                count = int(np.sum(finite))
                valid_counts[ib, iy] = count
                unique_counts[ib, iy] = int(np.unique(values[finite]).size) if count else 0
                valid_total += count
                if count == 0:
                    exclusions.append(
                        ConfounderPreprocessingExclusion(
                            reason=ConfounderExclusionReason.INSUFFICIENT_PERCENTILE_OBSERVATIONS,
                            bsa_id=bsa_id,
                            year=year,
                            detail="No valid projected candidate observations are available for this BSA-year.",
                        )
                    )
                    continue
                percentiles[ib, positions] = empirical_percentiles_right(values)
    return valid_total, tuple(exclusions)


def preprocess_candidate(
    candidate: ConfounderCandidateDefinition,
    *,
    canonical_root: str | Path,
    data_dir: str | Path,
    project_root: str | Path | None = None,
    log=print,
) -> ConfounderPreprocessingResult:
    """Project one hazard-format candidate and persist BSA/year percentiles."""
    canonical_root = Path(canonical_root).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    project_root = (
        canonical_root.parent if project_root is None else Path(project_root).expanduser().resolve()
    )
    try:
        canonical_root.relative_to(project_root)
    except ValueError as exc:
        raise ConfounderPreprocessingError("canonical_root must be contained by project_root") from exc

    _, bsa_ids, years, dates, _ = _canonical_context(canonical_root)
    B, n = len(bsa_ids), len(dates)
    base = project_root / "confounder"
    final_dir = base / candidate.candidate_id
    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=candidate.candidate_id + ".", suffix=".tmp", dir=base))

    source_hashes: dict[str, str] = {
        "canonical_metadata": sha256_file(canonical_root / "metadata.json"),
    }
    exclusions: list[ConfounderPreprocessingExclusion] = []
    try:
        hazard_file = _resolve_reference(candidate.hazard_file, data_dir=data_dir, project_root=project_root)
        crosswalk_file = _resolve_reference(candidate.crosswalk_file, data_dir=data_dir, project_root=project_root)
        _require_file(hazard_file, "candidate hazard-field")
        _require_file(crosswalk_file, "candidate crosswalk")
        source_hashes["candidate_hazard_file"] = sha256_file(hazard_file)
        source_hashes["candidate_crosswalk_file"] = sha256_file(crosswalk_file)

        grid_path = staging / "grid_hazard.npy"
        local_path = staging / "bsa.npy"
        territory_path = staging / "territory.npy"
        percentile_path = staging / "percentiles.npy"
        valid_counts_path = staging / "percentile_valid_counts.npy"
        unique_counts_path = staging / "percentile_unique_counts.npy"

        try:
            projection = project_hazard_field(
                source_file=hazard_file,
                crosswalk_file=crosswalk_file,
                date_field=candidate.hazard_date_field,
                crosswalk_bsa_field=candidate.crosswalk_bsa_field,
                grid_id_field=candidate.grid_id_field,
                intersection_area_field=candidate.intersection_area_field,
                dates=dates,
                bsa_ids=bsa_ids,
                raw_output=grid_path,
                bsa_output=local_path,
                territory_output=territory_path,
            )
        except SourceError as exc:
            raise ConfounderPreprocessingError(str(exc)) from exc

        percentile_valid_count, percentile_exclusions = _write_percentiles(
            local_path,
            percentile_path,
            valid_counts_path,
            unique_counts_path,
            bsa_ids=bsa_ids,
            years=years,
            dates=dates,
        )
        exclusions.extend(percentile_exclusions)

        with mapped(territory_path) as territory:
            territory_valid_count = int(np.sum(np.isfinite(territory)))
        with mapped(local_path) as local:
            bsa_valid_count = int(np.sum(np.isfinite(local)))
        if territory_valid_count == 0 or bsa_valid_count == 0:
            exclusions.insert(
                0,
                ConfounderPreprocessingExclusion(
                    reason=ConfounderExclusionReason.CANDIDATE_DATA_UNAVAILABLE,
                    detail=(
                        "Projected candidate data have no valid overlap with the canonical analytical panel "
                        "at one or both required analytical scales."
                    ),
                ),
            )

        relative_base = Path("confounder") / candidate.candidate_id
        percentile_panel = ConfounderPercentilePanel(
            candidate_id=candidate.candidate_id,
            array_path=(relative_base / "percentiles.npy").as_posix(),
            bsa_count=B,
            date_count=n,
            valid_observation_count=percentile_valid_count,
        )
        result = ConfounderPreprocessingResult(
            candidate_id=candidate.candidate_id,
            spatial_representation=ConfounderSpatialRepresentation.SPATIAL_FIELD,
            territory_array_path=(relative_base / "territory.npy").as_posix(),
            bsa_array_path=(relative_base / "bsa.npy").as_posix(),
            percentile_panel=percentile_panel,
            valid_counts_array_path=(relative_base / "percentile_valid_counts.npy").as_posix(),
            unique_counts_array_path=(relative_base / "percentile_unique_counts.npy").as_posix(),
            metadata_path=(relative_base / "metadata.json").as_posix(),
            bsa_count=B,
            date_count=n,
            years=tuple(years),
            territory_valid_observation_count=territory_valid_count,
            bsa_valid_observation_count=bsa_valid_count,
            source_hashes=source_hashes,
            exclusions=tuple(exclusions),
        )
        metadata = {
            "preprocessing": result.to_dict(),
            "candidate": candidate.to_dict(),
            "canonical": {
                "bsa_ids": bsa_ids,
                "years": years,
                "dates_ref": "../../canonical/metadata.json",
                "threshold_equality_convention": THRESHOLD_EQUALITY_CONVENTION,
            },
            "input": {
                "format": CONFOUNDER_HAZARD_FIELD_FORMAT,
                "hazard_file": candidate.hazard_file,
                "hazard_date_field": candidate.hazard_date_field,
                "crosswalk_file": candidate.crosswalk_file,
                "crosswalk_bsa_field": candidate.crosswalk_bsa_field,
                "grid_id_field": candidate.grid_id_field,
                "intersection_area_field": candidate.intersection_area_field,
            },
            "harmonization": {
                "rule": "APHREH_STANDARD_HAZARD_FIELD_PROJECTION_V1",
                "projection_rule": projection["projection_rule"],
                "grid_count": int(projection["grid_count"]),
                "intersection_count": int(projection["intersection_count"]),
                "missing_contributor_policy": projection["missing_contributor_policy"],
                "raw_grid_array": (relative_base / "grid_hazard.npy").as_posix(),
            },
            "percentiles": {
                "definition": PERCENTILE_DEFINITION,
                "formula": "U(x)=count(X<=x)/n_valid within BSA-year",
                "tie_rule": "RIGHT_CONTINUOUS_EMPIRICAL_CDF",
            },
        }
        atomic_json(staging / "metadata.json", metadata)

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging, final_dir)
        log(
            f"Confounder CA2: {candidate.candidate_id}, APHREH hazard-field projection, "
            f"grids={projection['grid_count']}, territory valid={territory_valid_count}/{n}, "
            f"BSA valid={bsa_valid_count}/{B*n}, percentile exclusions={len(percentile_exclusions)}"
        )
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_preprocessing_result(
    project_root: str | Path, candidate_id: str
) -> ConfounderPreprocessingResult:
    path = Path(project_root).expanduser().resolve() / "confounder" / candidate_id / "metadata.json"
    if not path.is_file():
        raise ConfounderPreprocessingError(f"Confounder preprocessing metadata is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ConfounderPreprocessingResult.from_dict(payload["preprocessing"])

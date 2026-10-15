"""Immutable contracts for candidate-confounder adjustment.

CA1 defines persistent configuration and later-stage calibration contracts. CA2
adds bounded-memory candidate-preprocessing descriptors, and CA3 adds cross-fitted profiling descriptors. Large date-by-BSA
panels remain disk-backed, matching the authoritative APHREH-ADSMap backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .configuration import DEFAULT_RELEVANCE_CALIPER_GRID, ConfounderAdjustmentSettings


CONFOUNDER_CONTRACT_SCHEMA = "APHREH_CONFOUNDER_CA1_V2_HAZARD_FIELD"
LEGACY_CONFOUNDER_CONTRACT_SCHEMAS = ("APHREH_CONFOUNDER_CA1_V1",)
CONFOUNDER_HAZARD_FIELD_FORMAT = "APHREH_HAZARD_FIELD_WIDE_V1"
PERCENTILE_PANEL_SCHEMA = "APHREH_CONFOUNDER_PERCENTILE_PANEL_V1"
CONFOUNDER_PREPROCESSING_SCHEMA = "APHREH_CONFOUNDER_PREPROCESSING_CA2_V1"
CONFOUNDER_PROFILING_SCHEMA = "APHREH_CONFOUNDER_PROFILING_CA3_V1"
PAIRWISE_SAMPLE_BUNDLE_SCHEMA = "APHREH_CONFOUNDER_PAIRWISE_SAMPLE_CA4_V1"
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ConfounderSpatialRepresentation(str, Enum):
    """Native representation presented to the CA2 harmonizer."""

    TERRITORY_WIDE = "TERRITORY_WIDE"
    BSA_SPECIFIC = "BSA_SPECIFIC"
    SPATIAL_FIELD = "SPATIAL_FIELD"


class ConfounderCalibrationStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NEUTRAL_ZERO_RELEVANCE = "NEUTRAL_ZERO_RELEVANCE"
    UNAVAILABLE = "UNAVAILABLE"


class ConfounderProfilingStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ConfounderExclusionReason(str, Enum):
    """Stable CA17 codes for persistence, diagnostics and export."""

    CANDIDATE_DATA_UNAVAILABLE = "CA17_01_CANDIDATE_DATA_UNAVAILABLE"
    INSUFFICIENT_PERCENTILE_OBSERVATIONS = "CA17_02_INSUFFICIENT_PERCENTILE_OBSERVATIONS"
    CANDIDATE_REPRESENTATION_UNAVAILABLE = "CA17_03_CANDIDATE_REPRESENTATION_UNAVAILABLE"
    INSUFFICIENT_YEARS_PRIMARY = "CA17_04_INSUFFICIENT_YEARS_PRIMARY"
    PROFILING_WMARM_CS_SURFACE_UNAVAILABLE = "CA17_05_PROFILING_WMARM_CS_SURFACE_UNAVAILABLE"
    PROFILING_POOLED_CI_UNAVAILABLE = "CA17_06_PROFILING_POOLED_CI_UNAVAILABLE"
    RELEVANCE_PROFILE_UNAVAILABLE = "CA17_07_RELEVANCE_PROFILE_UNAVAILABLE"
    REFERENCE_CONFIGURATION_UNAVAILABLE = "CA17_08_REFERENCE_CONFIGURATION_UNAVAILABLE"
    NO_ADMISSIBLE_CALIPER = "CA17_09_NO_ADMISSIBLE_CALIPER"
    NO_COMPATIBLE_PAIR = "CA17_10_NO_COMPATIBLE_PAIR"
    HELD_OUT_SUPPORT_BELOW_MINIMUM = "CA17_11_HELD_OUT_SUPPORT_BELOW_MINIMUM"
    ADJUSTED_BOOTSTRAP_PAIR_SUPPORT_EMPTY = "CA17_12_ADJUSTED_BOOTSTRAP_PAIR_SUPPORT_EMPTY"
    EXCESSIVE_INVALID_ADJUSTED_BOOTSTRAPS = "CA17_13_EXCESSIVE_INVALID_ADJUSTED_BOOTSTRAPS"
    INSUFFICIENT_ADJUSTED_YEARS_FOR_POOLING = "CA17_14_INSUFFICIENT_ADJUSTED_YEARS_FOR_POOLING"
    ADJUSTED_POOLED_CI_UNAVAILABLE = "CA17_15_ADJUSTED_POOLED_CI_UNAVAILABLE"
    NO_ELIGIBLE_ADJUSTED_POOLED_BSAS = "CA17_16_NO_ELIGIBLE_ADJUSTED_POOLED_BSAS"
    INSUFFICIENT_YEARS_REOPTIMIZATION = "CA17_17_INSUFFICIENT_YEARS_REOPTIMIZATION"
    NESTED_CALIBRATION_UNAVAILABLE = "CA17_18_NESTED_CALIBRATION_UNAVAILABLE"
    ADJUSTED_NULL_ENSEMBLE_INCOMPLETE = "CA17_19_ADJUSTED_NULL_ENSEMBLE_INCOMPLETE"
    ADJUSTED_REFERENCE_REGION_UNAVAILABLE = "CA17_20_ADJUSTED_REFERENCE_REGION_UNAVAILABLE"


@dataclass(frozen=True)
class ConfounderCandidateDefinition:
    """One candidate confounder using the standard APHREH hazard-field contract.

    The candidate source is a wide hazard CSV containing one date column and one
    column per source grid cell, exactly like ``sources.hazard_file``. The
    candidate is always projected to BSA and territory scales through a standard
    APHREH intersection-area crosswalk. No direct territory-wide or BSA-specific
    candidate input mode is part of the active contract.
    """

    candidate_id: str
    display_name: str
    hazard_file: str
    hazard_date_field: str
    crosswalk_file: str
    crosswalk_bsa_field: str
    grid_id_field: str
    intersection_area_field: str
    units: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    input_format: str = CONFOUNDER_HAZARD_FIELD_FORMAT
    schema_version: str = CONFOUNDER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported confounder contract schema: {self.schema_version}")
        if self.input_format != CONFOUNDER_HAZARD_FIELD_FORMAT:
            raise ValueError(f"unsupported candidate input format: {self.input_format}")
        if not _CANDIDATE_ID.fullmatch(self.candidate_id):
            raise ValueError("candidate_id must be a safe identifier using letters, numbers, '.', '_' or '-'")
        if self.candidate_id in (".", ".."):
            raise ValueError("candidate_id cannot be '.' or '..'")
        if not self.display_name.strip():
            raise ValueError("display_name must be nonempty")
        for name in (
            "hazard_file", "hazard_date_field", "crosswalk_file",
            "crosswalk_bsa_field", "grid_id_field", "intersection_area_field",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.units is not None and not str(self.units).strip():
            raise ValueError("units must be nonempty when supplied")

    @property
    def spatial_representation(self) -> ConfounderSpatialRepresentation:
        """Compatibility property for downstream CA3-CA8 contracts."""
        return ConfounderSpatialRepresentation.SPATIAL_FIELD

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_format": self.input_format,
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "hazard_file": self.hazard_file,
            "hazard_date_field": self.hazard_date_field,
            "crosswalk_file": self.crosswalk_file,
            "crosswalk_bsa_field": self.crosswalk_bsa_field,
            "grid_id_field": self.grid_id_field,
            "intersection_area_field": self.intersection_area_field,
            "units": self.units,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderCandidateDefinition":
        """Load the hazard-field contract, migrating legacy spatial-field mappings.

        Legacy CA1-CA9 ``SPATIAL_FIELD`` definitions are accepted so completed
        projects can be reopened. Legacy TERRITORY_WIDE and BSA_SPECIFIC input
        modes are intentionally rejected because candidate inputs now use the
        standard APHREH hazard-field data model.
        """
        representation = data.get("spatial_representation")
        if representation is not None and str(representation) != ConfounderSpatialRepresentation.SPATIAL_FIELD.value:
            raise ValueError(
                "legacy territory-wide/BSA-specific confounder sources are no longer supported; "
                "provide a standard APHREH hazard-field CSV and crosswalk"
            )
        hazard_file = data.get("hazard_file", data.get("source_reference"))
        crosswalk_file = data.get("crosswalk_file", data.get("crosswalk_source_reference"))
        if hazard_file is None:
            raise KeyError("hazard_file")
        if crosswalk_file is None:
            raise KeyError("crosswalk_file")
        return cls(
            candidate_id=str(data["candidate_id"]),
            display_name=str(data["display_name"]),
            hazard_file=str(hazard_file),
            hazard_date_field=str(data.get("hazard_date_field", data.get("source_date_field", "DATE_STR"))),
            crosswalk_file=str(crosswalk_file),
            crosswalk_bsa_field=str(data.get("crosswalk_bsa_field", "bsa_id")),
            grid_id_field=str(data.get("grid_id_field", data.get("crosswalk_grid_field", "grid_id"))),
            intersection_area_field=str(data.get("intersection_area_field", data.get("crosswalk_area_field", "intersection_area"))),
            units=None if data.get("units") is None else str(data["units"]),
            metadata=dict(data.get("metadata", {})),
            input_format=CONFOUNDER_HAZARD_FIELD_FORMAT,
            schema_version=CONFOUNDER_CONTRACT_SCHEMA,
        )


@dataclass(frozen=True)
class ConfounderPercentilePanel:
    """Disk-backed CA4 percentile-panel descriptor."""

    candidate_id: str
    array_path: str
    bsa_count: int
    date_count: int
    valid_observation_count: int
    dtype: str = "float64"
    percentile_definition: str = "EMPIRICAL_CDF_RIGHT"
    schema_version: str = PERCENTILE_PANEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PERCENTILE_PANEL_SCHEMA:
            raise ValueError(f"unsupported percentile-panel schema: {self.schema_version}")
        _validate_candidate_id(self.candidate_id)
        _validate_relative_path("array_path", self.array_path, suffix=".npy")
        if type(self.bsa_count) is not int or self.bsa_count < 1:
            raise ValueError("bsa_count must be a positive integer")
        if type(self.date_count) is not int or self.date_count < 1:
            raise ValueError("date_count must be a positive integer")
        maximum = self.bsa_count * self.date_count
        if type(self.valid_observation_count) is not int or not 0 <= self.valid_observation_count <= maximum:
            raise ValueError("valid_observation_count must lie in [0, bsa_count*date_count]")
        if self.dtype != "float64":
            raise ValueError("CA2 percentile panels use float64")
        if self.percentile_definition != "EMPIRICAL_CDF_RIGHT":
            raise ValueError("CA2 percentile_definition must be EMPIRICAL_CDF_RIGHT")

    @property
    def shape(self) -> tuple[int, int]:
        return self.bsa_count, self.date_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "array_path": self.array_path,
            "bsa_count": self.bsa_count,
            "date_count": self.date_count,
            "valid_observation_count": self.valid_observation_count,
            "dtype": self.dtype,
            "percentile_definition": self.percentile_definition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderPercentilePanel":
        return cls(
            candidate_id=str(data["candidate_id"]),
            array_path=str(data["array_path"]),
            bsa_count=int(data["bsa_count"]),
            date_count=int(data["date_count"]),
            valid_observation_count=int(data["valid_observation_count"]),
            dtype=str(data.get("dtype", "float64")),
            percentile_definition=str(data.get("percentile_definition", "EMPIRICAL_CDF_RIGHT")),
            schema_version=str(data.get("schema_version", PERCENTILE_PANEL_SCHEMA)),
        )


@dataclass(frozen=True)
class ConfounderPreprocessingExclusion:
    """Explicit CA2 candidate-data or BSA-year preprocessing exclusion."""

    reason: ConfounderExclusionReason
    bsa_id: str | None = None
    year: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.reason not in {
            ConfounderExclusionReason.CANDIDATE_DATA_UNAVAILABLE,
            ConfounderExclusionReason.INSUFFICIENT_PERCENTILE_OBSERVATIONS,
            ConfounderExclusionReason.CANDIDATE_REPRESENTATION_UNAVAILABLE,
        }:
            raise ValueError("CA2 preprocessing exclusions must use CA17_01, CA17_02 or CA17_03")
        if self.bsa_id is not None and not str(self.bsa_id).strip():
            raise ValueError("bsa_id must be nonempty when supplied")
        if self.year is not None and not 1000 <= int(self.year) <= 9999:
            raise ValueError("year must be a four-digit calendar year when supplied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "bsa_id": self.bsa_id,
            "year": self.year,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderPreprocessingExclusion":
        return cls(
            reason=ConfounderExclusionReason(data["reason"]),
            bsa_id=data.get("bsa_id"),
            year=None if data.get("year") is None else int(data["year"]),
            detail=data.get("detail"),
        )


@dataclass(frozen=True)
class ConfounderPreprocessingResult:
    """Persistent CA2 descriptor for harmonized candidate data and percentiles."""

    candidate_id: str
    spatial_representation: ConfounderSpatialRepresentation
    territory_array_path: str
    bsa_array_path: str
    percentile_panel: ConfounderPercentilePanel
    valid_counts_array_path: str
    unique_counts_array_path: str
    metadata_path: str
    bsa_count: int
    date_count: int
    years: tuple[int, ...]
    territory_valid_observation_count: int
    bsa_valid_observation_count: int
    source_hashes: Mapping[str, str]
    exclusions: tuple[ConfounderPreprocessingExclusion, ...] = ()
    schema_version: str = CONFOUNDER_PREPROCESSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_PREPROCESSING_SCHEMA:
            raise ValueError(f"unsupported confounder preprocessing schema: {self.schema_version}")
        _validate_candidate_id(self.candidate_id)
        if self.percentile_panel.candidate_id != self.candidate_id:
            raise ValueError("percentile_panel candidate_id must match preprocessing result")
        if self.percentile_panel.shape != (self.bsa_count, self.date_count):
            raise ValueError("percentile_panel shape must match preprocessing result")
        for name in ("territory_array_path", "bsa_array_path", "valid_counts_array_path", "unique_counts_array_path"):
            _validate_relative_path(name, getattr(self, name), suffix=".npy")
        _validate_relative_path("metadata_path", self.metadata_path, suffix=".json")
        if tuple(sorted(set(int(y) for y in self.years))) != self.years or not self.years:
            raise ValueError("years must be nonempty, unique and sorted")
        if type(self.bsa_count) is not int or self.bsa_count < 1:
            raise ValueError("bsa_count must be positive")
        if type(self.date_count) is not int or self.date_count < 1:
            raise ValueError("date_count must be positive")
        if not 0 <= self.territory_valid_observation_count <= self.date_count:
            raise ValueError("invalid territory_valid_observation_count")
        if not 0 <= self.bsa_valid_observation_count <= self.bsa_count * self.date_count:
            raise ValueError("invalid bsa_valid_observation_count")
        for name, digest in self.source_hashes.items():
            if not str(name).strip() or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise ValueError("source_hashes must contain lowercase SHA-256 digests")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "spatial_representation": self.spatial_representation.value,
            "territory_array_path": self.territory_array_path,
            "bsa_array_path": self.bsa_array_path,
            "percentile_panel": self.percentile_panel.to_dict(),
            "valid_counts_array_path": self.valid_counts_array_path,
            "unique_counts_array_path": self.unique_counts_array_path,
            "metadata_path": self.metadata_path,
            "bsa_count": self.bsa_count,
            "date_count": self.date_count,
            "years": list(self.years),
            "territory_valid_observation_count": self.territory_valid_observation_count,
            "bsa_valid_observation_count": self.bsa_valid_observation_count,
            "source_hashes": dict(self.source_hashes),
            "exclusions": [item.to_dict() for item in self.exclusions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderPreprocessingResult":
        return cls(
            candidate_id=str(data["candidate_id"]),
            spatial_representation=ConfounderSpatialRepresentation(data["spatial_representation"]),
            territory_array_path=str(data["territory_array_path"]),
            bsa_array_path=str(data["bsa_array_path"]),
            percentile_panel=ConfounderPercentilePanel.from_dict(data["percentile_panel"]),
            valid_counts_array_path=str(data["valid_counts_array_path"]),
            unique_counts_array_path=str(data["unique_counts_array_path"]),
            metadata_path=str(data["metadata_path"]),
            bsa_count=int(data["bsa_count"]),
            date_count=int(data["date_count"]),
            years=tuple(int(y) for y in data["years"]),
            territory_valid_observation_count=int(data["territory_valid_observation_count"]),
            bsa_valid_observation_count=int(data["bsa_valid_observation_count"]),
            source_hashes=dict(data.get("source_hashes", {})),
            exclusions=tuple(ConfounderPreprocessingExclusion.from_dict(v) for v in data.get("exclusions", [])),
            schema_version=str(data.get("schema_version", CONFOUNDER_PREPROCESSING_SCHEMA)),
        )


@dataclass(frozen=True)
class ConfounderProfilingResult:
    """CA3 descriptor for one held-out-year candidate-only APHREH surface.

    The complete surface is retained in the ordinary Store under
    ``surface_run_id``. This contract intentionally retains only small
    metadata and diagnostics so profiling remains bounded-memory.
    """

    candidate_id: str
    held_out_year: int
    training_years: tuple[int, ...]
    surface_run_id: str
    thresholds: tuple[float, ...]
    lags: tuple[int, ...]
    cell_count: int
    valid_cell_count: int
    status: ConfounderProfilingStatus
    wmarm_cs_max: float | None = None
    population_coverage_min: float | None = None
    population_coverage_max: float | None = None
    exclusion_reason: ConfounderExclusionReason | None = None
    reference_metric: str = "WMARM_CS"
    schema_version: str = CONFOUNDER_PROFILING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_PROFILING_SCHEMA:
            raise ValueError(f"unsupported confounder profiling schema: {self.schema_version}")
        _validate_candidate_id(self.candidate_id)
        _validate_crossfit_years(self.held_out_year, self.training_years)
        if not self.surface_run_id.strip():
            raise ValueError("surface_run_id must be nonempty")
        if not self.thresholds or tuple(sorted(set(float(v) for v in self.thresholds))) != self.thresholds:
            raise ValueError("thresholds must be nonempty, unique and sorted")
        if any(not isfinite(float(v)) or not 0.0 <= float(v) <= 100.0 for v in self.thresholds):
            raise ValueError("thresholds must be finite percentiles in [0, 100]")
        if not self.lags or tuple(sorted(set(int(v) for v in self.lags))) != self.lags:
            raise ValueError("lags must be nonempty, unique and sorted")
        if any(type(v) is not int or v < 0 for v in self.lags):
            raise ValueError("lags must contain nonnegative integers")
        expected = len(self.thresholds) * len(self.lags)
        if type(self.cell_count) is not int or self.cell_count != expected:
            raise ValueError("cell_count must equal len(thresholds)*len(lags)")
        if type(self.valid_cell_count) is not int or not 0 <= self.valid_cell_count <= self.cell_count:
            raise ValueError("valid_cell_count must lie in [0, cell_count]")
        if self.reference_metric != "WMARM_CS":
            raise ValueError("CA3 profiling reference metric must be WMARM_CS")
        for name in ("wmarm_cs_max", "population_coverage_min", "population_coverage_max"):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied")
        for name in ("population_coverage_min", "population_coverage_max"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.status is ConfounderProfilingStatus.AVAILABLE:
            if self.valid_cell_count < 1:
                raise ValueError("available profiling result requires at least one valid cell")
            if self.exclusion_reason is not None:
                raise ValueError("available profiling result cannot have exclusion_reason")
        else:
            if self.exclusion_reason is not ConfounderExclusionReason.PROFILING_WMARM_CS_SURFACE_UNAVAILABLE:
                raise ValueError("unavailable profiling result requires CA17_05 exclusion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "held_out_year": self.held_out_year,
            "training_years": list(self.training_years),
            "surface_run_id": self.surface_run_id,
            "thresholds": list(self.thresholds),
            "lags": list(self.lags),
            "cell_count": self.cell_count,
            "valid_cell_count": self.valid_cell_count,
            "status": self.status.value,
            "wmarm_cs_max": self.wmarm_cs_max,
            "population_coverage_min": self.population_coverage_min,
            "population_coverage_max": self.population_coverage_max,
            "exclusion_reason": None if self.exclusion_reason is None else self.exclusion_reason.value,
            "reference_metric": self.reference_metric,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderProfilingResult":
        exclusion = data.get("exclusion_reason")
        return cls(
            candidate_id=str(data["candidate_id"]),
            held_out_year=int(data["held_out_year"]),
            training_years=tuple(int(y) for y in data.get("training_years", [])),
            surface_run_id=str(data["surface_run_id"]),
            thresholds=tuple(float(v) for v in data.get("thresholds", [])),
            lags=tuple(int(v) for v in data.get("lags", [])),
            cell_count=int(data["cell_count"]),
            valid_cell_count=int(data["valid_cell_count"]),
            status=ConfounderProfilingStatus(data["status"]),
            wmarm_cs_max=None if data.get("wmarm_cs_max") is None else float(data["wmarm_cs_max"]),
            population_coverage_min=None if data.get("population_coverage_min") is None else float(data["population_coverage_min"]),
            population_coverage_max=None if data.get("population_coverage_max") is None else float(data["population_coverage_max"]),
            exclusion_reason=None if exclusion is None else ConfounderExclusionReason(exclusion),
            reference_metric=str(data.get("reference_metric", "WMARM_CS")),
            schema_version=str(data.get("schema_version", CONFOUNDER_PROFILING_SCHEMA)),
        )


@dataclass(frozen=True)
class PairwiseSampleBundle:
    """Disk-backed CA4 target-hazard original-sample panel for one cross-fit fold.

    The three arrays have canonical shape ``(B, n_dates)``. Candidate percentiles
    are not duplicated; ``candidate_percentile_array_path`` points to the CA2
    percentile panel. Held-out outcome dates are unavailable in the differential
    panel because extraction uses the same cross-fit masking rules as CA3.
    """

    candidate_id: str
    held_out_year: int
    training_years: tuple[int, ...]
    target_threshold: float
    target_lag: int
    differential_array_path: str
    raw_weight_array_path: str
    exposure_label_array_path: str
    candidate_percentile_array_path: str
    bsa_count: int
    date_count: int
    valid_differential_count: int
    positive_weight_count: int
    reference_configuration_source: str = "ROBUST_REGION_REPRESENTATIVE"
    schema_version: str = PAIRWISE_SAMPLE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PAIRWISE_SAMPLE_BUNDLE_SCHEMA:
            raise ValueError(f"unsupported pairwise-sample schema: {self.schema_version}")
        _validate_candidate_id(self.candidate_id)
        _validate_crossfit_years(self.held_out_year, self.training_years)
        if not isfinite(float(self.target_threshold)) or not 0.0 <= float(self.target_threshold) <= 100.0:
            raise ValueError("target_threshold must be a finite percentile in [0, 100]")
        if type(self.target_lag) is not int or self.target_lag < 0:
            raise ValueError("target_lag must be a nonnegative integer")
        for name in (
            "differential_array_path", "raw_weight_array_path",
            "exposure_label_array_path", "candidate_percentile_array_path",
        ):
            _validate_relative_path(name, getattr(self, name), suffix=".npy")
        if type(self.bsa_count) is not int or self.bsa_count < 1:
            raise ValueError("bsa_count must be positive")
        if type(self.date_count) is not int or self.date_count < 1:
            raise ValueError("date_count must be positive")
        maximum = self.bsa_count * self.date_count
        if type(self.valid_differential_count) is not int or not 0 <= self.valid_differential_count <= maximum:
            raise ValueError("valid_differential_count is outside the panel bounds")
        if type(self.positive_weight_count) is not int or not 0 <= self.positive_weight_count <= maximum:
            raise ValueError("positive_weight_count is outside the panel bounds")
        if not self.reference_configuration_source.strip():
            raise ValueError("reference_configuration_source must be nonempty")

    @property
    def shape(self) -> tuple[int, int]:
        return self.bsa_count, self.date_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "held_out_year": self.held_out_year,
            "training_years": list(self.training_years),
            "target_threshold": self.target_threshold,
            "target_lag": self.target_lag,
            "differential_array_path": self.differential_array_path,
            "raw_weight_array_path": self.raw_weight_array_path,
            "exposure_label_array_path": self.exposure_label_array_path,
            "candidate_percentile_array_path": self.candidate_percentile_array_path,
            "bsa_count": self.bsa_count,
            "date_count": self.date_count,
            "valid_differential_count": self.valid_differential_count,
            "positive_weight_count": self.positive_weight_count,
            "reference_configuration_source": self.reference_configuration_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairwiseSampleBundle":
        return cls(
            candidate_id=str(data["candidate_id"]),
            held_out_year=int(data["held_out_year"]),
            training_years=tuple(int(y) for y in data.get("training_years", [])),
            target_threshold=float(data["target_threshold"]),
            target_lag=int(data["target_lag"]),
            differential_array_path=str(data["differential_array_path"]),
            raw_weight_array_path=str(data["raw_weight_array_path"]),
            exposure_label_array_path=str(data["exposure_label_array_path"]),
            candidate_percentile_array_path=str(data["candidate_percentile_array_path"]),
            bsa_count=int(data["bsa_count"]),
            date_count=int(data["date_count"]),
            valid_differential_count=int(data["valid_differential_count"]),
            positive_weight_count=int(data["positive_weight_count"]),
            reference_configuration_source=str(data.get("reference_configuration_source", "ROBUST_REGION_REPRESENTATIVE")),
            schema_version=str(data.get("schema_version", PAIRWISE_SAMPLE_BUNDLE_SCHEMA)),
        )


@dataclass(frozen=True)
class ConfounderRelevanceProfile:
    """Cross-fitted CA4 relevance profile derived from reference WMARMcs."""

    candidate_id: str
    held_out_year: int
    training_years: tuple[int, ...]
    threshold_relevance: Mapping[float, float]
    normalized_relevance: Mapping[float, float]
    relevance_amplitude: float
    status: ConfounderCalibrationStatus
    reference_metric: str = "WMARM_CS"
    diagnostic_wmarm_by_threshold: Mapping[float, float] = field(default_factory=dict)
    population_coverage_by_threshold: Mapping[float, float] = field(default_factory=dict)
    schema_version: str = CONFOUNDER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported confounder contract schema: {self.schema_version}")
        _validate_crossfit_years(self.held_out_year, self.training_years)
        if self.reference_metric != "WMARM_CS":
            raise ValueError("CA4 primary reference metric must be WMARM_CS")
        _unit_interval("relevance_amplitude", self.relevance_amplitude)
        _validate_threshold_map("threshold_relevance", self.threshold_relevance)
        _validate_threshold_map("normalized_relevance", self.normalized_relevance)
        _validate_threshold_map(
            "diagnostic_wmarm_by_threshold", self.diagnostic_wmarm_by_threshold
        )
        _validate_threshold_map(
            "population_coverage_by_threshold", self.population_coverage_by_threshold
        )
        if self.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE:
            if self.relevance_amplitude != 0.0:
                raise ValueError("neutral profile requires relevance_amplitude == 0")
            if any(float(v) != 0.0 for v in self.threshold_relevance.values()):
                raise ValueError("neutral profile requires zero threshold relevance")
            if self.normalized_relevance:
                raise ValueError("neutral profile does not define normalized_relevance")
        elif self.status is ConfounderCalibrationStatus.AVAILABLE:
            if not self.normalized_relevance:
                raise ValueError("available profile requires normalized_relevance")
            if abs(sum(float(v) for v in self.normalized_relevance.values()) - 1.0) > 1e-9:
                raise ValueError("normalized_relevance must sum to 1")

    @property
    def is_neutral(self) -> bool:
        return self.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "held_out_year": self.held_out_year,
            "training_years": list(self.training_years),
            "threshold_relevance": _float_map_to_json(self.threshold_relevance),
            "normalized_relevance": _float_map_to_json(self.normalized_relevance),
            "relevance_amplitude": self.relevance_amplitude,
            "status": self.status.value,
            "reference_metric": self.reference_metric,
            "diagnostic_wmarm_by_threshold": _float_map_to_json(
                self.diagnostic_wmarm_by_threshold
            ),
            "population_coverage_by_threshold": _float_map_to_json(
                self.population_coverage_by_threshold
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderRelevanceProfile":
        return cls(
            candidate_id=str(data["candidate_id"]),
            held_out_year=int(data["held_out_year"]),
            training_years=tuple(int(y) for y in data.get("training_years", [])),
            threshold_relevance=_json_to_float_map(data.get("threshold_relevance", {})),
            normalized_relevance=_json_to_float_map(data.get("normalized_relevance", {})),
            relevance_amplitude=float(data["relevance_amplitude"]),
            status=ConfounderCalibrationStatus(data["status"]),
            reference_metric=str(data.get("reference_metric", "WMARM_CS")),
            diagnostic_wmarm_by_threshold=_json_to_float_map(
                data.get("diagnostic_wmarm_by_threshold", {})
            ),
            population_coverage_by_threshold=_json_to_float_map(
                data.get("population_coverage_by_threshold", {})
            ),
            schema_version=str(data.get("schema_version", CONFOUNDER_CONTRACT_SCHEMA)),
        )


@dataclass(frozen=True)
class ConfounderCalibrationResult:
    """CA4 cross-fitted caliper calibration result for one held-out year."""

    candidate_id: str
    held_out_year: int
    training_years: tuple[int, ...]
    relevance_profile: ConfounderRelevanceProfile | None
    selected_caliper: float | None
    residual_relevance_by_caliper: Mapping[float, float]
    retained_support_by_caliper: Mapping[float, float]
    status: ConfounderCalibrationStatus
    selected_caliper_by_tolerance: Mapping[float, float | None] = field(default_factory=dict)
    exclusion_reason: ConfounderExclusionReason | None = None
    schema_version: str = CONFOUNDER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported confounder contract schema: {self.schema_version}")
        _validate_crossfit_years(self.held_out_year, self.training_years)
        if self.relevance_profile is not None:
            if self.relevance_profile.candidate_id != self.candidate_id:
                raise ValueError("relevance_profile candidate_id must match calibration")
            if self.relevance_profile.held_out_year != self.held_out_year:
                raise ValueError("relevance_profile held_out_year must match calibration")
            if self.relevance_profile.training_years != self.training_years:
                raise ValueError("relevance_profile training_years must match calibration")
        if self.status is ConfounderCalibrationStatus.UNAVAILABLE:
            if self.exclusion_reason is None:
                raise ValueError("unavailable calibration requires exclusion_reason")
        elif self.exclusion_reason is not None:
            raise ValueError("available/neutral calibration cannot have exclusion_reason")
        if self.status is ConfounderCalibrationStatus.AVAILABLE:
            if self.relevance_profile is None:
                raise ValueError("available calibration requires relevance_profile")
            if self.selected_caliper is None:
                raise ValueError("available calibration requires selected_caliper")
            _unit_interval("selected_caliper", self.selected_caliper)
            if not _is_standard_caliper(self.selected_caliper):
                raise ValueError("selected_caliper must belong to the fixed CA8.1 grid")
        elif self.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE:
            if self.relevance_profile is None or not self.relevance_profile.is_neutral:
                raise ValueError("neutral calibration requires neutral relevance_profile")
            if self.selected_caliper is not None:
                raise ValueError("neutral calibration does not select a caliper")
        _validate_unit_interval_map(
            "residual_relevance_by_caliper", self.residual_relevance_by_caliper,
            key_is_caliper=True,
        )
        _validate_unit_interval_map(
            "retained_support_by_caliper", self.retained_support_by_caliper,
            key_is_caliper=True,
        )
        for tolerance, caliper in self.selected_caliper_by_tolerance.items():
            _unit_interval("selected_caliper_by_tolerance tolerance", tolerance)
            if caliper is not None:
                _unit_interval("selected_caliper_by_tolerance caliper", caliper)
                if not _is_standard_caliper(caliper):
                    raise ValueError("selected_caliper_by_tolerance values must belong to the fixed CA8.1 grid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "held_out_year": self.held_out_year,
            "training_years": list(self.training_years),
            "relevance_profile": None if self.relevance_profile is None else self.relevance_profile.to_dict(),
            "selected_caliper": self.selected_caliper,
            "residual_relevance_by_caliper": _float_map_to_json(
                self.residual_relevance_by_caliper
            ),
            "retained_support_by_caliper": _float_map_to_json(
                self.retained_support_by_caliper
            ),
            "selected_caliper_by_tolerance": {
                f"{float(k):.12g}": (None if v is None else float(v))
                for k, v in sorted(self.selected_caliper_by_tolerance.items())
            },
            "status": self.status.value,
            "exclusion_reason": None if self.exclusion_reason is None else self.exclusion_reason.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderCalibrationResult":
        profile = data.get("relevance_profile")
        exclusion = data.get("exclusion_reason")
        selected = data.get("selected_caliper")
        return cls(
            candidate_id=str(data["candidate_id"]),
            held_out_year=int(data["held_out_year"]),
            training_years=tuple(int(y) for y in data.get("training_years", [])),
            relevance_profile=None if profile is None else ConfounderRelevanceProfile.from_dict(profile),
            selected_caliper=None if selected is None else float(selected),
            residual_relevance_by_caliper=_json_to_float_map(
                data.get("residual_relevance_by_caliper", {})
            ),
            retained_support_by_caliper=_json_to_float_map(
                data.get("retained_support_by_caliper", {})
            ),
            status=ConfounderCalibrationStatus(data["status"]),
            selected_caliper_by_tolerance={
                float(k): (None if v is None else float(v))
                for k, v in dict(data.get("selected_caliper_by_tolerance", {})).items()
            },
            exclusion_reason=None if exclusion is None else ConfounderExclusionReason(exclusion),
            schema_version=str(data.get("schema_version", CONFOUNDER_CONTRACT_SCHEMA)),
        )


@dataclass(frozen=True)
class NestedConfounderCalibrationKey:
    """CA16.2 cache identity for leakage-free nested leave-one-year-out work."""

    candidate_id: str
    outer_deleted_year: int
    target_year: int

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be nonempty")
        if self.outer_deleted_year == self.target_year:
            raise ValueError("outer_deleted_year and target_year must differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "outer_deleted_year": self.outer_deleted_year,
            "target_year": self.target_year,
        }


@dataclass(frozen=True)
class ConfounderCompatibilityContext:
    """Frozen per-year context to be consumed later by CA5 estimator work."""

    candidate_id: str
    held_out_year: int
    training_years: tuple[int, ...]
    normalized_relevance: Mapping[float, float]
    relevance_amplitude: float
    selected_caliper: float | None
    continuous_percentile_safeguard: float
    minimum_retained_support: float
    neutral_compatibility: bool = False
    calibration_status: ConfounderCalibrationStatus = ConfounderCalibrationStatus.AVAILABLE
    outer_deleted_year: int | None = None
    schema_version: str = CONFOUNDER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFOUNDER_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported confounder contract schema: {self.schema_version}")
        _validate_crossfit_years(self.held_out_year, self.training_years)
        if self.outer_deleted_year is not None:
            if self.outer_deleted_year == self.held_out_year:
                raise ValueError("outer_deleted_year cannot equal held_out_year")
            if self.outer_deleted_year in self.training_years:
                raise ValueError("outer_deleted_year cannot occur in nested training_years")
        _unit_interval("relevance_amplitude", self.relevance_amplitude)
        _unit_interval(
            "continuous_percentile_safeguard", self.continuous_percentile_safeguard
        )
        _unit_interval("minimum_retained_support", self.minimum_retained_support)
        _validate_threshold_map("normalized_relevance", self.normalized_relevance)
        if self.neutral_compatibility:
            if self.relevance_amplitude != 0.0 or self.normalized_relevance:
                raise ValueError("neutral compatibility requires zero/empty relevance")
            if self.selected_caliper is not None:
                raise ValueError("neutral compatibility does not use selected_caliper")
            if self.calibration_status is not ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE:
                raise ValueError("neutral compatibility requires NEUTRAL_ZERO_RELEVANCE status")
        elif self.calibration_status is ConfounderCalibrationStatus.AVAILABLE:
            if self.selected_caliper is None:
                raise ValueError("available compatibility requires selected_caliper")
            _unit_interval("selected_caliper", self.selected_caliper)
            if not _is_standard_caliper(self.selected_caliper):
                raise ValueError("selected_caliper must belong to the fixed CA8.1 grid")
            if not self.normalized_relevance:
                raise ValueError("available compatibility requires normalized_relevance")

    @classmethod
    def from_calibration(
        cls,
        calibration: ConfounderCalibrationResult,
        settings: ConfounderAdjustmentSettings,
        *,
        outer_deleted_year: int | None = None,
    ) -> "ConfounderCompatibilityContext":
        if calibration.status is ConfounderCalibrationStatus.UNAVAILABLE:
            raise ValueError("cannot create compatibility context from unavailable calibration")
        profile = calibration.relevance_profile
        if profile is None:
            raise ValueError("calibration must carry a relevance_profile")
        return cls(
            candidate_id=calibration.candidate_id,
            held_out_year=calibration.held_out_year,
            training_years=calibration.training_years,
            normalized_relevance=dict(profile.normalized_relevance),
            relevance_amplitude=profile.relevance_amplitude,
            selected_caliper=calibration.selected_caliper,
            continuous_percentile_safeguard=settings.continuous_percentile_safeguard,
            minimum_retained_support=settings.minimum_retained_support,
            neutral_compatibility=(
                calibration.status is ConfounderCalibrationStatus.NEUTRAL_ZERO_RELEVANCE
            ),
            calibration_status=calibration.status,
            outer_deleted_year=outer_deleted_year,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "held_out_year": self.held_out_year,
            "training_years": list(self.training_years),
            "normalized_relevance": _float_map_to_json(self.normalized_relevance),
            "relevance_amplitude": self.relevance_amplitude,
            "selected_caliper": self.selected_caliper,
            "continuous_percentile_safeguard": self.continuous_percentile_safeguard,
            "minimum_retained_support": self.minimum_retained_support,
            "neutral_compatibility": self.neutral_compatibility,
            "calibration_status": self.calibration_status.value,
            "outer_deleted_year": self.outer_deleted_year,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderCompatibilityContext":
        selected = data.get("selected_caliper")
        outer = data.get("outer_deleted_year")
        return cls(
            candidate_id=str(data["candidate_id"]),
            held_out_year=int(data["held_out_year"]),
            training_years=tuple(int(y) for y in data.get("training_years", [])),
            normalized_relevance=_json_to_float_map(data.get("normalized_relevance", {})),
            relevance_amplitude=float(data["relevance_amplitude"]),
            selected_caliper=None if selected is None else float(selected),
            continuous_percentile_safeguard=float(data["continuous_percentile_safeguard"]),
            minimum_retained_support=float(data["minimum_retained_support"]),
            neutral_compatibility=bool(data.get("neutral_compatibility", False)),
            calibration_status=ConfounderCalibrationStatus(
                data.get("calibration_status", ConfounderCalibrationStatus.AVAILABLE.value)
            ),
            outer_deleted_year=None if outer is None else int(outer),
            schema_version=str(data.get("schema_version", CONFOUNDER_CONTRACT_SCHEMA)),
        )


def _is_standard_caliper(value: float) -> bool:
    return any(abs(float(value) - candidate) <= 1e-12 for candidate in DEFAULT_RELEVANCE_CALIPER_GRID)


def _validate_crossfit_years(held_out_year: int, training_years: tuple[int, ...]) -> None:
    if held_out_year in training_years:
        raise ValueError("held_out_year cannot occur in training_years")
    if tuple(sorted(set(training_years))) != training_years:
        raise ValueError("training_years must be unique and sorted")


def _unit_interval(name: str, value: float) -> None:
    if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _validate_threshold_map(name: str, values: Mapping[float, float]) -> None:
    for threshold, value in values.items():
        if not isfinite(float(threshold)) or not 0.0 < float(threshold) < 100.0:
            raise ValueError(f"{name} threshold keys must lie strictly between 0 and 100")
        _unit_interval(f"{name} value", value)


def _validate_unit_interval_map(
    name: str, values: Mapping[float, float], *, key_is_caliper: bool = False
) -> None:
    for key, value in values.items():
        _unit_interval(f"{name} {'caliper' if key_is_caliper else 'key'}", key)
        _unit_interval(f"{name} value", value)


def _float_map_to_json(values: Mapping[float, float]) -> dict[str, float]:
    return {format(float(k), ".12g"): float(v) for k, v in sorted(values.items())}


def _json_to_float_map(values: Mapping[str, Any]) -> dict[float, float]:
    return {float(k): float(v) for k, v in values.items()}


def _validate_candidate_id(candidate_id: str) -> None:
    if not _CANDIDATE_ID.fullmatch(str(candidate_id)) or str(candidate_id) in (".", ".."):
        raise ValueError("candidate_id must be a safe identifier")


def _validate_relative_path(name: str, value: str, *, suffix: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be nonempty")
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a project-relative path")
    if path.suffix.lower() != suffix:
        raise ValueError(f"{name} must reference a {suffix} file")

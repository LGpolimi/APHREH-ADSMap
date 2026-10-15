"""Immutable contracts for robustness/sensitivity methodological realizations.

Step RS-1 infrastructure only.  These contracts describe *what* a realization
is and where its first computational divergence from the reference occurs.  No
sensitivity estimator is implemented in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import isfinite
from typing import Any, Mapping

from ..hashing import hash_object


REALIZATION_SCHEMA = "APHREH_RS_REALIZATION_V1"
PLAN_SCHEMA = "APHREH_RS_PLAN_V1"


class RealizationKind(str, Enum):
    REFERENCE = "REFERENCE"
    SENSITIVITY = "SENSITIVITY"


class ComparisonMode(str, Enum):
    FIXED_PARAMETER = "FIXED_PARAMETER"
    REOPTIMIZED = "REOPTIMIZED"
    BOTH = "BOTH"


class FactorKind(str, Enum):
    NUMERICAL = "NUMERICAL"
    BINARY = "BINARY"
    CATEGORICAL = "CATEGORICAL"


class InvalidationStage(IntEnum):
    """Earliest stage that must be recomputed after a methodological change.

    Numeric ordering is intentional: a smaller value invalidates more of the
    downstream pipeline.  ``NONE`` is reserved for the reference realization.
    """

    NONE = 0
    CANONICAL_INPUT = 10
    EXPOSURE = 20
    INCIDENCE = 30
    BASELINE = 40
    YEARLY_RC = 50
    BOOTSTRAP = 60
    POOLING = 70
    TERRITORY = 80
    OPTIMIZATION = 90
    COMPARISON = 100


class RealizationState(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


class CheckpointState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class FactorSetting:
    """One realized methodological factor value."""

    name: str
    config_path: str
    value: Any
    invalidates_from: InvalidationStage

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("factor name must be nonempty")
        if not self.config_path.strip() or any(not p for p in self.config_path.split(".")):
            raise ValueError("config_path must be a nonempty dotted path")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("factor values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": self.config_path,
            "value": self.value,
            "invalidates_from": self.invalidates_from.name,
        }


@dataclass(frozen=True)
class RealizationSpec:
    """Immutable identity of one complete APHREH methodological realization."""

    realization_id: str
    kind: RealizationKind
    scenario_id: str
    analysis_id: str
    step_index: int | None
    trajectory_coordinate: float | None
    comparison_mode: ComparisonMode
    invalidates_from: InvalidationStage
    reference_realization_id: str | None
    resolved_settings_hash: str
    factors: tuple[FactorSetting, ...] = ()
    label: str = ""
    schema_version: str = REALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if not self.realization_id.strip():
            raise ValueError("realization_id must be nonempty")
        if not self.scenario_id.strip() or not self.analysis_id.strip():
            raise ValueError("scenario_id and analysis_id must be nonempty")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("step_index must be nonnegative")
        if self.trajectory_coordinate is not None and not 0.0 <= self.trajectory_coordinate <= 1.0:
            raise ValueError("trajectory_coordinate must be in [0, 1]")
        if len(self.resolved_settings_hash) != 64:
            raise ValueError("resolved_settings_hash must be a SHA-256 digest")
        names = [f.name for f in self.factors]
        paths = [f.config_path for f in self.factors]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("factor names and config paths must be unique within a realization")
        if self.kind is RealizationKind.REFERENCE:
            if self.factors or self.reference_realization_id is not None:
                raise ValueError("reference realization cannot contain factor overrides or a reference parent")
            if self.invalidates_from is not InvalidationStage.NONE:
                raise ValueError("reference realization must use invalidates_from=NONE")
        else:
            if not self.factors:
                raise ValueError("sensitivity realization requires at least one factor")
            if not self.reference_realization_id:
                raise ValueError("sensitivity realization requires reference_realization_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "realization_id": self.realization_id,
            "kind": self.kind.value,
            "scenario_id": self.scenario_id,
            "analysis_id": self.analysis_id,
            "step_index": self.step_index,
            "trajectory_coordinate": self.trajectory_coordinate,
            "comparison_mode": self.comparison_mode.value,
            "invalidates_from": self.invalidates_from.name,
            "reference_realization_id": self.reference_realization_id,
            "resolved_settings_hash": self.resolved_settings_hash,
            "factors": [f.to_dict() for f in self.factors],
            "label": self.label,
        }

    @property
    def specification_hash(self) -> str:
        return hash_object(self.to_dict())

    @property
    def overrides(self) -> dict[str, Any]:
        return {f.config_path: f.value for f in self.factors}


@dataclass(frozen=True)
class SensitivityPlan:
    """Deterministic reference plus sensitivity realization plan."""

    plan_id: str
    reference: RealizationSpec
    realizations: tuple[RealizationSpec, ...]
    schema_version: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.reference.kind is not RealizationKind.REFERENCE:
            raise ValueError("plan reference must be a REFERENCE realization")
        ids = [self.reference.realization_id, *(r.realization_id for r in self.realizations)]
        if len(ids) != len(set(ids)):
            raise ValueError("realization IDs must be unique within a plan")
        if any(r.reference_realization_id != self.reference.realization_id for r in self.realizations):
            raise ValueError("all sensitivity realizations must point to the plan reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "reference": self.reference.to_dict(),
            "realizations": [r.to_dict() for r in self.realizations],
        }

    @property
    def all_realizations(self) -> tuple[RealizationSpec, ...]:
        return (self.reference, *self.realizations)

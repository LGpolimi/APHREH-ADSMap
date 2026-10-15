"""Deterministic planner for robustness/sensitivity realizations.

The planner implements the common RS design mechanics only: one-factor or
joint scenarios, synchronized numerical trajectories, jointly applied
binary/categorical alternatives, duplicate-step removal after integer rounding,
and deterministic realization identities.  Analysis-specific execution consumes these plans; Analysis 2 support sensitivity is enabled in RS step 3.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from ..hashing import hash_object
from .models import (
    ComparisonMode,
    FactorKind,
    FactorSetting,
    InvalidationStage,
    RealizationKind,
    RealizationSpec,
    SensitivityPlan,
)


_VOLATILE_CONFIG_KEYS = {"output_dir", "data_dir", "config_path", "notes", "profile", "robustness", "confounder"}


def scientific_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the analytical configuration used in realization identity hashes.

    Paths/provenance, the robustness execution plan, and the separate
    candidate-confounder plan are intentionally not part of the realized
    unadjusted methodological settings. Source-file byte identities
    remain project-level provenance in ``manifest.json``.
    """
    return copy.deepcopy({k: v for k, v in config.items() if k not in _VOLATILE_CONFIG_KEYS})


def _get_path(target: Mapping[str, Any], path: str) -> Any:
    parts = path.split(".")
    if any(not p for p in parts):
        raise ValueError(f"invalid dotted config path {path!r}")
    node: Any = target
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"config path {path!r} does not exist in the reference configuration")
        node = node[part]
    return node


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if any(not p for p in parts):
        raise ValueError(f"invalid dotted config path {path!r}")
    node: dict[str, Any] = target
    for part in parts[:-1]:
        current = node.get(part)
        if not isinstance(current, dict):
            raise ValueError(f"config path {path!r} does not resolve through mapping {part!r}")
        node = current
    if parts[-1] not in node:
        raise ValueError(f"config path {path!r} does not exist in the reference configuration")
    node[parts[-1]] = copy.deepcopy(value)


def materialize_realization(reference_config: Mapping[str, Any], spec: RealizationSpec) -> dict[str, Any]:
    """Apply a realization's factor vector to a detached reference config."""
    result = copy.deepcopy(dict(reference_config))
    for factor in spec.factors:
        _set_path(result, factor.config_path, factor.value)
    return result


@dataclass(frozen=True)
class FactorSpec:
    name: str
    config_path: str
    analysis_id: str
    kind: FactorKind
    invalidates_from: InvalidationStage
    reference_value: Any
    lower: float | int | None = None
    upper: float | int | None = None
    alternative_value: Any = None
    integer: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.config_path.strip() or not self.analysis_id.strip():
            raise ValueError("factor name, config_path, and analysis_id must be nonempty")
        if self.kind is FactorKind.NUMERICAL:
            if type(self.lower) not in (int, float) or type(self.upper) not in (int, float):
                raise ValueError("numerical factors require lower and upper values")
            if not isfinite(float(self.lower)) or not isfinite(float(self.upper)):
                raise ValueError("numerical sensitivity bounds must be finite")
        elif self.kind is FactorKind.BINARY:
            if type(self.reference_value) is not bool or type(self.alternative_value) is not bool:
                raise ValueError("binary factors require boolean reference and alternative values")
            if self.reference_value == self.alternative_value:
                raise ValueError("binary alternative must differ from its reference value")
        elif self.kind is FactorKind.CATEGORICAL:
            if self.alternative_value == self.reference_value:
                raise ValueError("categorical alternative must differ from its reference value")

    def value_at(self, coordinate: float) -> Any:
        if self.kind is not FactorKind.NUMERICAL:
            return copy.deepcopy(self.alternative_value)
        value = float(self.lower) + coordinate * (float(self.upper) - float(self.lower))
        if self.integer:
            return int(round(value))
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": self.config_path,
            "analysis_id": self.analysis_id,
            "kind": self.kind.value,
            "invalidates_from": self.invalidates_from.name,
            "reference_value": self.reference_value,
            "lower": self.lower,
            "upper": self.upper,
            "alternative_value": self.alternative_value,
            "integer": self.integer,
        }


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    analysis_id: str
    factors: tuple[FactorSpec, ...]
    numerical_steps: int = 5
    comparison_mode: ComparisonMode = ComparisonMode.BOTH
    label: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.analysis_id.strip():
            raise ValueError("scenario_id and analysis_id must be nonempty")
        if not self.factors:
            raise ValueError("a sensitivity scenario requires at least one factor")
        if any(f.analysis_id != self.analysis_id for f in self.factors):
            raise ValueError("all factors in a scenario must use the scenario analysis_id")
        names = [f.name for f in self.factors]
        paths = [f.config_path for f in self.factors]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("scenario factor names and config paths must be unique")
        if any(f.kind is FactorKind.NUMERICAL for f in self.factors) and self.numerical_steps < 2:
            raise ValueError("numerical scenarios require at least two sensitivity steps")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "analysis_id": self.analysis_id,
            "factors": [f.to_dict() for f in self.factors],
            "numerical_steps": self.numerical_steps,
            "comparison_mode": self.comparison_mode.value,
            "label": self.label,
        }


class RealizationPlanner:
    """Create deterministic realization plans from a validated reference config."""

    def __init__(self, reference_config: Mapping[str, Any]):
        self.reference_config = copy.deepcopy(dict(reference_config))
        self.reference_science = scientific_configuration(self.reference_config)
        self.reference_hash = hash_object(self.reference_science)
        self.reference = RealizationSpec(
            realization_id=f"RS.REFERENCE.{self.reference_hash[:20]}",
            kind=RealizationKind.REFERENCE,
            scenario_id="REFERENCE",
            analysis_id="REFERENCE",
            step_index=None,
            trajectory_coordinate=None,
            comparison_mode=ComparisonMode.BOTH,
            invalidates_from=InvalidationStage.NONE,
            reference_realization_id=None,
            resolved_settings_hash=self.reference_hash,
            factors=(),
            label="Reference analytical realization",
        )

    def _realization_from_settings(
        self,
        scenario: ScenarioSpec,
        *,
        step_index: int,
        trajectory_coordinate: float,
        factors: tuple[FactorSetting, ...],
    ) -> RealizationSpec:
        realized = copy.deepcopy(self.reference_config)
        for factor in factors:
            _set_path(realized, factor.config_path, factor.value)
        settings_hash = hash_object(scientific_configuration(realized))
        earliest = min((f.invalidates_from for f in factors), key=int)
        identity_payload = {
            "reference_realization_id": self.reference.realization_id,
            "scenario_id": scenario.scenario_id,
            "analysis_id": scenario.analysis_id,
            "comparison_mode": scenario.comparison_mode.value,
            "step_index": step_index,
            "trajectory_coordinate": trajectory_coordinate,
            "resolved_settings_hash": settings_hash,
            "factors": [f.to_dict() for f in factors],
        }
        rid = f"RS.{scenario.analysis_id}.{hash_object(identity_payload)[:20]}"
        return RealizationSpec(
            realization_id=rid,
            kind=RealizationKind.SENSITIVITY,
            scenario_id=scenario.scenario_id,
            analysis_id=scenario.analysis_id,
            step_index=step_index,
            trajectory_coordinate=trajectory_coordinate,
            comparison_mode=scenario.comparison_mode,
            invalidates_from=earliest,
            reference_realization_id=self.reference.realization_id,
            resolved_settings_hash=settings_hash,
            factors=factors,
            label=scenario.label or scenario.scenario_id,
        )

    def expand_scenario(self, scenario: ScenarioSpec) -> tuple[RealizationSpec, ...]:
        for factor in scenario.factors:
            actual = _get_path(self.reference_config, factor.config_path)
            if actual != factor.reference_value:
                raise ValueError(
                    f"factor {factor.name!r} reference value {factor.reference_value!r} "
                    f"does not match configuration value {actual!r} at {factor.config_path}"
                )
        numerical = [f for f in scenario.factors if f.kind is FactorKind.NUMERICAL]
        coordinates = (
            [i / (scenario.numerical_steps - 1) for i in range(scenario.numerical_steps)]
            if numerical else [1.0]
        )
        result: list[RealizationSpec] = []
        # The reference realization is always retained separately; if a
        # sensitivity trajectory lands exactly on it, do not schedule a
        # duplicate computation.
        seen_settings: set[str] = {self.reference_hash}
        for raw_index, coordinate in enumerate(coordinates):
            settings = tuple(
                FactorSetting(
                    name=f.name,
                    config_path=f.config_path,
                    value=f.value_at(coordinate),
                    invalidates_from=f.invalidates_from,
                )
                for f in scenario.factors
            )
            candidate = self._realization_from_settings(
                scenario,
                step_index=len(result),
                trajectory_coordinate=coordinate,
                factors=settings,
            )
            # Integer interpolation can generate adjacent duplicate settings.
            if candidate.resolved_settings_hash in seen_settings:
                continue
            seen_settings.add(candidate.resolved_settings_hash)
            result.append(candidate)
        return tuple(result)

    def build(self, scenarios: Iterable[ScenarioSpec] = ()) -> SensitivityPlan:
        scenario_list = tuple(scenarios)
        ids = [s.scenario_id for s in scenario_list]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario_id values must be unique within a plan")
        realizations: list[RealizationSpec] = []
        for scenario in scenario_list:
            realizations.extend(self.expand_scenario(scenario))
        payload = {
            "reference_realization_id": self.reference.realization_id,
            "scenario_definitions": [s.to_dict() for s in scenario_list],
            "realization_ids": [r.realization_id for r in realizations],
        }
        plan_id = f"RSP.{hash_object(payload)[:24]}"
        return SensitivityPlan(plan_id=plan_id, reference=self.reference, realizations=tuple(realizations))

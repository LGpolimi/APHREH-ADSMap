"""Configuration contracts for candidate-confounder adjustment.

CA1-CA9 infrastructure. CA9 adds the production GUI, results workspace,
export/report integration, and final implementation validation without changing
the CA1-CA8 estimator or optimization mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_RELEVANCE_CALIPER_GRID = tuple(round(i * 0.025, 3) for i in range(41))
DEFAULT_RESIDUAL_RELEVANCE_SENSITIVITY = (0.05, 0.10, 0.15, 0.20)
CONFOUNDER_IMPLEMENTATION_STAGE = "ADJUSTED_OPTIMIZATION_INFERENCE"
LEGACY_CONFOUNDER_IMPLEMENTATION_STAGES = (
    "COMPLETE_ADJUSTED_SURFACE",
    "FIXED_PARAMETER_ADJUSTED_INFERENCE",
    "CANDIDATE_PREPROCESSING",
    "CROSS_FITTED_PROFILING",
    "PAIRWISE_CALIBRATION",
    "PAIRWISE_ESTIMATOR_INTEGRATION",
)


@dataclass(frozen=True)
class ConfounderAdjustmentSettings:
    """Frozen standard settings for one-candidate confounder adjustment.

    Method references: CA5.1, CA7.3, CA8.1, CA9.6, CA16.2 and CA18.
    CA8 permits the complete downstream confounder workflow, including nested
    leave-one-year-out calibration, conditional synchronized null search, robust-region
    resolution and adjusted representative selection when complete reoptimization is
    requested. CA9 exposes the frozen workflow through project-first GUI and result
    presentation layers.
    """

    enabled: bool = False
    request_complete_reoptimization: bool = False
    continuous_percentile_safeguard: float = 0.50
    relevance_caliper_grid: tuple[float, ...] = field(
        default=DEFAULT_RELEVANCE_CALIPER_GRID
    )
    residual_relevance_tolerance: float = 0.10
    residual_relevance_sensitivity_values: tuple[float, ...] = field(
        default=DEFAULT_RESIDUAL_RELEVANCE_SENSITIVITY
    )
    minimum_retained_support: float = 0.60
    minimum_years_primary: int = 3
    minimum_years_reoptimization: int = 4

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(self.request_complete_reoptimization) is not bool:
            raise ValueError("request_complete_reoptimization must be boolean")
        self._unit_interval(
            "continuous_percentile_safeguard", self.continuous_percentile_safeguard
        )
        self._unit_interval(
            "residual_relevance_tolerance", self.residual_relevance_tolerance
        )
        self._unit_interval("minimum_retained_support", self.minimum_retained_support)
        if tuple(float(v) for v in self.relevance_caliper_grid) != DEFAULT_RELEVANCE_CALIPER_GRID:
            raise ValueError(
                "relevance_caliper_grid is fixed by CA8.1/CA18 and must be "
                "{0, 0.025, ..., 1.000}"
            )
        if not self.residual_relevance_sensitivity_values:
            raise ValueError("residual_relevance_sensitivity_values cannot be empty")
        for value in self.residual_relevance_sensitivity_values:
            self._unit_interval("residual_relevance_sensitivity_values entry", value)
        if type(self.minimum_years_primary) is not int or self.minimum_years_primary < 3:
            raise ValueError("minimum_years_primary must be an integer >= 3")
        if type(self.minimum_years_reoptimization) is not int or self.minimum_years_reoptimization < 4:
            raise ValueError("minimum_years_reoptimization must be an integer >= 4")
        if self.minimum_years_reoptimization < self.minimum_years_primary:
            raise ValueError(
                "minimum_years_reoptimization cannot be smaller than minimum_years_primary"
            )

    @staticmethod
    def _unit_interval(name: str, value: float) -> None:
        if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")

    def supports_primary_adjustment(self, number_of_years: int) -> bool:
        return number_of_years >= self.minimum_years_primary

    def supports_standard_reoptimization(self, number_of_years: int) -> bool:
        return number_of_years >= self.minimum_years_reoptimization

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "request_complete_reoptimization": self.request_complete_reoptimization,
            "continuous_percentile_safeguard": float(self.continuous_percentile_safeguard),
            "relevance_caliper_grid": list(self.relevance_caliper_grid),
            "residual_relevance_tolerance": float(self.residual_relevance_tolerance),
            "residual_relevance_sensitivity_values": list(
                self.residual_relevance_sensitivity_values
            ),
            "minimum_retained_support": float(self.minimum_retained_support),
            "minimum_years_primary": self.minimum_years_primary,
            "minimum_years_reoptimization": self.minimum_years_reoptimization,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfounderAdjustmentSettings":
        return cls(
            enabled=data.get("enabled", False),
            request_complete_reoptimization=data.get(
                "request_complete_reoptimization", False
            ),
            continuous_percentile_safeguard=data.get(
                "continuous_percentile_safeguard", 0.50
            ),
            relevance_caliper_grid=tuple(
                data.get("relevance_caliper_grid", DEFAULT_RELEVANCE_CALIPER_GRID)
            ),
            residual_relevance_tolerance=data.get(
                "residual_relevance_tolerance", 0.10
            ),
            residual_relevance_sensitivity_values=tuple(
                data.get(
                    "residual_relevance_sensitivity_values",
                    DEFAULT_RESIDUAL_RELEVANCE_SENSITIVITY,
                )
            ),
            minimum_retained_support=data.get("minimum_retained_support", 0.60),
            minimum_years_primary=data.get("minimum_years_primary", 3),
            minimum_years_reoptimization=data.get(
                "minimum_years_reoptimization", 4
            ),
        )

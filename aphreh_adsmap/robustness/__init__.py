"""Robustness/sensitivity infrastructure.

Current release implements RS steps 1-10: common realization/planning/persistence infrastructure, all five analysis families, fixed-parameter comparison, complete reoptimization, explicit realization-specific LOYO/null-search components, and reoptimized fingerprint/surface comparison.
"""
from .models import (
    CheckpointState,
    ComparisonMode,
    FactorKind,
    FactorSetting,
    InvalidationStage,
    RealizationKind,
    RealizationSpec,
    RealizationState,
    SensitivityPlan,
)
from .planner import FactorSpec, RealizationPlanner, ScenarioSpec, materialize_realization, scientific_configuration
from .persistence import RobustnessPersistence
from .support_sensitivity import (
    ANALYSIS_ID as SUPPORT_SENSITIVITY_ANALYSIS_ID,
    SUPPORT_FACTOR_REGISTRY,
    SupportSensitivityRunner,
    build_robustness_plan,
    support_scenarios_from_config,
    realization_run_id,
    export_support_sensitivity,
)


from .exposure_sensitivity import (
    ANALYSIS_ID as EXPOSURE_SENSITIVITY_ANALYSIS_ID,
    ExposureSensitivityRunner,
    exposure_scenarios_from_config,
    export_exposure_sensitivity,
)

from .baseline_sensitivity import (
    ANALYSIS_ID as BASELINE_SENSITIVITY_ANALYSIS_ID,
    BASELINE_FACTOR_REGISTRY,
    BaselineSensitivityRunner,
    baseline_scenarios_from_config,
    export_baseline_sensitivity,
)

from .dependence_sensitivity import (
    ANALYSIS_ID as DEPENDENCE_SENSITIVITY_ANALYSIS_ID,
    DEPENDENCE_FACTOR_REGISTRY,
    DependenceSensitivityRunner,
    dependence_scenarios_from_config,
    export_dependence_sensitivity,
)


from .fixed_point_comparison import (
    ANALYSIS_ID as FIXED_POINT_COMPARISON_ANALYSIS_ID,
    FixedPointComparator,
    export_fixed_point_comparison,
)



from .realization_loyo import (
    ANALYSIS_ID as REALIZATION_LOYO_ANALYSIS_ID,
    CHECKPOINT_STAGE as REALIZATION_LOYO_CHECKPOINT_STAGE,
    RealizationLoyoRunner,
    export_realization_loyo,
)

from .realization_null import (
    ANALYSIS_ID as REALIZATION_NULL_ANALYSIS_ID,
    CHECKPOINT_STAGE as REALIZATION_NULL_CHECKPOINT_STAGE,
    RealizationNullSearchRunner,
    export_realization_null,
)

from .reoptimization import (
    ANALYSIS_ID as COMPLETE_REOPTIMIZATION_ANALYSIS_ID,
    CHECKPOINT_STAGE as COMPLETE_REOPTIMIZATION_CHECKPOINT_STAGE,
    ReoptimizationRunner,
    export_reoptimization,
)


from .fingerprint_surface_comparison import (
    ANALYSIS_ID as FINGERPRINT_SURFACE_COMPARISON_ANALYSIS_ID,
    CHECKPOINT_STAGE as FINGERPRINT_SURFACE_COMPARISON_CHECKPOINT_STAGE,
    FingerprintSurfaceComparator,
    export_fingerprint_surface_comparison,
)

from .region_stability import (
    ANALYSIS_ID as REGION_STABILITY_ANALYSIS_ID,
    RegionStabilityAnalyzer,
    compute_region_stability,
    export_region_stability,
)

__all__ = [
    "CheckpointState",
    "ComparisonMode",
    "FactorKind",
    "FactorSetting",
    "InvalidationStage",
    "RealizationKind",
    "RealizationSpec",
    "RealizationState",
    "SensitivityPlan",
    "FactorSpec",
    "ScenarioSpec",
    "RealizationPlanner",
    "materialize_realization",
    "scientific_configuration",
    "RobustnessPersistence",
    "SUPPORT_SENSITIVITY_ANALYSIS_ID",
    "SUPPORT_FACTOR_REGISTRY",
    "SupportSensitivityRunner",
    "build_robustness_plan",
    "support_scenarios_from_config",
    "realization_run_id",
    "export_support_sensitivity",
    "EXPOSURE_SENSITIVITY_ANALYSIS_ID",
    "ExposureSensitivityRunner",
    "exposure_scenarios_from_config",
    "export_exposure_sensitivity",
    "BASELINE_SENSITIVITY_ANALYSIS_ID",
    "BASELINE_FACTOR_REGISTRY",
    "BaselineSensitivityRunner",
    "baseline_scenarios_from_config",
    "export_baseline_sensitivity",
    "DEPENDENCE_SENSITIVITY_ANALYSIS_ID",
    "DEPENDENCE_FACTOR_REGISTRY",
    "DependenceSensitivityRunner",
    "dependence_scenarios_from_config",
    "export_dependence_sensitivity",
    "REALIZATION_LOYO_ANALYSIS_ID",
    "REALIZATION_LOYO_CHECKPOINT_STAGE",
    "RealizationLoyoRunner",
    "export_realization_loyo",
    "REALIZATION_NULL_ANALYSIS_ID",
    "REALIZATION_NULL_CHECKPOINT_STAGE",
    "RealizationNullSearchRunner",
    "export_realization_null",
    "FIXED_POINT_COMPARISON_ANALYSIS_ID",
    "FixedPointComparator",
    "export_fixed_point_comparison",
    "COMPLETE_REOPTIMIZATION_ANALYSIS_ID",
    "COMPLETE_REOPTIMIZATION_CHECKPOINT_STAGE",
    "ReoptimizationRunner",
    "export_reoptimization",
    "FINGERPRINT_SURFACE_COMPARISON_ANALYSIS_ID",
    "FINGERPRINT_SURFACE_COMPARISON_CHECKPOINT_STAGE",
    "FingerprintSurfaceComparator",
    "export_fingerprint_surface_comparison",
    "REGION_STABILITY_ANALYSIS_ID",
    "RegionStabilityAnalyzer",
    "compute_region_stability",
    "export_region_stability",
]

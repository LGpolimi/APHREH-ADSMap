"""Configuration validation. Unsupported scientific options fail before execution."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import math
from .support import SUPPORT_FIELDS, validate_support_policy
from .confounder import (
    CONFOUNDER_IMPLEMENTATION_STAGE,
    LEGACY_CONFOUNDER_IMPLEMENTATION_STAGES,
    ConfounderAdjustmentSettings,
    ConfounderCandidateDefinition,
)


class ConfigurationError(ValueError):
    pass


SUPPORTED_THRESHOLD_RESPONSE_MODES = ("STEP", "LINEAR_RAMP")
DEFAULT_THRESHOLD_RESPONSE_MODES = ("STEP", "LINEAR_RAMP")
DEFAULT_PRIMARY_THRESHOLD_RESPONSE_MODE = "STEP"
SUPPORTED_POPULATION_MODES = ("ANNUAL_NEAREST", "ANNUAL_STRICT", "STATIC_EXPLICIT")


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8-sig") as f:
        c = json.load(f)
    c = copy.deepcopy(c)
    c["data_dir"] = str((path.parent / c["data_dir"]).resolve())
    c["output_dir"] = str((path.parent / c["output_dir"]).resolve())
    c["config_path"] = str(path)
    validate(c)
    return c


def _normalize_threshold_response_modes(cal: dict, require) -> None:
    """Normalize legacy/single-mode and new multi-mode calibration configuration."""
    singular = cal.get("threshold_lag_response_mode")
    plural = cal.get("threshold_lag_response_modes")
    if singular is not None and plural is not None:
        require(list(plural) == [singular],
                "Use either threshold_lag_response_mode or threshold_lag_response_modes, not conflicting values")
    if plural is not None:
        require(isinstance(plural, list) and bool(plural),
                "Calibration threshold_lag_response_modes must be a nonempty list")
        modes = list(plural)
    elif singular is not None:
        modes = [singular]
    else:
        modes = list(DEFAULT_THRESHOLD_RESPONSE_MODES)
    require(all(isinstance(v, str) and v in SUPPORTED_THRESHOLD_RESPONSE_MODES for v in modes),
            "Calibration threshold response modes must be STEP or LINEAR_RAMP")
    require(len(set(modes)) == len(modes), "Duplicate calibration threshold response modes are not permitted")
    primary = cal.get("primary_threshold_lag_response_mode", DEFAULT_PRIMARY_THRESHOLD_RESPONSE_MODE if "STEP" in modes else modes[0])
    require(primary in modes, "Primary threshold-lag response mode must be one of threshold_lag_response_modes")
    cal["threshold_lag_response_modes"] = modes
    cal["primary_threshold_lag_response_mode"] = primary
    cal.pop("threshold_lag_response_mode", None)


def _normalize_sources(c: dict, require) -> None:
    """Canonicalize legacy source mappings without changing the tabular data model."""
    require(isinstance(c.get("sources"), dict), "sources must be a configuration object")
    s = c["sources"]
    legacy_date = s.get("date_field")
    if "hazard_date_field" not in s and legacy_date is not None:
        s["hazard_date_field"] = legacy_date
    if "outcome_date_field" not in s and legacy_date is not None:
        s["outcome_date_field"] = legacy_date
    required = (
        "hazard_file", "outcome_file", "population_file", "crosswalk_file",
        "hazard_date_field", "outcome_date_field", "bsa_id_field", "bsa_area_field",
        "crosswalk_bsa_field", "grid_id_field", "intersection_area_field", "population_fields",
    )
    for key in required:
        require(key in s, f"Missing source configuration field: {key}")
    for key in required[:-1]:
        require(isinstance(s[key], str) and bool(s[key].strip()), f"Source configuration {key} must be nonempty")
    require(isinstance(s["population_fields"], dict) and bool(s["population_fields"]),
            "sources.population_fields must map at least one population year to a column")
    normalized = {}
    for year, field in s["population_fields"].items():
        try:
            y = int(str(year))
        except Exception as exc:
            raise ConfigurationError(f"Invalid population year {year!r}") from exc
        require(1000 <= y <= 9999, f"Invalid population year {year!r}")
        require(isinstance(field, str) and bool(field.strip()), f"Population field for {y} must be nonempty")
        require(str(y) not in normalized, f"Duplicate population year {y}")
        normalized[str(y)] = field.strip()
    s["population_fields"] = dict(sorted(normalized.items(), key=lambda kv: int(kv[0])))
    holiday_file = s.get("holiday_calendar_file")
    if holiday_file is not None:
        require(isinstance(holiday_file, str) and bool(holiday_file.strip()),
                "sources.holiday_calendar_file must be a nonempty path when configured")
        holiday_date = s.get("holiday_date_field")
        require(isinstance(holiday_date, str) and bool(holiday_date.strip()),
                "sources.holiday_date_field is required with holiday_calendar_file")
        s["holiday_calendar_file"] = holiday_file.strip()
        s["holiday_date_field"] = holiday_date.strip()
        holiday_state = s.get("holiday_state_field")
        if holiday_state is not None:
            require(isinstance(holiday_state, str) and bool(holiday_state.strip()),
                    "sources.holiday_state_field must be a nonempty field name when configured")
            s["holiday_state_field"] = holiday_state.strip()
    else:
        s.pop("holiday_calendar_file", None)
        s.pop("holiday_date_field", None)
        s.pop("holiday_state_field", None)
    geometry_bsa_field = s.get("geometry_bsa_field", s.get("bsa_id_field"))
    require(isinstance(geometry_bsa_field, str) and bool(geometry_bsa_field.strip()),
            "sources.geometry_bsa_field must be a nonempty field name when geometry is configured")
    s["geometry_bsa_field"] = geometry_bsa_field.strip()
    # The old shared key is accepted on input, but resolved configurations use
    # separate hazard/outcome mappings because datasets may use different names.
    s.pop("date_field", None)


def _normalize_analysis_period(c: dict, require) -> None:
    period = c.get("analysis_period") or {"mode": "COMMON_DATES"}
    require(isinstance(period, dict), "analysis_period must be a configuration object")
    require(period.get("mode", "COMMON_DATES") == "COMMON_DATES",
            "The validated runtime uses all exact matching dates between hazard and outcome")
    c["analysis_period"] = {"mode": "COMMON_DATES"}
    # Legacy manually configured limits are intentionally ignored by the new
    # canonical period rule and removed from the resolved scientific settings.
    c.pop("start_date", None)
    c.pop("end_date", None)



def _normalize_confounder(c: dict, require) -> None:
    """Normalize CA1-CA9 confounder settings and candidate source mappings."""
    raw = c.get("confounder") or {}
    require(isinstance(raw, dict), "confounder must be a configuration object")
    allowed = {
        "enabled",
        "implementation_stage",
        "request_complete_reoptimization",
        "candidate",
        "continuous_percentile_safeguard",
        "relevance_caliper_grid",
        "residual_relevance_tolerance",
        "residual_relevance_sensitivity_values",
        "minimum_retained_support",
        "minimum_years_primary",
        "minimum_years_reoptimization",
    }
    unknown = set(raw) - allowed
    require(not unknown, "Unknown confounder fields: " + ", ".join(sorted(unknown)))

    stage = raw.get("implementation_stage", CONFOUNDER_IMPLEMENTATION_STAGE)
    require(stage in (CONFOUNDER_IMPLEMENTATION_STAGE, *LEGACY_CONFOUNDER_IMPLEMENTATION_STAGES),
            f"confounder.implementation_stage must be {CONFOUNDER_IMPLEMENTATION_STAGE}")
    try:
        settings = ConfounderAdjustmentSettings.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid confounder configuration: {exc}") from exc

    candidate_raw = raw.get("candidate")
    candidate = None
    if candidate_raw is not None:
        require(isinstance(candidate_raw, dict), "confounder.candidate must be null or a configuration object")
        # Active contract mirrors sources.hazard_file plus the standard APHREH
        # crosswalk. Legacy CA1-CA9 SPATIAL_FIELD keys are accepted only for
        # migration of existing projects and normalized immediately.
        candidate_allowed = {
            "schema_version", "input_format", "candidate_id", "display_name", "units", "metadata",
            "hazard_file", "hazard_date_field", "crosswalk_file", "crosswalk_bsa_field",
            "grid_id_field", "intersection_area_field",
            # Legacy SPATIAL_FIELD migration keys.
            "spatial_representation", "source_reference", "source_date_field",
            "harmonization_rule_id", "crosswalk_source_reference",
            "crosswalk_grid_field", "crosswalk_area_field",
            "territory_wide_source_reference", "bsa_source_reference", "source_value_field",
        }
        candidate_unknown = set(candidate_raw) - candidate_allowed
        require(not candidate_unknown,
                "Unknown confounder.candidate fields: " + ", ".join(sorted(candidate_unknown)))
        representation = candidate_raw.get("spatial_representation")
        if representation is not None:
            require(representation == "SPATIAL_FIELD",
                    "Candidate confounders must use the standard APHREH hazard-field format; "
                    "legacy TERRITORY_WIDE/BSA_SPECIFIC sources are not supported")
        require(not candidate_raw.get("territory_wide_source_reference") and not candidate_raw.get("bsa_source_reference")
                and not candidate_raw.get("source_value_field"),
                "Candidate confounders must be standard APHREH hazard-field CSVs; direct territory/BSA/value-field overrides are not supported")

        for key in ("candidate_id", "display_name"):
            require(isinstance(candidate_raw.get(key), str) and bool(candidate_raw[key].strip()),
                    f"confounder.candidate.{key} must be a nonempty string")
        hazard_file = candidate_raw.get("hazard_file") or candidate_raw.get("source_reference")
        require(isinstance(hazard_file, str) and bool(hazard_file.strip()),
                "confounder.candidate.hazard_file must be a nonempty string")
        legacy_specific_crosswalk = candidate_raw.get("crosswalk_source_reference")
        explicit_new_crosswalk = candidate_raw.get("crosswalk_file")
        crosswalk_file = explicit_new_crosswalk or legacy_specific_crosswalk or c["sources"]["crosswalk_file"]
        # Old SPATIAL_FIELD definitions carried generic bsa_id/grid_id/area defaults
        # even when they reused the canonical spatial database. In that migration
        # case the authoritative project crosswalk field mappings must be inherited.
        legacy_reuses_project_crosswalk = (
            representation == "SPATIAL_FIELD"
            and not explicit_new_crosswalk
            and not legacy_specific_crosswalk
        )
        if legacy_reuses_project_crosswalk:
            crosswalk_bsa_field = c["sources"]["crosswalk_bsa_field"]
            grid_id_field = c["sources"]["grid_id_field"]
            intersection_area_field = c["sources"]["intersection_area_field"]
        else:
            crosswalk_bsa_field = candidate_raw.get("crosswalk_bsa_field") or c["sources"]["crosswalk_bsa_field"]
            grid_id_field = candidate_raw.get("grid_id_field") or candidate_raw.get("crosswalk_grid_field") or c["sources"]["grid_id_field"]
            intersection_area_field = candidate_raw.get("intersection_area_field") or candidate_raw.get("crosswalk_area_field") or c["sources"]["intersection_area_field"]
        normalized_candidate = {
            "candidate_id": candidate_raw["candidate_id"].strip(),
            "display_name": candidate_raw["display_name"].strip(),
            "hazard_file": hazard_file.strip(),
            "hazard_date_field": str(candidate_raw.get("hazard_date_field") or candidate_raw.get("source_date_field") or c["sources"]["hazard_date_field"]).strip(),
            "crosswalk_file": str(crosswalk_file).strip(),
            "crosswalk_bsa_field": str(crosswalk_bsa_field).strip(),
            "grid_id_field": str(grid_id_field).strip(),
            "intersection_area_field": str(intersection_area_field).strip(),
            "units": candidate_raw.get("units"),
            "metadata": dict(candidate_raw.get("metadata", {})),
        }
        for key in ("hazard_date_field", "crosswalk_file", "crosswalk_bsa_field", "grid_id_field", "intersection_area_field"):
            require(bool(normalized_candidate[key]), f"confounder.candidate.{key} must be nonempty")
        require(normalized_candidate["units"] is None or (isinstance(normalized_candidate["units"], str) and bool(normalized_candidate["units"].strip())),
                "confounder.candidate.units must be null or a nonempty string")
        require(isinstance(normalized_candidate["metadata"], dict),
                "confounder.candidate.metadata must be a configuration object")
        try:
            candidate = ConfounderCandidateDefinition.from_dict(normalized_candidate)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid confounder candidate definition: {exc}") from exc

    # CA3 enables candidate-only cross-fitted profiling, CA4 calibration, CA5
    # the frozen-context original-sample adjusted estimator, CA6 fixed-point
    # bootstrap/pooling/WMARMcs inference, CA7 complete adjusted-surface generation,
    # CA8 nested adjusted optimization inference, and CA9 GUI/export presentation.
    require(not settings.enabled or candidate is not None,
            "confounder.candidate is required when confounder.enabled=true")
    normalized = settings.to_dict()
    normalized["implementation_stage"] = CONFOUNDER_IMPLEMENTATION_STAGE
    normalized["candidate"] = None if candidate is None else candidate.to_dict()
    c["confounder"] = normalized


def _normalize_robustness(c: dict, require) -> None:
    """Normalize robustness/sensitivity configuration used by RS steps 1-11."""
    rs=c.get("robustness") or {"enabled":False}; require(isinstance(rs,dict),"robustness must be a configuration object")
    enabled=rs.get("enabled",False); require(type(enabled) is bool,"robustness.enabled must be boolean")
    allowed_comparison_modes={"FIXED_PARAMETER","REOPTIMIZED","BOTH"}
    a1=rs.get("analysis_1_region_stability") or {}; require(isinstance(a1,dict),"robustness.analysis_1_region_stability must be a configuration object")
    a1_enabled=a1.get("enabled",True); require(type(a1_enabled) is bool,"robustness.analysis_1_region_stability.enabled must be boolean")
    minimum=a1.get("minimum_valid_cell_proportion"); require(minimum is None or (type(minimum) in (int,float) and math.isfinite(float(minimum)) and 0<=float(minimum)<=1),"robustness.analysis_1_region_stability.minimum_valid_cell_proportion must be null or in [0, 1]")

    def normalize_numeric_scenarios(section_name, allowed, analysis_label):
        sec=rs.get(section_name) or {}; require(isinstance(sec,dict),f"robustness.{section_name} must be a configuration object")
        sec_enabled=sec.get("enabled",False); require(type(sec_enabled) is bool,f"robustness.{section_name}.enabled must be boolean")
        require(not sec_enabled or enabled,f"robustness.enabled must be true when {analysis_label} is enabled")
        scenarios=sec.get("scenarios") or []; require(isinstance(scenarios,list),f"robustness.{section_name}.scenarios must be a list")
        require(not sec_enabled or bool(scenarios),f"{analysis_label} requires at least one sensitivity scenario")
        norm=[]; seen=set()
        for raw in scenarios:
            require(isinstance(raw,dict),f"Each {analysis_label} scenario must be an object")
            sid=raw.get("scenario_id"); require(isinstance(sid,str) and bool(sid.strip()),f"Each {analysis_label} scenario requires scenario_id")
            require(sid not in seen,f"Duplicate {analysis_label} scenario_id: {sid}"); seen.add(sid)
            steps=raw.get("steps",5); require(type(steps) is int and steps>=2,f"{analysis_label} scenario {sid} requires steps >= 2")
            mode=raw.get("comparison_mode","FIXED_PARAMETER"); require(mode in allowed_comparison_modes,f"{analysis_label} comparison_mode must be FIXED_PARAMETER, REOPTIMIZED, or BOTH")
            factors=raw.get("factors") or {}; require(isinstance(factors,dict) and bool(factors),f"{analysis_label} scenario {sid} requires at least one factor")
            nf={}
            for name,bounds in factors.items():
                require(name in allowed,f"Unsupported {analysis_label} factor: {name}"); require(isinstance(bounds,dict),f"{analysis_label} factor {name} must provide lower and upper")
                lo,hi=bounds.get("lower"),bounds.get("upper"); require(type(lo) in (int,float) and type(hi) in (int,float) and math.isfinite(float(lo)) and math.isfinite(float(hi)),f"{analysis_label} factor {name} lower/upper must be finite numbers")
                nf[name]={"lower":lo,"upper":hi}
            norm.append({"scenario_id":sid.strip(),"label":str(raw.get("label",sid)),"steps":steps,"comparison_mode":mode,"factors":nf})
        return sec_enabled,norm

    a2_enabled,a2_scenarios=normalize_numeric_scenarios("analysis_2_support_sensitivity",set(SUPPORT_FIELDS),"Analysis 2")

    # Analysis 3 is one structural alternative: BSA-specific yearly exposure
    # classification instead of the reference territory-wide calendar.
    a3=rs.get("analysis_3_bsa_specific_exposure") or {}; require(isinstance(a3,dict),"robustness.analysis_3_bsa_specific_exposure must be a configuration object")
    a3_enabled=a3.get("enabled",False); require(type(a3_enabled) is bool,"robustness.analysis_3_bsa_specific_exposure.enabled must be boolean")
    require(not a3_enabled or enabled,"robustness.enabled must be true when Analysis 3 is enabled")
    a3_mode=a3.get("comparison_mode","FIXED_PARAMETER"); require(a3_mode in allowed_comparison_modes,"Analysis 3 comparison_mode must be FIXED_PARAMETER, REOPTIMIZED, or BOTH")
    a3_label=str(a3.get("label","BSA-specific exposure classification"))
    if a3_enabled:
        require(c["core"]["exposure"].get("mode")=="TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE","Analysis 3 requires the reference TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE exposure mode")
        require(c["core"].get("support",{}).get("min_local_concordance") is None,"Analysis 3 is not compatible with an active min_local_concordance support threshold because local concordance is not applicable under BSA-specific exposure")

    dep_fields={"local_acf_interval_days","negligible_lag_persistence","annual_dependence_quantile","exposed_episode_quantile","final_block_length_multiplier"}
    a5_enabled,a5_scenarios=normalize_numeric_scenarios("analysis_5_dependence_sensitivity",dep_fields,"Analysis 5")
    dep_ref=c["core"]["dependence"]
    for raw in a5_scenarios:
        for name,bounds in raw["factors"].items():
            for value in (bounds["lower"],bounds["upper"]):
                if name in ("annual_dependence_quantile","exposed_episode_quantile"): require(0<=float(value)<=1,f"Analysis 5 factor {name} must be in [0, 1]")
                elif name=="final_block_length_multiplier": require(float(value)>0,f"Analysis 5 factor {name} must be > 0")
                elif name=="local_acf_interval_days": require(float(value).is_integer() and int(value)>=dep_ref["minimum_local_observations"] and int(value)>dep_ref["maximum_acf_lag_days"],"Analysis 5 local_acf_interval_days must be integer, >= minimum_local_observations, and > maximum_acf_lag_days")
                elif name=="negligible_lag_persistence": require(float(value).is_integer() and 1<=int(value)<=dep_ref["maximum_acf_lag_days"],"Analysis 5 negligible_lag_persistence must be integer in [1, maximum_acf_lag_days]")

    # Analysis 4 accepts numerical trajectories and binary methodological switches.
    a4=rs.get("analysis_4_baseline_sensitivity") or {}; require(isinstance(a4,dict),"robustness.analysis_4_baseline_sensitivity must be a configuration object")
    a4_enabled=a4.get("enabled",False); require(type(a4_enabled) is bool,"robustness.analysis_4_baseline_sensitivity.enabled must be boolean")
    require(not a4_enabled or enabled,"robustness.enabled must be true when Analysis 4 is enabled")
    raw_scenarios=a4.get("scenarios") or []; require(isinstance(raw_scenarios,list),"robustness.analysis_4_baseline_sensitivity.scenarios must be a list")
    require(not a4_enabled or bool(raw_scenarios),"Analysis 4 requires at least one sensitivity scenario")
    a4_scenarios=[]; seen=set(); numerical_names={"initial_half_width","maximum_half_width","minimum_eligible_dates"}; binary_names={"dynamic_enlargement","day_of_week_matching","holiday_matching"}
    for raw in raw_scenarios:
        require(isinstance(raw,dict),"Each Analysis 4 scenario must be an object")
        sid=raw.get("scenario_id"); require(isinstance(sid,str) and bool(sid.strip()),"Each Analysis 4 scenario requires scenario_id")
        require(sid not in seen,f"Duplicate Analysis 4 scenario_id: {sid}"); seen.add(sid)
        mode=raw.get("comparison_mode","FIXED_PARAMETER"); require(mode in allowed_comparison_modes,"Analysis 4 comparison_mode must be FIXED_PARAMETER, REOPTIMIZED, or BOTH")
        factors=raw.get("factors") or {}; require(isinstance(factors,dict) and bool(factors),f"Analysis 4 scenario {sid} requires at least one factor")
        has_numerical=any(name in numerical_names for name in factors)
        steps=raw.get("steps",5 if has_numerical else 1)
        if has_numerical: require(type(steps) is int and steps>=2,f"Analysis 4 scenario {sid} with numerical factors requires steps >= 2")
        else: require(type(steps) is int and steps>=1,f"Analysis 4 scenario {sid} requires steps >= 1")
        nf={}
        for name,definition in factors.items():
            require(name in numerical_names|binary_names,f"Unsupported Analysis 4 factor: {name}")
            require(isinstance(definition,dict),f"Analysis 4 factor {name} must be an object")
            if name in numerical_names:
                lo,hi=definition.get("lower"),definition.get("upper")
                require(type(lo) in (int,float) and type(hi) in (int,float) and math.isfinite(float(lo)) and math.isfinite(float(hi)),f"Analysis 4 factor {name} lower/upper must be finite numbers")
                require(float(lo).is_integer() and float(hi).is_integer(),f"Analysis 4 factor {name} must use integer bounds")
                require(int(lo)>=1 and int(hi)>=1,f"Analysis 4 factor {name} bounds must be >= 1")
                nf[name]={"kind":"NUMERICAL","lower":int(lo),"upper":int(hi)}
            else:
                alt=definition.get("alternative")
                require(type(alt) is bool,f"Analysis 4 binary factor {name} requires boolean alternative")
                require(alt is not bool(c["core"]["baseline"].get(name,False)),f"Analysis 4 binary factor {name} alternative must differ from the reference state")
                nf[name]={"kind":"BINARY","alternative":alt}
                if name=="holiday_matching" and alt:
                    require(bool(c.get("sources",{}).get("holiday_calendar_file")),"Analysis 4 holiday_matching=true requires sources.holiday_calendar_file and holiday_date_field")
        # Validate every synchronized numerical step against the baseline-policy
        # invariants now, before a long analysis is launched.
        refb=dict(c["core"]["baseline"])
        for i in range(steps if has_numerical else 1):
            u=i/(steps-1) if has_numerical else 1.0
            rb=dict(refb)
            for name,definition in nf.items():
                if definition["kind"]=="NUMERICAL":
                    rb[name]=int(round(definition["lower"]+u*(definition["upper"]-definition["lower"])))
                else:
                    rb[name]=definition["alternative"]
            require(rb["maximum_half_width"]>=rb["initial_half_width"],f"Analysis 4 scenario {sid} produces maximum_half_width < initial_half_width at sensitivity step {i}")
        a4_scenarios.append({"scenario_id":sid.strip(),"label":str(raw.get("label",sid)),"steps":steps,"comparison_mode":mode,"factors":nf})

    require(not enabled or a2_enabled or a3_enabled or a4_enabled or a5_enabled,"robustness.enabled=true requires Analysis 2, Analysis 3, Analysis 4, and/or Analysis 5")
    c["robustness"]={"enabled":enabled,"implementation_stage":"FINGERPRINT_SURFACE_COMPARISON",
      "analysis_1_region_stability":{"enabled":a1_enabled,"minimum_valid_cell_proportion":None if minimum is None else float(minimum)},
      "analysis_2_support_sensitivity":{"enabled":a2_enabled,"scenarios":a2_scenarios},
      "analysis_3_bsa_specific_exposure":{"enabled":a3_enabled,"label":a3_label,"comparison_mode":a3_mode},
      "analysis_4_baseline_sensitivity":{"enabled":a4_enabled,"scenarios":a4_scenarios},
      "analysis_5_dependence_sensitivity":{"enabled":a5_enabled,"scenarios":a5_scenarios}}

def validate(c: dict) -> None:
    def require(ok: bool, message: str) -> None:
        if not ok:
            raise ConfigurationError(message)

    require(c.get("schema") == "APHREH_DISK_DEV_V1", "Unknown project configuration schema")
    require(isinstance(c.get("data_dir"), str) and bool(c["data_dir"].strip()), "data_dir is required")
    require(isinstance(c.get("output_dir"), str) and bool(c["output_dir"].strip()), "output_dir is required")
    _normalize_analysis_period(c, require)
    _normalize_sources(c, require)
    _normalize_confounder(c, require)
    _normalize_robustness(c, require)
    qs, ls = c["thresholds"], c["lags"]
    require(bool(qs) and all(type(q) in (int, float) and 0 < q < 100 for q in qs),
            "Thresholds must be percentiles strictly between 0 and 100")
    require(len(set(qs)) == len(qs), "Duplicate thresholds are not permitted")
    require(bool(ls) and all(type(l) is int and l >= 0 for l in ls), "Lags must be nonnegative integers")
    require(len(set(ls)) == len(ls), "Duplicate lags are not permitted")
    c["thresholds"], c["lags"] = sorted(float(q) for q in qs), sorted(ls)
    require(type(c["seed"]) is int and c["seed"] >= 0, "Seed must be a nonnegative integer")
    core = c["core"]
    require(core["exposure"]["mode"] == "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE",
            "The validated DEV runtime supports territory-wide exposure with local concordance only")
    require(core["exposure"]["threshold_equality_convention"] == "GREATER_THAN_OR_EQUAL", "Unsupported threshold equality rule")
    require(core["pairwise_estimation"]["mode"] == "STANDARD", "Confounder-adjusted runtime is not enabled in the current production backend")
    require(core["pooling"]["method_version"] == "REML_MODIFIED_HK_V1", "MF4.1 requires REML, not DL")
    baseline = core["baseline"]
    require(type(baseline.get("dynamic_enlargement")) is bool, "baseline.dynamic_enlargement must be boolean")
    require(type(baseline.get("day_of_week_matching")) is bool, "baseline.day_of_week_matching must be boolean")
    require(type(baseline.get("holiday_matching")) is bool, "baseline.holiday_matching must be boolean")
    require(type(baseline["initial_half_width"]) is int and baseline["initial_half_width"] >= 1, "Baseline half-width must be positive")
    require(type(baseline["maximum_half_width"]) is int and baseline["maximum_half_width"] >= baseline["initial_half_width"], "Invalid maximum baseline half-width")
    minimum_baseline = baseline.get("minimum_eligible_dates")
    require(minimum_baseline is None or (type(minimum_baseline) is int and minimum_baseline >= 1), "baseline.minimum_eligible_dates must be null or a positive integer")
    require(not baseline.get("holiday_matching") or bool(c["sources"].get("holiday_calendar_file")),
            "Holiday matching requires sources.holiday_calendar_file and holiday_date_field")
    support = core.get("support", {})
    require(isinstance(support, dict), "core.support must be a configuration object")
    unknown_support = set(support) - set(SUPPORT_FIELDS)
    require(not unknown_support, "Unknown core.support fields: " + ", ".join(sorted(unknown_support)))
    for name in SUPPORT_FIELDS:
        support.setdefault(name, None)
    try:
        validate_support_policy(support)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    boot = core["bootstrap"]
    require(type(boot["iterations"]) is int and boot["iterations"] >= 2, "At least two bootstrap iterations are required")
    require(0 <= boot["maximum_invalid_realization_proportion"] < 1, "Invalid bootstrap invalidity limit")
    dep = core["dependence"]
    for k in ("trend_window_days", "local_acf_interval_days", "minimum_local_observations", "maximum_acf_lag_days", "negligible_lag_persistence", "smoothing_window_days"):
        require(type(dep[k]) is int and dep[k] > 0, f"{k} must be a positive integer")
    require(dep["trend_window_days"] >= 20, "MF3.2 trend requires at least 20 observations")
    require(dep["smoothing_window_days"] >= 5, "MF3.2 smoothing requires at least five estimates")
    require(dep["minimum_local_observations"] <= dep["local_acf_interval_days"], "Minimum ACF observations exceeds interval width")
    require(dep["negligible_lag_persistence"] <= dep["maximum_acf_lag_days"] < dep["local_acf_interval_days"], "Invalid ACF lag/persistence")
    require(type(dep["final_block_length_multiplier"]) in (int, float) and math.isfinite(float(dep["final_block_length_multiplier"])) and dep["final_block_length_multiplier"] > 0, "final_block_length_multiplier must be finite and > 0")
    for k in ("annual_dependence_quantile", "exposed_episode_quantile"):
        require(0 <= dep[k] <= 1, f"Invalid quantile: {k}")
    require(core["incidence"]["scaling_multiplier"] > 0, "Incidence scaling multiplier must be positive")
    opt = c["optimization"]
    require(type(opt["null_realizations_R"]) is int and opt["null_realizations_R"] >= 1, "At least one null realization is required")
    for k in ("minimum_bsa_coverage", "minimum_population_coverage", "minimum_persistence_pi", "near_optimal_relative_tolerance_epsilon"):
        require(0 <= opt[k] <= 1, f"Invalid optimization proportion: {k}")
    require(0 < opt["search_adjusted_alpha"] < 1, "Alpha must be strictly between zero and one")
    require(opt["null_randomization_rule"]["rule"] == "CIRCULAR_DISPLACEMENT", "Only within-year synchronized circular displacement is enabled")
    shift = opt["null_randomization_rule"]["parameters"]["minimum_shift_days"]
    require(type(shift) is int and shift >= 1, "Minimum circular shift must be positive")
    cal = c["calibration"]
    require(cal["mode"] in ("SOFT", "HARD"), "Unknown calibration mode")
    require(type(cal["number_of_realizations"]) is int and cal["number_of_realizations"] >= 1, "Invalid calibration realization count")
    _normalize_threshold_response_modes(cal, require)
    require(cal.get("calibration_threshold") is None or float(cal["calibration_threshold"]) in c["thresholds"], "Calibration threshold is not in the grid")
    classes = cal["effect_classes"]
    require(len({v["name"] for v in classes}) == len(classes), "Effect-class names must be unique")
    require(any(v["exact_null"] for v in classes), "An exact-null calibration class is required")
    for item in classes:
        if not item["exact_null"]:
            require(0 <= item["lower_percentile"] < item["upper_percentile"] <= 1, "Invalid effect-class interval")
    amp = [v for v in classes if v["name"] == cal["threshold_lag_amplitude_class"]]
    require(len(amp) == 1 and not amp[0]["exact_null"], "A non-null amplitude class is required")
    for pair in cal.get("generating_threshold_lag_pairs") or []:
        require(float(pair[0]) in c["thresholds"] and pair[1] in c["lags"], "Generating pair must be in the candidate grid")
    pop = c.setdefault("population_policy", {"mode": "ANNUAL_NEAREST"})
    require(isinstance(pop, dict), "population_policy must be a configuration object")
    mode = pop.get("mode", "ANNUAL_NEAREST")
    require(mode in SUPPORTED_POPULATION_MODES, "Population policy must be ANNUAL_NEAREST, ANNUAL_STRICT, or STATIC_EXPLICIT")
    pop["mode"] = mode
    if mode == "STATIC_EXPLICIT":
        require(type(pop.get("source_year")) is int, "STATIC_EXPLICIT population policy requires integer source_year")
        require(str(pop["source_year"]) in c["sources"]["population_fields"],
                "STATIC_EXPLICIT source_year must exist in sources.population_fields")
    runtime = c["runtime"]
    require(runtime["worker_count"] == 1, "This bounded-memory runtime requires worker_count=1")
    require(type(runtime["bootstrap_batch_size"]) is int and 1 <= runtime["bootstrap_batch_size"] <= 64, "Bootstrap batch size must be in 1..64")
    require(type(runtime["minimum_free_disk_mb"]) is int and runtime["minimum_free_disk_mb"] >= 0, "Invalid disk space guard")

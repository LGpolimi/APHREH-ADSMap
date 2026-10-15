"""Lightweight project-first desktop launcher. Numerical work runs in a child process.

No Qt, PyQGIS, or project-specific Python settings factory is needed. The text
console is bounded to 1000 lines; no numerical result objects cross into the GUI.
"""
from __future__ import annotations
import copy
import json
import math
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .config import load_config,validate
from .data import csv_headers, source_path
from .storage import atomic_json
from .results_overview import ResultsWorkspace, geometry_attribute_fields

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"


def new_release_configuration() -> dict:
    """Load the distributable scientific template and choose user-local paths.

    Neither the user data directory nor the output directory depends on the
    installation location, developer machine, or a bundled example dataset.
    """
    configuration = load_config(DEFAULT_CONFIG_PATH)
    configuration["data_dir"] = str(Path.home())
    configuration["output_dir"] = str(Path.home() / "APHREH-ADSMap-projects" / "new_project")
    return configuration

EFFECT_CLASS_ORDER = (
    "strong_protective",
    "moderate_protective",
    "negligible",
    "moderate_harmful",
    "strong_harmful",
)
EFFECT_CLASS_LABELS = {
    "strong_protective": "Strong protective",
    "moderate_protective": "Moderate protective",
    "negligible": "Negligible",
    "moderate_harmful": "Moderate harmful",
    "strong_harmful": "Strong harmful",
}


def effect_class_bounds(calibration: dict) -> dict[str, tuple[float, float]]:
    """Return GUI-editable percentile intervals keyed by calibration class name."""
    lookup = {item["name"]: item for item in calibration["effect_classes"] if not item["exact_null"]}
    missing = [name for name in EFFECT_CLASS_ORDER if name not in lookup]
    if missing:
        raise ValueError("GUI effect-class editor requires classes: " + ", ".join(EFFECT_CLASS_ORDER))
    return {name: (float(lookup[name]["lower_percentile"]), float(lookup[name]["upper_percentile"]))
            for name in EFFECT_CLASS_ORDER}


def validate_effect_class_bounds(bounds: dict[str, tuple[float, float]]) -> None:
    """Validate the GUI class semantics without changing the JSON schema.

    The calibration engine can technically read arbitrary intervals, but the named
    protective/negligible/harmful scenarios have directional meaning. The GUI
    therefore prevents edits that contradict those labels or make adjacent classes
    overlap. Advanced JSON remains available for expert configurations outside the
    standard named-class layout.
    """
    missing = [name for name in EFFECT_CLASS_ORDER if name not in bounds]
    if missing:
        raise ValueError("Missing calibration effect classes: " + ", ".join(missing))
    for name in EFFECT_CLASS_ORDER:
        lo, hi = bounds[name]
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"{EFFECT_CLASS_LABELS[name]} must satisfy 0 <= lower < upper <= 1")
    if bounds["strong_protective"][1] > 0.5 or bounds["moderate_protective"][1] > 0.5:
        raise ValueError("Protective effect classes must remain entirely at or below the median percentile 0.5")
    nlo, nhi = bounds["negligible"]
    if not (nlo <= 0.5 <= nhi):
        raise ValueError("The negligible class must contain the median percentile 0.5")
    if bounds["moderate_harmful"][0] < 0.5 or bounds["strong_harmful"][0] < 0.5:
        raise ValueError("Harmful effect classes must remain entirely at or above the median percentile 0.5")
    ordered = [bounds[name] for name in EFFECT_CLASS_ORDER]
    for left_name, right_name, left, right in zip(EFFECT_CLASS_ORDER, EFFECT_CLASS_ORDER[1:], ordered, ordered[1:]):
        if left[1] > right[0]:
            raise ValueError(f"{EFFECT_CLASS_LABELS[left_name]} overlaps {EFFECT_CLASS_LABELS[right_name]}")


def set_effect_class_bounds(calibration: dict, bounds: dict[str, tuple[float, float]]) -> None:
    """Apply validated GUI intervals to the existing calibration.effect_classes list."""
    validate_effect_class_bounds(bounds)
    lookup = {item["name"]: item for item in calibration["effect_classes"]}
    for name in EFFECT_CLASS_ORDER:
        lookup[name]["lower_percentile"], lookup[name]["upper_percentile"] = map(float, bounds[name])


def effect_class_summary(calibration: dict) -> str:
    bounds = effect_class_bounds(calibration)
    parts = []
    for name in EFFECT_CLASS_ORDER:
        lo, hi = bounds[name]
        parts.append(f"{EFFECT_CLASS_LABELS[name]} [{lo:g}, {hi:g}]")
    return " | ".join(parts)


def source_configuration_summary(c: dict) -> str:
    s = c["sources"]
    pops = ", ".join(f"{y}->{field}" for y,field in sorted(s["population_fields"].items(), key=lambda kv:int(kv[0])))
    return (f"Hazard {s['hazard_file']} [date={s['hazard_date_field']}] | "
            f"Outcome {s['outcome_file']} [date={s['outcome_date_field']}] | "
            f"Population {s['population_file']} [BSA={s['bsa_id_field']}; {pops}] | "
            f"Crosswalk {s['crosswalk_file']} [BSA={s['crosswalk_bsa_field']}; grid={s['grid_id_field']}; area={s['intersection_area_field']}] | "
            f"Geometry {s.get('geometry_file','none')} [BSA={s.get('geometry_bsa_field',s['bsa_id_field'])}]")



ROBUSTNESS_COMPARISON_MODES = ("FIXED_PARAMETER", "REOPTIMIZED", "BOTH")
ROBUSTNESS_SUPPORT_FACTORS = (
    ("min_exposed_observations", "Minimum exposed observations"),
    ("min_nonexposed_observations", "Minimum non-exposed observations"),
    ("min_event_contribution", "Minimum event contribution"),
    ("min_baseline_support", "Minimum baseline donors per target day"),
    ("min_effective_weighted_sample_size", "Minimum effective weighted sample size"),
    ("min_bootstrap_validity_rate", "Minimum bootstrap validity rate"),
    ("min_contributing_years", "Minimum contributing years"),
    ("min_local_concordance", "Minimum local concordance"),
)
ROBUSTNESS_DEPENDENCE_FACTORS = (
    ("local_acf_interval_days", "Local ACF interval, days"),
    ("negligible_lag_persistence", "Negligible-lag persistence"),
    ("annual_dependence_quantile", "Annual dependence quantile"),
    ("exposed_episode_quantile", "Exposed-episode quantile"),
    ("final_block_length_multiplier", "Final block-length multiplier"),
)
ROBUSTNESS_BASELINE_NUMERICAL = (
    ("initial_half_width", "Initial half-width"),
    ("maximum_half_width", "Maximum half-width"),
    ("minimum_eligible_dates", "Minimum eligible baseline dates"),
)
ROBUSTNESS_BASELINE_BINARY = (
    ("dynamic_enlargement", "Dynamic enlargement"),
    ("day_of_week_matching", "Exact day-of-week matching"),
    ("holiday_matching", "Holiday-state matching"),
)



def _reviewer_integer_levels(reference, *, moderate_default: int, strict_default: int, minimum: int = 0) -> tuple[int, int]:
    """Return two progressively stricter integer support thresholds.

    The default values are used when the reference policy is inactive.  If a
    project already has an active support threshold, both reviewer scenarios
    are moved above that reference so that the sensitivity run remains a true
    stress test rather than silently relaxing the production analysis.
    """
    if reference is None:
        return max(minimum, int(moderate_default)), max(minimum, int(strict_default))
    ref = int(reference)
    step = max(1, int(math.ceil(max(abs(ref), 1) * 0.25)))
    moderate = max(int(moderate_default), ref + step, int(minimum))
    strict = max(int(strict_default), ref + 2 * step, moderate + 1, int(minimum))
    return moderate, strict


def _reviewer_float_levels(reference, *, moderate_default: float, strict_default: float,
                            minimum: float = 0.0) -> tuple[float, float]:
    """Return two progressively stricter unbounded floating support levels."""
    if reference is None:
        return max(minimum, float(moderate_default)), max(minimum, float(strict_default))
    ref = float(reference)
    step = max(1.0, abs(ref) * 0.25)
    moderate = max(float(moderate_default), ref + step, float(minimum))
    strict = max(float(strict_default), ref + 2.0 * step, moderate + 1.0, float(minimum))
    return moderate, strict


def _reviewer_probability_levels(reference, *, moderate_default: float = 0.90,
                                  strict_default: float = 0.95) -> tuple[float, float]:
    """Return moderate/strict probability thresholds without exceeding one."""
    if reference is None:
        return float(moderate_default), float(strict_default)
    ref = float(reference)
    moderate = min(1.0, max(float(moderate_default), ref + max(0.02, (1.0 - ref) * 0.25)))
    strict = min(1.0, max(float(strict_default), ref + max(0.04, (1.0 - ref) * 0.50)))
    if strict < moderate:
        strict = moderate
    return moderate, strict


def _conservative_probability(reference: float, *, floor: float = 0.95) -> float:
    """One-sided conservative dependence quantile used by reviewer scenario R4."""
    ref = float(reference)
    return round(min(1.0, max(float(floor), ref + 0.05)), 12)


def default_robustness_configuration(c: dict) -> dict:
    """Build the minimal reviewer-oriented RS package used at first initialization.

    The preset corresponds to six analytical states: the unchanged reference
    realization R0, two support-quality states R1a/R1b, one BSA-specific
    exposure state R2, one temporally matched/strengthened baseline state R3,
    and one conservative-dependence state R4.  Analysis 1 is post-processing of
    R0 and therefore does not add another computational realization.
    """
    core = c["core"]
    support = core.get("support", {})
    baseline = core["baseline"]
    dependence = core["dependence"]

    # R1a/R1b: progressively stricter effective-support requirements.  Local
    # concordance is intentionally excluded because regional versus local
    # exposure classification is tested structurally by R2.  Baseline donor
    # support is handled in R3 rather than duplicated here.
    exp_moderate, exp_strict = _reviewer_integer_levels(
        support.get("min_exposed_observations"), moderate_default=10, strict_default=15, minimum=1
    )
    nonexp_moderate, nonexp_strict = _reviewer_integer_levels(
        support.get("min_nonexposed_observations"), moderate_default=40, strict_default=60, minimum=1
    )
    event_moderate, event_strict = _reviewer_integer_levels(
        support.get("min_event_contribution"), moderate_default=10, strict_default=20, minimum=0
    )
    effective_moderate, effective_strict = _reviewer_float_levels(
        support.get("min_effective_weighted_sample_size"), moderate_default=5.0, strict_default=10.0, minimum=0.0
    )
    validity_moderate, validity_strict = _reviewer_probability_levels(
        support.get("min_bootstrap_validity_rate"), moderate_default=0.90, strict_default=0.95
    )

    # R3: exact weekday matching plus stronger donor availability.  Holiday
    # matching is automatically included when a holiday calendar is configured;
    # it cannot be enabled safely when the required source is absent.
    maximum = int(baseline["maximum_half_width"])
    maximum_target = max(45, maximum + max(7, int(math.ceil(maximum * 0.50))))
    minimum_eligible = baseline.get("minimum_eligible_dates")
    if minimum_eligible is None:
        minimum_eligible_target = 4
    else:
        minimum_eligible_target = max(4, int(minimum_eligible) + max(1, int(math.ceil(int(minimum_eligible) * 0.25))))
    baseline_factors: dict[str, dict] = {
        "maximum_half_width": {"lower": maximum_target, "upper": maximum_target},
        "minimum_eligible_dates": {"lower": minimum_eligible_target, "upper": minimum_eligible_target},
    }
    if not bool(baseline.get("dynamic_enlargement", False)):
        baseline_factors["dynamic_enlargement"] = {"alternative": True}
    if not bool(baseline.get("day_of_week_matching", False)):
        baseline_factors["day_of_week_matching"] = {"alternative": True}
    holiday_available = bool(c.get("sources", {}).get("holiday_calendar_file"))
    if holiday_available and not bool(baseline.get("holiday_matching", False)):
        baseline_factors["holiday_matching"] = {"alternative": True}

    # R4: one-sided conservative dependence assumptions.  Equal lower/upper
    # bounds deliberately create a single alternative state after planner
    # duplicate removal while retaining the existing numerical-scenario schema.
    annual_target = _conservative_probability(dependence["annual_dependence_quantile"])
    episode_target = _conservative_probability(dependence["exposed_episode_quantile"])
    multiplier = float(dependence["final_block_length_multiplier"])
    multiplier_target = max(1.25, multiplier + 0.25, multiplier * 1.25)
    persistence = int(dependence["negligible_lag_persistence"])
    maximum_acf = int(dependence["maximum_acf_lag_days"])
    persistence_target = min(maximum_acf, max(persistence + 2, int(math.ceil(persistence * 1.50))))

    a3_compatible = (
        core.get("exposure", {}).get("mode") == "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE"
        and support.get("min_local_concordance") is None
    )

    return {
        "enabled": True,
        "implementation_stage": "FINGERPRINT_SURFACE_COMPARISON",
        "analysis_1_region_stability": {
            "enabled": True,
            "minimum_valid_cell_proportion": None,
        },
        "analysis_2_support_sensitivity": {
            "enabled": True,
            "scenarios": [{
                "scenario_id": "reviewer_support_quality",
                "label": "Reviewer R1a/R1b support quality: moderate to strict",
                "steps": 2,
                "comparison_mode": "BOTH",
                "factors": {
                    "min_exposed_observations": {"lower": exp_moderate, "upper": exp_strict},
                    "min_nonexposed_observations": {"lower": nonexp_moderate, "upper": nonexp_strict},
                    "min_event_contribution": {"lower": event_moderate, "upper": event_strict},
                    "min_effective_weighted_sample_size": {"lower": effective_moderate, "upper": effective_strict},
                    "min_bootstrap_validity_rate": {"lower": validity_moderate, "upper": validity_strict},
                },
            }],
        },
        "analysis_3_bsa_specific_exposure": {
            "enabled": bool(a3_compatible),
            "label": "Reviewer R2 BSA-specific exposure classification",
            "comparison_mode": "BOTH",
        },
        "analysis_4_baseline_sensitivity": {
            "enabled": True,
            "scenarios": [{
                "scenario_id": "reviewer_temporally_matched_baseline",
                "label": (
                    "Reviewer R3 weekday/holiday matched strengthened baseline"
                    if holiday_available else
                    "Reviewer R3 weekday-matched strengthened baseline"
                ),
                "steps": 2,
                "comparison_mode": "BOTH",
                "factors": baseline_factors,
            }],
        },
        "analysis_5_dependence_sensitivity": {
            "enabled": True,
            "scenarios": [{
                "scenario_id": "reviewer_conservative_dependence",
                "label": "Reviewer R4 conservative temporal-dependence specification",
                "steps": 2,
                "comparison_mode": "BOTH",
                "factors": {
                    "annual_dependence_quantile": {"lower": annual_target, "upper": annual_target},
                    "exposed_episode_quantile": {"lower": episode_target, "upper": episode_target},
                    "final_block_length_multiplier": {"lower": multiplier_target, "upper": multiplier_target},
                    "negligible_lag_persistence": {"lower": persistence_target, "upper": persistence_target},
                },
            }],
        },
    }

def robustness_configuration_is_uninitialized(c: dict) -> bool:
    """Return True only for the untouched empty RS setup from earlier builds."""
    rs = c.get("robustness")
    if not isinstance(rs, dict):
        return True
    if bool(rs.get("enabled", False)):
        return False
    for key in (
        "analysis_2_support_sensitivity",
        "analysis_4_baseline_sensitivity",
        "analysis_5_dependence_sensitivity",
    ):
        section = rs.get(key) or {}
        if bool(section.get("enabled", False)) or bool(section.get("scenarios")):
            return False
    a1 = rs.get("analysis_1_region_stability") or {}
    if bool(a1.get("enabled", True)) is not True or a1.get("minimum_valid_cell_proportion") is not None:
        return False
    a3 = rs.get("analysis_3_bsa_specific_exposure") or {}
    if bool(a3.get("enabled", False)):
        return False
    if a3.get("comparison_mode", "FIXED_PARAMETER") != "FIXED_PARAMETER":
        return False
    if str(a3.get("label", "BSA-specific exposure classification")) != "BSA-specific exposure classification":
        return False
    return True


def initialize_robustness_defaults(c: dict) -> dict:
    """Return a detached config with standard RS defaults when RS is untouched."""
    result = copy.deepcopy(c)
    if robustness_configuration_is_uninitialized(result):
        result["robustness"] = default_robustness_configuration(result)
    return result


def robustness_configuration_summary(c: dict) -> str:
    rs = c.get("robustness") or {}
    a1 = rs.get("analysis_1_region_stability") or {}
    parts = [f"A1 region stability: {'on' if a1.get('enabled', True) else 'off'}"]
    enabled = bool(rs.get("enabled", False))
    if not enabled:
        parts.append("alternative sensitivities: off")
        return " | ".join(parts)
    for key, label in (
        ("analysis_2_support_sensitivity", "A2 support"),
        ("analysis_3_bsa_specific_exposure", "A3 local exposure"),
        ("analysis_4_baseline_sensitivity", "A4 baseline"),
        ("analysis_5_dependence_sensitivity", "A5 dependence"),
    ):
        sec = rs.get(key) or {}
        if not sec.get("enabled"):
            continue
        if key == "analysis_3_bsa_specific_exposure":
            parts.append(f"{label}: {sec.get('comparison_mode','FIXED_PARAMETER')}")
        else:
            scenarios = sec.get("scenarios") or []
            parts.append(f"{label}: {len(scenarios)} scenario{'s' if len(scenarios)!=1 else ''}")
    return " | ".join(parts)




def confounder_configuration_summary(c: dict) -> str:
    ca = c.get("confounder") or {}
    candidate = ca.get("candidate") or {}
    if not ca.get("enabled"):
        if candidate:
            return f"off | configured candidate: {candidate.get('display_name') or candidate.get('candidate_id') or 'unnamed'}"
        return "off | no candidate configured"
    label = candidate.get("display_name") or candidate.get("candidate_id") or "unnamed candidate"
    source = candidate.get("hazard_file") or candidate.get("source_reference") or "—"
    mode = "complete adjusted reoptimization" if ca.get("request_complete_reoptimization") else "fixed-parameter adjustment"
    return (f"on | {label} | APHREH hazard field: {source} | {mode} | "
            f"tau={float(ca.get('residual_relevance_tolerance',0.10)):g} | "
            f"minimum retained support={float(ca.get('minimum_retained_support',0.60)):g}")

class RobustnessScenarioDialog(tk.Toplevel):
    """Structured editor for one robustness sensitivity scenario."""

    def __init__(self, parent, title, scenario, numerical_specs, binary_specs, reference_values, on_apply):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x640")
        self.minsize(820, 560)
        self.transient(parent)
        self.on_apply = on_apply
        self.numerical_specs = tuple(numerical_specs)
        self.binary_specs = tuple(binary_specs)
        self.reference_values = dict(reference_values)
        scenario = copy.deepcopy(scenario or {})
        outer = ttk.Frame(self, padding=14); outer.pack(fill="both", expand=True)
        meta = ttk.LabelFrame(outer, text="Scenario", padding=10); meta.pack(fill="x")
        meta.columnconfigure(1, weight=1)
        self.sid = tk.StringVar(value=scenario.get("scenario_id", ""))
        self.label = tk.StringVar(value=scenario.get("label", scenario.get("scenario_id", "")))
        self.steps = tk.StringVar(value=str(scenario.get("steps", 5 if self.numerical_specs else 1)))
        self.mode = tk.StringVar(value=scenario.get("comparison_mode", "FIXED_PARAMETER"))
        for row, (text, var) in enumerate((("Scenario ID", self.sid), ("Label", self.label), ("Steps", self.steps))):
            ttk.Label(meta, text=text).grid(row=row, column=0, sticky="w", padx=(0,10), pady=3)
            ttk.Entry(meta, textvariable=var).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Label(meta, text="Comparison mode").grid(row=3, column=0, sticky="w", padx=(0,10), pady=3)
        ttk.Combobox(meta, textvariable=self.mode, values=ROBUSTNESS_COMPARISON_MODES, state="readonly").grid(row=3, column=1, sticky="w", pady=3)

        factors = scenario.get("factors") or {}
        body = ttk.LabelFrame(outer, text="Sensitivity factors", padding=10); body.pack(fill="both", expand=True, pady=(12,0))
        body.columnconfigure(1, weight=1); body.columnconfigure(2, weight=1)
        ttk.Label(body, text="Use").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Factor / reference").grid(row=0, column=1, sticky="w")
        ttk.Label(body, text="Lower / alternative").grid(row=0, column=2, sticky="w")
        ttk.Label(body, text="Upper").grid(row=0, column=3, sticky="w")
        self.factor_vars = {}
        row = 1
        for name, label in self.numerical_specs:
            definition = factors.get(name) or {}
            active = tk.BooleanVar(value=name in factors)
            lo = tk.StringVar(value="" if definition.get("lower") is None else str(definition.get("lower")))
            hi = tk.StringVar(value="" if definition.get("upper") is None else str(definition.get("upper")))
            ref = self.reference_values.get(name)
            ttk.Checkbutton(body, variable=active).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(body, text=f"{label}  [ref: {ref}]").grid(row=row, column=1, sticky="w", pady=3)
            ttk.Entry(body, textvariable=lo, width=16).grid(row=row, column=2, sticky="ew", padx=(4,4), pady=3)
            ttk.Entry(body, textvariable=hi, width=16).grid(row=row, column=3, sticky="ew", pady=3)
            self.factor_vars[name] = ("NUMERICAL", active, lo, hi)
            row += 1
        for name, label in self.binary_specs:
            definition = factors.get(name) or {}
            active = tk.BooleanVar(value=name in factors)
            alt = tk.StringVar(value=str(definition.get("alternative", True)).lower())
            ref = self.reference_values.get(name)
            ttk.Checkbutton(body, variable=active).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(body, text=f"{label}  [ref: {ref}]").grid(row=row, column=1, sticky="w", pady=3)
            ttk.Combobox(body, textvariable=alt, values=("true", "false"), state="readonly", width=14).grid(row=row, column=2, sticky="w", padx=(4,4), pady=3)
            ttk.Label(body, text="—").grid(row=row, column=3, sticky="w", pady=3)
            self.factor_vars[name] = ("BINARY", active, alt, None)
            row += 1
        ttk.Label(body, text="Selected numerical factors move synchronously along the requested steps; they are not expanded factorially.", wraplength=800, justify="left").grid(row=row, column=0, columnspan=4, sticky="w", pady=(10,0))
        buttons = ttk.Frame(outer); buttons.pack(fill="x", pady=(10,0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=(8,0))
        ttk.Button(buttons, text="Apply", command=self._apply).pack(side="right")
        self.grab_set()

    def _apply(self):
        try:
            sid = self.sid.get().strip()
            if not sid:
                raise ValueError("Scenario ID is required")
            factors = {}
            has_numeric = False
            for name, values in self.factor_vars.items():
                kind, active, first, second = values
                if not active.get():
                    continue
                if kind == "NUMERICAL":
                    lo = float(first.get().strip()); hi = float(second.get().strip())
                    # Preserve integers when both values were entered as integers.
                    if lo.is_integer() and hi.is_integer():
                        lo, hi = int(lo), int(hi)
                    factors[name] = {"lower": lo, "upper": hi}
                    has_numeric = True
                else:
                    factors[name] = {"alternative": first.get().strip().lower() == "true"}
            if not factors:
                raise ValueError("Select at least one sensitivity factor")
            steps = int(self.steps.get().strip())
            if has_numeric and steps < 2:
                raise ValueError("Scenarios with numerical factors require at least 2 steps")
            if not has_numeric and steps < 1:
                raise ValueError("Steps must be at least 1")
            scenario = {
                "scenario_id": sid,
                "label": self.label.get().strip() or sid,
                "steps": steps,
                "comparison_mode": self.mode.get(),
                "factors": factors,
            }
            self.on_apply(scenario)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid robustness scenario", str(exc), parent=self)


class RobustnessScenarioTab(ttk.Frame):
    def __init__(self, parent, title, section, numerical_specs, binary_specs, reference_values):
        super().__init__(parent, padding=10)
        self.title_text = title
        self.section = copy.deepcopy(section or {})
        self.numerical_specs = tuple(numerical_specs)
        self.binary_specs = tuple(binary_specs)
        self.reference_values = dict(reference_values)
        self.enabled = tk.BooleanVar(value=bool(self.section.get("enabled", False)))
        self.scenarios = copy.deepcopy(self.section.get("scenarios") or [])
        top = ttk.Frame(self); top.pack(fill="x")
        ttk.Checkbutton(top, text=f"Enable {title}", variable=self.enabled).pack(side="left")
        ttk.Label(top, text="Each scenario is a prespecified methodological trajectory.", wraplength=620).pack(side="left", padx=(14,0))
        self.tree = ttk.Treeview(self, columns=("id","steps","mode","factors"), show="headings", height=9, selectmode="browse")
        for key, text, width in (("id","Scenario",170),("steps","Steps",60),("mode","Comparison",130),("factors","Factors",440)):
            self.tree.heading(key, text=text); self.tree.column(key, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(10,6))
        buttons = ttk.Frame(self); buttons.pack(fill="x")
        ttk.Button(buttons, text="Add...", command=self.add).pack(side="left")
        ttk.Button(buttons, text="Edit...", command=self.edit).pack(side="left", padx=(6,0))
        ttk.Button(buttons, text="Remove", command=self.remove).pack(side="left", padx=(6,0))
        self.refresh_tree()

    def _factor_summary(self, scenario):
        parts=[]
        for name, d in (scenario.get("factors") or {}).items():
            if "alternative" in d:
                parts.append(f"{name}={d['alternative']}")
            else:
                parts.append(f"{name}:{d.get('lower')}→{d.get('upper')}")
        return "; ".join(parts)

    def refresh_tree(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for idx, scenario in enumerate(self.scenarios):
            self.tree.insert("", "end", iid=str(idx), values=(scenario.get("scenario_id"), scenario.get("steps"), scenario.get("comparison_mode"), self._factor_summary(scenario)))

    def _open(self, index=None):
        scenario = None if index is None else self.scenarios[index]
        def save(value):
            if index is None: self.scenarios.append(value)
            else: self.scenarios[index] = value
            self.refresh_tree()
        RobustnessScenarioDialog(self, self.title_text, scenario, self.numerical_specs, self.binary_specs, self.reference_values, save)

    def add(self): self._open(None)
    def edit(self):
        sel=self.tree.selection()
        if sel: self._open(int(sel[0]))
    def remove(self):
        sel=self.tree.selection()
        if not sel: return
        self.scenarios.pop(int(sel[0])); self.refresh_tree()

    def value(self):
        return {"enabled": bool(self.enabled.get()), "scenarios": copy.deepcopy(self.scenarios)}


def relative_source_name(path: str | Path, data_dir: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    root = Path(data_dir).expanduser().resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


class Application:
    def __init__(self,root: tk.Tk):
        self.root = root
        self.process = None
        self.console_writer = self.console_reader = None
        self.close_after = False
        root.title("APHREH-ADSMap 2.0.1")
        root.geometry("1160x820")
        root.minsize(850,650)
        self.c = new_release_configuration()
        self.vars = {}
        top = ttk.Frame(root,padding=12);top.pack(fill="x")
        ttk.Label(top,text="APHREH-ADSMap",font=("Segoe UI",20,"bold")).pack(anchor="w")
        ttk.Label(top,text="Project-first bounded-memory workflow | Reference, calibration, optimization, robustness/sensitivity, and confounder adjustment").pack(anchor="w")
        actions = ttk.Frame(root,padding=(12,0));actions.pack(fill="x")
        for label,command in (("New project",self.new_project),("Open project",self.open_project),
                              ("Load JSON configuration",self.open_config),("Advanced JSON",self.edit_json)):
            ttk.Button(actions,text=label,command=command).pack(side="left",padx=(0,8),pady=5)
        form = ttk.LabelFrame(root,text="Project and analysis settings",padding=12);form.pack(fill="x",padx=12,pady=8)
        form.columnconfigure(1,weight=1)
        fields = [("data_dir","Input/source root directory"),("output_dir","Project/output directory"),
                  ("thresholds","Threshold percentiles"),("lags","Cumulative lags"),
                  ("bootstrap","Bootstrap iterations"),("calibration_count","Synthetic realizations per scenario"),("null_count","Null full-surface searches")]
        for row,(name,label) in enumerate(fields):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=3)
            v = tk.StringVar();self.vars[name] = v
            ttk.Entry(form,textvariable=v).grid(row=row,column=1,sticky="ew",pady=3)
            if name in ("data_dir","output_dir"):
                ttk.Button(form,text="Browse...",command=lambda n=name:self.browse(n)).grid(row=row,column=2,padx=(8,0))
        source_row = len(fields)
        ttk.Label(form,text="Data sources and field mapping").grid(row=source_row,column=0,sticky="nw",padx=(0,12),pady=3)
        self.source_summary = tk.StringVar()
        ttk.Label(form,textvariable=self.source_summary,wraplength=760,justify="left").grid(row=source_row,column=1,sticky="ew",pady=3)
        self.source_button = ttk.Button(form,text="Configure...",command=self.edit_sources)
        self.source_button.grid(row=source_row,column=2,padx=(8,0),sticky="n")
        effect_row = source_row + 1
        ttk.Label(form,text="Calibration effect percentiles").grid(row=effect_row,column=0,sticky="nw",padx=(0,12),pady=3)
        self.effect_summary = tk.StringVar()
        ttk.Label(form,textvariable=self.effect_summary,wraplength=720,justify="left").grid(row=effect_row,column=1,sticky="ew",pady=3)
        self.effect_button = ttk.Button(form,text="Edit...",command=self.edit_effect_classes)
        self.effect_button.grid(row=effect_row,column=2,padx=(8,0),sticky="n")
        robustness_row = effect_row + 1
        ttk.Label(form,text="Robustness / sensitivity").grid(row=robustness_row,column=0,sticky="nw",padx=(0,12),pady=3)
        self.robustness_summary = tk.StringVar()
        ttk.Label(form,textvariable=self.robustness_summary,wraplength=720,justify="left").grid(row=robustness_row,column=1,sticky="ew",pady=3)
        self.robustness_button = ttk.Button(form,text="Configure...",command=self.edit_robustness)
        self.robustness_button.grid(row=robustness_row,column=2,padx=(8,0),sticky="n")
        confounder_row = robustness_row + 1
        ttk.Label(form,text="Candidate confounder").grid(row=confounder_row,column=0,sticky="nw",padx=(0,12),pady=3)
        self.confounder_summary = tk.StringVar()
        ttk.Label(form,textvariable=self.confounder_summary,wraplength=720,justify="left").grid(row=confounder_row,column=1,sticky="ew",pady=3)
        self.confounder_button = ttk.Button(form,text="Configure...",command=self.edit_confounder)
        self.confounder_button.grid(row=confounder_row,column=2,padx=(8,0),sticky="n")
        self.phase = tk.StringVar(value="all")
        ttk.Label(form,text="Workflow").grid(row=confounder_row+1,column=0,sticky="w",pady=3)
        ttk.Combobox(form,textvariable=self.phase,values=("all","calibration","reference","robustness","confounder"),state="readonly").grid(row=confounder_row+1,column=1,sticky="w")
        self.notice = tk.StringVar()
        ttk.Label(root,textvariable=self.notice,wraplength=1000).pack(fill="x",padx=15,pady=4)
        controls = ttk.Frame(root,padding=(12,0));controls.pack(fill="x")
        self.run_button = ttk.Button(controls,text="Save project and run",command=self.start);self.run_button.pack(side="left",padx=(0,8))
        self.rs_run_button = ttk.Button(controls,text="Run robustness / sensitivity",command=self.start_robustness);self.rs_run_button.pack(side="left",padx=(0,8))
        self.ca_run_button = ttk.Button(controls,text="Run confounder adjustment",command=self.start_confounder);self.ca_run_button.pack(side="left",padx=(0,8))
        self.cancel_button = ttk.Button(controls,text="Cancel safely",command=self.cancel,state="disabled");self.cancel_button.pack(side="left",padx=(0,8))
        ttk.Button(controls,text="Results workspace",command=self.open_results_workspace).pack(side="left",padx=(0,8))
        ttk.Button(controls,text="Open static HTML report",command=self.open_report).pack(side="left")
        self.status = tk.StringVar(value="Ready. Review population and date assumptions before running.")
        ttk.Label(controls,textvariable=self.status).pack(side="right")
        box = ttk.Frame(root,padding=12);box.pack(fill="both",expand=True)
        self.console = tk.Text(box,height=13,wrap="word",state="disabled",font=("Consolas",10))
        scroll = ttk.Scrollbar(box,orient="vertical",command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side="left",fill="both",expand=True);scroll.pack(side="right",fill="y")
        self.refresh()
        root.protocol("WM_DELETE_WINDOW",self.close)

    def refresh(self):
        c = self.c
        for name in ("data_dir","output_dir"):
            self.vars[name].set(c[name])
        self.vars["thresholds"].set(", ".join(f"{q:g}" for q in c["thresholds"]))
        self.vars["lags"].set(", ".join(map(str,c["lags"])))
        self.vars["bootstrap"].set(c["core"]["bootstrap"]["iterations"])
        self.vars["calibration_count"].set(c["calibration"]["number_of_realizations"])
        self.vars["null_count"].set(c["optimization"]["null_realizations_R"])
        try:
            self.source_summary.set(source_configuration_summary(c))
        except Exception as exc:
            self.source_summary.set(f"Source configuration error: {exc}")
        try:
            self.effect_summary.set(effect_class_summary(c["calibration"]))
        except Exception as exc:
            self.effect_summary.set(f"Custom effect-class configuration: {exc}")
        try:
            self.robustness_summary.set(robustness_configuration_summary(c))
        except Exception as exc:
            self.robustness_summary.set(f"Robustness configuration error: {exc}")
        try:
            self.confounder_summary.set(confounder_configuration_summary(c))
        except Exception as exc:
            self.confounder_summary.set(f"Confounder configuration error: {exc}")
        self.notice.set(f"Analysis period: exact matching dates between hazard and outcome; calendar gaps are not bridged. Population policy: {c['population_policy']}. Threshold-lag calibration: {c['calibration']['threshold_lag_response_modes']} with {c['calibration']['primary_threshold_lag_response_mode']} primary. Robustness and candidate-confounder adjustment are post-reference workflows that may be attached to a loaded completed project. Changes to reference scientific settings require a new output directory.")

    def capture(self):
        c = copy.deepcopy(self.c)
        for name in ("data_dir","output_dir"):
            c[name] = self.vars[name].get().strip()
        for name in ("data_dir","output_dir"):
            c[name] = str(Path(c[name]).expanduser().resolve())
        c["thresholds"] = [float(v.strip()) for v in self.vars["thresholds"].get().split(",") if v.strip()]
        c["lags"] = [int(v.strip()) for v in self.vars["lags"].get().split(",") if v.strip()]
        c["core"]["bootstrap"]["iterations"] = int(self.vars["bootstrap"].get())
        c["calibration"]["number_of_realizations"] = int(self.vars["calibration_count"].get())
        c["optimization"]["null_realizations_R"] = int(self.vars["null_count"].get())
        validate(c)
        return c

    def browse(self,name):
        selected = filedialog.askdirectory(initialdir=self.vars[name].get(),title="Select "+name)
        if selected:
            self.vars[name].set(selected)

    def new_project(self):
        if self.process is not None:
            return
        self.c = new_release_configuration()
        self.phase.set("all")
        self.refresh()
        self.browse("output_dir")

    def open_project(self):
        directory = filedialog.askdirectory(title="Open APHREH-ADSMap project")
        if not directory:
            return
        for name in ("resolved_config.json","project_config.json"):
            path = Path(directory)/name
            if path.exists():
                self.load(path)
                self.vars["output_dir"].set(directory)
                manifest_path = Path(directory)/"manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest.get("reference_completed"):
                            ca = self.c.get("confounder") or {}
                            if ca.get("enabled") and ca.get("candidate"):
                                self.phase.set("confounder")
                            else:
                                self.phase.set("robustness")
                            self.status.set("Loaded completed reference project. Robustness and candidate-confounder workflows can be configured and run without redefining the reference analysis.")
                    except Exception:
                        pass
                return
        messagebox.showerror("Project not recognized","Select an APHREH project folder containing resolved_config.json or project_config.json. Use a new output folder when scientific settings differ from an existing completed project.")

    def open_config(self):
        path = filedialog.askopenfilename(initialdir=DEFAULT_CONFIG_PATH.parent,filetypes=[("JSON configuration","*.json")])
        if path:
            self.load(Path(path))

    def load(self,path):
        try:
            self.c = load_config(path);self.refresh()
        except Exception as exc:
            messagebox.showerror("Configuration error",str(exc))

    def edit_json(self):
        try:
            c = self.capture()
        except Exception as exc:
            messagebox.showerror("Configuration error",str(exc));return
        window = tk.Toplevel(self.root);window.title("Full project configuration");window.geometry("950x700")
        text = tk.Text(window,wrap="none");text.pack(fill="both",expand=True)
        text.insert("1.0",json.dumps(c,indent=2))
        def apply():
            try:
                value = json.loads(text.get("1.0","end"));validate(value)
                self.c = value;self.refresh();window.destroy()
            except Exception as exc:
                messagebox.showerror("Configuration error",str(exc),parent=window)
        ttk.Button(window,text="Apply",command=apply).pack(pady=8)

    def edit_sources(self):
        if self.process is not None:
            messagebox.showinfo("Analysis running","Source mappings are frozen while an analysis is running.")
            return
        try:
            # Capture pending main-form edits before opening a sub-dialog.  Any
            # later refresh must therefore preserve unrelated settings such as
            # output_dir, thresholds, lags, and run counts.
            base = self.capture()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc)); return
        data_dir = Path(base["data_dir"]).expanduser().resolve()
        current = copy.deepcopy(base["sources"])
        window = tk.Toplevel(self.root)
        window.title("Data sources and field configuration")
        window.geometry("980x720")
        window.minsize(900,620)
        window.transient(self.root)
        outer = ttk.Frame(window,padding=12);outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Configure files and field names for the standard APHREH tabular structure.",font=("Segoe UI",11,"bold")).pack(anchor="w")
        ttk.Label(outer,text="The analysis period is detected automatically from exact matching hazard/outcome dates. Population values use the nearest mapped year; equal-distance ties use the more recent year.",wraplength=930,justify="left").pack(anchor="w",pady=(3,10))
        grid = ttk.Frame(outer);grid.pack(fill="x")
        grid.columnconfigure(1,weight=1)
        file_vars = {}
        field_vars = {}
        header_cache = {}
        field_widgets = {}

        file_specs = [
            ("hazard_file","Hazard CSV"),
            ("outcome_file","Outcome CSV"),
            ("population_file","Population CSV"),
            ("crosswalk_file","Crosswalk CSV"),
            ("geometry_file","Geometry file, optional"),
        ]
        row = 0
        for key,label in file_specs:
            ttk.Label(grid,text=label).grid(row=row,column=0,sticky="w",padx=(0,10),pady=3)
            var=tk.StringVar(value=current.get(key,""));file_vars[key]=var
            ttk.Entry(grid,textvariable=var).grid(row=row,column=1,sticky="ew",pady=3)
            ttk.Button(grid,text="Browse...",command=lambda k=key:self._browse_source_file(window,data_dir,file_vars,k,header_cache,field_widgets)).grid(row=row,column=2,padx=(8,0),pady=3)
            row += 1

        ttk.Separator(grid,orient="horizontal").grid(row=row,column=0,columnspan=3,sticky="ew",pady=8);row+=1
        mapping_specs = [
            ("hazard_date_field","Hazard date field","hazard_file"),
            ("outcome_date_field","Outcome date field","outcome_file"),
            ("bsa_id_field","Population BSA ID field","population_file"),
            ("bsa_area_field","Population BSA area field","population_file"),
            ("crosswalk_bsa_field","Crosswalk BSA ID field","crosswalk_file"),
            ("grid_id_field","Crosswalk grid ID field","crosswalk_file"),
            ("intersection_area_field","Crosswalk intersection area field","crosswalk_file"),
            ("geometry_bsa_field","Geometry BSA ID field","geometry_file"),
        ]
        for key,label,filekey in mapping_specs:
            ttk.Label(grid,text=label).grid(row=row,column=0,sticky="w",padx=(0,10),pady=3)
            var=tk.StringVar(value=current.get(key,""));field_vars[key]=var
            combo=ttk.Combobox(grid,textvariable=var,state="normal")
            combo.grid(row=row,column=1,sticky="ew",pady=3)
            field_widgets.setdefault(filekey,[]).append(combo)
            row+=1

        pop_frame = ttk.LabelFrame(outer,text="Population year to field mapping",padding=10);pop_frame.pack(fill="both",expand=True,pady=(12,0))
        tree = ttk.Treeview(pop_frame,columns=("year","field"),show="headings",height=7,selectmode="browse")
        tree.heading("year",text="Population year");tree.heading("field",text="CSV field")
        tree.column("year",width=140,anchor="center");tree.column("field",width=420,anchor="w")
        tree.pack(fill="both",expand=True)
        for y,field in sorted(current.get("population_fields",{}).items(),key=lambda kv:int(kv[0])):
            tree.insert("", "end", values=(y,field))
        edit = ttk.Frame(pop_frame);edit.pack(fill="x",pady=(8,0))
        ttk.Label(edit,text="Year").pack(side="left")
        year_var=tk.StringVar();ttk.Entry(edit,textvariable=year_var,width=10).pack(side="left",padx=(5,12))
        ttk.Label(edit,text="Field").pack(side="left")
        pop_field_var=tk.StringVar();pop_combo=ttk.Combobox(edit,textvariable=pop_field_var,state="normal",width=35);pop_combo.pack(side="left",padx=(5,8))
        field_widgets.setdefault("population_file",[]).append(pop_combo)
        def add_population_mapping():
            try:
                y=int(year_var.get().strip());field=pop_field_var.get().strip()
                if not 1000<=y<=9999 or not field: raise ValueError
            except Exception:
                messagebox.showerror("Population mapping","Enter a four-digit year and a nonempty population field.",parent=window);return
            for item in tree.get_children():
                values=tree.item(item,"values")
                if int(values[0])==y:
                    tree.item(item,values=(str(y),field));break
            else:
                tree.insert("","end",values=(str(y),field))
            year_var.set("");pop_field_var.set("")
        def remove_population_mapping():
            for item in tree.selection(): tree.delete(item)
        ttk.Button(edit,text="Add / update",command=add_population_mapping).pack(side="left",padx=(0,6))
        ttk.Button(edit,text="Remove selected",command=remove_population_mapping).pack(side="left")

        def reload_headers(filekey):
            configured=file_vars[filekey].get().strip()
            headers=[]
            if configured:
                path=source_path(data_dir,configured)
                if path.is_file():
                    try:
                        headers=csv_headers(path) if path.suffix.lower()==".csv" else geometry_attribute_fields(path)
                    except Exception:
                        headers=[]
            header_cache[filekey]=headers
            for widget in field_widgets.get(filekey,[]): widget.configure(values=headers)
        for key,_ in file_specs:
            reload_headers(key)

        buttons=ttk.Frame(outer);buttons.pack(fill="x",pady=(10,0))
        def apply():
            try:
                updated=copy.deepcopy(base)
                updated["data_dir"]=str(data_dir)
                src=updated["sources"]
                for key,_ in file_specs:
                    value=file_vars[key].get().strip()
                    if value or key!="geometry_file": src[key]=value
                    elif key in src: src.pop(key)
                for key,_,_ in mapping_specs: src[key]=field_vars[key].get().strip()
                mapping={}
                for item in tree.get_children():
                    y,field=tree.item(item,"values");mapping[str(int(y))]=str(field).strip()
                src["population_fields"]=mapping
                # Source mapping is independent from the population-selection
                # policy.  Preserve the previously configured policy instead
                # of silently resetting it to ANNUAL_NEAREST.
                validate(updated)
                # File and column checks are performed here for immediate GUI feedback.
                for filekey in ("hazard_file","outcome_file","population_file","crosswalk_file"):
                    reload_headers(filekey)
                    path=source_path(data_dir,src[filekey])
                    if not path.is_file(): raise ValueError(f"Required source does not exist: {path}")
                checks={
                    "hazard_file":[src["hazard_date_field"]],
                    "outcome_file":[src["outcome_date_field"]],
                    "population_file":[src["bsa_id_field"],src["bsa_area_field"],*src["population_fields"].values()],
                    "crosswalk_file":[src["crosswalk_bsa_field"],src["grid_id_field"],src["intersection_area_field"]],
                }
                for filekey,fields in checks.items():
                    headers=csv_headers(source_path(data_dir,src[filekey]))
                    missing=[v for v in fields if v not in headers]
                    if missing: raise ValueError(f"{filekey}: configured fields not found: {', '.join(missing)}")
                if src.get("geometry_file"):
                    geometry_path=source_path(data_dir,src["geometry_file"])
                    if not geometry_path.is_file(): raise ValueError(f"Configured geometry does not exist: {geometry_path}")
                    geometry_fields=geometry_attribute_fields(geometry_path)
                    if src["geometry_bsa_field"] not in geometry_fields:
                        raise ValueError(f"geometry_file: configured BSA field not found: {src['geometry_bsa_field']}")
                self.c=updated
                self.vars["data_dir"].set(str(data_dir))
                self.refresh();window.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid source configuration",str(exc),parent=window)
        ttk.Button(buttons,text="Cancel",command=window.destroy).pack(side="right",padx=(8,0))
        ttk.Button(buttons,text="Apply",command=apply).pack(side="right")
        window.grab_set()

    def _browse_source_file(self,parent,data_dir,file_vars,key,header_cache,field_widgets):
        current=file_vars[key].get().strip()
        initial=data_dir
        if current:
            p=source_path(data_dir,current)
            if p.parent.exists(): initial=p.parent
        types=[("CSV files","*.csv"),("All files","*.*")] if key!="geometry_file" else [("Spatial files","*.gpkg *.geojson *.json"),("GeoPackage","*.gpkg"),("GeoJSON","*.geojson"),("All files","*.*")]
        selected=filedialog.askopenfilename(parent=parent,initialdir=initial,filetypes=types)
        if not selected: return
        file_vars[key].set(relative_source_name(selected,data_dir))
        try:
            headers=csv_headers(Path(selected)) if key!="geometry_file" else geometry_attribute_fields(Path(selected))
        except Exception as exc:
            messagebox.showerror("Cannot read source fields",str(exc),parent=parent);headers=[]
        header_cache[key]=headers
        for widget in field_widgets.get(key,[]): widget.configure(values=headers)

    def edit_robustness(self):
        if self.process is not None:
            messagebox.showinfo("Analysis running", "Robustness settings are frozen while an analysis is running.")
            return
        try:
            base = self.capture()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc)); return
        base = initialize_robustness_defaults(base)
        current = copy.deepcopy(base.get("robustness") or {})
        window = tk.Toplevel(self.root)
        window.title("Robustness / sensitivity configuration")
        window.geometry("1080x760")
        window.minsize(940, 650)
        window.transient(self.root)
        outer = ttk.Frame(window, padding=12); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Robustness / sensitivity", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text=("Configure the prespecified robustness realizations. FIXED_PARAMETER compares at the reference robust representative cell; "
                  "REOPTIMIZED performs realization-specific LOYO and maximum-null searches; BOTH performs both. Numerical factors move synchronously within each scenario."),
            wraplength=1020, justify="left"
        ).pack(anchor="w", pady=(3, 8))
        master = tk.BooleanVar(value=bool(current.get("enabled", False)))
        master_frame = ttk.Frame(outer); master_frame.pack(fill="x", pady=(0,8))
        ttk.Checkbutton(master_frame, text="Enable alternative sensitivity analyses (Analyses 2–5)", variable=master).pack(side="left")
        ttk.Label(master_frame, text="Analysis 1 is reference post-processing and can remain active independently.").pack(side="left", padx=(12,0))

        notebook = ttk.Notebook(outer); notebook.pack(fill="both", expand=True)
        # Analysis 1
        a1 = current.get("analysis_1_region_stability") or {}
        a1_page = ttk.Frame(notebook, padding=12); notebook.add(a1_page, text="A1 Region stability")
        a1_enabled = tk.BooleanVar(value=bool(a1.get("enabled", True)))
        a1_min = tk.StringVar(value="" if a1.get("minimum_valid_cell_proportion") is None else str(a1.get("minimum_valid_cell_proportion")))
        ttk.Checkbutton(a1_page, text="Run stability summaries across the reference robust region", variable=a1_enabled).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        ttk.Label(a1_page, text="Minimum valid-cell proportion, optional").grid(row=1,column=0,sticky="w",padx=(0,10))
        ttk.Entry(a1_page, textvariable=a1_min, width=14).grid(row=1,column=1,sticky="w")
        ttk.Label(a1_page, text="Leave blank to report all BSA summaries. A value in [0,1] flags BSAs with insufficient support across the robust region.", wraplength=760, justify="left").grid(row=2,column=0,columnspan=2,sticky="w",pady=(8,0))

        # Analysis 2
        support_ref = base["core"].get("support", {})
        a2_tab = RobustnessScenarioTab(notebook, "Analysis 2 · support sensitivity", current.get("analysis_2_support_sensitivity"), ROBUSTNESS_SUPPORT_FACTORS, (), support_ref)
        notebook.add(a2_tab, text="A2 Support")

        # Analysis 3
        a3 = current.get("analysis_3_bsa_specific_exposure") or {}
        a3_page = ttk.Frame(notebook, padding=12); notebook.add(a3_page, text="A3 BSA exposure")
        a3_enabled = tk.BooleanVar(value=bool(a3.get("enabled", False)))
        a3_label = tk.StringVar(value=a3.get("label", "BSA-specific exposure classification"))
        a3_mode = tk.StringVar(value=a3.get("comparison_mode", "FIXED_PARAMETER"))
        ttk.Checkbutton(a3_page, text="Enable BSA-specific yearly exposure classification", variable=a3_enabled).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        ttk.Label(a3_page, text="Label").grid(row=1,column=0,sticky="w",padx=(0,10),pady=3)
        ttk.Entry(a3_page, textvariable=a3_label, width=55).grid(row=1,column=1,sticky="w",pady=3)
        ttk.Label(a3_page, text="Comparison mode").grid(row=2,column=0,sticky="w",padx=(0,10),pady=3)
        ttk.Combobox(a3_page, textvariable=a3_mode, values=ROBUSTNESS_COMPARISON_MODES, state="readonly", width=20).grid(row=2,column=1,sticky="w",pady=3)
        ttk.Label(a3_page, text="This structural realization replaces the common territory-wide exposure calendar with BSA-year-specific percentile thresholds, local exposure labels, local baseline eligibility, and a synchronized bootstrap with BSA-specific draw validity.", wraplength=800, justify="left").grid(row=3,column=0,columnspan=2,sticky="w",pady=(12,0))

        # Analysis 4
        baseline_ref = base["core"].get("baseline", {})
        a4_tab = RobustnessScenarioTab(notebook, "Analysis 4 · baseline sensitivity", current.get("analysis_4_baseline_sensitivity"), ROBUSTNESS_BASELINE_NUMERICAL, ROBUSTNESS_BASELINE_BINARY, baseline_ref)
        notebook.add(a4_tab, text="A4 Baseline")

        # Analysis 5
        dependence_ref = base["core"].get("dependence", {})
        a5_tab = RobustnessScenarioTab(notebook, "Analysis 5 · dependence sensitivity", current.get("analysis_5_dependence_sensitivity"), ROBUSTNESS_DEPENDENCE_FACTORS, (), dependence_ref)
        notebook.add(a5_tab, text="A5 Dependence")

        buttons = ttk.Frame(outer); buttons.pack(fill="x", pady=(10,0))
        def apply():
            try:
                minimum_text = a1_min.get().strip()
                minimum = None if not minimum_text else float(minimum_text)
                rs = {
                    "enabled": bool(master.get()),
                    "analysis_1_region_stability": {
                        "enabled": bool(a1_enabled.get()),
                        "minimum_valid_cell_proportion": minimum,
                    },
                    "analysis_2_support_sensitivity": a2_tab.value(),
                    "analysis_3_bsa_specific_exposure": {
                        "enabled": bool(a3_enabled.get()),
                        "label": a3_label.get().strip() or "BSA-specific exposure classification",
                        "comparison_mode": a3_mode.get(),
                    },
                    "analysis_4_baseline_sensitivity": a4_tab.value(),
                    "analysis_5_dependence_sensitivity": a5_tab.value(),
                }
                updated = copy.deepcopy(base)
                updated["robustness"] = rs
                validate(updated)
                self.c = updated
                self.refresh()
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid robustness configuration", str(exc), parent=window)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right", padx=(8,0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="right")
        window.grab_set()

    def edit_confounder(self):
        if self.process is not None:
            messagebox.showinfo("Analysis running", "Confounder settings are frozen while an analysis is running.")
            return
        try:
            base = self.capture()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc)); return
        current = copy.deepcopy(base.get("confounder") or {})
        candidate = copy.deepcopy(current.get("candidate") or {})
        window = tk.Toplevel(self.root)
        window.title("Candidate-confounder adjustment")
        window.geometry("980x760")
        window.minsize(860, 650)
        window.transient(self.root)
        outer = ttk.Frame(window, padding=12); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Candidate-confounder adjustment", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(outer, text=("Configure one candidate confounder. The primary analysis compares the adjusted and unadjusted target hazard at the reference representative configuration. Complete adjusted reoptimization additionally requires at least four usable years."), wraplength=930, justify="left").pack(anchor="w", pady=(3,8))

        enabled = tk.BooleanVar(value=bool(current.get("enabled", False)))
        reopt = tk.BooleanVar(value=bool(current.get("request_complete_reoptimization", False)))
        flags = ttk.Frame(outer); flags.pack(fill="x", pady=(0,8))
        ttk.Checkbutton(flags, text="Enable candidate-confounder adjustment", variable=enabled).pack(side="left")
        ttk.Checkbutton(flags, text="Request complete adjusted reoptimization", variable=reopt).pack(side="left", padx=(18,0))

        notebook = ttk.Notebook(outer); notebook.pack(fill="both", expand=True)
        candidate_page = ttk.Frame(notebook, padding=12); notebook.add(candidate_page, text="Candidate")
        method_page = ttk.Frame(notebook, padding=12); notebook.add(method_page, text="Method")
        candidate_page.columnconfigure(1, weight=1); method_page.columnconfigure(1, weight=1)

        source_defaults = base.get("sources") or {}
        defaults = {
            "candidate_id": candidate.get("candidate_id", "candidate"),
            "display_name": candidate.get("display_name", "Candidate confounder"),
            "hazard_file": candidate.get("hazard_file", candidate.get("source_reference", "")),
            "units": candidate.get("units") or "",
            "hazard_date_field": candidate.get("hazard_date_field", candidate.get("source_date_field", source_defaults.get("hazard_date_field", "DATE_STR"))),
            "crosswalk_file": candidate.get("crosswalk_file", candidate.get("crosswalk_source_reference", source_defaults.get("crosswalk_file", ""))),
            "crosswalk_bsa_field": candidate.get("crosswalk_bsa_field", source_defaults.get("crosswalk_bsa_field", "bsa_id")),
            "grid_id_field": candidate.get("grid_id_field", candidate.get("crosswalk_grid_field", source_defaults.get("grid_id_field", "grid_id"))),
            "intersection_area_field": candidate.get("intersection_area_field", candidate.get("crosswalk_area_field", source_defaults.get("intersection_area_field", "intersection_area"))),
        }
        vars_ = {key: tk.StringVar(value=str(value)) for key,value in defaults.items()}
        rows = (
            ("candidate_id", "Candidate ID"),
            ("display_name", "Display name"),
            ("hazard_file", "Candidate hazard CSV"),
            ("units", "Units, optional"),
            ("hazard_date_field", "Hazard date field"),
            ("crosswalk_file", "BSA-grid crosswalk CSV"),
            ("crosswalk_bsa_field", "Crosswalk BSA field"),
            ("grid_id_field", "Crosswalk grid ID field"),
            ("intersection_area_field", "Intersection-area field"),
        )
        source_keys = {"hazard_file", "crosswalk_file"}
        for row,(key,label) in enumerate(rows):
            ttk.Label(candidate_page,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=3)
            widget = ttk.Entry(candidate_page,textvariable=vars_[key])
            widget.grid(row=row,column=1,sticky="ew",pady=3)
            if key in source_keys:
                def browse_candidate_source(k=key):
                    data_dir = Path(self.vars["data_dir"].get()).expanduser().resolve()
                    selected = filedialog.askopenfilename(parent=window, initialdir=data_dir, filetypes=[("CSV files","*.csv"),("All files","*.*")])
                    if selected:
                        vars_[k].set(relative_source_name(selected, data_dir))
                ttk.Button(candidate_page,text="Browse...",command=browse_candidate_source).grid(row=row,column=2,padx=(8,0))
        ttk.Label(
            candidate_page,
            text=("Candidate confounders use the same data model as the APHREH hazard field: "
                  "one date column followed by one column per source grid cell. The same "
                  "intersection-area projection and strict missing-contributor rule are used "
                  "to derive BSA and territory series. By default, reuse the project hazard "
                  "date and crosswalk field mappings."),
            wraplength=820, justify="left"
        ).grid(row=len(rows),column=0,columnspan=3,sticky="w",pady=(10,0))

        method_defaults = {
            "continuous_percentile_safeguard": current.get("continuous_percentile_safeguard",0.50),
            "residual_relevance_tolerance": current.get("residual_relevance_tolerance",0.10),
            "minimum_retained_support": current.get("minimum_retained_support",0.60),
            "minimum_years_primary": current.get("minimum_years_primary",3),
            "minimum_years_reoptimization": current.get("minimum_years_reoptimization",4),
        }
        method_vars = {key: tk.StringVar(value=str(value)) for key,value in method_defaults.items()}
        for row,(key,label) in enumerate((
            ("continuous_percentile_safeguard","Continuous percentile safeguard"),
            ("residual_relevance_tolerance","Residual relevance tolerance"),
            ("minimum_retained_support","Minimum retained support"),
            ("minimum_years_primary","Minimum years for primary adjustment"),
            ("minimum_years_reoptimization","Minimum years for reoptimization"),
        )):
            ttk.Label(method_page,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=4)
            ttk.Entry(method_page,textvariable=method_vars[key],width=18).grid(row=row,column=1,sticky="w",pady=4)
        ttk.Label(method_page,text="The relevance-caliper grid 0, 0.025, …, 1.000 and sensitivity tolerances 0.05, 0.10, 0.15, 0.20 are fixed by the methodological specification and are not edited here.", wraplength=800, justify="left").grid(row=6,column=0,columnspan=2,sticky="w",pady=(12,0))

        buttons = ttk.Frame(outer); buttons.pack(fill="x", pady=(10,0))
        def apply():
            try:
                cand = {
                    "candidate_id": vars_["candidate_id"].get().strip(),
                    "display_name": vars_["display_name"].get().strip(),
                    "hazard_file": vars_["hazard_file"].get().strip(),
                    "units": vars_["units"].get().strip() or None,
                    "hazard_date_field": vars_["hazard_date_field"].get().strip() or source_defaults.get("hazard_date_field", "DATE_STR"),
                    "crosswalk_file": vars_["crosswalk_file"].get().strip() or source_defaults.get("crosswalk_file", ""),
                    "crosswalk_bsa_field": vars_["crosswalk_bsa_field"].get().strip() or source_defaults.get("crosswalk_bsa_field", "bsa_id"),
                    "grid_id_field": vars_["grid_id_field"].get().strip() or source_defaults.get("grid_id_field", "grid_id"),
                    "intersection_area_field": vars_["intersection_area_field"].get().strip() or source_defaults.get("intersection_area_field", "intersection_area"),
                    "metadata": copy.deepcopy(candidate.get("metadata") or {}),
                }
                if not bool(enabled.get()) and not cand["hazard_file"] and not candidate:
                    cand = None
                updated = copy.deepcopy(base)
                updated["confounder"] = {
                    "enabled": bool(enabled.get()),
                    "request_complete_reoptimization": bool(reopt.get()),
                    "candidate": cand,
                    "continuous_percentile_safeguard": float(method_vars["continuous_percentile_safeguard"].get()),
                    "residual_relevance_tolerance": float(method_vars["residual_relevance_tolerance"].get()),
                    "minimum_retained_support": float(method_vars["minimum_retained_support"].get()),
                    "minimum_years_primary": int(method_vars["minimum_years_primary"].get()),
                    "minimum_years_reoptimization": int(method_vars["minimum_years_reoptimization"].get()),
                }
                validate(updated)
                self.c = updated
                self.refresh()
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid confounder configuration", str(exc), parent=window)

        ttk.Button(buttons,text="Cancel",command=window.destroy).pack(side="right",padx=(8,0))
        ttk.Button(buttons,text="Apply",command=apply).pack(side="right")
        window.grab_set()

    def edit_effect_classes(self):
        if self.process is not None:
            messagebox.showinfo("Analysis running","Calibration effect classes are frozen while an analysis is running.")
            return
        try:
            # Work from a complete snapshot of the current form, not from the
            # last committed self.c object.  This prevents changing calibration
            # intervals from reverting unrelated unsaved GUI fields.
            base = self.capture()
            current = effect_class_bounds(base["calibration"])
        except Exception as exc:
            messagebox.showerror("Effect-class configuration",str(exc))
            return
        window = tk.Toplevel(self.root)
        window.title("Calibration effect-magnitude percentile classes")
        window.geometry("780x430")
        window.minsize(700,390)
        window.transient(self.root)
        frame = ttk.Frame(window,padding=14);frame.pack(fill="both",expand=True)
        ttk.Label(frame,text="Edit percentile positions on the 0..1 scale (0.25 = 25th percentile).",
                  font=("Segoe UI",11,"bold")).grid(row=0,column=0,columnspan=4,sticky="w",pady=(0,5))
        ttk.Label(frame,text="The named classes must preserve protective / negligible / harmful direction and may touch but not overlap. Exact null remains delta = 0.",
                  wraplength=735,justify="left").grid(row=1,column=0,columnspan=4,sticky="w",pady=(0,12))
        ttk.Label(frame,text="Class").grid(row=2,column=0,sticky="w",padx=(0,20))
        ttk.Label(frame,text="Lower percentile").grid(row=2,column=1,sticky="w")
        ttk.Label(frame,text="Upper percentile").grid(row=2,column=2,sticky="w")
        ttk.Label(frame,text="Interpretation").grid(row=2,column=3,sticky="w",padx=(15,0))
        variables = {}
        interpretation = {
            "strong_protective":"strong decrease relative to median",
            "moderate_protective":"moderate decrease relative to median",
            "negligible":"near-median alteration",
            "moderate_harmful":"moderate increase relative to median",
            "strong_harmful":"strong increase relative to median",
        }
        for row,name in enumerate(EFFECT_CLASS_ORDER,start=3):
            lo,hi = current[name]
            vlo,vhi = tk.StringVar(value=f"{lo:g}"),tk.StringVar(value=f"{hi:g}")
            variables[name] = (vlo,vhi)
            ttk.Label(frame,text=EFFECT_CLASS_LABELS[name]).grid(row=row,column=0,sticky="w",pady=4)
            ttk.Entry(frame,textvariable=vlo,width=14).grid(row=row,column=1,sticky="w",pady=4)
            ttk.Entry(frame,textvariable=vhi,width=14).grid(row=row,column=2,sticky="w",pady=4)
            ttk.Label(frame,text=interpretation[name]).grid(row=row,column=3,sticky="w",padx=(15,0),pady=4)
        amp = self.c["calibration"].get("threshold_lag_amplitude_class")
        ttk.Label(frame,text=f"Threshold-lag calibration amplitude class: {EFFECT_CLASS_LABELS.get(amp,amp)}",
                  wraplength=735).grid(row=8,column=0,columnspan=4,sticky="w",pady=(12,4))
        ttk.Label(frame,text="These values are saved in calibration.effect_classes in project_config.json and resolved_config.json.",
                  wraplength=735).grid(row=9,column=0,columnspan=4,sticky="w",pady=(0,10))
        buttons = ttk.Frame(frame);buttons.grid(row=10,column=0,columnspan=4,sticky="e",pady=(6,0))
        def apply():
            try:
                bounds = {name:(float(variables[name][0].get().strip()),float(variables[name][1].get().strip()))
                          for name in EFFECT_CLASS_ORDER}
                updated = copy.deepcopy(base)
                set_effect_class_bounds(updated["calibration"],bounds)
                validate(updated)
                self.c = updated
                self.refresh()
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid effect-class percentiles",str(exc),parent=window)
        ttk.Button(buttons,text="Cancel",command=window.destroy).pack(side="right",padx=(8,0))
        ttk.Button(buttons,text="Apply",command=apply).pack(side="right")
        window.grab_set()

    def append(self,text):
        self.console.configure(state="normal")
        self.console.insert("end",text)
        lines = int(self.console.index("end-1c").split(".")[0])
        if lines>1000:
            self.console.delete("1.0",f"{lines-1000}.0")
        self.console.see("end");self.console.configure(state="disabled")

    def start_robustness(self):
        if self.process is not None:
            return
        try:
            current = self.capture()
            initialized = initialize_robustness_defaults(current)
            if initialized != current:
                self.c = initialized
                self.refresh()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
            return
        self.phase.set("robustness")
        self.start()

    def start_confounder(self):
        if self.process is not None:
            return
        try:
            current = self.capture()
            ca = current.get("confounder") or {}
            if not ca.get("candidate"):
                raise ValueError("Configure a candidate confounder before running the confounder workflow")
            if not ca.get("enabled"):
                current["confounder"]["enabled"] = True
                validate(current)
                self.c = current
                self.refresh()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
            return
        self.phase.set("confounder")
        self.start()

    def start(self):
        if self.process is not None:
            return
        try:
            c = self.capture()
            phase = self.phase.get()
            # Reviewer-oriented sensitivity scenarios are opt-in for a general
            # release.  The explicit robustness workflow initializes its
            # standard scenario set; ALL honors only what the user configured.
            if phase == "robustness":
                c = initialize_robustness_defaults(c)
                validate(c)
            # Commit the complete current GUI state before refresh/save so that
            # no independent field can fall back to an older in-memory value.
            self.c = copy.deepcopy(c)
            self.refresh()
            out = Path(c["output_dir"]);out.mkdir(parents=True,exist_ok=True)
            path = out/"project_config.json"
            atomic_json(path,c)
            log = out/"gui_console.log"
            self.console_writer = log.open("w",encoding="utf-8")
            command = [sys.executable,"-u",str(ROOT/"main.py"),"--headless","--config",str(path),"--phase",phase]
            flags = subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
            self.process = subprocess.Popen(command,cwd=ROOT,stdout=self.console_writer,stderr=subprocess.STDOUT,creationflags=flags)
            self.console_reader = log.open("r",encoding="utf-8",errors="replace")
            self.active_output = out
            self.run_button.configure(state="disabled");self.rs_run_button.configure(state="disabled");self.ca_run_button.configure(state="disabled");self.cancel_button.configure(state="normal")
            self.status.set("Running; committed cells are saved on disk.")
            self.append("\n--- Starting project workflow ---\n")
            self.root.after(200,self.poll)
        except Exception as exc:
            messagebox.showerror("Cannot start analysis",str(exc))

    def poll(self):
        if self.process is None:
            return
        chunk = self.console_reader.read(65536)
        if chunk:
            self.append(chunk)
        code = self.process.poll()
        if code is None:
            self.root.after(250,self.poll);return
        rest = self.console_reader.read(65536)
        if rest:
            self.append(rest)
        self.console_reader.close();self.console_writer.close()
        self.process = None
        self.run_button.configure(state="normal");self.rs_run_button.configure(state="normal");self.ca_run_button.configure(state="normal");self.cancel_button.configure(state="disabled")
        self.status.set("Completed." if code==0 else "Stopped or failed; inspect the console and error.log.")
        if self.close_after:
            self.root.destroy()

    def cancel(self):
        if self.process is not None:
            (self.active_output/"CANCEL").touch()
            self.status.set("Cancellation requested; preserving completed cells.")

    def open_results_workspace(self):
        project = Path(self.vars["output_dir"].get()).expanduser().resolve()
        try:
            ResultsWorkspace(self.root,project)
        except Exception as exc:
            messagebox.showinfo("Results unavailable",str(exc))

    def open_report(self):
        path = Path(self.vars["output_dir"].get())/"results/report.html"
        if path.exists():
            webbrowser.open(path.resolve().as_uri())
        else:
            messagebox.showinfo("Results unavailable","No completed report in this project folder yet.")

    def close(self):
        if self.process is not None:
            if messagebox.askyesno("Analysis running","Cancel this analysis and close after the current safe checkpoint?"):
                self.close_after = True;self.cancel()
        else:
            self.root.destroy()


def main() -> int:
    try:
        root = tk.Tk()
        Application(root)
        root.mainloop()
        return 0
    except tk.TclError as exc:
        print(f"Desktop display unavailable: {exc}\nUse python main.py --headless --config configs/default.json --input DATA_DIR --output PROJECT_DIR for headless execution.",file=sys.stderr)
        return 1

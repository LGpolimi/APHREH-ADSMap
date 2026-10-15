"""Executable project workflow, immutable provenance, checkpoints, and CSV exports."""
from __future__ import annotations
import hashlib
import html
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
import psutil
from . import __version__
from .config import validate
from .data import build_canonical,Data,source_path
from .storage import Store,atomic_json,sha256_file,dumps
from .engine import FixedParametersEngine,SurfaceRunner
from .optimization import Optimizer
from .calibration import Calibration
from .confounder import (
    ConfounderAdjustmentSettings, ConfounderCandidateDefinition,
    ConfounderPersistence, ConfounderProfilingRunner, ConfounderCalibrationRunner, ConfounderPointEstimator,
    ConfounderAdjustedInferenceRunner, ConfounderAdjustedSurfaceRunner,
    ConfounderAdjustedOptimizationRunner,
    confounder_code_identity, confounder_identity, confounder_run_id,
    preprocess_candidate,
)
from .robustness import (
    RealizationState, RobustnessPersistence,
    RegionStabilityAnalyzer, export_region_stability,
    build_robustness_plan, SupportSensitivityRunner, ExposureSensitivityRunner, BaselineSensitivityRunner, DependenceSensitivityRunner, FixedPointComparator, ReoptimizationRunner, FingerprintSurfaceComparator,
)


class ProjectLock:
    """OS lock is released even after a crash; stale lockfiles do not block resume."""
    def __init__(self,path: Path):
        self.path = path
    def __enter__(self):
        self.file = self.path.open("a+b")
        if self.file.tell()==0:
            self.file.write(b"0");self.file.flush()
        self.file.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError) as exc:
            self.file.close()
            raise RuntimeError("Another process is already using this project folder") from exc
        return self
    def __exit__(self,*args):
        self.file.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    if os.name!="nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/(1024**2 if sys.platform=="darwin" else 1024)
    info = psutil.Process().memory_info()
    return getattr(info,"peak_wset",info.rss)/1024**2


def code_identity() -> dict:
    base = Path(__file__).resolve().parent.parent
    paths = sorted((base/"aphreh_adsmap").rglob("*.py"))
    return {str(p.relative_to(base)):sha256_file(p) for p in paths}


def source_identity(c: dict) -> dict:
    fields = c["sources"]
    result = {}
    for k in ("hazard_file","outcome_file","population_file","crosswalk_file","geometry_file","holiday_calendar_file"):
        if k not in fields:
            continue
        p = source_path(c["data_dir"],fields[k])
        if not p.is_file():
            if k in ("geometry_file","holiday_calendar_file"):
                continue
            raise FileNotFoundError(f"Missing source: {p}")
        result[k] = {"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)}
    return result


REFERENCE_IDENTITY_SCHEMA = "APHREH_REFERENCE_IDENTITY_V2"
ROBUSTNESS_IDENTITY_SCHEMA = "APHREH_ROBUSTNESS_PLAN_IDENTITY_V1"
CONFOUNDER_MANIFEST_SCHEMA = "APHREH_CONFOUNDER_MANIFEST_CA8_V1"

# Exact aggregate hashes of the two server-side baselines that are explicitly
# approved for in-place RS migration.  This is intentionally narrow: a project
# produced by an arbitrary build with the same version string is not accepted.
KNOWN_LEGACY_CODE_AGGREGATES = {
    "2.1.9": "282aa8b7fc8b0339116619c094a00722f87cf4849af3521fd3c22e261acb41a1",
    "2.3.0": "7e90c066ab5f2aade0d39788384d43f9ae1d8ed7ba0233678d15ee489594a646",
}

# Exact downstream-compatible builds that may attach newer confounder stages
# without recomputing an already-completed unadjusted reference analysis.
# Admission remains hash-pinned and additionally checks sources and scientific
# reference configuration.
KNOWN_COMPATIBLE_DOWNSTREAM_CODE_AGGREGATES = {
    "2.3.8": "d39265e8f1742333defa34f7710b789bccf5260237fd3ad96bec6f835cde48c6",
    "2.3.10": "39b32f5a129d031fb7cbc069cce001463afcd7bed6a9a90b13ab15a8699580cb",
    "2.3.11": "13f9980cf30a387cea4fa91f30da5ac26f69b57d5de797b6c75bc566f3709282",
    "2.3.12": "0305d94e75f6489b0d3c7c606843a70aa9cde7cb0eb2bb24ce8e26326a5ada6c",
    "2.3.13": "648b2b093f2ccb795b544a45366a16cfb6e8e6c818a3758b7f9ea2b1457b10db",
    "2.3.14": "e4a2ad5737894e9e2b1cb38dc43e756b38cca12dde7b1aeedd4e9e910d73a32d",
    "2.3.15": "762b2a3ff3255bd0fbe136550c17964099305a9985c1be0e4e125f80a63c2de2",
}

# Exact uploaded 2.3.16 build validated against the accelerated kernels.
# Do not extend this allowlist based only on a version label.
KNOWN_EQUIVALENT_PERFORMANCE_BUILDS = {
    "2.3.16": "15ff34306c70d7e0b1e9019c418740232ae28e78fe45c4dd77f5b48c7c581536",
    "2.3.17": "9f55246277645401c4281cda46331d4abdf2f5a17180911cedfb5df5543e6fa9",
    "2.3.18": "d8b5f9995d07d052173a1e64e4c2fa8a42be56be1e1a31a014c93aec60eacd7b",
}

# Exact Windows build used for 2.3.19 calibration. 2.3.20 changes only the
# synthetic-calibration BSA-year exclusion propagation inside the estimator;
# ordinary observed reference calculations remain definitionally unchanged.
# A 2.3.19 project may therefore reuse its completed observed reference while
# calibration checkpoints are explicitly invalidated and recomputed.
KNOWN_CALIBRATION_FIX_PREDECESSORS = {
    "2.3.19": "b562fd1b82368c50b60e3acaac1d5ded996648a4689dcd4e7f641c08b9357fd8",
}

# Exact 2.3.20 build validated from the user's corrected calibration run.
# 2.3.21 changes only synthetic threshold/lag optimization admissibility and
# recovery-validity reporting. The observed reference and RC magnitude
# calibration remain scientifically unchanged, so those checkpoints may be
# retained while TL calibration records are recomputed.
KNOWN_CALIBRATION_ADMISSIBILITY_FIX_PREDECESSORS = {
    "2.3.20": "2c9fd674f1eb9cb045dd302be5ea60b2dd32d7c9d54683eae9c6273eb295e5bf",
}

_REFERENCE_CODE_FILES = (
    "config.py", "data.py", "engine.py", "hashing.py", "numerics.py",
    "optimization.py", "randomness.py", "storage.py", "support.py", "performance.py",
)
_VOLATILE_REFERENCE_KEYS = {"output_dir", "data_dir", "config_path", "notes", "profile", "robustness", "confounder"}


def _aggregate_hash(mapping: dict) -> str:
    return hashlib.sha256(dumps(mapping).encode()).hexdigest()


def _normalized_code_hashes(mapping: dict | None) -> dict:
    """Normalize manifest code paths across Windows and POSIX separators."""
    if not isinstance(mapping, dict):
        return {}
    return {str(k).replace("\\", "/"): v for k,v in mapping.items()}


def reference_scientific_configuration(c: dict) -> dict:
    """Configuration that defines the persisted reference analysis.

    Robustness and candidate-confounder execution settings are deliberately
    excluded. A loaded project may therefore attach either downstream plan
    without redefining its completed unadjusted reference surface.
    """
    return {k:v for k,v in c.items() if k not in _VOLATILE_REFERENCE_KEYS}


def reference_code_identity() -> dict:
    base = Path(__file__).resolve().parent
    result = {}
    for name in _REFERENCE_CODE_FILES:
        path = base/name
        if path.is_file():
            result[f"aphreh_adsmap/{name}"] = sha256_file(path)
    return result


def robustness_code_identity() -> dict:
    base = Path(__file__).resolve().parent.parent
    paths = sorted((base/"aphreh_adsmap"/"robustness").glob("*.py"))
    return {str(p.relative_to(base)):sha256_file(p) for p in paths}


def reference_identity(c: dict, sources: dict) -> str:
    payload = {
        "schema": REFERENCE_IDENTITY_SCHEMA,
        "configuration": reference_scientific_configuration(c),
        "sources": sources,
        "reference_code": reference_code_identity(),
    }
    return hashlib.sha256(dumps(payload).encode()).hexdigest()


def robustness_identity(c: dict, ref_identity: str, plan) -> str:
    payload = {
        "schema": ROBUSTNESS_IDENTITY_SCHEMA,
        "reference_identity": ref_identity,
        "robustness_configuration": c.get("robustness") or {},
        "plan_id": plan.plan_id,
        "robustness_code": robustness_code_identity(),
    }
    return hashlib.sha256(dumps(payload).encode()).hexdigest()


def _compatible_performance_upgrade(old: dict, c: dict, sources: dict, root: Path) -> dict:
    """Reuse exact predecessor checkpoints after a result-equivalent speedup.

    A matching version alone is insufficient. Verify the complete predecessor
    code map, scientific configuration, source contents and reference identity.
    Unlike downstream-only migrations, interrupted calibration/reference cells
    can resume because the estimator and random streams remain equivalent.
    """
    version = str(old.get("version") or "")
    expected = KNOWN_EQUIVALENT_PERFORMANCE_BUILDS.get(version)
    code = old.get("source_code_hashes")
    actual = _aggregate_hash(code) if isinstance(code, dict) else None
    if expected is None or actual != expected:
        raise ValueError("This predecessor code build is not approved for performance-equivalent checkpoint reuse.")
    if dumps(old.get("source_files") or {}) != dumps(sources):
        raise ValueError("The loaded project source identities do not match the current configuration.")
    previous_config = old.get("configuration")
    if not isinstance(previous_config, dict) or dumps(reference_scientific_configuration(previous_config)) != dumps(reference_scientific_configuration(c)):
        raise ValueError("The loaded project's reference scientific settings differ from the current configuration.")
    normalized_code = _normalized_code_hashes(code)
    previous_reference_code = {k: normalized_code[k] for k in old.get("reference_code_hashes", {}) if k in normalized_code}
    if not previous_reference_code or previous_reference_code != old.get("reference_code_hashes"):
        raise ValueError("The predecessor reference code identity is inconsistent.")
    previous_identity = _aggregate_hash({
        "schema": REFERENCE_IDENTITY_SCHEMA,
        "configuration": reference_scientific_configuration(previous_config),
        "sources": sources,
        "reference_code": previous_reference_code,
    })
    if previous_identity != old.get("reference_identity"):
        raise ValueError("The predecessor reference identity is inconsistent.")
    if not (root/"canonical"/"metadata.json").is_file() or not (root/"results.sqlite").is_file():
        raise ValueError("The loaded project is missing canonical data or results.sqlite.")
    return {
        "migrated_utc": now(), "origin_version": version,
        "origin_identity": previous_identity, "origin_code_aggregate": actual,
        "target_version": __version__,
        "basis": "exact approved predecessor code, identical inputs/settings, tested numerical and RNG equivalence",
        "scope": "PERFORMANCE_EQUIVALENT_CHECKPOINT_REUSE",
    }


def _compatible_calibration_fix_upgrade(old: dict, c: dict, sources: dict, root: Path) -> dict:
    """Reuse an exact 2.3.19 observed reference while invalidating calibration.

    The 2.3.20 estimator change is dormant unless a calibration BSA-year
    eligibility sidecar is supplied. Therefore the ordinary observed reference
    is unchanged, while all 2.3.19 synthetic-calibration checkpoints must be
    discarded because they were affected by the exclusion-propagation defect.
    """
    version = str(old.get("version") or "")
    expected = KNOWN_CALIBRATION_FIX_PREDECESSORS.get(version)
    code = old.get("source_code_hashes")
    actual = _aggregate_hash(code) if isinstance(code, dict) else None
    if expected is None or actual != expected:
        raise ValueError("This predecessor code build is not approved for calibration-fix reference reuse.")
    if dumps(old.get("source_files") or {}) != dumps(sources):
        raise ValueError("The loaded project source identities do not match the current configuration.")
    previous_config = old.get("configuration")
    if not isinstance(previous_config, dict) or dumps(reference_scientific_configuration(previous_config)) != dumps(reference_scientific_configuration(c)):
        raise ValueError("The loaded project's reference scientific settings differ from the current configuration.")
    normalized_code = _normalized_code_hashes(code)
    previous_reference_code = {k: normalized_code[k] for k in old.get("reference_code_hashes", {}) if k in normalized_code}
    if not previous_reference_code or previous_reference_code != old.get("reference_code_hashes"):
        raise ValueError("The predecessor reference code identity is inconsistent.")
    previous_identity = _aggregate_hash({
        "schema": REFERENCE_IDENTITY_SCHEMA,
        "configuration": reference_scientific_configuration(previous_config),
        "sources": sources,
        "reference_code": previous_reference_code,
    })
    if previous_identity != old.get("reference_identity"):
        raise ValueError("The predecessor reference identity is inconsistent.")
    if not (root/"canonical"/"metadata.json").is_file() or not (root/"results.sqlite").is_file():
        raise ValueError("The loaded project is missing canonical data or results.sqlite.")
    return {
        "migrated_utc": now(), "origin_version": version,
        "origin_identity": previous_identity, "origin_code_aggregate": actual,
        "target_version": __version__,
        "basis": "exact 2.3.19 predecessor, identical inputs/settings, observed reference unchanged; calibration invalidated",
        "scope": "CALIBRATION_FIX_REFERENCE_REUSE",
        "reset_calibration": True,
    }


def _compatible_calibration_admissibility_upgrade(old: dict, c: dict, sources: dict, root: Path) -> dict:
    """Reuse exact 2.3.20 reference and RC calibration, recompute TL calibration.

    The 2.3.21 change affects only synthetic threshold/lag support coverage and
    the derived recovery-validity layer. Full-territory observed analysis and
    RC magnitude calibration are unchanged. Admission is hash-pinned and also
    verifies sources, scientific settings, and predecessor reference identity.
    """
    version = str(old.get("version") or "")
    expected = KNOWN_CALIBRATION_ADMISSIBILITY_FIX_PREDECESSORS.get(version)
    code = old.get("source_code_hashes")
    actual = _aggregate_hash(code) if isinstance(code, dict) else None
    if expected is None or actual != expected:
        raise ValueError("This predecessor code build is not approved for TL-calibration admissibility-fix reuse.")
    if dumps(old.get("source_files") or {}) != dumps(sources):
        raise ValueError("The loaded project source identities do not match the current configuration.")
    previous_config = old.get("configuration")
    if not isinstance(previous_config, dict) or dumps(reference_scientific_configuration(previous_config)) != dumps(reference_scientific_configuration(c)):
        raise ValueError("The loaded project's reference scientific settings differ from the current configuration.")
    normalized_code = _normalized_code_hashes(code)
    previous_reference_code = {k: normalized_code[k] for k in old.get("reference_code_hashes", {}) if k in normalized_code}
    if not previous_reference_code or previous_reference_code != old.get("reference_code_hashes"):
        raise ValueError("The predecessor reference code identity is inconsistent.")
    previous_identity = _aggregate_hash({
        "schema": REFERENCE_IDENTITY_SCHEMA,
        "configuration": reference_scientific_configuration(previous_config),
        "sources": sources,
        "reference_code": previous_reference_code,
    })
    if previous_identity != old.get("reference_identity"):
        raise ValueError("The predecessor reference identity is inconsistent.")
    if not (root/"canonical"/"metadata.json").is_file() or not (root/"results.sqlite").is_file():
        raise ValueError("The loaded project is missing canonical data or results.sqlite.")
    return {
        "migrated_utc": now(), "origin_version": version,
        "origin_identity": previous_identity, "origin_code_aggregate": actual,
        "target_version": __version__,
        "basis": "exact 2.3.20 predecessor, identical inputs/settings, observed reference and RC calibration unchanged; TL calibration invalidated",
        "scope": "TL_CALIBRATION_ADMISSIBILITY_FIX_REFERENCE_AND_RC_REUSE",
        "reset_tl_calibration": True,
    }


def _reset_tl_calibration_checkpoints(store: Store) -> int:
    """Delete only synthetic threshold/lag calibration products.

    RC magnitude calibration and the observed reference are preserved. This is
    intentionally narrower than the 2.3.19 to 2.3.20 migration.
    """
    before = store.db.total_changes
    store.db.execute("DELETE FROM records WHERE scope LIKE 'CAL_TL_%'")
    store.db.execute("DELETE FROM records WHERE kind='calibration_instance' AND scope='TL'")
    store.db.execute("DELETE FROM records WHERE kind IN ('calibration_recovery','calibration_mean_surface')")
    store.db.execute("DELETE FROM records WHERE kind='workflow' AND scope='CALIBRATION'")
    store.commit()
    return store.db.total_changes - before


def _reset_calibration_checkpoints(store: Store) -> int:
    """Delete only calibration-derived records; preserve observed reference/RS/CA."""
    before = store.db.total_changes
    store.db.execute(
        """DELETE FROM records WHERE
           scope LIKE 'CAL_RC_%' OR scope LIKE 'CAL_TL_%' OR
           scope LIKE 'CAL_EXACT_NULL_SURFACE_%'"""
    )
    calibration_kinds = (
        "calibration_instance", "calibration_distribution",
        "calibration_operating_characteristics", "calibration_ordering",
        "calibration_recovery", "calibration_mean_surface",
        "synthetic_support", "synthetic_generation",
    )
    placeholders = ",".join("?" for _ in calibration_kinds)
    store.db.execute(f"DELETE FROM records WHERE kind IN ({placeholders})", calibration_kinds)
    store.db.execute("DELETE FROM records WHERE kind='workflow' AND scope='CALIBRATION'")
    store.commit()
    return store.db.total_changes - before


def _remove_calibration_exports(root: Path) -> int:
    """Remove stale exported calibration summaries before corrected recomputation."""
    out = Path(root)/"results"
    if not out.is_dir():
        return 0
    removed = 0
    for pattern in ("calibration_*.csv", "calibration_*.json", "calibration_*.md", "synthetic_*.csv"):
        for path in out.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _compatible_downstream_reference_upgrade(old: dict, c: dict, sources: dict, root: Path) -> dict:
    """Accept an explicitly hash-pinned build for downstream confounder attachment.

    This is deliberately narrow. Reference scientific settings, exact source
    identities, completed reference/optimization checkpoints, and the aggregate
    package hash must all match an approved predecessor build.
    """
    version = str(old.get("version") or "")
    expected = KNOWN_COMPATIBLE_DOWNSTREAM_CODE_AGGREGATES.get(version)
    actual_hashes = old.get("source_code_hashes")
    actual = _aggregate_hash(actual_hashes) if isinstance(actual_hashes, dict) else None
    if expected is None or actual != expected:
        raise ValueError(
            "This project's completed reference was produced by a code build that is not "
            "approved for downstream confounder reuse. Use the matching predecessor project or rerun "
            "the reference analysis with this build."
        )
    if dumps(old.get("source_files") or {}) != dumps(sources):
        raise ValueError("The loaded project source identities do not match the current configuration.")
    old_configuration = old.get("configuration")
    if not isinstance(old_configuration, dict) or dumps(reference_scientific_configuration(old_configuration)) != dumps(reference_scientific_configuration(c)):
        raise ValueError(
            "The loaded project's reference scientific settings differ from the current configuration."
        )
    if not (root/"canonical"/"metadata.json").is_file() or not (root/"results.sqlite").is_file():
        raise ValueError("The loaded project is missing canonical data or results.sqlite.")
    probe = Store(root/"results.sqlite")
    try:
        completed = probe.get("workflow", "REFERENCE", "result")
        optimization = probe.get("optimization", "REFERENCE", "result")
        if not completed or completed.get("status") != "COMPLETED" or optimization is None:
            raise ValueError("The loaded project does not contain a completed reference analysis and optimization.")
    finally:
        probe.close()
    return {
        "migrated_utc": now(),
        "origin_version": version,
        "origin_identity": old.get("reference_identity") or old.get("identity"),
        "origin_code_aggregate": actual,
        "target_version": __version__,
        "basis": "exact approved predecessor build, identical sources, identical reference scientific settings, completed reference checkpoint",
        "scope": "DOWNSTREAM_CONFOUNDER_REUSE",
    }


def _legacy_reference_migration(old: dict, c: dict, sources: dict, root: Path) -> dict:
    """Validate an explicitly approved noRS/RSinit project for RS-only reuse."""
    version = str(old.get("version") or "")
    expected = KNOWN_LEGACY_CODE_AGGREGATES.get(version)
    actual_hashes = old.get("source_code_hashes")
    actual = _aggregate_hash(actual_hashes) if isinstance(actual_hashes, dict) else None
    if expected is None or actual != expected:
        raise ValueError(
            "This legacy project was not produced by an approved compatible noRS/RSinit build. "
            "Its completed reference results cannot be reused in place."
        )
    if dumps(old.get("source_files") or {}) != dumps(sources):
        raise ValueError(
            "The loaded project source identities do not match the current configuration; "
            "robustness cannot be attached to this reference analysis."
        )
    old_configuration = old.get("configuration")
    if not isinstance(old_configuration, dict):
        raise ValueError("The legacy manifest does not contain the reference scientific configuration.")
    if dumps(reference_scientific_configuration(old_configuration)) != dumps(reference_scientific_configuration(c)):
        raise ValueError(
            "The loaded project reference scientific settings differ from the current configuration. "
            "Only robustness settings may change when reusing a completed reference analysis."
        )
    if not (root/"canonical"/"metadata.json").is_file() or not (root/"results.sqlite").is_file():
        raise ValueError("The loaded legacy project is missing canonical data or results.sqlite.")
    probe = Store(root/"results.sqlite")
    try:
        completed = probe.get("workflow", "REFERENCE", "result")
        optimization = probe.get("optimization", "REFERENCE", "result")
        if not completed or completed.get("status") != "COMPLETED" or optimization is None:
            raise ValueError("The loaded legacy project does not contain a completed reference analysis and optimization.")
    finally:
        probe.close()
    return {
        "migrated_utc": now(),
        "origin_version": version,
        "origin_identity": old.get("identity"),
        "origin_code_aggregate": actual,
        "target_version": __version__,
        "basis": "exact approved build, identical sources, identical reference scientific settings, completed reference checkpoint",
    }


def _robustness_infrastructure(c: dict, plan, reference_id: str, plan_identity: str) -> dict:
    return {
        "implementation_stage":"FINGERPRINT_SURFACE_COMPARISON",
        "alternative_realization_execution_enabled":bool(c["robustness"]["enabled"] and (c["robustness"]["analysis_2_support_sensitivity"]["enabled"] or c["robustness"]["analysis_3_bsa_specific_exposure"]["enabled"] or c["robustness"]["analysis_4_baseline_sensitivity"]["enabled"] or c["robustness"]["analysis_5_dependence_sensitivity"]["enabled"])),
        "implemented_analyses":["ANALYSIS_1_ROBUST_REGION_STABILITY","ANALYSIS_2_SUPPORT_SENSITIVITY","ANALYSIS_3_BSA_SPECIFIC_EXPOSURE","ANALYSIS_4_BASELINE_SENSITIVITY","ANALYSIS_5_DEPENDENCE_SENSITIVITY","FIXED_PARAMETER_COMPARISON","REALIZATION_SPECIFIC_LOYO","REALIZATION_SPECIFIC_NULL_SEARCH","COMPLETE_REOPTIMIZATION","FINGERPRINT_SURFACE_COMPARISON"],
        "explicit_realization_loyo_null_components":True,
        "analysis_1_enabled":c["robustness"]["analysis_1_region_stability"]["enabled"],
        "analysis_2_enabled":c["robustness"]["analysis_2_support_sensitivity"]["enabled"],
        "analysis_2_realization_count":len([r for r in plan.realizations if r.analysis_id=="ANALYSIS_2_SUPPORT_SENSITIVITY"]),
        "analysis_3_enabled":c["robustness"]["analysis_3_bsa_specific_exposure"]["enabled"],
        "analysis_3_realization_count":len([r for r in plan.realizations if r.analysis_id=="ANALYSIS_3_BSA_SPECIFIC_EXPOSURE"]),
        "analysis_4_enabled":c["robustness"]["analysis_4_baseline_sensitivity"]["enabled"],
        "analysis_4_realization_count":len([r for r in plan.realizations if r.analysis_id=="ANALYSIS_4_BASELINE_SENSITIVITY"]),
        "analysis_5_enabled":c["robustness"]["analysis_5_dependence_sensitivity"]["enabled"],
        "analysis_5_realization_count":len([r for r in plan.realizations if r.analysis_id=="ANALYSIS_5_DEPENDENCE_SENSITIVITY"]),
        "reference_realization_id":reference_id,
        "plan_id":plan.plan_id,
        "plan_identity":plan_identity,
        "plan_file":"results/robustness_plan.json",
        "analysis_1_summary_file":"results/reference_region_stability.json",
        "analysis_2_summary_file":"results/robustness_analysis2_summary.json",
        "analysis_3_summary_file":"results/robustness_analysis3_summary.json",
        "analysis_4_summary_file":"results/robustness_analysis4_summary.json",
        "analysis_5_summary_file":"results/robustness_analysis5_summary.json",
        "fixed_point_comparison_summary_file":"results/robustness_fixed_point_summary.json",
        "realization_loyo_summary_file":"results/robustness_realization_loyo_summary.json",
        "realization_null_summary_file":"results/robustness_realization_null_summary.json",
        "reoptimization_summary_file":"results/robustness_reoptimization_summary.json",
        "fingerprint_surface_comparison_summary_file":"results/robustness_fingerprint_surface_summary.json",
    }


def _run_robustness(c: dict, d: Data, s: Store, root: Path, log):
    reference = s.get("workflow", "REFERENCE", "result")
    optimization = s.get("optimization", "REFERENCE", "result")
    if not reference or reference.get("status") != "COMPLETED" or optimization is None:
        raise ValueError(
            "Robustness/sensitivity requires a completed reference analysis in this project. "
            "Run the reference workflow first."
        )

    plan = build_robustness_plan(c)
    persistence = RobustnessPersistence(s)
    persistence.register_plan(plan)
    reference_id = plan.reference.realization_id
    persistence.set_realization_state(reference_id, RealizationState.COMPLETED)
    atomic_json(root/"results"/"robustness_plan.json", plan.to_dict())

    a1 = c["robustness"]["analysis_1_region_stability"]
    if a1["enabled"]:
        log("RS Analysis 1: stability across the complete reference robust region")
        rs1 = RegionStabilityAnalyzer(
            s, persistence, reference_id,
            minimum_valid_cell_proportion=a1["minimum_valid_cell_proportion"],
        ).run(optimization)
        export_region_stability(rs1, root/"results")
        log(f"RS Analysis 1 complete: {rs1['status']}; robust-region cells={rs1['robust_region_cell_count']}")

    a2 = c["robustness"]["analysis_2_support_sensitivity"]
    if c["robustness"]["enabled"] and a2["enabled"]:
        log("RS Analysis 2: sparse-data and effective-support sensitivity")
        rs2 = SupportSensitivityRunner(d, s, persistence, root, log).run(c, plan)
        log(f"RS Analysis 2 complete: realizations={rs2['realization_count']}")
    a3 = c["robustness"]["analysis_3_bsa_specific_exposure"]
    if c["robustness"]["enabled"] and a3["enabled"]:
        log("RS Analysis 3: BSA-specific exposure classification")
        rs3 = ExposureSensitivityRunner(d, s, persistence, root, log).run(c, plan)
        log(f"RS Analysis 3 complete: realizations={rs3['realization_count']}")
    a4 = c["robustness"]["analysis_4_baseline_sensitivity"]
    if c["robustness"]["enabled"] and a4["enabled"]:
        log("RS Analysis 4: baseline-window and baseline-matching sensitivity")
        rs4 = BaselineSensitivityRunner(d, s, persistence, root, log).run(c, plan)
        log(f"RS Analysis 4 complete: realizations={rs4['realization_count']}")
    a5 = c["robustness"]["analysis_5_dependence_sensitivity"]
    if c["robustness"]["enabled"] and a5["enabled"]:
        log("RS Analysis 5: temporal-dependence and block-length sensitivity")
        rs5 = DependenceSensitivityRunner(d, s, persistence, root, log).run(c, plan)
        log(f"RS Analysis 5 complete: realizations={rs5['realization_count']}")
    if c["robustness"]["enabled"]:
        log("RS Step 7: fixed-parameter reference-versus-sensitivity comparison")
        rs_fixed = FixedPointComparator(s, persistence, root, log).run(c, plan)
        log(f"RS fixed-point comparison complete: {rs_fixed['status']}; completed={rs_fixed['completed_comparison_count']}/{rs_fixed['requested_comparison_count']}")
        log("RS Steps 8-9: explicit realization-specific LOYO/null searches and complete reoptimization")
        rs_reopt = ReoptimizationRunner(d, s, persistence, root, log).run(c, plan)
        log(f"RS complete reoptimization: {rs_reopt['status']}; completed={rs_reopt['completed_realization_count']}/{rs_reopt['requested_realization_count']}")
        log("RS Step 10: reoptimized fingerprint and complete-surface comparison")
        rs_surface = FingerprintSurfaceComparator(s, persistence, root, log).run(c, plan)
        log(f"RS fingerprint/surface comparison: {rs_surface['status']}; completed={rs_surface['completed_comparison_count']}/{rs_surface['requested_comparison_count']}")
    return plan, persistence, reference_id


def _confounder_infrastructure(c: dict, run_id: str, identity: str, preprocessing, summary: dict) -> dict:
    return {
        "schema": CONFOUNDER_MANIFEST_SCHEMA,
        "implementation_stage": c["confounder"]["implementation_stage"],
        "candidate_id": preprocessing.candidate_id,
        "analysis_run_id": run_id,
        "analysis_identity": identity,
        "reference_metric": "WMARM_CS",
        "reference_configuration": summary.get("reference_configuration"),
        "reference_configuration_source": summary.get("reference_configuration_source"),
        "reference_configuration_inference_status": summary.get("reference_configuration_inference_status"),
        "reference_configuration_supported": bool(summary.get("reference_configuration_supported", False)),
        "reference_configuration_selection_rule": summary.get("reference_configuration_selection_rule"),
        "reference_optimization_status": summary.get("reference_optimization_status"),
        "candidate_preprocessing_metadata": preprocessing.metadata_path,
        "profile_count": int(summary.get("profile_count", 0)),
        "available_profile_count": int(summary.get("available_profile_count", 0)),
        "summary_file": "results/confounder_ca8_summary.json",
        "surface_summary_file": "results/confounder_ca7_summary.json",
        "fixed_inference_summary_file": "results/confounder_ca6_summary.json",
        "point_summary_file": "results/confounder_ca5_summary.json",
        "calibration_summary_file": "results/confounder_ca4_summary.json",
        "profiling_summary_file": "results/confounder_ca3_summary.json",
        "profile_csv": "results/confounder_profile.csv",
        "profile_surface_csv": "results/confounder_profile_surface.csv",
        "calibration_count": int(summary.get("calibration_count", 0)),
        "available_calibration_count": int(summary.get("available_calibration_count", 0)),
        "neutral_calibration_count": int(summary.get("neutral_calibration_count", 0)),
        "unavailable_calibration_count": int(summary.get("unavailable_calibration_count", 0)),
        "calibration_csv": "results/confounder_calibration.csv",
        "calibration_curve_csv": "results/confounder_calibration_curve.csv",
        "calibration_annual_csv": "results/confounder_calibration_annual.csv",
        "compatibility_context_csv": "results/confounder_compatibility_context.csv",
        "adjusted_point_csv": "results/confounder_adjusted_point.csv",
        "adjusted_yearly_csv": "results/confounder_adjusted_yearly.csv",
        "adjusted_pooled_csv": "results/confounder_adjusted_pooled.csv",
        "fixed_comparison_csv": "results/confounder_fixed_comparison.csv",
        "fixed_comparison_bsa_csv": "results/confounder_fixed_comparison_bsa.csv",
        "adjusted_surface_csv": "results/confounder_adjusted_surface.csv",
        "adjusted_loyo_csv": "results/confounder_adjusted_loyo.csv",
        "adjusted_null_maxima_csv": "results/confounder_adjusted_null_maximum.csv",
        "adjusted_null_cells_csv": "results/confounder_adjusted_null_cell.csv",
        "adjusted_optimization_cells_csv": "results/confounder_adjusted_optimization_cell.csv",
        "adjusted_optimization_file": "results/confounder_adjusted_optimization.json",
        "adjusted_fingerprint_file": "results/confounder_adjusted_fingerprint.json",
        "surface_comparison_file": "results/confounder_surface_comparison.json",
        "ca9_summary_file": "results/confounder_ca9_summary.json",
        "ca9_gui_configuration_enabled": True,
        "ca9_results_workspace_enabled": True,
        "ca9_export_catalog_enabled": True,
        "ca9_static_report_enabled": True,
        "complete_adjusted_surface_requested": bool(summary.get("complete_adjusted_surface_requested", False)),
        "complete_adjusted_surface_enabled": bool(summary.get("complete_adjusted_surface_enabled", False)),
        "surface_cell_count": int(summary.get("surface_cell_count", 0)),
        "valid_surface_cell_count": int(summary.get("valid_surface_cell_count", 0)),
        "adjusted_reoptimization_inference_enabled": bool(summary.get("adjusted_reoptimization_inference_enabled", False)),
        "adjusted_surface_run_id": summary.get("adjusted_run_id"),
        "nested_loyo_row_count": int(summary.get("nested_loyo_row_count", 0)),
        "nested_calibration_count": int(summary.get("nested_calibration_count", 0)),
        "nested_context_count": int(summary.get("nested_context_count", 0)),
        "adjusted_null_realizations_valid": int(summary.get("adjusted_null_realizations_valid", 0)),
        "adjusted_null_complete": bool(summary.get("adjusted_null_complete", False)),
        "robust_region_status": summary.get("robust_region_status"),
        "robust_region_cell_count": int(summary.get("robust_region_cell_count", 0)),
        "adjusted_representative": summary.get("adjusted_representative"),
        "adjusted_p_max": summary.get("adjusted_p_max"),
        "adjusted_estimator_enabled": bool(summary.get("adjusted_estimator_enabled", False)),
        "adjusted_bootstrap_enabled": bool(summary.get("adjusted_bootstrap_enabled", False)),
        "adjusted_pooling_enabled": bool(summary.get("adjusted_pooling_enabled", False)),
        "adjusted_point_rows": int(summary.get("point_rows", 0)),
        "eligible_adjusted_point_rows": int(summary.get("eligible_point_rows", 0)),
        "adjusted_yearly_rows": int(summary.get("adjusted_yearly_rows", 0)),
        "adjusted_pooled_rows": int(summary.get("adjusted_pooled_rows", 0)),
        "fixed_parameter_comparison_enabled": bool(summary.get("fixed_parameter_comparison_enabled", False)),
        "adjusted_wmarm_cs": summary.get("adjusted_wmarm_cs"),
        "delta_wmarm_cs": summary.get("delta_wmarm_cs"),
    }



def _valid_confounder_reference_cell(value) -> dict | None:
    """Normalize one persisted threshold-lag cell for confounder reference use."""
    if not isinstance(value, dict):
        return None
    try:
        q = float(value["q"])
        lag = int(value["lag"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(q):
        return None
    out = {"q": q, "lag": lag}
    for name in ("wmarm_cs", "wmarm"):
        raw = value.get(name)
        try:
            number = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            number = None
        out[name] = number if number is not None and np.isfinite(number) else None
    return out


def _confounder_reference_selection(optimization: dict | None) -> dict:
    """Choose the fixed reference configuration without changing reference inference.

    The ordinary supported representative remains first priority. If the
    reference optimization has no robust region, CA4 and later stages may still
    run diagnostically at the strongest *admissible observed* WMARMcs cell.
    That fallback is explicitly labelled exploratory and does not alter the
    persisted unadjusted optimization result or its support status.
    """
    optimization = optimization if isinstance(optimization, dict) else {}
    supported = _valid_confounder_reference_cell(optimization.get("representative"))
    if supported is not None:
        return {
            "configuration": supported,
            "source": "SUPPORTED_REPRESENTATIVE",
            "inference_status": "SUPPORTED_REFERENCE",
            "supported": True,
            "selection_rule": "REFERENCE_ROBUST_REGION_REPRESENTATIVE",
            "reference_optimization_status": optimization.get("status"),
            "reference_robust_region_cell_count": len(optimization.get("robust_region") or []),
        }

    peak = _valid_confounder_reference_cell(optimization.get("observed_peak"))
    if peak is None:
        # Compatibility fallback for older persisted optimization payloads that
        # predate the explicit observed_peak field. Use only cells already marked
        # admissible by the completed reference optimization.
        candidates = []
        for row in optimization.get("diagnostics") or []:
            if not isinstance(row, dict) or not bool(row.get("admissible_support")):
                continue
            candidate = _valid_confounder_reference_cell(row)
            if candidate is None or candidate.get("wmarm_cs") is None:
                continue
            candidates.append(candidate)
        if candidates:
            peak = min(candidates, key=lambda row: (-float(row["wmarm_cs"]), float(row["q"]), int(row["lag"])))

    if peak is not None:
        return {
            "configuration": peak,
            "source": "EXPLORATORY_OBSERVED_PEAK",
            "inference_status": "EXPLORATORY_UNSUPPORTED_REFERENCE",
            "supported": False,
            "selection_rule": "MAX_ADMISSIBLE_OBSERVED_WMARM_CS",
            "reference_optimization_status": optimization.get("status"),
            "reference_robust_region_cell_count": len(optimization.get("robust_region") or []),
        }

    return {
        "configuration": None,
        "source": "UNAVAILABLE",
        "inference_status": "REFERENCE_CONFIGURATION_UNAVAILABLE",
        "supported": False,
        "selection_rule": None,
        "reference_optimization_status": optimization.get("status"),
        "reference_robust_region_cell_count": len(optimization.get("robust_region") or []),
    }


def _reset_legacy_confounder_reference_gate(
    s: Store, persistence: ConfounderPersistence, run_id: str, log
) -> bool:
    """Resume an older CA3-complete project that stopped at the old CA4 gate.

    This migration is deliberately narrow. It activates only when the existing
    CA4 summary has no reference representative. CA3 profile records and CA3
    checkpoints are preserved; CA4 and later products are removed so they can
    be rebuilt against the new exploratory fallback configuration.
    """
    previous = s.get("workflow", run_id, "calibration_result")
    if not isinstance(previous, dict) or previous.get("reference_representative") is not None:
        return False

    downstream_kinds = (
        "workflow",
        "confounder_relevance_profile",
        "confounder_pairwise_bundle",
        "confounder_calibration",
        "confounder_calibration_curve",
        "confounder_calibration_annual",
        "confounder_compatibility_context",
        "confounder_adjusted_point",
        "confounder_adjusted_yearly",
        "confounder_adjusted_pooled",
        "confounder_fixed_comparison",
        "confounder_fixed_comparison_bsa",
        "confounder_adjusted_surface",
        "confounder_nested_calibration",
        "confounder_nested_context",
        "confounder_adjusted_loyo",
        "confounder_adjusted_null_maximum",
        "confounder_adjusted_null_cell",
        "confounder_adjusted_optimization",
        "confounder_adjusted_optimization_cell",
        "confounder_adjusted_fingerprint",
        "confounder_surface_comparison",
    )
    placeholders = ",".join("?" for _ in downstream_kinds)
    s.db.execute(
        f"DELETE FROM records WHERE scope=? AND kind IN ({placeholders})",
        (run_id, *downstream_kinds),
    )
    # Preserve CA3 cross-fitted profiles and their checkpoints. Everything from
    # relevance profiling onward is derived from the chosen fixed reference cell.
    s.db.execute(
        "DELETE FROM ca_checkpoints WHERE run_id=? AND stage<>?",
        (run_id, "CROSS_FITTED_PROFILE"),
    )
    s.db.execute(
        "UPDATE ca_runs SET state='PLANNED', error=NULL WHERE run_id=?",
        (run_id,),
    )
    s.commit()
    log(
        "Confounder compatibility migration: preserved completed CA3 profiles and "
        "reset only CA4+ records previously unavailable because no supported reference representative existed"
    )
    return True


def _run_confounder(c: dict, d: Data, s: Store, root: Path, log, ref_identity: str):
    reference = s.get("workflow", "REFERENCE", "result")
    optimization = s.get("optimization", "REFERENCE", "result")
    if not reference or reference.get("status") != "COMPLETED" or optimization is None:
        raise ValueError(
            "Candidate-confounder profiling requires a completed reference analysis in this project. "
            "Run the reference workflow first."
        )
    reference_selection = _confounder_reference_selection(optimization)
    reference_configuration = reference_selection["configuration"]
    if reference_configuration is not None:
        if reference_selection["supported"]:
            log(
                f"Confounder reference configuration: supported representative "
                f"q={reference_configuration['q']:g}, lag={reference_configuration['lag']}"
            )
        else:
            log(
                f"Confounder reference configuration: no supported reference region; using exploratory "
                f"observed admissible WMARMcs peak q={reference_configuration['q']:g}, "
                f"lag={reference_configuration['lag']}. Confounder outputs remain exploratory and do not "
                f"change the unadjusted reference support status."
            )
    else:
        log(
            "Confounder reference configuration unavailable: the completed reference analysis has no "
            "admissible observed threshold-lag cell"
        )

    settings = ConfounderAdjustmentSettings.from_dict(c["confounder"])
    if not settings.enabled:
        raise ValueError("confounder.enabled must be true to run the confounder phase")
    candidate_raw = c["confounder"].get("candidate")
    if candidate_raw is None:
        raise ValueError("confounder.candidate is required to run the confounder phase")
    candidate = ConfounderCandidateDefinition.from_dict(candidate_raw)

    log(f"Confounder CA2-CA3: preprocess candidate {candidate.candidate_id}")
    preprocessing = preprocess_candidate(
        candidate, canonical_root=root/"canonical", data_dir=c["data_dir"],
        project_root=root, log=log,
    )
    identity = confounder_identity(ref_identity, settings, candidate, preprocessing)
    run_id = confounder_run_id(candidate.candidate_id, identity)
    persistence = ConfounderPersistence(s)
    persistence.register_run(
        run_id=run_id, candidate_id=candidate.candidate_id,
        reference_identity=ref_identity, confounder_identity=identity,
        payload={
            "candidate": candidate.to_dict(),
            "settings": settings.to_dict(),
            "preprocessing": preprocessing.to_dict(),
            "reference_identity": ref_identity,
        },
    )
    if reference_configuration is not None and not reference_selection["supported"]:
        _reset_legacy_confounder_reference_gate(s, persistence, run_id, log)
    profile_summary = ConfounderProfilingRunner(
        d, c, s, root, persistence, run_id, ref_identity, candidate,
        preprocessing, log,
    ).run()
    atomic_json(root/"results"/"confounder_ca3_summary.json", profile_summary)
    confounder_reference_optimization = dict(optimization)
    confounder_reference_optimization["representative"] = reference_configuration
    calibration_summary = ConfounderCalibrationRunner(
        d, c, s, root, persistence, run_id, candidate, preprocessing, log
    ).run(confounder_reference_optimization)
    calibration_summary.update({
        "reference_configuration": reference_configuration,
        "reference_configuration_source": reference_selection["source"],
        "reference_configuration_inference_status": reference_selection["inference_status"],
        "reference_configuration_supported": reference_selection["supported"],
        "reference_configuration_selection_rule": reference_selection["selection_rule"],
        "reference_optimization_status": reference_selection["reference_optimization_status"],
        "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
    })
    s.put("workflow", run_id, "calibration_result", calibration_summary)
    s.commit()
    atomic_json(root/"results"/"confounder_ca4_summary.json", {
        **profile_summary,
        "status": calibration_summary["status"],
        "stage": calibration_summary["stage"],
        "profiling_status": profile_summary["status"],
        "reference_representative": calibration_summary.get("reference_representative"),
        "reference_configuration": reference_configuration,
        "reference_configuration_source": reference_selection["source"],
        "reference_configuration_inference_status": reference_selection["inference_status"],
        "reference_configuration_supported": reference_selection["supported"],
        "reference_configuration_selection_rule": reference_selection["selection_rule"],
        "reference_optimization_status": reference_selection["reference_optimization_status"],
        "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
        "calibration_count": calibration_summary.get("calibration_count", 0),
        "available_calibration_count": calibration_summary.get("available_calibration_count", 0),
        "neutral_calibration_count": calibration_summary.get("neutral_calibration_count", 0),
        "unavailable_calibration_count": calibration_summary.get("unavailable_calibration_count", 0),
        "adjusted_estimator_enabled": False,
        "calibrations": calibration_summary.get("calibrations", []),
    })
    point_summary = ConfounderPointEstimator(
        d, c, s, root, persistence, run_id, preprocessing, log
    ).run(calibration_summary.get("reference_representative"))
    ca5_summary = {
        **profile_summary,
        "status": point_summary["status"],
        "stage": point_summary["stage"],
        "profiling_status": profile_summary["status"],
        "calibration_status": calibration_summary["status"],
        "reference_representative": calibration_summary.get("reference_representative"),
        "reference_configuration": reference_configuration,
        "reference_configuration_source": reference_selection["source"],
        "reference_configuration_inference_status": reference_selection["inference_status"],
        "reference_configuration_supported": reference_selection["supported"],
        "reference_configuration_selection_rule": reference_selection["selection_rule"],
        "reference_optimization_status": reference_selection["reference_optimization_status"],
        "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
        "calibration_count": calibration_summary.get("calibration_count", 0),
        "available_calibration_count": calibration_summary.get("available_calibration_count", 0),
        "neutral_calibration_count": calibration_summary.get("neutral_calibration_count", 0),
        "unavailable_calibration_count": calibration_summary.get("unavailable_calibration_count", 0),
        "adjusted_estimator_enabled": point_summary.get("adjusted_estimator_enabled", True),
        "adjusted_bootstrap_enabled": point_summary.get("adjusted_bootstrap_enabled", False),
        "adjusted_pooling_enabled": point_summary.get("adjusted_pooling_enabled", False),
        "point_rows": point_summary.get("point_rows", 0),
        "eligible_point_rows": point_summary.get("eligible_point_rows", 0),
        "excluded_point_rows": point_summary.get("excluded_point_rows", 0),
        "reference_point_mismatch_count": point_summary.get("reference_point_mismatch_count", 0),
        "calibrations": calibration_summary.get("calibrations", []),
    }
    atomic_json(root/"results"/"confounder_ca5_summary.json", ca5_summary)
    inference_summary = ConfounderAdjustedInferenceRunner(
        d, c, s, root, persistence, run_id, preprocessing, log
    ).run(calibration_summary.get("reference_representative"))
    summary = {
        **ca5_summary,
        **inference_summary,
        "profiling_status": profile_summary["status"],
        "calibration_status": calibration_summary["status"],
        "reference_configuration": reference_configuration,
        "reference_configuration_source": reference_selection["source"],
        "reference_configuration_inference_status": reference_selection["inference_status"],
        "reference_configuration_supported": reference_selection["supported"],
        "reference_configuration_selection_rule": reference_selection["selection_rule"],
        "reference_optimization_status": reference_selection["reference_optimization_status"],
        "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
        "calibration_count": calibration_summary.get("calibration_count", 0),
        "available_calibration_count": calibration_summary.get("available_calibration_count", 0),
        "neutral_calibration_count": calibration_summary.get("neutral_calibration_count", 0),
        "unavailable_calibration_count": calibration_summary.get("unavailable_calibration_count", 0),
        "point_rows": point_summary.get("point_rows", 0),
        "eligible_point_rows": point_summary.get("eligible_point_rows", 0),
        "excluded_point_rows": point_summary.get("excluded_point_rows", 0),
        "reference_point_mismatch_count": point_summary.get("reference_point_mismatch_count", 0),
        "calibrations": calibration_summary.get("calibrations", []),
    }
    s.put("workflow", run_id, "result", summary)
    s.commit()
    atomic_json(root/"results"/"confounder_ca6_summary.json", summary)

    surface_summary = ConfounderAdjustedSurfaceRunner(
        d, c, s, root, persistence, run_id, preprocessing, log
    ).run(requested=bool(settings.request_complete_reoptimization))
    fixed_inference_status = summary.get("status")
    surface_status = surface_summary.get("status")
    summary = {
        **summary,
        **surface_summary,
        "status": surface_status if settings.request_complete_reoptimization else fixed_inference_status,
        "fixed_inference_status": fixed_inference_status,
        "adjusted_surface_status": surface_status,
        "profiling_status": profile_summary["status"],
        "calibration_status": calibration_summary["status"],
        "reference_configuration": reference_configuration,
        "reference_configuration_source": reference_selection["source"],
        "reference_configuration_inference_status": reference_selection["inference_status"],
        "reference_configuration_supported": reference_selection["supported"],
        "reference_configuration_selection_rule": reference_selection["selection_rule"],
        "reference_optimization_status": reference_selection["reference_optimization_status"],
        "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
        "calibration_count": calibration_summary.get("calibration_count", 0),
        "available_calibration_count": calibration_summary.get("available_calibration_count", 0),
        "neutral_calibration_count": calibration_summary.get("neutral_calibration_count", 0),
        "unavailable_calibration_count": calibration_summary.get("unavailable_calibration_count", 0),
        "point_rows": point_summary.get("point_rows", 0),
        "eligible_point_rows": point_summary.get("eligible_point_rows", 0),
        "excluded_point_rows": point_summary.get("excluded_point_rows", 0),
        "reference_point_mismatch_count": point_summary.get("reference_point_mismatch_count", 0),
        "calibrations": calibration_summary.get("calibrations", []),
    }
    s.put("workflow", run_id, "result", summary)
    s.commit()
    atomic_json(root/"results"/"confounder_ca7_summary.json", summary)

    optimization_summary = ConfounderAdjustedOptimizationRunner(
        d, c, s, root, persistence, run_id, candidate, preprocessing, log
    ).run(
        requested=bool(settings.request_complete_reoptimization),
        representative=calibration_summary.get("reference_representative"),
    )
    ca7_status = summary.get("status")
    summary = {
        **summary,
        **optimization_summary,
        "status": optimization_summary.get("status")
        if settings.request_complete_reoptimization
        else ca7_status,
        "adjusted_surface_status": surface_summary.get("status"),
        "fixed_inference_status": fixed_inference_status,
    }
    s.put("workflow", run_id, "result", summary)
    s.commit()
    atomic_json(root/"results"/"confounder_ca8_summary.json", summary)
    adjusted_optimization = s.get("confounder_adjusted_optimization", run_id, "result")
    if adjusted_optimization is not None:
        atomic_json(root/"results"/"confounder_adjusted_optimization.json", adjusted_optimization)
    adjusted_fingerprint = s.get("confounder_adjusted_fingerprint", run_id, "result")
    if adjusted_fingerprint is not None:
        atomic_json(root/"results"/"confounder_adjusted_fingerprint.json", adjusted_fingerprint)
    surface_comparison = s.get("confounder_surface_comparison", run_id, "result")
    if surface_comparison is not None:
        surface_comparison = dict(surface_comparison)
        surface_comparison.update({
            "confounder_reference_configuration": reference_configuration,
            "reference_configuration_source": reference_selection["source"],
            "reference_configuration_inference_status": reference_selection["inference_status"],
            "reference_configuration_supported": reference_selection["supported"],
        })
        adjusted_rep = surface_comparison.get("adjusted_representative")
        if reference_configuration is not None:
            surface_comparison["reference_representative"] = {
                "q": float(reference_configuration["q"]),
                "lag": int(reference_configuration["lag"]),
            }
            if isinstance(adjusted_rep, dict) and adjusted_rep.get("q") is not None and adjusted_rep.get("lag") is not None:
                surface_comparison["representative_delta_q"] = float(adjusted_rep["q"]) - float(reference_configuration["q"])
                surface_comparison["representative_delta_lag"] = int(adjusted_rep["lag"]) - int(reference_configuration["lag"])
        s.put("confounder_surface_comparison", run_id, "result", surface_comparison)
        s.commit()
        summary.update({
            "reference_configuration": reference_configuration,
            "reference_configuration_source": reference_selection["source"],
            "reference_configuration_inference_status": reference_selection["inference_status"],
            "reference_configuration_supported": reference_selection["supported"],
            "reference_configuration_selection_rule": reference_selection["selection_rule"],
            "reference_optimization_status": reference_selection["reference_optimization_status"],
            "reference_robust_region_cell_count": reference_selection["reference_robust_region_cell_count"],
            "reference_representative": reference_configuration,
        })
        s.put("workflow", run_id, "result", summary)
        s.commit()
        atomic_json(root/"results"/"confounder_ca8_summary.json", summary)
        atomic_json(root/"results"/"confounder_surface_comparison.json", surface_comparison)
    return summary, persistence, run_id, identity, preprocessing


def export_results(root: Path,s: Store,c: dict,data: Data,phase: str) -> None:
    out = root/"results"
    out.mkdir(exist_ok=True)
    for kind in ("cell","yearly","pooled","support","loyo","null_maximum","null_cell","optimization_cell",
                 "calibration_instance","calibration_distribution","calibration_operating_characteristics",
                 "calibration_ordering","calibration_recovery","calibration_mean_surface","synthetic_support","synthetic_generation",
                 "confounder_profile","confounder_profile_cell",
                 "confounder_relevance_profile","confounder_pairwise_bundle",
                 "confounder_calibration","confounder_calibration_curve",
                 "confounder_calibration_annual","confounder_compatibility_context",
                 "confounder_adjusted_point","confounder_adjusted_yearly",
                 "confounder_adjusted_pooled","confounder_fixed_comparison",
                 "confounder_fixed_comparison_bsa","confounder_adjusted_surface",
                 "confounder_nested_calibration","confounder_nested_context",
                 "confounder_adjusted_loyo","confounder_adjusted_null_maximum",
                 "confounder_adjusted_null_cell","confounder_adjusted_optimization",
                 "confounder_adjusted_optimization_cell","confounder_adjusted_fingerprint",
                 "confounder_surface_comparison"):
        s.export_csv(kind,out/f"{kind}.csv")
    for kind,name in (("cell","reference_surface"),("yearly","reference_yearly"),("pooled","reference_pooled"),
                      ("loyo","reference_loyo"),("optimization_cell","reference_optimization_cells"),
                      ("null_maximum","reference_null_maxima")):
        s.export_csv(kind,out/f"{name}.csv",run_filter="REFERENCE")
    s.export_csv("confounder_profile",out/"confounder_profile.csv")
    s.export_csv("confounder_profile_cell",out/"confounder_profile_surface.csv")
    s.export_csv("confounder_relevance_profile",out/"confounder_relevance_profile.csv")
    s.export_csv("confounder_pairwise_bundle",out/"confounder_pairwise_bundle.csv")
    s.export_csv("confounder_calibration",out/"confounder_calibration.csv")
    s.export_csv("confounder_calibration_curve",out/"confounder_calibration_curve.csv")
    s.export_csv("confounder_calibration_annual",out/"confounder_calibration_annual.csv")
    s.export_csv("confounder_compatibility_context",out/"confounder_compatibility_context.csv")
    s.export_csv("confounder_adjusted_point",out/"confounder_adjusted_point.csv")
    s.export_csv("confounder_adjusted_yearly",out/"confounder_adjusted_yearly.csv")
    s.export_csv("confounder_adjusted_pooled",out/"confounder_adjusted_pooled.csv")
    s.export_csv("confounder_fixed_comparison",out/"confounder_fixed_comparison.csv")
    s.export_csv("confounder_fixed_comparison_bsa",out/"confounder_fixed_comparison_bsa.csv")
    s.export_csv("confounder_adjusted_surface",out/"confounder_adjusted_surface.csv")
    s.export_csv("confounder_nested_calibration",out/"confounder_nested_calibration.csv")
    s.export_csv("confounder_nested_context",out/"confounder_nested_context.csv")
    s.export_csv("confounder_adjusted_loyo",out/"confounder_adjusted_loyo.csv")
    s.export_csv("confounder_adjusted_null_maximum",out/"confounder_adjusted_null_maximum.csv")
    s.export_csv("confounder_adjusted_null_cell",out/"confounder_adjusted_null_cell.csv")
    s.export_csv("confounder_adjusted_optimization_cell",out/"confounder_adjusted_optimization_cell.csv")
    optimization = s.get("optimization","REFERENCE","result")
    if optimization is not None:
        atomic_json(out/"reference_optimization.json",optimization)
    fp = s.get("fingerprint","REFERENCE","result")
    if fp:
        atomic_json(out/"reference_fingerprint.json",fp)
    calibration = s.get("workflow","CALIBRATION","result")
    if calibration:
        atomic_json(out/"calibration_summary.json",calibration)
    # Preserve optional geometry for downstream joining. The statistical
    # workflow itself does not require a geometry engine.
    geometry_name = c["sources"].get("geometry_file","")
    geometry = source_path(c["data_dir"],geometry_name) if geometry_name else Path()
    if geometry.is_file():
        (root/"geometry").mkdir(exist_ok=True)
        shutil.copyfile(geometry,root/"geometry"/geometry.name)
    cells = list(s.rows("cell","REFERENCE"))
    body = []
    for r in cells:
        value_cs = "unavailable" if r.get("wmarm_cs") is None else f"{r['wmarm_cs']:.8f}"
        value_point = "unavailable" if r.get("wmarm") is None else f"{r['wmarm']:.8f}"
        kcs = "unavailable" if r.get("confidence_support_retention_ratio") is None else f"{r['confidence_support_retention_ratio']:.3f}"
        body.append(f"<tr><td>{r['q']:g}</td><td>{r['lag']}</td><td>{value_cs}</td><td>{value_point}</td><td>{kcs}</td><td>{r['bsa_coverage']:.1%}</td><td>{r['population_coverage']:.1%}</td></tr>" if r["population_coverage"] is not None else f"<tr><td>{r['q']:g}</td><td>{r['lag']}</td><td>{value_cs}</td><td>{value_point}</td><td>{kcs}</td><td colspan='2'>unavailable</td></tr>")
    notice = "DEV Monte Carlo counts are for software execution checks, not publication-level inference." if c["profile"]=="DEV" else "The statistical support and Monte Carlo precision must be assessed before interpretation."
    confounder_html = ""
    ca = c.get("confounder") or {}
    candidate = ca.get("candidate") or {}
    if candidate:
        candidate_id = str(candidate.get("candidate_id") or "")
        ca8_path = out / "confounder_ca8_summary.json"
        ca8_summary = None
        if ca8_path.is_file():
            try:
                ca8_summary = json.loads(ca8_path.read_text(encoding="utf-8-sig"))
            except Exception:
                ca8_summary = None
        analysis_run_id = str((ca8_summary or {}).get("analysis_run_id") or "")
        fixed = s.get("confounder_fixed_comparison", analysis_run_id, "result") if analysis_run_id else None
        adjusted_opt = s.get("confounder_adjusted_optimization", analysis_run_id, "result") if analysis_run_id else None
        comparison = s.get("confounder_surface_comparison", analysis_run_id, "result") if analysis_run_id else None
        fixed_rows = ""
        if fixed:
            for label,key in (("Reference WMARMcs","reference_wmarm_cs"),("Adjusted WMARMcs","adjusted_wmarm_cs"),("Delta WMARMcs","delta_wmarm_cs"),("Reference population coverage","reference_population_coverage"),("Adjusted population coverage","adjusted_population_coverage"),("BSA sign agreement","bsa_sign_agreement"),("Pooled RC Spearman","pooled_rc_spearman")):
                value = fixed.get(key)
                if value is None:
                    rendered = "unavailable"
                elif "coverage" in key or "agreement" in key:
                    rendered = f"{float(value):.1%}"
                else:
                    rendered = f"{float(value):.6g}"
                fixed_rows += f"<tr><td>{html.escape(label)}</td><td>{html.escape(rendered)}</td></tr>"
        opt_text = "not requested or unavailable"
        if adjusted_opt:
            rep = adjusted_opt.get("representative")
            rep_text = "none" if not rep else f"q={rep.get('q')}, lag={rep.get('lag')}"
            opt_text = f"status={adjusted_opt.get('status')}; representative={rep_text}; robust cells={len(adjusted_opt.get('robust_region') or [])}"
        comparison_text = ""
        if comparison:
            comparison_text = (f"<p>Complete-surface comparison: WMARMcs RMSE={comparison.get('wmarm_cs_surface_rmse')}; "
                               f"rank concordance={comparison.get('wmarm_cs_surface_spearman')}; "
                               f"robust-region Jaccard={comparison.get('robust_region_jaccard')}.</p>")
        confounder_html = (
            f"<h2>Candidate-confounder adjustment</h2><p>Candidate: {html.escape(str(candidate.get('display_name') or candidate_id))}. "
            f"Input: standard APHREH hazard-field CSV ({html.escape(str(candidate.get('hazard_file') or candidate.get('source_reference') or ''))}), "
            f"projected with the standard intersection-area hazard projection. The reference territorial metric is WMARMcs.</p>"
            + (f"<table><tr><th>Fixed-parameter diagnostic</th><th>Value</th></tr>{fixed_rows}</table>" if fixed_rows else "<p>Fixed-parameter adjusted inference has not been completed.</p>")
            + f"<p>Adjusted optimization: {html.escape(opt_text)}.</p>" + comparison_text
        )
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>APHREH DEV results</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:45px auto;line-height:1.5;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px 12px;border-bottom:1px solid #bbb;text-align:right}}pre{{white-space:pre-wrap}}.notice{{border:1px solid #888;padding:15px}}</style>
<h1>APHREH-ADSMap</h1><p class="notice">{html.escape(notice)}</p>
<p>Data: {data.meta['actual_start']} to {data.meta['actual_end']}; {data.B} BSAs; {data.n} exact hazard/outcome matching days across years {html.escape(str(data.meta['years']))}; contiguous calendar segments: {data.meta.get('contiguous_segment_count',1)}. Population policy: {html.escape(dumps(c['population_policy']))}. Workflow phase: {html.escape(phase)}.</p>
<h2>Observed threshold-lag surface</h2><p>WMARMcs is the reference confidence-scaled relevance metric using continuous relative-uncertainty attenuation. Standard WMARM is retained as a diagnostic point-estimate magnitude.</p><table><tr><th>Threshold percentile</th><th>Cumulative lag</th><th>WMARMcs</th><th>Diagnostic WMARM</th><th>Kcs</th><th>BSA coverage</th><th>Population coverage</th></tr>{''.join(body)}</table>
<h2>Optimization</h2><p>Status: {html.escape(optimization['status']) if optimization else 'not executed'}.
A descriptive surface peak is not a forced robust optimum.</p>
<pre>{html.escape(json.dumps({k:optimization.get(k) for k in ('observed_peak','p_max','null_cutoff','representative')} if optimization else {},indent=2))}</pre>
{confounder_html}
<h2>Files</h2><p>Use reference_surface.csv, reference_yearly.csv, reference_pooled.csv, reference_loyo.csv,
reference_optimization.json and reference_fingerprint.json. Calibration distributions, recovery diagnostics,
null maxima and generation provenance are retained in the accompanying CSVs. Candidate-confounder outputs use the confounder_ prefix and include cross-fitted calibration, fixed adjusted inference, complete adjusted surfaces, and adjusted optimization artifacts when requested. results.sqlite is the authoritative checkpoint store.</p>
<p>Plots and maps: open this project in the APHREH-ADSMap Results workspace. BSA-specific figure export requires Matplotlib.</p>
<p>Mathematical formulation (MF, PO, RS, CA, CM): <a href="https://doi.org/10.5281/zenodo.18671472">Zenodo DOI 10.5281/zenodo.18671472</a>.</p></html>"""
    (out/"report.html").write_text(text,encoding="utf-8")


def run_project(c: dict, *, phase: str="all", emit=print) -> dict:
    validate(c)
    if phase not in ("all","calibration","reference","robustness","confounder"):
        raise ValueError("phase must be all, calibration, reference, robustness, or confounder")
    root = Path(c["output_dir"]).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    with ProjectLock(root/".run.lock"):
        (root/"CANCEL").unlink(missing_ok=True)
        sources = source_identity(c)
        full_code = code_identity()
        ref_identity = reference_identity(c, sources)
        manifest_path = root/"manifest.json"
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        migration = None

        if old is None and phase in ("robustness", "confounder"):
            label = "Robustness/sensitivity" if phase == "robustness" else "Candidate-confounder profiling"
            raise ValueError(f"{label} can only be launched from a project with a completed reference analysis.")
        if old is not None:
            previous_reference_identity = old.get("reference_identity")
            if previous_reference_identity is not None:
                if previous_reference_identity != ref_identity:
                    if str(old.get("version") or "") in KNOWN_CALIBRATION_ADMISSIBILITY_FIX_PREDECESSORS:
                        migration = _compatible_calibration_admissibility_upgrade(old, c, sources, root)
                    elif str(old.get("version") or "") in KNOWN_CALIBRATION_FIX_PREDECESSORS:
                        migration = _compatible_calibration_fix_upgrade(old, c, sources, root)
                    elif str(old.get("version") or "") in KNOWN_EQUIVALENT_PERFORMANCE_BUILDS:
                        migration = _compatible_performance_upgrade(old, c, sources, root)
                    elif (
                        phase in ("robustness", "confounder")
                        and str(old.get("version") or "") in KNOWN_COMPATIBLE_DOWNSTREAM_CODE_AGGREGATES
                    ):
                        migration = _compatible_downstream_reference_upgrade(old, c, sources, root)
                    else:
                        raise ValueError(
                            "This output folder belongs to different code, inputs, or scientific settings for the reference analysis. "
                            "Only robustness or confounder-module settings may change in place; use a new output folder for reference changes."
                        )
            elif phase == "robustness":
                migration = _legacy_reference_migration(old, c, sources, root)
            else:
                # Preserve the pre-2.3.1 overwrite guard for legacy projects
                # unless the explicit RS-only migration path is requested.
                science = {k:v for k,v in c.items() if k not in ("output_dir","data_dir","config_path","notes","profile")}
                legacy_identity = hashlib.sha256(dumps({"configuration":science,"sources":sources,"code":full_code}).encode()).hexdigest()
                if old.get("identity") != legacy_identity:
                    raise ValueError(
                        "This output folder belongs to different code, inputs, or scientific settings. "
                        "Choose a new output folder, or use the robustness workflow to attach RS analyses to an approved completed legacy reference project."
                    )

        manifest = old or {"created_utc":now()}
        if migration is not None:
            history = list(manifest.get("migration_history") or [])
            history.append(migration)
            manifest["migration_history"] = history
        manifest.update(
            identity=ref_identity,
            reference_identity=ref_identity,
            reference_identity_schema=REFERENCE_IDENTITY_SCHEMA,
            version=__version__,
            source_files=sources,
            source_code_hashes=full_code,
            reference_code_hashes=reference_code_identity(),
            status="RUNNING",last_started_utc=now(),phase=phase,
            environment={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__},
            configuration=c,
        )
        atomic_json(manifest_path,manifest)
        atomic_json(root/"resolved_config.json",c)
        log_path = root/"execution.jsonl"
        start = time.perf_counter()
        def log(message):
            event = {"utc":now(),"message":message,"rss_mb":psutil.Process().memory_info().rss/1024**2}
            with log_path.open("a",encoding="utf-8") as f:
                f.write(dumps(event)+"\n")
            emit(message,flush=True) if emit is print else emit(message)
        s = None
        rs_persistence = None
        rs_reference_id = None
        rs_plan = None
        rs_plan_identity = None
        ca_summary = None
        ca_persistence = None
        ca_run_id = None
        ca_identity = None
        ca_preprocessing = None
        try:
            log(f"APHREH {__version__}; profile={c['profile']}; phase={phase}")
            if migration is not None:
                log(f"Compatible project checkpoints accepted: scope={migration['scope']}; version={migration['origin_version']}; identity={migration['origin_identity']}")
            log(f"Population assumption: {c['population_policy']}")
            log(f"Monte Carlo: bootstrap={c['core']['bootstrap']['iterations']}, calibration={c['calibration']['number_of_realizations']} per scenario, TL modes={c['calibration']['threshold_lag_response_modes']} (primary={c['calibration']['primary_threshold_lag_response_mode']}), null={c['optimization']['null_realizations_R']}")

            canonical_metadata = root/"canonical"/"metadata.json"
            if phase in ("robustness", "confounder"):
                if not canonical_metadata.exists() or not (root/"results.sqlite").is_file():
                    label = "Robustness/sensitivity" if phase == "robustness" else "Candidate-confounder profiling"
                    raise ValueError(f"{label} requires the completed project's canonical data and results.sqlite.")
            elif not canonical_metadata.exists():
                build_canonical(c,root/"canonical",log)

            from .performance import memory_budget
            budget = memory_budget()
            budget.check(force=True)
            log(f"Performance budget: cache={budget.cache_limit/1024**2:g} MiB; scratch RAM={budget.workspace_limit/1024**2:g} MiB; process RSS guard={budget.rss_guard/1024**2:g} MiB")
            d = Data(root/"canonical")
            s = Store(root/"results.sqlite")
            if migration is not None and migration.get("reset_tl_calibration"):
                removed = _reset_tl_calibration_checkpoints(s)
                exported = _remove_calibration_exports(root)
                log(
                    f"TL-calibration admissibility migration: removed {removed} affected 2.3.20 TL calibration checkpoint records "
                    f"and {exported} stale calibration exports; RC calibration and observed reference checkpoints preserved"
                )
            if migration is not None and migration.get("reset_calibration") and phase in ("all", "calibration"):
                removed = _reset_calibration_checkpoints(s)
                exported = _remove_calibration_exports(root)
                log(
                    f"Calibration-fix migration: removed {removed} affected 2.3.19 calibration checkpoint records "
                    f"and {exported} stale calibration exports; observed reference checkpoints preserved"
                )
            e = FixedParametersEngine(d,c,s,root,log)
            runner = SurfaceRunner(e)
            opt = Optimizer(runner)

            if phase in ("all","calibration"):
                Calibration(runner,opt).run()
                # Export every successfully completed stage immediately.  This
                # keeps upstream results usable if a later stage in an ``all``
                # workflow fails.  The SQLite store remains authoritative.
                export_results(root,s,c,d,"calibration")
                log(f"Calibration exports refreshed: {root/'results'}")

            if phase in ("all","reference"):
                if phase=="reference" and s.get("workflow","CALIBRATION","result") is None:
                    log("Reference-only diagnostic run: pre-analysis calibration has NOT been completed")
                log("Observed reference: complete threshold-lag surface")
                cells = runner.run("REFERENCE")
                result = opt.run("REFERENCE",cells)
                log(f"Observed reference complete: {result['status']}; p_max={result['p_max']}")
                s.put("workflow","REFERENCE","result",{"status":"COMPLETED","optimization_status":result["status"]})
                s.commit()
                export_results(root,s,c,d,"reference")
                log(f"Reference exports refreshed: {root/'results'}")

            if phase in ("all","robustness"):
                rs_plan, rs_persistence, rs_reference_id = _run_robustness(c,d,s,root,log)
                rs_plan_identity = robustness_identity(c, ref_identity, rs_plan)
                export_results(root,s,c,d,"robustness")
                log(f"Robustness/sensitivity exports refreshed: {root/'results'}")

            if phase == "confounder" or (phase == "all" and c["confounder"]["enabled"]):
                ca_summary, ca_persistence, ca_run_id, ca_identity, ca_preprocessing = _run_confounder(
                    c, d, s, root, log, ref_identity
                )
                log(
                    f"Confounder CA6 complete: status={ca_summary['status']}; "
                    f"profiles={ca_summary.get('available_profile_count',0)}/{ca_summary.get('profile_count',0)}; "
                    f"calibrations={ca_summary.get('available_calibration_count',0)} available, "
                    f"{ca_summary.get('neutral_calibration_count',0)} neutral"
                )
                export_results(root,s,c,d,"confounder")
                log(f"Confounder exports refreshed: {root/'results'}")

            # Final refresh preserves the historical complete-workflow export
            # behavior and report phase label.  Earlier stage exports above are
            # intentionally durable if execution never reaches this point.
            export_results(root,s,c,d,phase)
            if ca_summary is not None:
                atomic_json(root/"results"/"confounder_ca9_summary.json", {
                    "stage": "CA9_GUI_EXPORTS_FINAL_VALIDATION",
                    "status": "COMPLETED",
                    "candidate_id": ca_preprocessing.candidate_id,
                    "analysis_run_id": ca_run_id,
                    "reference_metric": "WMARM_CS",
                    "reference_configuration": ca_summary.get("reference_configuration"),
                    "reference_configuration_source": ca_summary.get("reference_configuration_source"),
                    "reference_configuration_inference_status": ca_summary.get("reference_configuration_inference_status"),
                    "reference_configuration_supported": bool(ca_summary.get("reference_configuration_supported", False)),
                    "reference_configuration_selection_rule": ca_summary.get("reference_configuration_selection_rule"),
                    "reference_optimization_status": ca_summary.get("reference_optimization_status"),
                    "computational_implementation_stage": c["confounder"]["implementation_stage"],
                    "scientific_results_recomputed_by_ca9": False,
                    "gui_configuration_enabled": True,
                    "results_workspace_enabled": True,
                    "export_catalog_enabled": True,
                    "static_html_report_enabled": True,
                    "complete_adjusted_surface_requested": bool(c["confounder"].get("request_complete_reoptimization")),
                    "adjusted_reoptimization_inference_enabled": bool(ca_summary.get("adjusted_reoptimization_inference_enabled", False)),
                })
            retained_cells = s.db.execute("SELECT COUNT(*) FROM records WHERE kind='cell'").fetchone()[0]
            null_cells = s.db.execute("SELECT COUNT(*) FROM records WHERE kind='null_cell'").fetchone()[0]
            counts = {"retained_fixed_cells":retained_cells,"null_search_cells":null_cells,
                      "total_fixed_parameter_evaluations":retained_cells+null_cells,
                      "yearly_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='yearly'").fetchone()[0],
                      "pooled_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='pooled'").fetchone()[0],
                      "confounder_profile_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_profile'").fetchone()[0],
                      "confounder_profile_cell_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_profile_cell'").fetchone()[0],
                      "confounder_calibration_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_calibration'").fetchone()[0],
                      "confounder_calibration_curve_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_calibration_curve'").fetchone()[0],
                      "confounder_adjusted_point_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_adjusted_point'").fetchone()[0],
                      "confounder_adjusted_yearly_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_adjusted_yearly'").fetchone()[0],
                      "confounder_adjusted_pooled_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_adjusted_pooled'").fetchone()[0],
                      "confounder_fixed_comparison_bsa_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_fixed_comparison_bsa'").fetchone()[0],
                      "confounder_adjusted_surface_rows":s.db.execute("SELECT COUNT(*) FROM records WHERE kind='confounder_adjusted_surface'").fetchone()[0]}
            manifest.update(
                status="COMPLETED",completed_utc=now(),invocation_elapsed_seconds=time.perf_counter()-start,
                peak_process_rss_mb=peak_rss_mb(),
                performance=budget.snapshot(),
                data_summary={k:d.meta[k] for k in ("actual_start","actual_end","date_count","bsa_count","grid_count","intersection_count")},
                execution_counts=counts,
                calibration_completed=s.get("workflow","CALIBRATION","result") is not None,
                reference_completed=s.get("workflow","REFERENCE","result") is not None,
            )
            if rs_plan is not None:
                manifest["robustness_identity_schema"] = ROBUSTNESS_IDENTITY_SCHEMA
                manifest["robustness_identity"] = rs_plan_identity
                manifest["robustness_code_hashes"] = robustness_code_identity()
                manifest["robustness_infrastructure"] = _robustness_infrastructure(c,rs_plan,rs_reference_id,rs_plan_identity)
                manifest["robustness_completed_utc"] = now()
            if ca_summary is not None:
                manifest["confounder_identity_schema"] = CONFOUNDER_MANIFEST_SCHEMA
                manifest["confounder_identity"] = ca_identity
                manifest["confounder_code_hashes"] = confounder_code_identity()
                manifest["confounder_infrastructure"] = _confounder_infrastructure(
                    c, ca_run_id, ca_identity, ca_preprocessing, ca_summary
                )
                manifest["confounder_completed_utc"] = now()
                manifest["confounder_completed"] = True
            atomic_json(manifest_path,manifest)
            log(f"COMPLETED; peak process RSS={manifest['peak_process_rss_mb']:.1f} MiB; results={root/'results'}")
            return manifest
        except BaseException as exc:
            # Sensitivity realization runners maintain their own states.  The
            # reference realization represents the already-completed reference
            # analysis and must not be marked failed by an RS-only failure.
            manifest.update(status="CANCELLED" if type(exc).__name__ in ("Cancelled","KeyboardInterrupt") else "FAILED",
                            last_error=f"{type(exc).__name__}: {exc}",invocation_elapsed_seconds=time.perf_counter()-start,
                            peak_process_rss_mb=peak_rss_mb())
            atomic_json(manifest_path,manifest)
            with (root/"error.log").open("a",encoding="utf-8") as f:
                f.write(traceback.format_exc()+"\n")
            raise
        finally:
            if s is not None:
                s.close()
            # No acceleration cache survives a completed/failed project task.
            from .performance import memory_budget
            memory_budget().clear()


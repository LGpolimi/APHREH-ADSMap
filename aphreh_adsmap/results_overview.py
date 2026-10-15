"""Read-only Results Workspace for completed APHREH-ADSMap analyses.

Implemented pages
-----------------
* Overview: high-level result state, paired WMARMcs / diagnostic WMARM previews,
  and selected-cell support.
* Surface: detailed threshold-lag exploration with synchronized 2D heatmaps or
  3D rotated surfaces, optimization overlays, metric switching, calibration
  relevance references, and selected-cell diagnostics.
* Calibration: effect-magnitude distributions, operating characteristics, paired
  STEP/LINEAR_RAMP recovery summaries, mean threshold-lag calibration surfaces,
  and exact synthetic-versus-observed signal inspection.
* Diagnostics: selected-cell support, pooled-BSA heterogeneity, LOYO stability,
  and estimator exclusion/data-quality summaries.
* Spatial: interactive pooled and yearly RCy BSA maps, effect layers, and selected-BSA detail.
* BSA detail: pooled/yearly effect inspection, forest plot, bootstrap and support diagnostics, plus selected/all-BSA figure export.
* Robustness / sensitivity: realization selection, fixed-point BSA changes, reoptimized surfaces, factor trajectories, region stability, and fingerprint comparisons.
* Candidate confounder: cross-fitted calibration, fixed-point adjusted inference, complete adjusted surfaces, and adjusted optimization comparison.
* Outputs: grouped result artifacts, metadata, preview, and safe open/copy actions.

The workspace is read-only with respect to the project. Ordinary navigation uses
exported results. Synthetic signal inspection additionally reconstructs one
calibration outcome from the retained canonical arrays and generation metadata
inside an isolated temporary directory, verifies its stored SHA-256 identity,
and never mutates or resumes the project analysis.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .engine import confidence_scaled_magnitude, pooled_ci_excludes_zero


ZERO_TOLERANCE = 1e-15


def _float(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(value):
    value = _float(value)
    return None if value is None else int(value)


def _bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt_number(value, digits=3):
    value = _float(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_percent(value, digits=1):
    value = _float(value)
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _fmt_q(value):
    value = _float(value)
    if value is None:
        return "—"
    return f"q{value:g}"


def _fmt_bool(value):
    value = _bool(value)
    if value is None:
        return "—"
    return "Yes" if value else "No"


def _surface_row(raw: dict) -> dict:
    numeric = {
        "q", "lag", "wmarm_cs", "wmarm", "marm",
        "confidence_support_retention_ratio", "bsa_coverage",
        "population_coverage", "ci_excluding_zero_population_share",
        "eligible_bsas", "total_bsas", "mean_i_squared_percent",
        "directional_balance", "directional_balance_cs",
        "signed_coefficient", "signed_coefficient_cs",
        "spatial_magnitude_dispersion", "spatial_magnitude_dispersion_cs",
    }
    result = dict(raw)
    for key in numeric:
        if key in result:
            result[key] = _float(result[key])
    if result.get("lag") is not None:
        result["lag"] = int(result["lag"])
    if "valid" in result:
        result["valid"] = _bool(result["valid"])
    return result


@dataclass
class OverviewModel:
    project_dir: Path
    surface: list[dict]
    optimization: dict | None
    manifest: dict | None

    @property
    def results_dir(self) -> Path:
        return self.project_dir / "results"

    @property
    def reference_maximum(self):
        if self.optimization:
            value = self.optimization.get("observed_admissible_maximum_wmarm_cs")
            if _float(value) is not None:
                return _float(value)
        values = [_float(r.get("wmarm_cs")) for r in self.surface]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    @property
    def diagnostic_maximum(self):
        values = [_float(r.get("wmarm")) for r in self.surface]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    @property
    def all_zero_supported(self) -> bool:
        maximum = self.reference_maximum
        return maximum is not None and abs(maximum) <= ZERO_TOLERANCE

    @property
    def status_code(self) -> str:
        if not self.surface:
            return "RESULTS_UNAVAILABLE"
        if self.all_zero_supported:
            return "ZERO_SUPPORTED_SURFACE"
        if self.optimization and self.optimization.get("status") == "ROBUST_REGION":
            return "ROBUST_REGION"
        if self.optimization and self.optimization.get("status") == "NO_ROBUST_REGION":
            return "NO_ROBUST_REGION"
        return "SURFACE_AVAILABLE"

    @property
    def status_title(self) -> str:
        return {
            "ROBUST_REGION": "SUPPORTED ROBUST REGION",
            "NO_ROBUST_REGION": "NO ROBUST REGION",
            "ZERO_SUPPORTED_SURFACE": "DEGENERATE ZERO REFERENCE-RELEVANCE SURFACE",
            "SURFACE_AVAILABLE": "REFERENCE SURFACE AVAILABLE",
            "RESULTS_UNAVAILABLE": "RESULTS NOT AVAILABLE",
        }[self.status_code]

    @property
    def status_detail(self) -> str:
        if self.status_code == "ROBUST_REGION":
            rep = self.optimization.get("representative") if self.optimization else None
            if rep:
                return (
                    "A robust threshold-lag region was identified. Representative configuration: "
                    f"{_fmt_q(rep.get('q'))}, lag {_int(rep.get('lag'))}."
                )
            return "A robust threshold-lag region was identified."
        if self.status_code == "ZERO_SUPPORTED_SURFACE":
            return (
                "WMARMcs is exactly zero throughout the evaluated admissible threshold-lag domain. "
                "This is a degenerate zero reference-relevance surface, not a statistical-significance statement. "
                "Diagnostic WMARM and CI-support diagnostics remain available for interpretation."
            )
        if self.status_code == "NO_ROBUST_REGION":
            return (
                "A confidence-scaled reference surface exists, but no threshold-lag cell satisfies "
                "all robustness criteria."
            )
        if self.status_code == "SURFACE_AVAILABLE":
            return "The reference surface is available. Search-adjusted optimization is absent or incomplete."
        return "No completed reference surface was found in this project."

    @property
    def default_cell_key(self):
        target = None
        if self.optimization:
            target = self.optimization.get("representative") or self.optimization.get("observed_peak")
        if target:
            q, lag = _float(target.get("q")), _int(target.get("lag"))
            if q is not None and lag is not None:
                return q, lag
        candidates = [r for r in self.surface if _float(r.get("wmarm_cs")) is not None]
        if candidates:
            best = min(
                candidates,
                key=lambda r: (-_float(r.get("wmarm_cs")), _float(r.get("q")), _int(r.get("lag"))),
            )
            return _float(best.get("q")), _int(best.get("lag"))
        return None

    def cell(self, q, lag):
        q = _float(q)
        lag = _int(lag)
        for row in self.surface:
            if _float(row.get("q")) == q and _int(row.get("lag")) == lag:
                return row
        return None

    def optimization_cell(self, q, lag) -> dict | None:
        """Return the optimization diagnostics for one surface cell, if available."""
        if not self.optimization:
            return None
        q = _float(q)
        lag = _int(lag)
        for row in self.optimization.get("diagnostics") or []:
            if _float(row.get("q")) == q and _int(row.get("lag")) == lag:
                return row
        return None

    @staticmethod
    def _key_from_dict(row) -> tuple[float, int] | None:
        if not isinstance(row, dict):
            return None
        q, lag = _float(row.get("q")), _int(row.get("lag"))
        return None if q is None or lag is None else (q, lag)

    @property
    def representative_key(self):
        return self._key_from_dict((self.optimization or {}).get("representative"))

    @property
    def observed_peak_key(self):
        return self._key_from_dict((self.optimization or {}).get("observed_peak"))

    @property
    def robust_keys(self) -> set[tuple[float, int]]:
        result = set()
        if not self.optimization:
            return result
        for item in self.optimization.get("robust_region") or []:
            key = self._key_from_dict(item)
            if key:
                result.add(key)
        # Some result versions carry only the per-cell diagnostic flag.
        if not result:
            for item in self.optimization.get("diagnostics") or []:
                if _bool(item.get("robust")):
                    key = self._key_from_dict(item)
                    if key:
                        result.add(key)
        return result

    @property
    def null_cutoff(self):
        if not self.optimization:
            return None
        return _float(
            self.optimization.get("null_cutoff_wmarm_cs", self.optimization.get("null_cutoff"))
        )

    @property
    def surface_coherence(self):
        return _float((self.optimization or {}).get("surface_coherence"))

    @property
    def representative_text(self) -> str:
        rep = self.optimization.get("representative") if self.optimization else None
        if not rep:
            return "—"
        return f"{_fmt_q(rep.get('q'))}, lag {_int(rep.get('lag'))}"

    @property
    def p_max(self):
        return _float(self.optimization.get("p_max")) if self.optimization else None

    @property
    def date_summary(self) -> str:
        if not self.manifest:
            return ""
        data = self.manifest.get("data_summary") or {}
        start, end = data.get("actual_start"), data.get("actual_end")
        count = data.get("date_count")
        if start and end:
            suffix = f" · {count} matched days" if count is not None else ""
            return f"{start} to {end}{suffix}"
        return ""


SECONDARY_SURFACE_METRICS = {
    "Diagnostic WMARM": ("wmarm", "Diagnostic WMARM", "number"),
    "Kcs": ("confidence_support_retention_ratio", "Uncertainty-retention Kcs", "number"),
    "CI-supported population": (
        "ci_excluding_zero_population_share",
        "CI-supported population",
        "percent",
    ),
    "BSA coverage": ("bsa_coverage", "BSA coverage", "percent"),
    "Population coverage": ("population_coverage", "Population coverage", "percent"),
    "LOYO persistence": ("persistence", "LOYO near-optimal persistence", "percent"),
}


def _reference_checkpoint_fallback(project_dir: Path) -> tuple[list[dict], dict | None]:
    """Read committed reference checkpoints when exports are unavailable.

    ``results.sqlite`` is the authoritative checkpoint store.  Older interrupted
    ``all`` workflows could contain a fully committed reference analysis without
    having reached the final CSV export step.  The Results Workspace is read-only,
    so this fallback opens SQLite in read-only mode and does not resume, mutate, or
    materialize any analysis result.
    """
    db_path = project_dir / "results.sqlite"
    if not db_path.is_file():
        return [], None
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        db = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        return [], None
    try:
        workflow_row = db.execute(
            "SELECT payload FROM records WHERE kind=? AND scope=? AND key=?",
            ("workflow", "REFERENCE", "result"),
        ).fetchone()
        if workflow_row is None:
            return [], None
        try:
            workflow = json.loads(workflow_row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return [], None
        if str(workflow.get("status") or "").upper() != "COMPLETED":
            return [], None

        surface = []
        for (payload,) in db.execute(
            "SELECT payload FROM records WHERE kind=? AND scope=? ORDER BY key",
            ("cell", "REFERENCE"),
        ):
            try:
                surface.append(_surface_row(json.loads(payload)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

        optimization = None
        row = db.execute(
            "SELECT payload FROM records WHERE kind=? AND scope=? AND key=?",
            ("optimization", "REFERENCE", "result"),
        ).fetchone()
        if row is not None:
            try:
                optimization = json.loads(row[0])
            except (TypeError, ValueError, json.JSONDecodeError):
                optimization = None
        return surface, optimization
    except sqlite3.Error:
        return [], None
    finally:
        db.close()


def load_overview(project_dir: str | Path) -> OverviewModel:
    project_dir = Path(project_dir).expanduser().resolve()
    results = project_dir / "results"
    surface_path = results / "reference_surface.csv"
    surface = []
    if surface_path.is_file():
        with surface_path.open("r", encoding="utf-8-sig", newline="") as handle:
            surface = [_surface_row(row) for row in csv.DictReader(handle)]
    optimization = None
    optimization_path = results / "reference_optimization.json"
    if optimization_path.is_file():
        optimization = json.loads(optimization_path.read_text(encoding="utf-8"))

    # A completed reference may predate the module-level export fix, or an
    # ``all`` workflow may have stopped in a later module before the historical
    # final export.  In either case use the committed authoritative checkpoints.
    if not surface:
        surface, checkpoint_optimization = _reference_checkpoint_fallback(project_dir)
        if optimization is None:
            optimization = checkpoint_optimization

    manifest = None
    manifest_path = project_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not surface:
        raise FileNotFoundError(f"No completed reference surface found in {results}")
    return OverviewModel(
        project_dir=project_dir,
        surface=surface,
        optimization=optimization,
        manifest=manifest,
    )

CALIBRATION_CLASS_ORDER = (
    "exact_null",
    "negligible",
    "moderate_protective",
    "moderate_harmful",
    "strong_protective",
    "strong_harmful",
)

CALIBRATION_METRICS = {
    "WMARMcs": ("wmarm_cs", "WMARMcs", "number"),
    "Diagnostic WMARM": ("wmarm", "Diagnostic WMARM", "number"),
    "Kcs": ("confidence_support_retention_ratio", "Uncertainty-retention Kcs", "number"),
    "CI-supported population": ("ci_share", "CI-supported population", "percent"),
    "Directional balance (CS)": ("directional_balance_cs", "Confidence-supported directional balance", "number"),
}


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _reference_result_rows(project_dir: Path, kind: str, export_name: str) -> list[dict]:
    """Load persisted REFERENCE rows without recomputing the analysis.

    Existing projects normally provide ``results/reference_*.csv``.  Projects
    produced by older builds, or interrupted after the reference stage but before
    export, can still contain the authoritative rows in ``results.sqlite``.  This
    helper is deliberately read-only and therefore makes the new Results Workspace
    views usable on results computed in an earlier run.
    """
    rows = _read_csv_rows(project_dir / "results" / export_name)
    if rows:
        return rows
    db_path = project_dir / "results.sqlite"
    if not db_path.is_file():
        return []
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        db = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        return []
    try:
        result = []
        for (payload,) in db.execute(
            "SELECT payload FROM records WHERE kind=? AND scope=? ORDER BY key",
            (str(kind), "REFERENCE"),
        ):
            try:
                item = json.loads(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                result.append(item)
        return result
    except sqlite3.Error:
        return []
    finally:
        db.close()


def _human_class(name: str) -> str:
    return str(name or "").replace("_", " ").strip().title()


def _human_mode(name: str) -> str:
    return "LINEAR RAMP" if str(name) == "LINEAR_RAMP" else str(name or "—")


def _json_object(value) -> dict | None:
    """Return a JSON object stored directly or serialized in a CSV cell."""
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def distribution_summary(values) -> dict:
    """Descriptive distribution summary used by synthetic signal inspection."""
    import numpy as np

    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    if not finite.size:
        return {
            "valid_count": 0,
            "missing_count": int(a.size),
            "mean": None,
            "sd": None,
            "minimum": None,
            "q025": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q975": None,
            "maximum": None,
            "iqr": None,
        }
    q025, q25, median, q75, q975 = np.quantile(
        finite, [0.025, 0.25, 0.5, 0.75, 0.975], method="linear"
    )
    return {
        "valid_count": int(finite.size),
        "missing_count": int(a.size - finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
        "minimum": float(np.min(finite)),
        "q025": float(q025),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q975": float(q975),
        "maximum": float(np.max(finite)),
        "iqr": float(q75 - q25),
    }


def signal_comparison_diagnostics(observed, synthetic) -> dict:
    """Paired and distributional diagnostics for two equal-length signals.

    These are descriptive diagnostics only. ``ecdf_max_gap`` is the maximum
    absolute difference between the two empirical CDFs, not a hypothesis test.
    """
    import numpy as np

    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    if observed.shape != synthetic.shape:
        raise ValueError("Observed and synthetic signals must have the same shape")
    paired = np.isfinite(observed) & np.isfinite(synthetic)
    if np.any(paired):
        delta = synthetic[paired] - observed[paired]
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta * delta)))
        if np.sum(paired) >= 2 and np.std(observed[paired]) > 0 and np.std(synthetic[paired]) > 0:
            correlation = float(np.corrcoef(observed[paired], synthetic[paired])[0, 1])
        else:
            correlation = None
    else:
        mae = rmse = correlation = None

    a = np.sort(observed[np.isfinite(observed)])
    b = np.sort(synthetic[np.isfinite(synthetic)])
    if a.size and b.size:
        grid = np.unique(np.concatenate((a, b)))
        fa = np.searchsorted(a, grid, side="right") / a.size
        fb = np.searchsorted(b, grid, side="right") / b.size
        ecdf_gap = float(np.max(np.abs(fa - fb)))
    else:
        ecdf_gap = None
    return {
        "paired_count": int(np.sum(paired)),
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "ecdf_max_gap": ecdf_gap,
    }


def reconstruct_synthetic_outcome(project_dir: str | Path, metadata: dict):
    """Reconstruct one calibration outcome in an isolated temporary workspace.

    The implementation intentionally reuses :class:`Calibration.generate`, so
    the Results Workspace does not maintain a second synthetic-data generator.
    The reconstructed file must match the SHA-256 stored by the original run.
    """
    import numpy as np

    from .calibration import Calibration
    from .data import Data
    from .engine import FixedParametersEngine, SurfaceRunner
    from .optimization import Optimizer
    from .storage import Store, sha256_file

    project_dir = Path(project_dir).expanduser().resolve()
    config_path = project_dir / "resolved_config.json"
    canonical = project_dir / "canonical"
    if not config_path.is_file() or not (canonical / "metadata.json").is_file():
        raise FileNotFoundError("Synthetic inspection requires resolved_config.json and canonical project data")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run = str(metadata.get("run") or "").strip()
    class_specification = _json_object(metadata.get("class_specification"))
    q = _float(metadata.get("q_generating"))
    lag = _int(metadata.get("lag_generating"))
    if not run or class_specification is None or q is None:
        raise ValueError("Synthetic generation metadata are incomplete")
    response_mode = metadata.get("threshold_lag_response_mode")
    if response_mode in (None, "", "NOT_APPLICABLE"):
        response_mode = None
    rng_identity = str(metadata.get("paired_rng_identity") or run)
    expected_sha = str(metadata.get("synthetic_file_sha256") or "").strip()

    with tempfile.TemporaryDirectory(prefix="aphreh_results_synthetic_") as tmp:
        work = Path(tmp)
        store = Store(work / "reconstruction.sqlite")
        try:
            data = Data(canonical)
            engine = FixedParametersEngine(data, config, store, work, lambda message: None)
            runner = SurfaceRunner(engine)
            calibration = Calibration(runner, Optimizer(runner))

            # Reuse retained scaffolds when possible, but copy them into the
            # temporary workspace so threshold-response files can never modify
            # the loaded project.
            source_scaffold = project_dir / "calibration_scaffolds" / f"q{q:g}"
            target_scaffold = work / "calibration_scaffolds" / f"q{q:g}"
            if (source_scaffold / "complete.json").is_file():
                target_scaffold.mkdir(parents=True, exist_ok=True)
                for name in ("complete.json", "baseline.npy", "half_width.npy", "donors.npy"):
                    source = source_scaffold / name
                    if source.is_file():
                        shutil.copy2(source, target_scaffold / name)

            path = calibration.generate(
                run,
                class_specification,
                q,
                lag,
                response_mode=response_mode,
                rng_identity=rng_identity,
            )
            actual_sha = sha256_file(path)
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    "Reconstructed synthetic SHA-256 does not match the stored calibration realization"
                )
            outcome = np.array(np.load(path, mmap_mode="r", allow_pickle=False), dtype=float)
            return outcome, actual_sha
        finally:
            store.close()


@dataclass
class CalibrationModel:
    project_dir: Path
    summary: dict | None
    distributions: list[dict]
    operating: list[dict]
    ordering: list[dict]
    recovery: list[dict]
    mean_surface: list[dict]
    instances: list[dict]
    generation: list[dict]
    synthetic_support: list[dict]
    review: dict | None = None

    @property
    def available(self) -> bool:
        return bool(self.summary) or bool(self.distributions) or bool(self.recovery) or bool(self.mean_surface)

    @property
    def status(self) -> str:
        return str((self.summary or {}).get("status") or ("AVAILABLE" if self.available else "NOT_AVAILABLE"))

    @property
    def integrity_status(self) -> str:
        return str((self.review or {}).get("calibration_integrity_status") or "NOT_REVIEWED")

    @property
    def concerns(self) -> list[dict]:
        return list((self.review or {}).get("concerns_requiring_inspection") or [])

    @property
    def reference_metric(self) -> str:
        return str((self.summary or {}).get("reference_territory_metric") or "WMARM_CS")

    @property
    def diagnostic_metric(self) -> str:
        return str((self.summary or {}).get("diagnostic_territory_metric") or "WMARM")

    @property
    def calibration_threshold(self):
        return _float((self.summary or {}).get("calibration_threshold"))

    @property
    def realizations_per_scenario(self):
        return _int((self.summary or {}).get("realizations_per_scenario"))

    @property
    def primary_mode(self) -> str | None:
        value = (self.summary or {}).get("primary_threshold_lag_response_mode")
        if value:
            return str(value)
        modes = self.response_modes
        return modes[0] if modes else None

    @property
    def response_modes(self) -> list[str]:
        modes = list((self.summary or {}).get("threshold_lag_response_modes") or [])
        if not modes:
            modes = [str(r.get("threshold_lag_response_mode")) for r in self.recovery if r.get("threshold_lag_response_mode")]
        result = []
        for mode in modes:
            if mode not in result:
                result.append(mode)
        return result

    @property
    def generating_pairs(self) -> list[tuple[float, int]]:
        pairs = []
        for raw in (self.summary or {}).get("generating_pairs") or []:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                q, lag = _float(raw[0]), _int(raw[1])
                if q is not None and lag is not None and (q, lag) not in pairs:
                    pairs.append((q, lag))
        if not pairs:
            for row in self.recovery:
                q, lag = _float(row.get("q_generating")), _int(row.get("lag_generating"))
                if q is not None and lag is not None and (q, lag) not in pairs:
                    pairs.append((q, lag))
        return sorted(pairs, key=lambda x: (x[0], x[1]))

    @property
    def effect_classes(self) -> list[str]:
        found = []
        for row in self.operating + self.distributions:
            name = row.get("class")
            if name and name not in found:
                found.append(name)
        ordered = [name for name in CALIBRATION_CLASS_ORDER if name in found]
        ordered.extend(name for name in found if name not in ordered)
        return ordered

    @property
    def synthetic_runs(self) -> list[dict]:
        """Completed seed-reconstructible calibration outcomes."""
        def sort_key(row):
            class_name = str(row.get("class") or "")
            try:
                class_rank = CALIBRATION_CLASS_ORDER.index(class_name)
            except ValueError:
                class_rank = len(CALIBRATION_CLASS_ORDER)
            lag = _int(row.get("lag_generating"))
            return (
                0 if lag is None else 1,
                class_rank,
                str(row.get("threshold_lag_response_mode") or ""),
                _float(row.get("q_generating")) or 0.0,
                -1 if lag is None else lag,
                str(row.get("run") or ""),
            )
        return sorted((dict(row) for row in self.generation if row.get("run")), key=sort_key)

    def synthetic_run(self, run: str) -> dict | None:
        for row in self.generation:
            if str(row.get("run")) == str(run):
                return row
        return None

    def synthetic_support_rows(self, run: str) -> list[dict]:
        return [row for row in self.synthetic_support if str(row.get("run")) == str(run)]

    def negligible_reference_boundary(self, metric: str = "wmarm_cs", quantile: str = "q975"):
        """Upper empirical reference limit for the configured negligible class.

        This is a descriptive calibration reference only. It does not alter the
        estimator, null calibration, optimization rules, or inferential status.
        """
        row = self.territory_distribution("negligible", metric)
        return None if row is None else _float(row.get(quantile))

    def territory_distribution(self, class_name: str, metric: str) -> dict | None:
        for row in self.distributions:
            if row.get("class") == class_name and row.get("metric") == metric and not row.get("bsa_id") and not row.get("year"):
                return row
        return None

    def operating_row(self, class_name: str) -> dict | None:
        for row in self.operating:
            if row.get("class") == class_name:
                return row
        return None

    def ordering_row(self, direction: str) -> dict | None:
        for row in self.ordering:
            if str(row.get("direction")) == direction:
                return row
        return None

    def recovery_row(self, mode: str, q_generating, lag_generating) -> dict | None:
        qg, lg = _float(q_generating), _int(lag_generating)
        for row in self.recovery:
            if (
                row.get("threshold_lag_response_mode") == mode
                and _float(row.get("q_generating")) == qg
                and _int(row.get("lag_generating")) == lg
            ):
                return row
        return None

    def mean_surface_rows(self, mode: str, q_generating, lag_generating) -> list[dict]:
        qg, lg = _float(q_generating), _int(lag_generating)
        result = []
        for raw in self.mean_surface:
            if (
                raw.get("threshold_lag_response_mode") == mode
                and _float(raw.get("q_generating")) == qg
                and _int(raw.get("lag_generating")) == lg
            ):
                row = dict(raw)
                for key in (
                    "q", "lag", "q_generating", "lag_generating", "mean", "median", "sd",
                    "q025", "q25", "q75", "q975", "valid_count", "wmarm_cs_mean",
                    "wmarm_cs_median", "wmarm_cs_sd", "wmarm_cs_q025", "wmarm_cs_q25",
                    "wmarm_cs_q75", "wmarm_cs_q975", "wmarm_cs_valid_count",
                    "diagnostic_wmarm_mean", "diagnostic_wmarm_median", "diagnostic_wmarm_sd",
                    "diagnostic_wmarm_q025", "diagnostic_wmarm_q25", "diagnostic_wmarm_q75",
                    "diagnostic_wmarm_q975", "diagnostic_wmarm_valid_count",
                ):
                    if key in row:
                        row[key] = _float(row.get(key))
                if row.get("lag") is not None:
                    row["lag"] = int(row["lag"])
                result.append(row)
        return sorted(result, key=lambda r: (_float(r.get("q")) or 0, _int(r.get("lag")) or 0))

    def mean_surface_cell(self, mode: str, q_generating, lag_generating, q, lag) -> dict | None:
        q, lag = _float(q), _int(lag)
        for row in self.mean_surface_rows(mode, q_generating, lag_generating):
            if _float(row.get("q")) == q and _int(row.get("lag")) == lag:
                return row
        return None

    def mode_summary(self, mode: str) -> dict:
        rows = [r for r in self.recovery if r.get("threshold_lag_response_mode") == mode]
        def valid_count(row):
            value = _int(row.get("valid_realizations"))
            return (_int(row.get("total_realizations")) or 0) if value is None else value

        total = sum(_int(r.get("total_realizations")) or 0 for r in rows)
        valid = sum(valid_count(r) for r in rows)
        invalid = sum(
            (_int(r.get("invalid_realizations"))
             if _int(r.get("invalid_realizations")) is not None
             else max(0, (_int(r.get("total_realizations")) or 0) - valid_count(r)))
            for r in rows
        )
        nonempty = sum(_int(r.get("nonempty_robust_regions")) or 0 for r in rows)
        null_incomplete = sum(_int(r.get("null_incomplete_count")) or 0 for r in rows)

        def weighted(field, weight_field="valid_realizations"):
            numerator = denominator = 0.0
            for row in rows:
                value = _float(row.get(field))
                weight = valid_count(row) if weight_field == "valid_realizations" else _float(row.get(weight_field))
                if value is not None and weight is not None and weight > 0:
                    numerator += value * weight
                    denominator += weight
            return None if denominator <= 0 else numerator / denominator

        role = next((r.get("response_role") for r in rows if r.get("response_role")), None)
        return {
            "mode": mode,
            "response_role": role,
            "generating_pair_count": len(rows),
            "total_realizations": total,
            "valid_realizations": valid,
            "invalid_realizations": invalid,
            "nonempty_robust_regions": nonempty,
            "robust_region_rate": None if valid <= 0 else nonempty / valid,
            "recovery_rate_denominator": "VALID_REALIZATIONS",
            "null_incomplete_count": null_incomplete,
            "threshold_recovered_unconditional": weighted("threshold_recovered_unconditional"),
            "lag_recovered_unconditional": weighted("lag_recovered_unconditional"),
            "joint_recovered_unconditional": weighted("joint_recovered_unconditional"),
            "threshold_recovered_conditional_nonempty": weighted(
                "threshold_recovered_conditional_nonempty", "nonempty_robust_regions"
            ),
            "lag_recovered_conditional_nonempty": weighted(
                "lag_recovered_conditional_nonempty", "nonempty_robust_regions"
            ),
            "joint_recovered_conditional_nonempty": weighted(
                "joint_recovered_conditional_nonempty", "nonempty_robust_regions"
            ),
        }


def load_calibration(project_dir: str | Path) -> CalibrationModel:
    project_dir = Path(project_dir).expanduser().resolve()
    results = project_dir / "results"
    summary = None
    path = results / "calibration_summary.json"
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
    review = None
    path = results / "calibration_review.json"
    if path.is_file():
        review = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationModel(
        project_dir=project_dir,
        summary=summary,
        distributions=_read_csv_rows(results / "calibration_distribution.csv"),
        operating=_read_csv_rows(results / "calibration_operating_characteristics.csv"),
        ordering=_read_csv_rows(results / "calibration_ordering.csv"),
        recovery=_read_csv_rows(results / "calibration_recovery.csv"),
        mean_surface=_read_csv_rows(results / "calibration_mean_surface.csv"),
        instances=_read_csv_rows(results / "calibration_instance.csv"),
        generation=_read_csv_rows(results / "synthetic_generation.csv"),
        synthetic_support=_read_csv_rows(results / "synthetic_support.csv"),
        review=review,
    )


@dataclass
class DiagnosticsModel:
    """Read-only diagnostic rows exported by the reference analysis."""

    project_dir: Path
    pooled: list[dict]
    yearly: list[dict]
    loyo: list[dict]

    @property
    def available(self) -> bool:
        return bool(self.pooled) or bool(self.yearly) or bool(self.loyo)

    @staticmethod
    def _matching(rows: list[dict], q, lag) -> list[dict]:
        q, lag = _float(q), _int(lag)
        if q is None or lag is None:
            return []
        return [
            row for row in rows
            if _float(row.get("q")) == q and _int(row.get("lag")) == lag
        ]

    def pooled_rows(self, q, lag) -> list[dict]:
        return sorted(self._matching(self.pooled, q, lag), key=lambda r: str(r.get("bsa_id", "")))

    def yearly_rows(self, q, lag) -> list[dict]:
        return sorted(
            self._matching(self.yearly, q, lag),
            key=lambda r: (_int(r.get("year")) or 0, str(r.get("bsa_id", ""))),
        )

    def loyo_rows(self, q, lag) -> list[dict]:
        return sorted(
            self._matching(self.loyo, q, lag),
            key=lambda r: _int(r.get("excluded_year")) or 0,
        )

    @staticmethod
    def supported_magnitude(row: dict):
        """Revised MF4.5 confidence-scaled magnitude used by WMARMcs."""
        return confidence_scaled_magnitude(
            _float(row.get("pooled_rc")), _float(row.get("ci_low")), _float(row.get("ci_high"))
        )

    @staticmethod
    def ci_excludes_zero(row: dict) -> bool | None:
        return pooled_ci_excludes_zero(_float(row.get("ci_low")), _float(row.get("ci_high")))

    def pooled_supported_count(self, q, lag) -> tuple[int, int]:
        rows = [r for r in self.pooled_rows(q, lag) if _bool(r.get("eligible")) is not False]
        supported = sum(1 for r in rows if self.ci_excludes_zero(r) is True)
        return supported, len(rows)

    def year_summaries(self, q, lag) -> list[dict]:
        grouped: dict[int, list[dict]] = {}
        for row in self.yearly_rows(q, lag):
            year = _int(row.get("year"))
            if year is not None:
                grouped.setdefault(year, []).append(row)

        def mean(rows, field):
            vals = [_float(r.get(field)) for r in rows]
            vals = [v for v in vals if v is not None]
            return None if not vals else sum(vals) / len(vals)

        result = []
        for year, rows in sorted(grouped.items()):
            valid_ratio = []
            for row in rows:
                requested = _float(row.get("bootstrap_requested"))
                valid = _float(row.get("bootstrap_valid"))
                if requested and requested > 0 and valid is not None:
                    valid_ratio.append(valid / requested)
            result.append({
                "year": year,
                "eligible": sum(_bool(r.get("eligible")) is True for r in rows),
                "total": len(rows),
                "bootstrap_validity": None if not valid_ratio else sum(valid_ratio) / len(valid_ratio),
                "n_exposed": mean(rows, "n_exposed"),
                "effective_exposed": mean(rows, "effective_exposed"),
                "baseline_donors": mean(rows, "baseline_donor_count_mean"),
                "missing_incidence": sum(_int(r.get("missing_incidence_days")) or 0 for r in rows),
                "missing_hazard": sum(_int(r.get("missing_local_hazard_days")) or 0 for r in rows),
                "missing_baseline": sum(_int(r.get("unavailable_baseline_days")) or 0 for r in rows),
            })
        return result

    def exclusion_counts(self, q, lag) -> list[tuple[str, str, int]]:
        counts: dict[tuple[str, str], int] = {}
        for stage, rows in (("Yearly RC", self.yearly_rows(q, lag)), ("Pooled BSA", self.pooled_rows(q, lag))):
            for row in rows:
                if _bool(row.get("eligible")) is True:
                    continue
                reason = str(row.get("reason") or "UNSPECIFIED").strip()
                counts[(stage, reason)] = counts.get((stage, reason), 0) + 1
        return [(stage, reason, count) for (stage, reason), count in sorted(counts.items())]

    def data_quality_totals(self, q, lag) -> dict:
        rows = self.yearly_rows(q, lag)
        invalid = [_float(r.get("invalid_proportion")) for r in rows]
        invalid = [v for v in invalid if v is not None]
        return {
            "missing_incidence_days": sum(_int(r.get("missing_incidence_days")) or 0 for r in rows),
            "missing_local_hazard_days": sum(_int(r.get("missing_local_hazard_days")) or 0 for r in rows),
            "unavailable_baseline_days": sum(_int(r.get("unavailable_baseline_days")) or 0 for r in rows),
            "zero_weight_valid_days": sum(_int(r.get("zero_weight_valid_days")) or 0 for r in rows),
            "global_redraws": sum(_int(r.get("global_redraws")) or 0 for r in rows),
            "mean_invalid_bootstrap_proportion": None if not invalid else sum(invalid) / len(invalid),
            "max_invalid_bootstrap_proportion": None if not invalid else max(invalid),
        }


def load_diagnostics(project_dir: str | Path) -> DiagnosticsModel:
    project_dir = Path(project_dir).expanduser().resolve()
    results = project_dir / "results"
    return DiagnosticsModel(
        project_dir=project_dir,
        pooled=_reference_result_rows(project_dir, "pooled", "reference_pooled.csv"),
        yearly=_reference_result_rows(project_dir, "yearly", "reference_yearly.csv"),
        loyo=_reference_result_rows(project_dir, "loyo", "reference_loyo.csv"),
    )



# ---------------------------------------------------------------------------
# Spatial results
# ---------------------------------------------------------------------------

SPATIAL_LAYERS = (
    ("pooled_rc", "Pooled RC"),
    ("absolute_rc", "Absolute RC"),
    ("supported_magnitude", "Confidence-scaled magnitude S_b"),
    ("ci_width", "95% CI width"),
    ("i_squared_percent", "I² heterogeneity (%)"),
    ("population", "Population"),
)
SPATIAL_LAYER_LABELS = dict(SPATIAL_LAYERS)


def _join_key(value) -> str:
    """Stable geometry/result join key while preserving nonnumeric identifiers."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _project_sources(project_dir: Path) -> dict:
    for name in ("resolved_config.json", "project_config.json"):
        path = project_dir / name
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(payload.get("sources"), dict):
                    return payload["sources"]
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _geometry_path(project_dir: Path, sources: dict) -> Path | None:
    geometry_dir = project_dir / "geometry"
    configured = str(sources.get("geometry_file") or "").strip()
    if geometry_dir.is_dir():
        if configured:
            direct = geometry_dir / Path(configured).name
            if direct.is_file():
                return direct
        candidates = sorted(
            (p for p in geometry_dir.iterdir() if p.is_file()),
            key=lambda p: (p.suffix.lower() not in {".gpkg", ".geojson", ".json"}, p.name.lower()),
        )
        if candidates:
            return candidates[0]
    # A project opened before export may still point at the original source.
    data_dir = None
    for name in ("resolved_config.json", "project_config.json"):
        path = project_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            data_dir = payload.get("data_dir")
        except (OSError, json.JSONDecodeError):
            pass
        if data_dir:
            break
    if configured and data_dir:
        candidate = Path(data_dir).expanduser() / configured
        if candidate.is_file():
            return candidate.resolve()
    return None


def geometry_attribute_fields(path: str | Path) -> list[str]:
    """Return non-geometry attribute fields for supported geometry sources.

    This helper is also used by the source-mapping GUI. It deliberately relies
    only on the Python standard library so the results map does not introduce a
    GIS runtime dependency.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        try:
            con = sqlite3.connect(str(path))
            try:
                row = con.execute(
                    "SELECT table_name FROM gpkg_contents WHERE data_type='features' ORDER BY table_name LIMIT 1"
                ).fetchone()
                if not row:
                    return []
                table = row[0]
                geom = con.execute(
                    "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", (table,)
                ).fetchone()
                geom_col = geom[0] if geom else None
                fields = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
                return [f for f in fields if f != geom_col]
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return []
    if suffix in {".geojson", ".json"}:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return []
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            return []
        fields = []
        for feature in features[:200]:
            props = feature.get("properties") if isinstance(feature, dict) else None
            if isinstance(props, dict):
                for key in props:
                    if key not in fields:
                        fields.append(key)
        return fields
    return []


class _WKBReader:
    """Minimal Polygon/MultiPolygon WKB reader used for GeoPackage display."""
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    def _read(self, fmt: str, endian: str):
        size = struct.calcsize(endian + fmt)
        if self.pos + size > len(self.data):
            raise ValueError("Truncated WKB geometry")
        value = struct.unpack_from(endian + fmt, self.data, self.pos)
        self.pos += size
        return value[0] if len(value) == 1 else value

    def geometry(self) -> list[list[tuple[float, float]]]:
        if self.pos >= len(self.data):
            return []
        byte_order = self._read("B", "<")
        endian = "<" if byte_order == 1 else ">"
        raw_type = int(self._read("I", endian))
        has_z = bool(raw_type & 0x80000000)
        has_m = bool(raw_type & 0x40000000)
        has_srid = bool(raw_type & 0x20000000)
        base_type = raw_type & 0x0FFFFFFF
        if 3000 <= base_type < 4000:
            has_z = has_m = True; base_type -= 3000
        elif 2000 <= base_type < 3000:
            has_m = True; base_type -= 2000
        elif 1000 <= base_type < 2000:
            has_z = True; base_type -= 1000
        if has_srid:
            self._read("I", endian)
        dims = 2 + int(has_z) + int(has_m)

        def point():
            coords = [self._read("d", endian) for _ in range(dims)]
            return float(coords[0]), float(coords[1])

        if base_type == 3:  # Polygon
            ring_count = int(self._read("I", endian))
            outer = []
            for ring_index in range(ring_count):
                n = int(self._read("I", endian))
                ring = [point() for _ in range(n)]
                if ring_index == 0:
                    outer = ring
            return [outer] if len(outer) >= 3 else []
        if base_type in (6, 7):  # MultiPolygon or GeometryCollection
            count = int(self._read("I", endian))
            parts = []
            for _ in range(count):
                parts.extend(self.geometry())
            return parts
        if base_type == 1:  # Point
            point(); return []
        if base_type == 2:  # LineString
            n = int(self._read("I", endian))
            for _ in range(n): point()
            return []
        if base_type in (4, 5):  # MultiPoint / MultiLineString
            count = int(self._read("I", endian))
            parts = []
            for _ in range(count):
                parts.extend(self.geometry())
            return parts
        raise ValueError(f"Unsupported WKB geometry type {base_type}")


def _gpkg_parts(blob: bytes) -> list[list[tuple[float, float]]]:
    if not blob or len(blob) < 8 or blob[:2] != b"GP":
        return []
    flags = blob[3]
    envelope_code = (flags >> 1) & 0x07
    envelope_doubles = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}.get(envelope_code)
    if envelope_doubles is None:
        return []
    offset = 8 + envelope_doubles * 8
    if offset >= len(blob):
        return []
    try:
        return _WKBReader(blob[offset:]).geometry()
    except (ValueError, struct.error):
        return []


@dataclass
class SpatialFeature:
    bsa_id: str
    parts: list[list[tuple[float, float]]]

    @property
    def bounds(self):
        points = [pt for part in self.parts for pt in part]
        if not points:
            return None
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)


def _choose_geometry_id_field(columns: list[str], configured: str | None,
                              values_by_field: dict[str, list], expected_ids: set[str]) -> str | None:
    if configured and configured in columns:
        return configured
    best = None; best_overlap = -1
    for field in columns:
        values = {_join_key(v) for v in values_by_field.get(field, []) if _join_key(v)}
        overlap = len(values & expected_ids) if expected_ids else 0
        id_bonus = 0.25 if "id" in field.lower() else 0.0
        score = overlap + id_bonus
        if score > best_overlap:
            best, best_overlap = field, score
    return best


def _load_geojson_features(path: Path, configured_field: str | None,
                           expected_ids: set[str]) -> tuple[list[SpatialFeature], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(raw_features, list):
        raise ValueError("GeoJSON must contain a FeatureCollection")
    columns = geometry_attribute_fields(path)
    values_by_field = {field: [] for field in columns}
    for feature in raw_features:
        props = feature.get("properties") or {}
        for field in columns:
            values_by_field[field].append(props.get(field))
    id_field = _choose_geometry_id_field(columns, configured_field, values_by_field, expected_ids)
    if not id_field:
        raise ValueError("Could not determine a geometry BSA ID field")
    features = []
    for feature in raw_features:
        props = feature.get("properties") or {}
        bid = _join_key(props.get(id_field))
        geom = feature.get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        parts = []
        if gtype == "Polygon" and isinstance(coords, list) and coords:
            parts = [[(float(x), float(y)) for x, y, *_ in coords[0]]]
        elif gtype == "MultiPolygon" and isinstance(coords, list):
            parts = [[(float(x), float(y)) for x, y, *_ in poly[0]] for poly in coords if poly]
        if bid and any(len(part) >= 3 for part in parts):
            features.append(SpatialFeature(bid, [part for part in parts if len(part) >= 3]))
    return features, id_field


def _load_gpkg_features(path: Path, configured_field: str | None,
                        expected_ids: set[str]) -> tuple[list[SpatialFeature], str | None]:
    con = sqlite3.connect(str(path))
    try:
        content = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features' ORDER BY table_name LIMIT 1"
        ).fetchone()
        if not content:
            raise ValueError("GeoPackage contains no feature layer")
        table = content[0]
        geom_row = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", (table,)
        ).fetchone()
        if not geom_row:
            raise ValueError("GeoPackage feature layer has no registered geometry column")
        geom_col = geom_row[0]
        columns = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall() if r[1] != geom_col]
        values_by_field = {}
        for field in columns:
            try:
                values_by_field[field] = [r[0] for r in con.execute(f'SELECT "{field}" FROM "{table}"').fetchall()]
            except sqlite3.Error:
                values_by_field[field] = []
        id_field = _choose_geometry_id_field(columns, configured_field, values_by_field, expected_ids)
        if not id_field:
            raise ValueError("Could not determine a geometry BSA ID field")
        rows = con.execute(f'SELECT "{id_field}", "{geom_col}" FROM "{table}"').fetchall()
    finally:
        con.close()
    features = []
    for raw_id, blob in rows:
        bid = _join_key(raw_id)
        parts = _gpkg_parts(blob) if blob is not None else []
        if bid and parts:
            features.append(SpatialFeature(bid, parts))
    return features, id_field


def load_geometry_features(path: str | Path, configured_field: str | None = None,
                           expected_ids: set[str] | None = None) -> tuple[list[SpatialFeature], str | None]:
    path = Path(path)
    expected_ids = set(expected_ids or ())
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        return _load_gpkg_features(path, configured_field, expected_ids)
    if suffix in {".geojson", ".json"}:
        return _load_geojson_features(path, configured_field, expected_ids)
    raise ValueError(f"Unsupported in-app geometry format: {path.suffix or path.name}. Use GeoPackage or GeoJSON.")


@dataclass
class SpatialModel:
    project_dir: Path
    pooled: list[dict]
    yearly: list[dict]
    geometry_path: Path | None
    geometry_bsa_field: str | None
    features: list[SpatialFeature]
    geometry_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.pooled) or bool(self.yearly)

    @property
    def map_available(self) -> bool:
        return bool(self.features) and not self.geometry_error

    @staticmethod
    def effect_value(row: dict | None):
        if not row:
            return None
        value = _float(row.get("pooled_rc"))
        return _float(row.get("rc")) if value is None else value

    @classmethod
    def supported_magnitude(cls, row: dict | None):
        if not row:
            return None
        return confidence_scaled_magnitude(
            cls.effect_value(row), _float(row.get("ci_low")), _float(row.get("ci_high"))
        )

    def pooled_rows(self, q, lag) -> list[dict]:
        q, lag = _float(q), _int(lag)
        rows = [r for r in self.pooled if _float(r.get("q")) == q and _int(r.get("lag")) == lag]
        return sorted(rows, key=lambda r: _join_key(r.get("bsa_id")))

    def years(self, q, lag) -> list[int]:
        q, lag = _float(q), _int(lag)
        years = {
            _int(r.get("year")) for r in self.yearly
            if _float(r.get("q")) == q and _int(r.get("lag")) == lag and _int(r.get("year")) is not None
        }
        return sorted(years)

    def yearly_rows(self, q, lag, year) -> list[dict]:
        q, lag, year = _float(q), _int(lag), _int(year)
        rows = [
            r for r in self.yearly
            if _float(r.get("q")) == q and _int(r.get("lag")) == lag and _int(r.get("year")) == year
        ]
        return sorted(rows, key=lambda r: _join_key(r.get("bsa_id")))

    def rows(self, q, lag, year=None) -> list[dict]:
        return self.pooled_rows(q, lag) if year is None else self.yearly_rows(q, lag, year)

    def row_lookup(self, q, lag, year=None) -> dict[str, dict]:
        return {_join_key(r.get("bsa_id")): r for r in self.rows(q, lag, year)}

    def row_for_bsa(self, q, lag, bsa_id, year=None) -> dict | None:
        return self.row_lookup(q, lag, year).get(_join_key(bsa_id))

    def layer_value(self, row: dict | None, layer: str):
        if not row:
            return None
        if layer in {"pooled_rc", "yearly_rc"}:
            return self.effect_value(row)
        if layer == "absolute_rc":
            rc = self.effect_value(row); return None if rc is None else abs(rc)
        if layer == "supported_magnitude":
            return self.supported_magnitude(row)
        if layer == "ci_width":
            lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
            return None if lo is None or hi is None else max(0.0, hi - lo)
        return _float(row.get(layer))

    def map_summary(self, q, lag, year=None) -> dict:
        rows = self.rows(q, lag, year)
        lookup = self.row_lookup(q, lag, year)
        feature_ids = {_join_key(f.bsa_id) for f in self.features}
        joined = len(feature_ids & set(lookup))
        eligible = sum(_bool(r.get("eligible")) is True for r in rows)
        supported = sum(
            1 for r in rows
            if _bool(r.get("eligible")) is not False and DiagnosticsModel.ci_excludes_zero(r) is True
        )
        return {
            "geometry_features": len(feature_ids),
            "joined_features": joined,
            "result_bsas": len(rows),
            "eligible_bsas": eligible,
            "supported_bsas": supported,
        }


def load_spatial(project_dir: str | Path) -> SpatialModel:
    project_dir = Path(project_dir).expanduser().resolve()
    pooled = _reference_result_rows(project_dir, "pooled", "reference_pooled.csv")
    yearly = _reference_result_rows(project_dir, "yearly", "reference_yearly.csv")
    expected = {
        _join_key(r.get("bsa_id")) for r in (pooled + yearly) if _join_key(r.get("bsa_id"))
    }
    sources = _project_sources(project_dir)
    path = _geometry_path(project_dir, sources)
    configured = str(sources.get("geometry_bsa_field") or sources.get("bsa_id_field") or "").strip() or None
    if path is None:
        return SpatialModel(project_dir, pooled, yearly, None, configured, [], "No project geometry is available. The BSA results table remains usable.")
    try:
        features, actual_field = load_geometry_features(path, configured, expected)
        error = None if features else "No polygon features could be read from the project geometry."
        return SpatialModel(project_dir, pooled, yearly, path, actual_field, features, error)
    except Exception as exc:
        return SpatialModel(project_dir, pooled, yearly, path, configured, [], str(exc))


# ---------------------------------------------------------------------------
# BSA detail results
# ---------------------------------------------------------------------------

@dataclass
class BsaDetailModel:
    """Read-only pooled and yearly records for one BSA at one q-lag cell."""

    project_dir: Path
    pooled: list[dict]
    yearly: list[dict]

    @property
    def available(self) -> bool:
        return bool(self.pooled) or bool(self.yearly)

    def pooled_rows(self, q, lag) -> list[dict]:
        q, lag = _float(q), _int(lag)
        rows = [r for r in self.pooled if _float(r.get("q")) == q and _int(r.get("lag")) == lag]
        return sorted(rows, key=lambda r: _join_key(r.get("bsa_id")))

    def bsa_ids(self, q, lag) -> list[str]:
        return [_join_key(r.get("bsa_id")) for r in self.pooled_rows(q, lag) if _join_key(r.get("bsa_id"))]

    def pooled_row(self, q, lag, bsa_id) -> dict | None:
        key = _join_key(bsa_id)
        return next((r for r in self.pooled_rows(q, lag) if _join_key(r.get("bsa_id")) == key), None)

    def yearly_rows(self, q, lag, bsa_id) -> list[dict]:
        q, lag, key = _float(q), _int(lag), _join_key(bsa_id)
        rows = [
            r for r in self.yearly
            if _float(r.get("q")) == q and _int(r.get("lag")) == lag and _join_key(r.get("bsa_id")) == key
        ]
        return sorted(rows, key=lambda r: _int(r.get("year")) or 0)

    def default_bsa(self, q, lag) -> str | None:
        rows = self.pooled_rows(q, lag)
        eligible = [r for r in rows if _bool(r.get("eligible")) is True]
        source = eligible or rows
        return _join_key(source[0].get("bsa_id")) if source else None

    @staticmethod
    def supported_magnitude(row: dict | None):
        return None if row is None else DiagnosticsModel.supported_magnitude(row)


def load_bsa_detail(project_dir: str | Path) -> BsaDetailModel:
    project_dir = Path(project_dir).expanduser().resolve()
    return BsaDetailModel(
        project_dir=project_dir,
        pooled=_reference_result_rows(project_dir, "pooled", "reference_pooled.csv"),
        yearly=_reference_result_rows(project_dir, "yearly", "reference_yearly.csv"),
    )


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def _mix_color(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    aa, bb = _hex_rgb(a), _hex_rgb(b)
    rgb = tuple(round(x + (y - x) * t) for x, y in zip(aa, bb))
    return "#%02x%02x%02x" % rgb


def _spatial_color(value, minimum, maximum, signed=False) -> str:
    value = _float(value)
    if value is None:
        return "#d8d8d8"
    if signed:
        extent = max(abs(minimum or 0), abs(maximum or 0))
        if extent <= 0:
            return "#f7f7f7"
        ratio = max(-1.0, min(1.0, value / extent))
        return _mix_color("#f7f7f7", "#b2182b", ratio) if ratio >= 0 else _mix_color("#f7f7f7", "#2166ac", -ratio)
    if minimum is None or maximum is None or maximum <= minimum:
        return "#5b8db8"
    ratio = (value - minimum) / (maximum - minimum)
    return _mix_color("#eef4f8", "#285f8f", ratio)


class SpatialMapCanvas(tk.Canvas):
    def __init__(self, parent, on_select):
        super().__init__(parent, background="#fbfbfb", highlightthickness=1, highlightbackground="#b8b8b8")
        self.on_select = on_select
        self.features = []
        self.lookup = {}
        self.layer = "pooled_rc"
        self.selected_bsa = None
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<Button-1>", self._clicked)

    def set_data(self, features, lookup, layer, selected_bsa=None):
        self.features = features or []
        self.lookup = lookup or {}
        self.layer = layer
        self.selected_bsa = _join_key(selected_bsa) if selected_bsa is not None else None
        self.redraw()

    def _clicked(self, event):
        item = self.find_closest(event.x, event.y)
        if not item:
            return
        tags = self.gettags(item[0])
        for tag in tags:
            if tag.startswith("bsa::"):
                self.on_select(tag[5:]); return

    @staticmethod
    def _screen_points(part, transform):
        points = []
        last = None
        for x, y in part:
            sx, sy = transform(x, y)
            candidate = (round(sx, 1), round(sy, 1))
            if last is None or abs(candidate[0]-last[0]) >= 0.7 or abs(candidate[1]-last[1]) >= 0.7:
                points.extend(candidate); last = candidate
        return points

    @staticmethod
    def _layer_value(row: dict | None, layer: str):
        if not row:
            return None
        if layer in {"pooled_rc", "yearly_rc"}:
            return SpatialModel.effect_value(row)
        if layer == "absolute_rc":
            rc = SpatialModel.effect_value(row); return None if rc is None else abs(rc)
        if layer == "supported_magnitude":
            return SpatialModel.supported_magnitude(row)
        if layer == "ci_width":
            lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
            return None if lo is None or hi is None else max(0.0, hi - lo)
        return _float(row.get(layer))

    def redraw(self):
        self.delete("all")
        if not self.features:
            self.create_text(max(10, self.winfo_width()/2), max(10, self.winfo_height()/2),
                             text="Spatial geometry unavailable\nBSA results remain available in the table.",
                             justify="center", fill="#555555", font=("Segoe UI", 11))
            return
        bounds = [f.bounds for f in self.features if f.bounds]
        if not bounds:
            return
        xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
        xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)
        width = max(100, self.winfo_width()); height = max(100, self.winfo_height())
        pad = 24
        dx, dy = max(xmax-xmin, 1e-12), max(ymax-ymin, 1e-12)
        scale = min((width-2*pad)/dx, (height-2*pad)/dy)
        ox = (width - dx*scale)/2; oy = (height - dy*scale)/2
        def transform(x, y): return ox+(x-xmin)*scale, height-(oy+(y-ymin)*scale)
        values = []
        for feature in self.features:
            row = self.lookup.get(_join_key(feature.bsa_id))
            value = self._layer_value(row, self.layer)
            if value is not None:
                values.append(value)
        minimum, maximum = (min(values), max(values)) if values else (None, None)
        signed = self.layer in {"pooled_rc", "yearly_rc"}
        for feature in self.features:
            bid = _join_key(feature.bsa_id); row = self.lookup.get(bid)
            value = self._layer_value(row, self.layer)
            fill = _spatial_color(value, minimum, maximum, signed)
            selected = bid == self.selected_bsa
            for part in feature.parts:
                coords = self._screen_points(part, transform)
                if len(coords) >= 6:
                    self.create_polygon(coords, fill=fill, outline="#202020" if selected else "#ffffff",
                                        width=2.5 if selected else 0.8, tags=(f"bsa::{bid}",))
        # Compact continuous legend.
        lx0, ly0, lx1, ly1 = 28, height-24, min(width-28, 248), height-12
        if values and lx1 > lx0:
            steps = 60
            for i in range(steps):
                t=i/(steps-1)
                if signed:
                    val=(2*t-1)*max(abs(minimum or 0),abs(maximum or 0))
                else:
                    val=(minimum or 0)+t*((maximum or 0)-(minimum or 0))
                color=_spatial_color(val,minimum,maximum,signed)
                x0=lx0+(lx1-lx0)*i/steps; x1=lx0+(lx1-lx0)*(i+1)/steps
                self.create_rectangle(x0,ly0,x1,ly1,fill=color,outline=color)
            self.create_text(lx0,ly0-3,text=_fmt_number(minimum,3),anchor="sw",fill="#333333",font=("Segoe UI",8))
            self.create_text(lx1,ly0-3,text=_fmt_number(maximum,3),anchor="se",fill="#333333",font=("Segoe UI",8))


class SpatialPanel(ttk.Frame):
    ANNUAL_LAYERS = (
        ("yearly_rc", "Yearly RC (RCy)"),
        ("absolute_rc", "Absolute RCy"),
        ("supported_magnitude", "Confidence-scaled magnitude"),
        ("ci_width", "95% CI width"),
        ("population", "Population"),
    )

    def __init__(self, parent, overview_model: OverviewModel, model: SpatialModel, on_select_bsa=None):
        super().__init__(parent, padding=16)
        self.overview_model = overview_model
        self.model = model
        self.on_select_bsa = on_select_bsa
        self.selected = overview_model.default_cell_key
        self.selected_bsa = None
        self.selected_year = None
        self.period_label = tk.StringVar(value="Pooled")
        self.layer = tk.StringVar(value="pooled_rc")
        self.layer_label = tk.StringVar(value=SPATIAL_LAYER_LABELS["pooled_rc"])
        self.cards = {}
        self.detail_vars = {}
        self._tree_items = {}
        self._build()
        if self.selected:
            self.set_selected_cell(*self.selected)

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="x")
        ttk.Label(top, text="Spatial", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.cell_label = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.cell_label).pack(side="left", padx=(12,0))
        control = ttk.Frame(top); control.pack(side="right")
        ttk.Label(control, text="Estimate").pack(side="left", padx=(0,6))
        self.period_combo = ttk.Combobox(control, textvariable=self.period_label, state="readonly", width=12)
        self.period_combo.pack(side="left", padx=(0,12))
        self.period_combo.bind("<<ComboboxSelected>>", self._period_changed)
        ttk.Label(control,text="Map layer").pack(side="left",padx=(0,6))
        self.layer_combo=ttk.Combobox(control,textvariable=self.layer_label,state="readonly",width=27)
        self.layer_combo.pack(side="left")
        self.layer_combo.bind("<<ComboboxSelected>>", self._layer_changed)

        self.geometry_note = tk.StringVar(value="")
        ttk.Label(self,textvariable=self.geometry_note,wraplength=1180,justify="left").pack(fill="x",pady=(5,9))

        cards=ttk.Frame(self); cards.pack(fill="x",pady=(0,10))
        for key,title in (("features","Geometry BSAs"),("joined","Joined BSAs"),("eligible","Eligible estimates"),("supported","CI-supported")):
            frame=ttk.LabelFrame(cards,text=title,padding=8); frame.pack(side="left",fill="x",expand=True,padx=(0,8))
            var=tk.StringVar(value="—"); ttk.Label(frame,textvariable=var,font=("Segoe UI",15,"bold")).pack(anchor="w")
            self.cards[key]=var

        body=ttk.Panedwindow(self,orient="horizontal"); body.pack(fill="both",expand=True)
        map_frame=ttk.LabelFrame(body,text="BSA map",padding=6)
        side=ttk.Frame(body,padding=(10,0,0,0)); body.add(map_frame,weight=3); body.add(side,weight=2)
        self.map=SpatialMapCanvas(map_frame,self.select_bsa); self.map.pack(fill="both",expand=True)

        detail=ttk.LabelFrame(side,text="Selected BSA",padding=9); detail.pack(fill="x")
        detail_specs=(
            ("bsa","BSA"),("rc","RC / RCy"),("ci","95% CI"),("support","Confidence-scaled magnitude"),
            ("population","Population"),("years","Year / contributing years"),("tau","Tau²"),("i2","I²"),
            ("eligible","Eligible"),("reason","Reason"),
        )
        for r,(key,label) in enumerate(detail_specs):
            ttk.Label(detail,text=label).grid(row=r,column=0,sticky="w",padx=(0,12),pady=1)
            var=tk.StringVar(value="—"); self.detail_vars[key]=var
            ttk.Label(detail,textvariable=var).grid(row=r,column=1,sticky="w",pady=1)

        table_frame=ttk.LabelFrame(side,text="BSA results",padding=6); table_frame.pack(fill="both",expand=True,pady=(10,0))
        cols=("bsa","value","rc","support","ci","eligible")
        self.tree=ttk.Treeview(table_frame,columns=cols,show="headings",height=14,selectmode="browse")
        labels={"bsa":"BSA","value":"Layer value","rc":"RC / RCy","support":"Supported magnitude","ci":"95% CI","eligible":"Eligible"}
        widths={"bsa":75,"value":90,"rc":85,"support":105,"ci":150,"eligible":65}
        for c in cols:
            self.tree.heading(c,text=labels[c]); self.tree.column(c,width=widths[c],anchor="center")
        yscroll=ttk.Scrollbar(table_frame,orient="vertical",command=self.tree.yview); self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left",fill="both",expand=True); yscroll.pack(side="right",fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)

    def _period_changed(self, event=None):
        text = str(self.period_label.get() or "Pooled")
        self.selected_year = None if text == "Pooled" else _int(text)
        self._refresh_layer_options()
        self._refresh_view()

    def _layer_options(self):
        return SPATIAL_LAYERS if self.selected_year is None else self.ANNUAL_LAYERS

    def _refresh_layer_options(self):
        options = self._layer_options()
        keys = [key for key, _ in options]
        labels = [label for _, label in options]
        self.layer_combo.configure(values=labels)
        if self.layer.get() not in keys:
            self.layer.set(keys[0])
        self.layer_label.set(dict(options)[self.layer.get()])

    def _layer_changed(self,event=None):
        label=self.layer_label.get()
        for key,known in self._layer_options():
            if known==label:
                self.layer.set(key); break
        self._refresh_view()

    def set_selected_cell(self,q,lag):
        if not self.overview_model.cell(q,lag): return
        self.selected=(_float(q),_int(lag))
        years = self.model.years(*self.selected)
        values = ["Pooled"] + [str(year) for year in years]
        self.period_combo.configure(values=values)
        if self.selected_year not in years:
            self.selected_year = None
            self.period_label.set("Pooled")
        else:
            self.period_label.set(str(self.selected_year))
        self._refresh_layer_options()
        rows=self.model.rows(*self.selected,self.selected_year)
        if self.selected_bsa is None or self.model.row_for_bsa(*self.selected,self.selected_bsa,self.selected_year) is None:
            eligible=[r for r in rows if _bool(r.get("eligible")) is True]
            source=eligible or rows
            self.selected_bsa=_join_key(source[0].get("bsa_id")) if source else None
        self._refresh_view()

    def _refresh_view(self):
        if not self.selected: return
        q,lag=self.selected
        rows=self.model.rows(q,lag,self.selected_year)
        lookup=self.model.row_lookup(q,lag,self.selected_year)
        summary=self.model.map_summary(q,lag,self.selected_year)
        suffix = "Pooled" if self.selected_year is None else f"RCy {self.selected_year}"
        self.cell_label.set(f"{_fmt_q(q)}, lag {_int(lag)} · {suffix}")
        self.cards["features"].set(str(summary["geometry_features"]) if self.model.geometry_path else "—")
        self.cards["joined"].set(f"{summary['joined_features']} / {summary['result_bsas']}" if self.model.geometry_path else "—")
        self.cards["eligible"].set(f"{summary['eligible_bsas']} / {summary['result_bsas']}")
        self.cards["supported"].set(f"{summary['supported_bsas']} / {summary['eligible_bsas']}" if summary['eligible_bsas'] else "0")
        if self.model.map_available:
            self.geometry_note.set(
                f"Geometry: {self.model.geometry_path.name} · join field: {self.model.geometry_bsa_field} · display: {suffix}"
            )
        else:
            self.geometry_note.set(self.model.geometry_error or "Spatial geometry unavailable.")
        if self.selected_bsa not in lookup:
            eligible=[r for r in rows if _bool(r.get("eligible")) is True]
            source=eligible or rows
            self.selected_bsa=_join_key(source[0].get("bsa_id")) if source else None
        self.map.set_data(self.model.features,lookup,self.layer.get(),self.selected_bsa)
        current_selection=self.selected_bsa
        for item in self.tree.get_children(): self.tree.delete(item)
        self._tree_items={}
        for row in rows:
            bid=_join_key(row.get("bsa_id")); value=self.model.layer_value(row,self.layer.get()); support=self.model.supported_magnitude(row)
            lo,hi=_float(row.get("ci_low")),_float(row.get("ci_high"))
            ci="—" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"
            rc=self.model.effect_value(row)
            item=self.tree.insert("","end",values=(bid,_fmt_number(value,3),_fmt_number(rc,3),_fmt_number(support,3),ci,_fmt_bool(row.get("eligible"))))
            self._tree_items[bid]=item
        if current_selection in self._tree_items:
            item=self._tree_items[current_selection]; self.tree.selection_set(item); self.tree.focus(item); self.tree.see(item)
        self._update_detail()

    def _tree_selected(self,event=None):
        selection=self.tree.selection()
        if not selection: return
        values=self.tree.item(selection[0],"values")
        if values: self.select_bsa(values[0],from_tree=True)

    def select_bsa(self,bsa_id,from_tree=False,notify=True):
        bid=_join_key(bsa_id)
        if not self.selected or self.model.row_for_bsa(*self.selected,bid,self.selected_year) is None: return
        changed = bid != self.selected_bsa
        self.selected_bsa=bid
        if not from_tree and bid in self._tree_items:
            item=self._tree_items[bid]; self.tree.selection_set(item); self.tree.focus(item); self.tree.see(item)
        self.map.selected_bsa=bid; self.map.redraw(); self._update_detail()
        if notify and changed and self.on_select_bsa:
            self.on_select_bsa(bid)

    def set_selected_bsa(self,bsa_id):
        self.select_bsa(bsa_id, notify=False)

    def _update_detail(self):
        row=self.model.row_for_bsa(*self.selected,self.selected_bsa,self.selected_year) if self.selected and self.selected_bsa else None
        if not row:
            for var in self.detail_vars.values(): var.set("—")
            return
        lo,hi=_float(row.get("ci_low")),_float(row.get("ci_high")); support=self.model.supported_magnitude(row)
        if self.selected_year is None:
            year_text = str(_int(row.get("contributing_years"))) if _int(row.get("contributing_years")) is not None else "—"
            tau = _fmt_number(row.get("tau_squared"),4)
            i2 = _fmt_number(row.get("i_squared_percent"),1)
        else:
            year_text = str(self.selected_year)
            tau = "—"
            i2 = "—"
        values={
            "bsa":_join_key(row.get("bsa_id")),"rc":_fmt_number(self.model.effect_value(row),4),
            "ci":"—" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]",
            "support":_fmt_number(support,4),"population":_fmt_number(row.get("population"),0),
            "years":year_text,"tau":tau,"i2":i2,
            "eligible":_fmt_bool(row.get("eligible")),"reason":str(row.get("reason") or "—"),
        }
        for key,var in self.detail_vars.items(): var.set(values[key])


class BsaForestPlot(tk.Canvas):
    """Dependency-free forest plot for yearly bootstrap CIs and pooled HK CI."""

    def __init__(self, parent):
        super().__init__(parent, background="white", highlightthickness=1, highlightbackground="#d0d0d0", height=300)
        self.yearly = []
        self.pooled = None
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, yearly: list[dict], pooled: dict | None):
        self.yearly = list(yearly)
        self.pooled = pooled
        self.redraw()

    @staticmethod
    def _interval(row, pooled=False):
        rc = _float(row.get("pooled_rc" if pooled else "rc"))
        lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
        return rc, lo, hi

    def redraw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), 720), max(self.winfo_height(), 280)
        records = []
        for row in self.yearly:
            rc, lo, hi = self._interval(row)
            records.append((str(_int(row.get("year")) or row.get("year") or "—"), rc, lo, hi, False, row))
        if self.pooled:
            rc, lo, hi = self._interval(self.pooled, pooled=True)
            records.append(("Pooled", rc, lo, hi, True, self.pooled))
        finite = []
        for _, rc, lo, hi, _, row in records:
            finite.extend(v for v in (rc, lo, hi) if v is not None)
            if row is self.pooled:
                finite.extend(v for v in (_float(row.get("prediction_low")), _float(row.get("prediction_high"))) if v is not None)
        if not records or not finite:
            self.create_text(w/2, h/2, text="No estimable yearly or pooled RC values for this BSA and cell.", fill="#555555", font=("Segoe UI",10))
            return
        xmin, xmax = min(finite + [0.0]), max(finite + [0.0])
        if xmax <= xmin:
            xmin -= 0.5; xmax += 0.5
        pad = max((xmax-xmin)*0.08, 0.02)
        xmin -= pad; xmax += pad
        left, right, top, bottom = 92, w-30, 26, h-42
        def sx(v): return left + (v-xmin)/(xmax-xmin)*(right-left)
        self.create_line(left,bottom,right,bottom,fill="#777777")
        if xmin <= 0 <= xmax:
            x0=sx(0); self.create_line(x0,top,x0,bottom,fill="#999999",dash=(4,3))
        for t in range(5):
            value=xmin+(xmax-xmin)*t/4; x=sx(value)
            self.create_line(x,bottom,x,bottom+4,fill="#777777")
            self.create_text(x,bottom+17,text=f"{value:.2f}",fill="#555555",font=("Segoe UI",8))
        self.create_text((left+right)/2,h-5,text="Risk coefficient",fill="#333333",font=("Segoe UI",9))
        n=len(records); usable=max(bottom-top,1); step=usable/max(n,1)
        for i,(label,rc,lo,hi,is_pooled,row) in enumerate(records):
            y=top+step*(i+0.5)
            self.create_text(left-10,y,text=label,anchor="e",font=("Segoe UI",9,"bold" if is_pooled else "normal"),fill="#222222")
            if is_pooled:
                pl,ph=_float(row.get("prediction_low")),_float(row.get("prediction_high"))
                if pl is not None and ph is not None:
                    self.create_line(sx(pl),y,sx(ph),y,fill="#888888",width=2,dash=(5,3))
            if lo is not None and hi is not None:
                self.create_line(sx(lo),y,sx(hi),y,fill="#333333",width=3 if is_pooled else 2)
                self.create_line(sx(lo),y-4,sx(lo),y+4,fill="#333333")
                self.create_line(sx(hi),y-4,sx(hi),y+4,fill="#333333")
            if rc is not None:
                x=sx(rc)
                if is_pooled:
                    d=7; self.create_polygon(x-d,y,x,y-d,x+d,y,x,y+d,fill="#202020",outline="#202020")
                else:
                    self.create_oval(x-4,y-4,x+4,y+4,fill="#3b6f9c",outline="#20435f")
        self.create_text(right,8,text="Yearly: bootstrap 95% CI   ·   pooled: modified HK 95% CI   ·   dashed: prediction interval",anchor="ne",fill="#666666",font=("Segoe UI",8))


def _safe_file_component(value) -> str:
    text = str(value or "").strip() or "unknown"
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "."} else "_" for ch in text)
    return cleaned[:120] or "unknown"


def _new_figure(width=7.4, height=4.8):
    """Create a backend-independent Matplotlib figure lazily.

    Matplotlib remains an optional Results Workspace dependency.  The scientific
    model and ordinary workspace navigation do not import it.  Exporting figures
    requires ``requirements-plots.txt``.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise RuntimeError(
            "BSA figure export requires Matplotlib. Install requirements-plots.txt first."
        ) from exc
    figure = Figure(figsize=(width, height), dpi=120, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def _save_figure_formats(figure, base_path: Path, formats: tuple[str, ...]) -> list[Path]:
    outputs = []
    for fmt in formats:
        suffix = str(fmt).strip().lower()
        if suffix not in {"png", "svg"}:
            continue
        path = base_path.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    return outputs


def _bsa_forest_figure(model: BsaDetailModel, q, lag, bsa_id):
    figure = _new_figure(7.6, 4.8)
    ax = figure.add_subplot(111)
    yearly = model.yearly_rows(q, lag, bsa_id)
    pooled = model.pooled_row(q, lag, bsa_id)
    records = []
    for row in yearly:
        records.append((str(_int(row.get("year")) or row.get("year") or "—"), row, False))
    if pooled:
        records.append(("Pooled", pooled, True))
    y_positions = list(range(len(records)))
    finite_any = False
    for y, (label, row, is_pooled) in zip(y_positions, records):
        rc = _float(row.get("pooled_rc" if is_pooled else "rc"))
        lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
        if rc is None:
            continue
        finite_any = True
        if lo is not None and hi is not None and lo <= rc <= hi:
            ax.errorbar(
                [rc], [y], xerr=[[rc - lo], [hi - rc]], fmt="D" if is_pooled else "o",
                capsize=4, linewidth=1.6, markersize=6 if is_pooled else 5,
            )
        else:
            ax.plot([rc], [y], marker="D" if is_pooled else "o", linestyle="none")
        if is_pooled:
            pl, ph = _float(row.get("prediction_low")), _float(row.get("prediction_high"))
            if pl is not None and ph is not None:
                ax.hlines(y, pl, ph, linestyles="dashed", linewidth=1.2)
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions, [label for label, _, _ in records])
    ax.invert_yaxis()
    ax.set_xlabel("Risk coefficient")
    ax.set_title(f"BSA {bsa_id}, yearly RCy and pooled RC, {_fmt_q(q)}, lag {_int(lag)}")
    ax.grid(axis="x", alpha=0.25)
    if not finite_any:
        ax.text(0.5, 0.5, "No estimable RC values", transform=ax.transAxes, ha="center", va="center")
    return figure


def _bsa_yearly_trend_figure(model: BsaDetailModel, q, lag, bsa_id):
    figure = _new_figure(7.6, 4.5)
    ax = figure.add_subplot(111)
    rows = model.yearly_rows(q, lag, bsa_id)
    points = []
    for row in rows:
        year, rc = _int(row.get("year")), _float(row.get("rc"))
        if year is not None and rc is not None:
            points.append((year, rc, _float(row.get("ci_low")), _float(row.get("ci_high"))))
    points.sort()
    if points:
        years = [p[0] for p in points]
        rcs = [p[1] for p in points]
        lower = [max(0.0, p[1] - p[2]) if p[2] is not None else 0.0 for p in points]
        upper = [max(0.0, p[3] - p[1]) if p[3] is not None else 0.0 for p in points]
        has_ci = any(p[2] is not None and p[3] is not None for p in points)
        if has_ci:
            ax.errorbar(years, rcs, yerr=[lower, upper], marker="o", linestyle="-", capsize=4)
        else:
            ax.plot(years, rcs, marker="o")
        ax.set_xticks(years)
    else:
        ax.text(0.5, 0.5, "No estimable yearly RCy values", transform=ax.transAxes, ha="center", va="center")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("RCy")
    ax.set_title(f"BSA {bsa_id}, yearly risk coefficient, {_fmt_q(q)}, lag {_int(lag)}")
    ax.grid(alpha=0.25)
    return figure


def _bsa_location_figure(spatial_model: SpatialModel, q, lag, bsa_id):
    if not spatial_model.map_available:
        return None
    try:
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError as exc:
        raise RuntimeError(
            "BSA figure export requires Matplotlib. Install requirements-plots.txt first."
        ) from exc
    figure = _new_figure(6.4, 6.0)
    ax = figure.add_subplot(111)
    target = _join_key(bsa_id)
    target_feature = None
    for feature in spatial_model.features:
        bid = _join_key(feature.bsa_id)
        if bid == target:
            target_feature = feature
        for part in feature.parts:
            if len(part) < 3:
                continue
            patch = MplPolygon(
                part, closed=True,
                facecolor="#3f78a8" if bid == target else "#ececec",
                edgecolor="#202020" if bid == target else "#ffffff",
                linewidth=1.7 if bid == target else 0.6,
            )
            ax.add_patch(patch)
    if target_feature is None:
        return None
    bounds = [f.bounds for f in spatial_model.features if f.bounds]
    if bounds:
        xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
        xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)
        dx, dy = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
        pad_x, pad_y = dx * 0.03, dy * 0.03
        ax.set_xlim(xmin - pad_x, xmax + pad_x); ax.set_ylim(ymin - pad_y, ymax + pad_y)
    row = spatial_model.row_for_bsa(q, lag, bsa_id)
    rc = spatial_model.effect_value(row)
    rc_text = "—" if rc is None else f"{rc:.4f}"
    ax.set_title(f"BSA {bsa_id}, location, pooled RC {rc_text}, {_fmt_q(q)}, lag {_int(lag)}")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return figure


def export_bsa_figures(
    detail_model: BsaDetailModel,
    spatial_model: SpatialModel | None,
    q,
    lag,
    bsa_ids,
    destination: str | Path,
    *,
    figure_types=("forest", "trend", "map"),
    formats=("png",),
) -> list[Path]:
    """Export BSA-specific figures from already persisted reference results.

    No estimator or workflow object is created.  The function reads only the
    Results Workspace models, so it can be applied to any previously computed
    project that contains reference exports or committed REFERENCE checkpoints.
    """
    q, lag = _float(q), _int(lag)
    if q is None or lag is None:
        raise ValueError("A valid reference threshold and lag must be selected")
    available = detail_model.bsa_ids(q, lag)
    available_lookup = {_join_key(item): item for item in available}
    if isinstance(bsa_ids, str):
        requested = available if bsa_ids.strip().upper() == "ALL" else [bsa_ids]
    else:
        requested = list(bsa_ids or [])
    selected = []
    for raw in requested:
        key = _join_key(raw)
        if key in available_lookup and key not in selected:
            selected.append(key)
    if not selected:
        raise ValueError("No BSA with persisted results matches the requested selection")
    requested_types = tuple(t for t in figure_types if t in {"forest", "trend", "map"})
    requested_formats = tuple(f for f in (str(x).lower() for x in formats) if f in {"png", "svg"})
    if not requested_types:
        raise ValueError("Select at least one figure type")
    if not requested_formats:
        raise ValueError("Select at least one export format")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    multiple = len(selected) > 1
    exported = []
    manifest_rows = []
    for bid in selected:
        target_dir = root / f"BSA_{_safe_file_component(bid)}" if multiple else root
        target_dir.mkdir(parents=True, exist_ok=True)
        makers = {
            "forest": lambda: _bsa_forest_figure(detail_model, q, lag, bid),
            "trend": lambda: _bsa_yearly_trend_figure(detail_model, q, lag, bid),
            "map": lambda: _bsa_location_figure(spatial_model, q, lag, bid) if spatial_model else None,
        }
        for figure_type in requested_types:
            figure = makers[figure_type]()
            if figure is None:
                continue
            base = target_dir / f"BSA_{_safe_file_component(bid)}_{figure_type}"
            paths = _save_figure_formats(figure, base, requested_formats)
            figure.clear()
            for path in paths:
                exported.append(path)
                manifest_rows.append({
                    "bsa_id": bid,
                    "q": f"{q:g}",
                    "lag": str(lag),
                    "figure_type": figure_type,
                    "format": path.suffix.lstrip("."),
                    "file": str(path.relative_to(root)),
                })
    manifest = root / "bsa_figure_export_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bsa_id", "q", "lag", "figure_type", "format", "file"])
        writer.writeheader(); writer.writerows(manifest_rows)
    exported.append(manifest)
    return exported


class BsaFigureExportDialog(tk.Toplevel):
    """Small selection dialog for read-only BSA figure export."""

    def __init__(self, parent, bsa_ids, current_bsa=None):
        super().__init__(parent)
        self.title("Export BSA figures")
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        values = ["ALL"] + list(bsa_ids)
        self.bsa_var = tk.StringVar(value=current_bsa if current_bsa in values else (values[1] if len(values) > 1 else "ALL"))
        self.forest_var = tk.BooleanVar(value=True)
        self.trend_var = tk.BooleanVar(value=True)
        self.map_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=True)
        self.svg_var = tk.BooleanVar(value=False)
        frame = ttk.Frame(self, padding=14); frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="BSA").grid(row=0, column=0, sticky="w", padx=(0,10), pady=4)
        combo = ttk.Combobox(frame, textvariable=self.bsa_var, values=values, state="readonly", width=28)
        combo.grid(row=0, column=1, sticky="ew", pady=4)
        figures = ttk.LabelFrame(frame, text="Figures", padding=8); figures.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8,4))
        ttk.Checkbutton(figures, text="Forest plot, yearly RCy plus pooled RC", variable=self.forest_var).pack(anchor="w")
        ttk.Checkbutton(figures, text="Yearly RCy trend", variable=self.trend_var).pack(anchor="w")
        ttk.Checkbutton(figures, text="BSA location map", variable=self.map_var).pack(anchor="w")
        formats = ttk.LabelFrame(frame, text="Formats", padding=8); formats.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Checkbutton(formats, text="PNG", variable=self.png_var).pack(side="left", padx=(0,12))
        ttk.Checkbutton(formats, text="SVG", variable=self.svg_var).pack(side="left")
        buttons = ttk.Frame(frame); buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10,0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=(6,0))
        ttk.Button(buttons, text="Choose folder and export", command=self._accept).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    def _accept(self):
        types = []
        if self.forest_var.get(): types.append("forest")
        if self.trend_var.get(): types.append("trend")
        if self.map_var.get(): types.append("map")
        formats = []
        if self.png_var.get(): formats.append("png")
        if self.svg_var.get(): formats.append("svg")
        if not types or not formats:
            messagebox.showwarning("Export BSA figures", "Select at least one figure type and one format.", parent=self)
            return
        destination = filedialog.askdirectory(parent=self, title="Choose BSA figure export folder")
        if not destination:
            return
        self.result = {
            "bsa": self.bsa_var.get(),
            "types": tuple(types),
            "formats": tuple(formats),
            "destination": destination,
        }
        self.destroy()

class BsaDetailPanel(ttk.Frame):
    def __init__(self, parent, overview_model: OverviewModel, model: BsaDetailModel, on_select_bsa=None, spatial_model: SpatialModel | None = None):
        super().__init__(parent, padding=16)
        self.overview_model=overview_model
        self.model=model
        self.spatial_model=spatial_model
        self.on_select_bsa=on_select_bsa
        self.selected=overview_model.default_cell_key
        self.selected_bsa=None
        self.bsa_var=tk.StringVar(value="")
        self.card_vars={}
        self.pooled_vars={}
        self._build()
        if self.selected:
            self.set_selected_cell(*self.selected)

    def _build(self):
        top=ttk.Frame(self); top.pack(fill="x")
        ttk.Label(top,text="BSA detail",font=("Segoe UI",16,"bold")).pack(side="left")
        self.cell_label=tk.StringVar(value=""); ttk.Label(top,textvariable=self.cell_label).pack(side="left",padx=(12,0))
        control=ttk.Frame(top); control.pack(side="right")
        ttk.Label(control,text="BSA").pack(side="left",padx=(0,6))
        self.bsa_combo=ttk.Combobox(control,textvariable=self.bsa_var,state="readonly",width=18)
        self.bsa_combo.pack(side="left"); self.bsa_combo.bind("<<ComboboxSelected>>",self._combo_selected)
        ttk.Button(control,text="◀",width=3,command=lambda:self._step_bsa(-1)).pack(side="left",padx=(6,2))
        ttk.Button(control,text="▶",width=3,command=lambda:self._step_bsa(1)).pack(side="left")
        ttk.Button(control,text="Export BSA figures…",command=self.export_figures).pack(side="left",padx=(12,0))

        cards=ttk.Frame(self); cards.pack(fill="x",pady=(10,10))
        for key,title in (("rc","Pooled RC"),("ci","95% pooled CI"),("support","Supported S_b"),("i2","I² heterogeneity"),("years","Contributing years"),("population","Population")):
            frame=ttk.LabelFrame(cards,text=title,padding=7); frame.pack(side="left",fill="x",expand=True,padx=(0,7))
            var=tk.StringVar(value="—"); ttk.Label(frame,textvariable=var,font=("Segoe UI",13,"bold")).pack(anchor="w"); self.card_vars[key]=var

        upper=ttk.Panedwindow(self,orient="horizontal"); upper.pack(fill="both",expand=False)
        plot_frame=ttk.LabelFrame(upper,text="Yearly and pooled risk coefficients",padding=6)
        pooled_frame=ttk.LabelFrame(upper,text="Pooled model diagnostics",padding=9)
        upper.add(plot_frame,weight=3); upper.add(pooled_frame,weight=2)
        self.forest=BsaForestPlot(plot_frame); self.forest.pack(fill="both",expand=True)
        pooled_specs=(("eligible","Eligible"),("reason","Reason"),("tau","Tau²"),("i2","I²"),("q","Cochran Q"),("hkq","HK q"),("hkse","HK standard error"),("prediction","Prediction interval"),("year_ids","Contributing year IDs"))
        for r,(key,label) in enumerate(pooled_specs):
            ttk.Label(pooled_frame,text=label).grid(row=r,column=0,sticky="nw",padx=(0,12),pady=2)
            var=tk.StringVar(value="—"); self.pooled_vars[key]=var
            ttk.Label(pooled_frame,textvariable=var,wraplength=260,justify="left").grid(row=r,column=1,sticky="nw",pady=2)

        self.notebook=ttk.Notebook(self); self.notebook.pack(fill="both",expand=True,pady=(10,0))
        self.yearly_tab=ttk.Frame(self.notebook,padding=6); self.bootstrap_tab=ttk.Frame(self.notebook,padding=6)
        self.support_tab=ttk.Frame(self.notebook,padding=6); self.concordance_tab=ttk.Frame(self.notebook,padding=6)
        self.notebook.add(self.yearly_tab,text="Yearly RC"); self.notebook.add(self.bootstrap_tab,text="Bootstrap")
        self.notebook.add(self.support_tab,text="Exposure support"); self.notebook.add(self.concordance_tab,text="Local concordance")
        self.yearly_tree=self._tree(self.yearly_tab,("year","rc","ci","variance","eligible","reason"),
            {"year":"Year","rc":"RC","ci":"Bootstrap 95% CI","variance":"Variance","eligible":"Eligible","reason":"Reason"},
            {"year":65,"rc":85,"ci":155,"variance":90,"eligible":70,"reason":180})
        self.bootstrap_tree=self._tree(self.bootstrap_tab,("year","valid","invalid","stability","pos","neg","zero","block","redraws"),
            {"year":"Year","valid":"Valid / requested","invalid":"Invalid %","stability":"Directional stability","pos":"Positive %","neg":"Negative %","zero":"Zero %","block":"Block length","redraws":"Global redraws"},
            {"year":60,"valid":110,"invalid":80,"stability":120,"pos":80,"neg":80,"zero":70,"block":85,"redraws":95})
        self.support_tree=self._tree(self.support_tab,("year","pop","pop_year","exposed","nonexposed","eff_e","eff_n","donors","missing_i","missing_b"),
            {"year":"Year","pop":"Population","pop_year":"Pop source year","exposed":"Exposed days","nonexposed":"Non-exposed days","eff_e":"Effective exposed","eff_n":"Effective non-exposed","donors":"Mean baseline donors","missing_i":"Missing incidence","missing_b":"Unavailable baseline"},
            {"year":55,"pop":90,"pop_year":95,"exposed":90,"nonexposed":105,"eff_e":100,"eff_n":110,"donors":115,"missing_i":95,"missing_b":115})
        self.concordance_tree=self._tree(self.concordance_tab,("year","thr_local","thr_territory","conc_e","conc_n","missing_h","zero_weight"),
            {"year":"Year","thr_local":"Local threshold","thr_territory":"Territory threshold","conc_e":"Concordance exposed","conc_n":"Concordance non-exposed","missing_h":"Missing local hazard","zero_weight":"Zero-weight valid days"},
            {"year":60,"thr_local":110,"thr_territory":115,"conc_e":120,"conc_n":135,"missing_h":115,"zero_weight":125})

    @staticmethod
    def _tree(parent,columns,labels,widths):
        frame=ttk.Frame(parent); frame.pack(fill="both",expand=True)
        tree=ttk.Treeview(frame,columns=columns,show="headings",height=10)
        for c in columns:
            tree.heading(c,text=labels[c]); tree.column(c,width=widths[c],anchor="center" if c not in {"reason"} else "w")
        y=ttk.Scrollbar(frame,orient="vertical",command=tree.yview); x=ttk.Scrollbar(frame,orient="horizontal",command=tree.xview)
        tree.configure(yscrollcommand=y.set,xscrollcommand=x.set)
        tree.grid(row=0,column=0,sticky="nsew"); y.grid(row=0,column=1,sticky="ns"); x.grid(row=1,column=0,sticky="ew")
        frame.rowconfigure(0,weight=1); frame.columnconfigure(0,weight=1)
        return tree

    def _combo_selected(self,event=None):
        self.set_selected_bsa(self.bsa_var.get(),notify=True)

    def export_figures(self):
        if not self.selected:
            messagebox.showinfo("Export BSA figures", "Select a threshold-lag cell first.", parent=self)
            return
        ids = self.model.bsa_ids(*self.selected)
        if not ids:
            messagebox.showinfo("Export BSA figures", "No BSA-specific reference results are available for this cell.", parent=self)
            return
        dialog = BsaFigureExportDialog(self, ids, self.selected_bsa)
        self.wait_window(dialog)
        if not dialog.result:
            return
        request = dialog.result
        selection = "ALL" if request["bsa"] == "ALL" else [request["bsa"]]
        try:
            paths = export_bsa_figures(
                self.model, self.spatial_model, *self.selected, selection, request["destination"],
                figure_types=request["types"], formats=request["formats"],
            )
        except Exception as exc:
            messagebox.showerror("Export BSA figures", f"Could not export the selected figures:\n{exc}", parent=self)
            return
        figure_count = max(0, len(paths) - 1)
        messagebox.showinfo(
            "Export BSA figures",
            f"Export complete. {figure_count} figure files were written.\n\n{request['destination']}",
            parent=self,
        )

    def _step_bsa(self,direction):
        values=list(self.bsa_combo.cget("values"))
        if not values: return
        try: index=values.index(self.selected_bsa)
        except ValueError: index=0
        self.set_selected_bsa(values[(index+direction)%len(values)],notify=True)

    def set_selected_cell(self,q,lag):
        if not self.overview_model.cell(q,lag): return
        self.selected=(_float(q),_int(lag)); self.cell_label.set(f"{_fmt_q(q)}, lag {_int(lag)}")
        ids=self.model.bsa_ids(*self.selected); self.bsa_combo.configure(values=ids)
        if self.selected_bsa not in ids:
            self.selected_bsa=self.model.default_bsa(*self.selected)
        self.bsa_var.set(self.selected_bsa or "")
        self._refresh()

    def set_selected_bsa(self,bsa_id,notify=False):
        bid=_join_key(bsa_id)
        if not self.selected or self.model.pooled_row(*self.selected,bid) is None: return
        self.selected_bsa=bid; self.bsa_var.set(bid); self._refresh()
        if notify and self.on_select_bsa:
            self.on_select_bsa(bid)

    @staticmethod
    def _clear(tree):
        for item in tree.get_children(): tree.delete(item)

    def _refresh(self):
        pooled=self.model.pooled_row(*self.selected,self.selected_bsa) if self.selected and self.selected_bsa else None
        yearly=self.model.yearly_rows(*self.selected,self.selected_bsa) if self.selected and self.selected_bsa else []
        if not pooled:
            for var in self.card_vars.values(): var.set("—")
            for var in self.pooled_vars.values(): var.set("—")
            self.forest.set_data(yearly,None)
            return
        lo,hi=_float(pooled.get("ci_low")),_float(pooled.get("ci_high")); support=self.model.supported_magnitude(pooled)
        card={"rc":_fmt_number(pooled.get("pooled_rc"),4),"ci":"—" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]",
              "support":_fmt_number(support,4),"i2":"—" if _float(pooled.get("i_squared_percent")) is None else f"{_float(pooled.get('i_squared_percent')):.1f}%",
              "years":str(_int(pooled.get("contributing_years"))) if _int(pooled.get("contributing_years")) is not None else "—",
              "population":_fmt_number(pooled.get("population"),0)}
        for k,v in card.items(): self.card_vars[k].set(v)
        pl,ph=_float(pooled.get("prediction_low")),_float(pooled.get("prediction_high"))
        year_ids=pooled.get("year_ids")
        if isinstance(year_ids,str):
            try: year_ids=json.loads(year_ids)
            except Exception: pass
        pvals={"eligible":_fmt_bool(pooled.get("eligible")),"reason":str(pooled.get("reason") or "—"),"tau":_fmt_number(pooled.get("tau_squared"),5),
               "i2":"—" if _float(pooled.get("i_squared_percent")) is None else f"{_float(pooled.get('i_squared_percent')):.1f}%",
               "q":_fmt_number(pooled.get("cochran_q"),4),"hkq":_fmt_number(pooled.get("hk_q"),4),"hkse":_fmt_number(pooled.get("hk_standard_error"),5),
               "prediction":"—" if pl is None or ph is None else f"[{pl:.4f}, {ph:.4f}]",
               "year_ids":", ".join(map(str,year_ids)) if isinstance(year_ids,list) else str(year_ids or "—")}
        for k,v in pvals.items(): self.pooled_vars[k].set(v)
        self.forest.set_data(yearly,pooled)
        for tree in (self.yearly_tree,self.bootstrap_tree,self.support_tree,self.concordance_tree): self._clear(tree)
        for row in yearly:
            year=str(_int(row.get("year")) or row.get("year") or "—"); rlo,rhi=_float(row.get("ci_low")),_float(row.get("ci_high"))
            ci="—" if rlo is None or rhi is None else f"[{rlo:.3f}, {rhi:.3f}]"
            self.yearly_tree.insert("","end",values=(year,_fmt_number(row.get("rc"),4),ci,_fmt_number(row.get("variance"),5),_fmt_bool(row.get("eligible")),str(row.get("reason") or "—")))
            requested=_int(row.get("bootstrap_requested")); valid=_int(row.get("bootstrap_valid")); valid_text="—" if requested is None or valid is None else f"{valid} / {requested}"
            self.bootstrap_tree.insert("","end",values=(year,valid_text,_fmt_percent(row.get("invalid_proportion"),1),_fmt_percent(row.get("directional_stability"),1),
                _fmt_percent(row.get("bootstrap_positive_share"),1),_fmt_percent(row.get("bootstrap_negative_share"),1),_fmt_percent(row.get("bootstrap_zero_share"),1),
                str(_int(row.get("block_length"))) if _int(row.get("block_length")) is not None else "—",str(_int(row.get("global_redraws"))) if _int(row.get("global_redraws")) is not None else "—"))
            self.support_tree.insert("","end",values=(year,_fmt_number(row.get("population"),0),str(_int(row.get("population_source_year"))) if _int(row.get("population_source_year")) is not None else "—",
                str(_int(row.get("n_exposed"))) if _int(row.get("n_exposed")) is not None else "—",str(_int(row.get("n_nonexposed"))) if _int(row.get("n_nonexposed")) is not None else "—",
                _fmt_number(row.get("effective_exposed"),1),_fmt_number(row.get("effective_nonexposed"),1),_fmt_number(row.get("baseline_donor_count_mean"),1),
                str(_int(row.get("missing_incidence_days")) or 0),str(_int(row.get("unavailable_baseline_days")) or 0)))
            self.concordance_tree.insert("","end",values=(year,_fmt_number(row.get("threshold_local"),3),_fmt_number(row.get("threshold_territory"),3),
                _fmt_percent(row.get("concordance_exposed"),1),_fmt_percent(row.get("concordance_nonexposed"),1),str(_int(row.get("missing_local_hazard_days")) or 0),str(_int(row.get("zero_weight_valid_days")) or 0)))


ROBUSTNESS_ANALYSIS_LABELS = {
    "ANALYSIS_2_SUPPORT_SENSITIVITY": "A2 Support sensitivity",
    "ANALYSIS_3_BSA_SPECIFIC_EXPOSURE": "A3 BSA-specific exposure",
    "ANALYSIS_4_BASELINE_SENSITIVITY": "A4 Baseline sensitivity",
    "ANALYSIS_5_DEPENDENCE_SENSITIVITY": "A5 Dependence sensitivity",
}


def _read_json_file(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


@dataclass
class RobustnessModel:
    project_dir: Path
    plan: dict | None
    fixed_summary: dict | None
    fixed_bsa: list[dict]
    reoptimization_summary: dict | None
    surface_summary: dict | None
    surface_cells: list[dict]
    fingerprint_components: list[dict]
    region_summary: dict | None
    region_bsa: list[dict]
    region_territory: list[dict]

    @property
    def available(self) -> bool:
        return bool(self.plan) or bool(self.fixed_summary) or bool(self.reoptimization_summary) or bool(self.surface_summary)

    @property
    def realizations(self) -> list[dict]:
        return list((self.plan or {}).get("realizations") or [])

    @property
    def reference(self) -> dict | None:
        ref = (self.plan or {}).get("reference")
        return ref if isinstance(ref, dict) else None

    @property
    def default_realization_id(self):
        rows = self.realizations
        return rows[0].get("realization_id") if rows else None

    def realization(self, rid: str | None):
        if not rid:
            return None
        return next((row for row in self.realizations if row.get("realization_id") == rid), None)

    def label(self, rid: str | None) -> str:
        row = self.realization(rid)
        if not row:
            return "—"
        analysis = ROBUSTNESS_ANALYSIS_LABELS.get(row.get("analysis_id"), row.get("analysis_id", "Sensitivity"))
        step = row.get("step_index")
        step_text = "" if step is None else f" · step {int(step)+1}"
        return f"{analysis} · {row.get('scenario_id','—')}{step_text}"

    def factor_text(self, rid: str | None) -> str:
        row = self.realization(rid) or {}
        factors = row.get("factors") or []
        if not factors:
            return "—"
        return "; ".join(f"{f.get('name')}={f.get('value')}" for f in factors)

    def fixed_row(self, rid):
        rows = (self.fixed_summary or {}).get("summary_rows") or []
        return next((r for r in rows if r.get("sensitivity_realization_id") == rid), None)

    def reoptimization_row(self, rid):
        rows = (self.reoptimization_summary or {}).get("summary_rows") or []
        return next((r for r in rows if r.get("realization_id") == rid), None)

    def surface_row(self, rid):
        rows = (self.surface_summary or {}).get("summary_rows") or []
        return next((r for r in rows if r.get("sensitivity_realization_id") == rid), None)

    def bsa_rows(self, rid):
        return [r for r in self.fixed_bsa if r.get("sensitivity_realization_id") == rid]

    def comparison_surface_rows(self, rid):
        return [r for r in self.surface_cells if r.get("sensitivity_realization_id") == rid]

    def fingerprint_rows(self, rid):
        return [r for r in self.fingerprint_components if r.get("sensitivity_realization_id") == rid]

    def trajectory_rows(self, rid):
        selected = self.realization(rid)
        if not selected:
            return []
        return [r for r in self.realizations
                if r.get("analysis_id") == selected.get("analysis_id") and r.get("scenario_id") == selected.get("scenario_id")]


def load_robustness(project_dir: str | Path) -> RobustnessModel:
    project_dir = Path(project_dir).resolve()
    results = project_dir / "results"
    return RobustnessModel(
        project_dir=project_dir,
        plan=_read_json_file(results / "robustness_plan.json"),
        fixed_summary=_read_json_file(results / "robustness_fixed_point_summary.json"),
        fixed_bsa=_read_csv_rows(results / "robustness_fixed_point_bsa.csv"),
        reoptimization_summary=_read_json_file(results / "robustness_reoptimization_summary.json"),
        surface_summary=_read_json_file(results / "robustness_fingerprint_surface_summary.json"),
        surface_cells=_read_csv_rows(results / "robustness_surface_cells.csv"),
        fingerprint_components=_read_csv_rows(results / "robustness_fingerprint_components.csv"),
        region_summary=_read_json_file(results / "reference_region_stability.json"),
        region_bsa=_read_csv_rows(results / "reference_region_stability_bsa.csv"),
        region_territory=_read_csv_rows(results / "reference_region_stability_territory.csv"),
    )




@dataclass
class ConfounderModel:
    project_dir: Path
    summary: dict | None
    fixed_summary: dict | None
    calibration_rows: list[dict]
    fixed_bsa_rows: list[dict]
    surface_rows: list[dict]
    optimization: dict | None
    fingerprint: dict | None
    surface_comparison: dict | None

    @property
    def available(self) -> bool:
        return bool(self.summary) or bool(self.fixed_summary) or bool(self.calibration_rows) or bool(self.surface_rows)

    @property
    def analysis_run_id(self) -> str | None:
        for source in (self.summary, self.fixed_summary, self.optimization):
            if isinstance(source, dict) and source.get("analysis_run_id"):
                return str(source["analysis_run_id"])
        return None

    @property
    def candidate_id(self) -> str:
        for source in (self.summary, self.fixed_summary):
            if isinstance(source, dict) and source.get("candidate_id"):
                return str(source["candidate_id"])
        return "—"

    @property
    def status(self) -> str:
        return str((self.summary or {}).get("status") or (self.fixed_summary or {}).get("status") or "NOT_AVAILABLE")

    @property
    def representative(self) -> dict | None:
        value = (self.optimization or {}).get("representative")
        return value if isinstance(value, dict) else None

    @property
    def robust_region_count(self) -> int | None:
        if self.optimization is None:
            return None
        region = self.optimization.get("robust_region")
        return len(region) if isinstance(region, list) else 0


def _filter_analysis_rows(rows: list[dict], analysis_run_id: str | None) -> list[dict]:
    if not analysis_run_id:
        return rows
    selected = [row for row in rows if str(row.get("analysis_run_id") or "") == analysis_run_id]
    return selected if selected else rows


def load_confounder(project_dir: str | Path) -> ConfounderModel:
    project_dir = Path(project_dir).resolve()
    results = project_dir / "results"
    summary = _read_json_file(results / "confounder_ca8_summary.json")
    fixed_summary = _read_json_file(results / "confounder_ca6_summary.json")
    run_id = None
    for source in (summary, fixed_summary):
        if isinstance(source, dict) and source.get("analysis_run_id"):
            run_id = str(source["analysis_run_id"]); break
    return ConfounderModel(
        project_dir=project_dir,
        summary=summary,
        fixed_summary=fixed_summary,
        calibration_rows=_filter_analysis_rows(_read_csv_rows(results / "confounder_calibration.csv"), run_id),
        fixed_bsa_rows=_filter_analysis_rows(_read_csv_rows(results / "confounder_fixed_comparison_bsa.csv"), run_id),
        surface_rows=_filter_analysis_rows(_read_csv_rows(results / "confounder_adjusted_surface.csv"), run_id),
        optimization=_read_json_file(results / "confounder_adjusted_optimization.json"),
        fingerprint=_read_json_file(results / "confounder_adjusted_fingerprint.json"),
        surface_comparison=_read_json_file(results / "confounder_surface_comparison.json"),
    )

OUTPUT_GROUP_ORDER = (
    "Observed analysis",
    "Calibration",
    "Null and optimization",
    "Robustness / sensitivity",
    "Candidate confounder",
    "Reports and figures",
    "Technical all-runs tables",
    "Provenance and checkpoint",
    "Project support",
    "Other results",
)

_OUTPUT_DESCRIPTIONS = {
    "reference_surface.csv": "Observed threshold-lag reference surface with WMARMcs and diagnostic territorial metrics.",
    "reference_yearly.csv": "Observed yearly BSA-level RC estimates, bootstrap support, and data-quality diagnostics.",
    "reference_pooled.csv": "Observed pooled BSA coefficients, confidence intervals, heterogeneity, and eligibility.",
    "reference_loyo.csv": "Leave-one-year-out territorial surfaces for temporal stability assessment.",
    "reference_optimization_cells.csv": "Cell-level optimization diagnostics for the observed threshold-lag surface.",
    "reference_null_maxima.csv": "Search-adjusted null maxima for the observed optimization domain.",
    "reference_optimization.json": "Observed robust-region decision, representative cell, null cutoff, and optimization diagnostics.",
    "reference_fingerprint.json": "Observed multidimensional APHREH analysis fingerprint.",
    "calibration_summary.json": "Calibration workflow summary and reference/diagnostic metric declarations.",
    "calibration_distribution.csv": "Effect-magnitude calibration distributions across synthetic classes.",
    "calibration_operating_characteristics.csv": "Calibration power, Type I error, sign recovery, and related operating characteristics.",
    "calibration_ordering.csv": "Strong, moderate, and negligible calibration ordering diagnostics.",
    "calibration_recovery.csv": "Threshold and lag recovery results by response mode and generating pair.",
    "calibration_mean_surface.csv": "Mean STEP/LINEAR_RAMP calibration surfaces across candidate threshold-lag cells.",
    "calibration_instance.csv": "Per-realization calibration results and selected configurations.",
    "synthetic_support.csv": "Synthetic calibration support diagnostics.",
    "synthetic_generation.csv": "Synthetic outcome generation provenance and sampled scenario parameters.",
    "null_maximum.csv": "All null-search maximum statistics retained across analysis runs.",
    "null_cell.csv": "All null-search cell-level results retained across analysis runs.",
    "optimization_cell.csv": "All optimization-cell diagnostics retained across runs.",
    "cell.csv": "All threshold-lag cell results retained across reference, null, and calibration runs.",
    "yearly.csv": "All yearly BSA-level estimator rows retained across runs.",
    "pooled.csv": "All pooled BSA-level rows retained across runs.",
    "support.csv": "All support diagnostics retained across runs.",
    "loyo.csv": "All leave-one-year-out rows retained across runs.",
    "report.html": "Static HTML summary of the completed APHREH analysis.",
    "reference_surface_heatmap.png": "Optional rendered WMARMcs threshold-lag heatmap.",
    "reference_surface_wmarm_diagnostic_heatmap.png": "Optional rendered diagnostic WMARM threshold-lag heatmap.",
    "reference_surface_3d.png": "Optional rendered three-dimensional WMARMcs surface.",
    "manifest.json": "Immutable project identity, run status, environment, source hashes, and resolved configuration provenance.",
    "resolved_config.json": "Complete resolved project configuration used by the computational engine.",
    "execution.jsonl": "Structured execution log including timestamps and process memory observations.",
    "results.sqlite": "Authoritative checkpoint database used for resumable execution and result persistence.",
    "metadata.json": "Canonicalized input-date and support metadata produced during data preparation.",
    "robustness_plan.json": "Robustness/sensitivity realization plan and deterministic reference-realization provenance.",
    "reference_region_stability.json": "RS Analysis 1 summary for stability across the complete reference robust threshold-lag region.",
    "reference_region_stability_bsa.csv": "RS Analysis 1 BSA-level RC, confidence-support, sign, and rank stability across the robust region.",
    "reference_region_stability_territory.csv": "RS Analysis 1 territory-level and disconnected-component stability summaries.",
    "robustness_analysis2_summary.json": "RS Analysis 2 sparse-data and effective-support sensitivity summary and realization provenance.",
    "robustness_analysis3_summary.json": "RS Analysis 3 BSA-specific exposure-classification sensitivity summary and realization provenance.",
    "robustness_analysis3_realizations.csv": "RS Analysis 3 realization-level local-exposure and confidence-scaled surface summaries.",
    "robustness_analysis3_surfaces.csv": "RS Analysis 3 threshold-lag surface under BSA-specific exposure classification.",
    "robustness_analysis3_yearly.csv": "RS Analysis 3 yearly local thresholds, exposure support, bootstrap, and eligibility diagnostics.",
    "robustness_analysis3_exclusions.csv": "RS Analysis 3 yearly and pooled exclusion counts under BSA-specific exposure classification.",
    "robustness_analysis4_summary.json": "RS Analysis 4 baseline-window and baseline-matching sensitivity summary and realization provenance.",
    "robustness_analysis4_realizations.csv": "RS Analysis 4 realization-level baseline policies and confidence-scaled surface summaries.",
    "robustness_analysis4_surfaces.csv": "RS Analysis 4 threshold-lag surfaces for every baseline-sensitivity realization.",
    "robustness_analysis4_yearly.csv": "RS Analysis 4 yearly baseline-window, donor-support, bootstrap, and eligibility diagnostics.",
    "robustness_analysis4_exclusions.csv": "RS Analysis 4 target-day/yearly/pooled exclusion counts by baseline-sensitivity realization.",
    "robustness_analysis5_summary.json": "RS Analysis 5 temporal-dependence and block-length sensitivity summary and realization provenance.",
    "robustness_analysis5_realizations.csv": "RS Analysis 5 realization-level dependence policies and confidence-scaled surface summaries.",
    "robustness_analysis5_surfaces.csv": "RS Analysis 5 threshold-lag surfaces for every dependence-sensitivity realization.",
    "robustness_analysis5_yearly.csv": "RS Analysis 5 yearly dependence, block-length, bootstrap, and RC-invariance diagnostics.",
    "robustness_analysis5_exclusions.csv": "RS Analysis 5 yearly eligibility/exclusion counts by dependence-sensitivity realization.",
    "robustness_fixed_point_summary.json": "RS Step 7 fixed-parameter comparison status and reference representative configuration.",
    "robustness_fixed_point_comparisons.csv": "RS Step 7 realization-level fixed-point changes in WMARMcs, diagnostic WMARM, Kcs, R_CI, coverage, sign/rank agreement, and support transitions.",
    "robustness_fixed_point_bsa.csv": "RS Step 7 BSA-level pooled RC and confidence-scaled magnitude shifts plus eligibility/CI-support transition classes at the reference representative cell.",
    "robustness_realization_loyo_summary.json": "RS Step 9 realization-specific LOYO reconstruction summary and provenance.",
    "robustness_realization_loyo.csv": "RS Step 9 independently checkpointed leave-one-year-out WMARMcs surfaces for reoptimized sensitivity realizations.",
    "robustness_realization_null_summary.json": "RS Step 9 realization-specific maximum-statistic null-search summaries, cutoffs, and search-adjusted p-values.",
    "robustness_realization_null_maxima.csv": "RS Step 9 independently checkpointed maximum-null distributions for reoptimized sensitivity realizations.",
    "robustness_realization_null_cells.csv": "RS Step 9 cell-level null WMARMcs surfaces for all realization-specific null replicates.",
    "robustness_reoptimization_summary.json": "RS Step 8 complete realization-specific reoptimization summary, including realization-specific null calibration and robust-region outcomes.",
    "robustness_reoptimization_realizations.csv": "RS Step 8 realization-level optimization, null, robust-region, and representative-configuration summaries.",
    "robustness_reoptimization_cells.csv": "RS Step 8 cell-level near-optimality, LOYO stability, null support, and robust-region diagnostics for reoptimized sensitivity realizations.",
    "robustness_reoptimization_loyo.csv": "RS Step 8 realization-specific leave-one-year-out WMARMcs surfaces.",
    "robustness_reoptimization_null_maxima.csv": "RS Step 8 realization-specific search-adjusted maximum-null distributions.",
    "robustness_reoptimization_regions.csv": "RS Step 8 robust-region cells for each reoptimized sensitivity realization.",
    "robustness_fingerprint_surface_summary.json": "RS Step 10 reoptimized surface-state, robust-region, representative, and fingerprint comparison summary.",
    "robustness_surface_comparisons.csv": "RS Step 10 realization-level WMARMcs/diagnostic-WMARM surface RMSE and rank concordance, robust-region Jaccard overlap, and representative displacement.",
    "robustness_surface_cells.csv": "RS Step 10 cell-level reference-versus-sensitivity surface values, admissibility flags, and WMARMcs/WMARM differences.",
    "robustness_fingerprint_components.csv": "RS Step 10 scalar confidence-scaled fingerprint component values and signed sensitivity-minus-reference differences.",
    "robustness_analysis2_realizations.csv": "RS Analysis 2 realization-level support policies and confidence-scaled surface summaries.",
    "robustness_analysis2_surfaces.csv": "RS Analysis 2 threshold-lag surfaces for every support-sensitivity realization.",
    "robustness_analysis2_exclusions.csv": "RS Analysis 2 yearly and pooled eligibility/exclusion counts by sensitivity realization.",
}


@dataclass(frozen=True)
class OutputArtifact:
    path: Path
    project_dir: Path
    group: str
    description: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def relative_path(self) -> str:
        try:
            return str(self.path.relative_to(self.project_dir))
        except ValueError:
            return str(self.path)

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def type_label(self) -> str:
        labels = {
            ".csv": "CSV table", ".json": "JSON", ".jsonl": "JSONL log",
            ".html": "HTML report", ".sqlite": "SQLite database", ".db": "Database",
            ".png": "PNG figure", ".jpg": "JPEG figure", ".jpeg": "JPEG figure",
            ".pdf": "PDF", ".txt": "Text", ".md": "Markdown",
        }
        return labels.get(self.suffix, (self.suffix[1:].upper() + " file") if self.suffix else "File")

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    @property
    def modified_timestamp(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0


@dataclass
class OutputsModel:
    project_dir: Path
    artifacts: list[OutputArtifact]

    @property
    def available(self) -> bool:
        return bool(self.artifacts)

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.artifacts)

    @property
    def groups(self) -> list[str]:
        present = {item.group for item in self.artifacts}
        return [name for name in OUTPUT_GROUP_ORDER if name in present]

    def count(self, group: str) -> int:
        return sum(item.group == group for item in self.artifacts)

    def filtered(self, group: str = "All", search: str = "") -> list[OutputArtifact]:
        search = str(search or "").strip().lower()
        rows = self.artifacts
        if group and group != "All":
            rows = [item for item in rows if item.group == group]
        if search:
            rows = [
                item for item in rows
                if search in item.name.lower()
                or search in item.relative_path.lower()
                or search in item.description.lower()
                or search in item.group.lower()
            ]
        order = {name: i for i, name in enumerate(OUTPUT_GROUP_ORDER)}
        return sorted(rows, key=lambda item: (order.get(item.group, 999), item.name.lower()))


def _artifact_group(path: Path, project_dir: Path) -> str:
    name = path.name.lower()
    try:
        parent = path.parent.relative_to(project_dir)
    except ValueError:
        parent = path.parent
    if str(parent).replace("\\", "/") == "results":
        if name.startswith("robustness_analysis2_") or name.startswith("robustness_analysis3_") or name.startswith("robustness_analysis4_") or name.startswith("robustness_analysis5_") or name.startswith("robustness_fixed_point_") or name.startswith("robustness_realization_loyo") or name.startswith("robustness_realization_null") or name.startswith("robustness_reoptimization_") or name.startswith("robustness_fingerprint_") or name.startswith("robustness_surface_") or name.startswith("reference_region_stability"):
            return "Robustness / sensitivity"
        if name.startswith("confounder_"):
            return "Candidate confounder"
        if name.startswith("reference_") and path.suffix.lower() in {".csv", ".json"}:
            return "Observed analysis"
        if name.startswith("calibration_") or name.startswith("synthetic_"):
            return "Calibration"
        if name.startswith("null_") or name == "optimization_cell.csv":
            return "Null and optimization"
        if name == "report.html" or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            return "Reports and figures"
        if name == "robustness_plan.json":
            return "Provenance and checkpoint"
        if name in {"cell.csv", "yearly.csv", "pooled.csv", "support.csv", "loyo.csv"}:
            return "Technical all-runs tables"
        return "Other results"
    if name in {"manifest.json", "resolved_config.json", "execution.jsonl", "results.sqlite"}:
        return "Provenance and checkpoint"
    if str(parent).replace("\\", "/").startswith("canonical"):
        return "Provenance and checkpoint"
    if str(parent).replace("\\", "/").startswith("geometry"):
        return "Project support"
    return "Other results"


def _artifact_description(path: Path) -> str:
    if path.name in _OUTPUT_DESCRIPTIONS:
        return _OUTPUT_DESCRIPTIONS[path.name]
    if path.parent.name == "geometry":
        return "Geometry copied into the project for downstream spatial joining and visualization."
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
        return "Rendered figure associated with the APHREH project results."
    if path.suffix.lower() == ".csv":
        return "Exported APHREH result table."
    if path.suffix.lower() == ".json":
        return "APHREH structured result or provenance metadata."
    return "APHREH project output artifact."


def load_outputs(project_dir: str | Path) -> OutputsModel:
    project_dir = Path(project_dir).expanduser().resolve()
    candidates: list[Path] = []
    results = project_dir / "results"
    if results.is_dir():
        candidates.extend(path for path in results.iterdir() if path.is_file())
    for name in ("manifest.json", "resolved_config.json", "execution.jsonl", "results.sqlite"):
        path = project_dir / name
        if path.is_file():
            candidates.append(path)
    canonical_metadata = project_dir / "canonical" / "metadata.json"
    if canonical_metadata.is_file():
        candidates.append(canonical_metadata)
    geometry = project_dir / "geometry"
    if geometry.is_dir():
        candidates.extend(path for path in geometry.iterdir() if path.is_file())
    seen = set()
    artifacts = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        artifacts.append(OutputArtifact(
            path=resolved,
            project_dir=project_dir,
            group=_artifact_group(resolved, project_dir),
            description=_artifact_description(resolved),
        ))
    return OutputsModel(project_dir=project_dir, artifacts=artifacts)


def _format_file_size(size: int) -> str:
    value = float(max(0, size))
    units = ("B", "KB", "MB", "GB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def _format_modified(timestamp: float) -> str:
    if not timestamp:
        return "—"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
    except (ValueError, OSError, OverflowError):
        return "—"


def _preview_artifact(artifact: OutputArtifact, max_chars: int = 18000, max_rows: int = 35) -> str:
    path = artifact.path
    suffix = artifact.suffix
    if suffix == ".html":
        return (
            "Static HTML report. Use 'Open selected' to open this artifact explicitly in the web browser.\n\n"
            + artifact.description
        )
    if suffix in {".sqlite", ".db"}:
        return (
            "Binary checkpoint database. It is intentionally not rendered in the Results Workspace.\n\n"
            + artifact.description
        )
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg"}:
        return "Binary/visual artifact. Use 'Open selected' to inspect it.\n\n" + artifact.description
    try:
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(obj, indent=2, ensure_ascii=False)
        elif suffix == ".csv":
            lines = []
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        lines.append("… preview truncated …")
                        break
                    lines.append("\t".join(str(value) for value in row))
            text = "\n".join(lines)
        elif suffix == ".jsonl":
            lines = []
            with path.open("r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if i >= max_rows:
                        lines.append("… preview truncated …")
                        break
                    raw = line.rstrip("\n")
                    try:
                        lines.append(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
                    except json.JSONDecodeError:
                        lines.append(raw)
            text = "\n".join(lines)
        elif suffix in {".txt", ".md"} or not suffix:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            return "Preview is not available for this file type.\n\n" + artifact.description
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"Preview unavailable: {type(exc).__name__}: {exc}"
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n… preview truncated …"
    return text


def _open_external(path: Path, *, force_browser: bool = False) -> None:
    path = Path(path).resolve()
    if force_browser or path.suffix.lower() == ".html":
        webbrowser.open(path.as_uri())
        return
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class MetricCard(ttk.Frame):
    def __init__(self, parent, title: str, value: str = "—", subtitle: str = ""):
        super().__init__(parent, padding=(10, 8), relief="ridge")
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=title, style="Overview.CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.value = tk.StringVar(value=value)
        self.subtitle = tk.StringVar(value=subtitle)
        ttk.Label(self, textvariable=self.value, style="Overview.CardValue.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )
        ttk.Label(
            self,
            textvariable=self.subtitle,
            style="Overview.CardSub.TLabel",
            wraplength=220,
        ).grid(row=2, column=0, sticky="w", pady=(3, 0))


class HeatmapPreview(ttk.Frame):
    """Dependency-free threshold-lag heatmap used by Overview and Surface."""

    def __init__(
        self,
        parent,
        title: str,
        metric: str,
        on_select=None,
        value_format: str = "number",
        show_overlay_legend: bool = False,
    ):
        super().__init__(parent, padding=8, relief="ridge")
        self.metric = metric
        self.title = tk.StringVar(value=title)
        self.on_select = on_select
        self.value_format = value_format
        self.rows = []
        self.selected = None
        self.diagnostics = {}
        self.overlays = {}
        self.representative = None
        self.robust_keys = set()
        self.calibration_boundary = None
        ttk.Label(self, textvariable=self.title, style="Overview.SectionTitle.TLabel").pack(
            anchor="w"
        )
        self.canvas = tk.Canvas(
            self,
            width=390,
            height=260,
            highlightthickness=0,
            background="white",
        )
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        if show_overlay_legend:
            self.overlay_legend = tk.StringVar(
                value="★ representative   R robust   O near-optimal   N null-supported   T LOYO-stable"
            )
            ttk.Label(
                self,
                textvariable=self.overlay_legend,
                style="Overview.CardSub.TLabel",
            ).pack(anchor="w", pady=(4, 0))
        else:
            self.overlay_legend = None

    def set_metric(self, metric: str, title: str | None = None, value_format: str | None = None):
        self.metric = metric
        if title is not None:
            self.title.set(title)
        if value_format is not None:
            self.value_format = value_format
        self.redraw()

    def set_data(
        self,
        rows,
        selected=None,
        diagnostics=None,
        overlays=None,
        representative=None,
        robust_keys=None,
        calibration_boundary=None,
    ):
        self.rows = list(rows)
        self.selected = selected
        self.diagnostics = diagnostics or {}
        self.overlays = dict(overlays or {})
        self.representative = representative
        self.robust_keys = set(robust_keys or ())
        self.calibration_boundary = _float(calibration_boundary)
        if self.overlay_legend is not None:
            text = "★ representative   R robust   O near-optimal   N null-supported   T LOYO-stable"
            if self.calibration_boundary is not None and self.metric == "wmarm_cs":
                text += "   C above negligible q97.5"
            self.overlay_legend.set(text)
        self.redraw()

    @staticmethod
    def _color(value, minimum, maximum):
        if value is None:
            return "#eceff1"
        if maximum <= minimum + ZERO_TOLERANCE:
            t = 0.0 if abs(maximum) <= ZERO_TOLERANCE else 0.65
        else:
            t = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
        lo = (238, 244, 248)
        hi = (44, 95, 134)
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(lo, hi))
        return "#%02x%02x%02x" % rgb

    def _display_value(self, value):
        if value is None:
            return "—"
        if self.value_format == "percent":
            return f"{100.0 * value:.0f}%"
        if abs(value) < 100:
            return f"{value:.3f}"
        return f"{value:.1f}"

    def _metric_value(self, row, key):
        if self.metric == "persistence":
            return _float((self.diagnostics.get(key) or {}).get("persistence"))
        return _float(row.get(self.metric))

    def _draw_overlay_badges(self, c, x0, y0, x1, y1, key, diagnostic, value):
        badges = []
        if self.overlays.get("representative") and key == self.representative:
            badges.append("★")
        if self.overlays.get("robust") and key in self.robust_keys:
            badges.append("R")
        if self.overlays.get("near_optimal") and _bool((diagnostic or {}).get("near_optimal")):
            badges.append("O")
        if self.overlays.get("null_supported") and _bool((diagnostic or {}).get("null_supported")):
            badges.append("N")
        if self.overlays.get("stable") and _bool((diagnostic or {}).get("stable")):
            badges.append("T")
        if (
            self.overlays.get("calibration_relevance")
            and self.metric == "wmarm_cs"
            and self.calibration_boundary is not None
            and value is not None
            and value > self.calibration_boundary
        ):
            badges.append("C")
        if not badges:
            return
        text = " ".join(badges)
        c.create_text(
            x0 + 4,
            y0 + 3,
            anchor="nw",
            text=text,
            fill="#5b1a1a",
            font=("Segoe UI", 8, "bold"),
        )

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if not self.rows:
            c.create_text(15, 20, anchor="nw", text="No surface data", fill="#666")
            return
        qs = sorted({_float(r.get("q")) for r in self.rows if _float(r.get("q")) is not None})
        lags = sorted({_int(r.get("lag")) for r in self.rows if _int(r.get("lag")) is not None})
        if not qs or not lags:
            return
        by_key = {(_float(r.get("q")), _int(r.get("lag"))): r for r in self.rows}
        finite = []
        for key, row in by_key.items():
            value = self._metric_value(row, key)
            if value is not None:
                finite.append(value)
        minimum, maximum = (min(finite), max(finite)) if finite else (0.0, 0.0)
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 220)
        left, top, right, bottom = 56, 22, 18, 42
        cw = max(34, (w - left - right) / len(lags))
        ch = max(30, (h - top - bottom) / len(qs))
        for i, q in enumerate(qs):
            y0 = top + i * ch
            y1 = y0 + ch
            c.create_text(left - 8, (y0 + y1) / 2, anchor="e", text=f"q{q:g}", fill="#333")
            for j, lag in enumerate(lags):
                x0 = left + j * cw
                x1 = x0 + cw
                key = (q, lag)
                row = by_key.get(key, {})
                diagnostic = self.diagnostics.get(key) or {}
                value = self._metric_value(row, key)
                selected = self.selected == key
                tag = f"cell_{i}_{j}"
                c.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=self._color(value, minimum, maximum),
                    outline="#111" if selected else "#d0d5d8",
                    width=3 if selected else 1,
                    tags=(tag, "cell"),
                )
                if self.overlays.get("robust") and key in self.robust_keys:
                    c.create_rectangle(
                        x0 + 3,
                        y0 + 3,
                        x1 - 3,
                        y1 - 3,
                        outline="#187d3c",
                        width=3,
                    )
                if (
                    self.overlays.get("calibration_relevance")
                    and self.metric == "wmarm_cs"
                    and self.calibration_boundary is not None
                    and value is not None
                    and value > self.calibration_boundary
                ):
                    c.create_rectangle(
                        x0 + 6,
                        y0 + 6,
                        x1 - 6,
                        y1 - 6,
                        outline="#6a3d9a",
                        width=2,
                        dash=(4, 2),
                    )
                text_color = "#222"
                if value is not None and maximum > minimum:
                    if (value - minimum) / (maximum - minimum) > 0.58:
                        text_color = "white"
                c.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    text=self._display_value(value),
                    fill=text_color,
                    tags=(tag, "cell"),
                )
                self._draw_overlay_badges(c, x0, y0, x1, y1, key, diagnostic, value)
                if self.on_select:
                    c.tag_bind(
                        tag,
                        "<Button-1>",
                        lambda event, qq=q, ll=lag: self.on_select(qq, ll),
                    )
        for j, lag in enumerate(lags):
            x = left + (j + 0.5) * cw
            c.create_text(x, top + len(qs) * ch + 8, anchor="n", text=str(lag), fill="#333")
        c.create_text(
            left + (len(lags) * cw) / 2,
            h - 5,
            anchor="s",
            text="Cumulative lag",
            fill="#555",
        )
        if (
            self.overlays.get("calibration_relevance")
            and self.metric == "wmarm_cs"
            and self.calibration_boundary is not None
        ):
            c.create_text(
                w - right,
                5,
                anchor="ne",
                text=f"Negligible q97.5 = {self._display_value(self.calibration_boundary)}",
                fill="#5a2f7a",
                font=("Segoe UI", 8, "bold"),
            )


class SupportPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10, relief="ridge")
        ttk.Label(self, text="Selected-cell support", style="Overview.SectionTitle.TLabel").pack(
            anchor="w"
        )
        self.cell_label = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.cell_label, style="Overview.CardValue.TLabel").pack(
            anchor="w", pady=(5, 8)
        )
        self.items = {}
        for key, label in (
            ("bsa_coverage", "BSA coverage"),
            ("population_coverage", "Population coverage"),
            ("ci_excluding_zero_population_share", "CI-supported population"),
        ):
            frame = ttk.Frame(self)
            frame.pack(fill="x", pady=4)
            ttk.Label(frame, text=label).pack(anchor="w")
            value = tk.StringVar(value="—")
            ttk.Label(frame, textvariable=value).pack(anchor="e")
            bar = ttk.Progressbar(frame, maximum=1.0, mode="determinate")
            bar.pack(fill="x", pady=(2, 0))
            self.items[key] = (value, bar)
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=8)
        self.kcs = tk.StringVar(value="—")
        self.wmarmcs = tk.StringVar(value="—")
        self.wmarm = tk.StringVar(value="—")
        for label, var in (
            ("WMARMcs", self.wmarmcs),
            ("Diagnostic WMARM", self.wmarm),
            ("Kcs", self.kcs),
        ):
            row = ttk.Frame(self)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Label(row, textvariable=var).pack(side="right")

    def set_cell(self, row):
        if not row:
            self.cell_label.set("—")
            return
        self.cell_label.set(f"{_fmt_q(row.get('q'))}, lag {_int(row.get('lag'))}")
        for key, (value, bar) in self.items.items():
            x = _float(row.get(key))
            value.set(_fmt_percent(x))
            bar.configure(value=0 if x is None else max(0, min(1, x)))
        self.kcs.set(_fmt_number(row.get("confidence_support_retention_ratio"), 3))
        self.wmarmcs.set(_fmt_number(row.get("wmarm_cs"), 4))
        self.wmarm.set(_fmt_number(row.get("wmarm"), 4))


class OverviewPanel(ttk.Frame):
    def __init__(self, parent, model: OverviewModel, on_select=None, calibration_model: CalibrationModel | None = None):
        super().__init__(parent, padding=14)
        self.model = model
        self.calibration_model = calibration_model
        self.calibration_boundary = (
            calibration_model.negligible_reference_boundary()
            if calibration_model is not None else None
        )
        self.calibration_overlay = tk.BooleanVar(value=False)
        self.on_select = on_select
        self.selected = model.default_cell_key
        self._configure_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_header()
        self._build_cards()
        self._build_overlay_controls()
        self._build_visuals()
        self.set_selected_cell(*(self.selected or (None, None)))

    def _configure_styles(self):
        style = ttk.Style(self)
        style.configure("Overview.CardTitle.TLabel", font=("Segoe UI", 9))
        style.configure("Overview.CardValue.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Overview.CardSub.TLabel", font=("Segoe UI", 8))
        style.configure("Overview.SectionTitle.TLabel", font=("Segoe UI", 11, "bold"))

    def _build_header(self):
        palette = {
            "ROBUST_REGION": ("#e8f5e9", "#1b5e20"),
            "NO_ROBUST_REGION": ("#fff8e1", "#7a4b00"),
            "ZERO_SUPPORTED_SURFACE": ("#eef2f5", "#263238"),
            "SURFACE_AVAILABLE": ("#e8f1fb", "#174a7e"),
            "RESULTS_UNAVAILABLE": ("#f5f5f5", "#555"),
        }
        bg, fg = palette[self.model.status_code]
        banner = tk.Frame(self, bg=bg, padx=14, pady=10)
        banner.grid(row=0, column=0, sticky="ew")
        tk.Label(
            banner,
            text=self.model.status_title,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            banner,
            text=self.model.status_detail,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9),
            wraplength=1050,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

    def _build_cards(self):
        cards = ttk.Frame(self)
        cards.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for i in range(6):
            cards.columnconfigure(i, weight=1, uniform="overview-card")
        entries = (
            ("Reference metric", "WMARMcs", "confidence-scaled magnitude"),
            ("Representative", self.model.representative_text, "robust-region representative"),
            ("Maximum WMARMcs", _fmt_number(self.model.reference_maximum, 4), "admissible observed maximum"),
            ("Diagnostic WMARM max", _fmt_number(self.model.diagnostic_maximum, 4), "point-estimate magnitude"),
            ("Search-adjusted p", _fmt_number(self.model.p_max, 4), "maximum-statistic inference"),
            ("Project period", self.model.date_summary or "—", "exact hazard/outcome matching dates"),
        )
        for i, (title, value, sub) in enumerate(entries):
            MetricCard(cards, title, value, sub).grid(
                row=0,
                column=i,
                sticky="nsew",
                padx=(0 if i == 0 else 5, 0),
            )

    def _user_select(self, q, lag):
        if self.on_select:
            self.on_select(q, lag)
        else:
            self.set_selected_cell(q, lag)

    def _build_overlay_controls(self):
        if self.calibration_boundary is None:
            return
        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Show calibration relevance reference",
            variable=self.calibration_overlay,
            command=self._refresh_visuals,
        ).pack(side="left")
        ttk.Label(
            controls,
            text=(
                f"negligible synthetic WMARMcs upper 95% reference limit, q97.5 = "
                f"{_fmt_number(self.calibration_boundary, 4)}, descriptive only"
            ),
            style="Overview.CardSub.TLabel",
        ).pack(side="left", padx=(10, 0))

    def _build_visuals(self):
        visuals = ttk.Frame(self)
        visuals.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        visuals.columnconfigure(0, weight=4)
        visuals.columnconfigure(1, weight=4)
        visuals.columnconfigure(2, weight=3)
        visuals.rowconfigure(0, weight=1)
        self.reference_heatmap = HeatmapPreview(
            visuals,
            "Reference WMARMcs",
            "wmarm_cs",
            self._user_select,
            show_overlay_legend=self.calibration_boundary is not None,
        )
        self.reference_heatmap.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.diagnostic_heatmap = HeatmapPreview(
            visuals, "Diagnostic WMARM", "wmarm", self._user_select
        )
        self.diagnostic_heatmap.grid(row=0, column=1, sticky="nsew", padx=6)
        self.support = SupportPanel(visuals)
        self.support.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

    def _refresh_visuals(self):
        overlays = {"calibration_relevance": bool(self.calibration_overlay.get())}
        self.reference_heatmap.set_data(
            self.model.surface,
            self.selected,
            overlays=overlays,
            calibration_boundary=self.calibration_boundary,
        )
        self.diagnostic_heatmap.set_data(self.model.surface, self.selected)

    def set_selected_cell(self, q, lag):
        if q is None or lag is None:
            row = None
            selected = None
        else:
            selected = (_float(q), _int(lag))
            row = self.model.cell(*selected)
            if row is None:
                return
        self.selected = selected
        self._refresh_visuals()
        self.support.set_cell(row)


class SurfaceDetailPanel(ttk.Frame):
    """Detailed numerical and optimization diagnostics for one surface cell."""

    def __init__(self, parent, model: OverviewModel):
        super().__init__(parent, padding=10, relief="ridge")
        self.model = model
        ttk.Label(self, text="Selected cell", style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.cell_title = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.cell_title, style="Overview.CardValue.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        self.values = {}
        sections = (
            (
                "Territorial metrics",
                (
                    ("wmarm_cs", "WMARMcs"),
                    ("wmarm", "Diagnostic WMARM"),
                    ("kcs", "Kcs"),
                    ("ci_share", "CI-supported population"),
                ),
            ),
            (
                "Coverage",
                (
                    ("bsa_coverage", "BSA coverage"),
                    ("population_coverage", "Population coverage"),
                    ("admissible", "Admissible support"),
                ),
            ),
            (
                "Optimization",
                (
                    ("near_optimal", "Near-optimal"),
                    ("null_supported", "Null-supported"),
                    ("stable", "LOYO-stable"),
                    ("robust", "Robust cell"),
                    ("persistence", "LOYO persistence"),
                ),
            ),
            (
                "LOYO summary",
                (
                    ("loyo_mean", "Mean WMARMcs"),
                    ("loyo_sd", "SD WMARMcs"),
                    ("loyo_max_deviation", "Max deviation"),
                ),
            ),
        )
        for section, rows in sections:
            ttk.Label(self, text=section, font=("Segoe UI", 9, "bold")).pack(
                anchor="w", pady=(7, 2)
            )
            for key, label in rows:
                frame = ttk.Frame(self)
                frame.pack(fill="x", pady=1)
                ttk.Label(frame, text=label).pack(side="left")
                var = tk.StringVar(value="—")
                ttk.Label(frame, textvariable=var).pack(side="right")
                self.values[key] = var
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=8)
        self.context = tk.StringVar(value="")
        ttk.Label(
            self,
            textvariable=self.context,
            wraplength=270,
            justify="left",
            style="Overview.CardSub.TLabel",
        ).pack(anchor="w")

    def set_cell(self, row, diagnostic):
        if not row:
            self.cell_title.set("—")
            for var in self.values.values():
                var.set("—")
            return
        self.cell_title.set(f"{_fmt_q(row.get('q'))}, lag {_int(row.get('lag'))}")
        diag = diagnostic or {}
        self.values["wmarm_cs"].set(_fmt_number(row.get("wmarm_cs"), 4))
        self.values["wmarm"].set(_fmt_number(row.get("wmarm"), 4))
        self.values["kcs"].set(_fmt_number(row.get("confidence_support_retention_ratio"), 3))
        self.values["ci_share"].set(_fmt_percent(row.get("ci_excluding_zero_population_share")))
        self.values["bsa_coverage"].set(_fmt_percent(row.get("bsa_coverage")))
        self.values["population_coverage"].set(_fmt_percent(row.get("population_coverage")))
        self.values["admissible"].set(_fmt_bool(diag.get("admissible_support")))
        self.values["near_optimal"].set(_fmt_bool(diag.get("near_optimal")))
        self.values["null_supported"].set(_fmt_bool(diag.get("null_supported")))
        self.values["stable"].set(_fmt_bool(diag.get("stable")))
        self.values["robust"].set(_fmt_bool(diag.get("robust")))
        self.values["persistence"].set(_fmt_percent(diag.get("persistence")))
        self.values["loyo_mean"].set(
            _fmt_number(diag.get("loyo_mean_wmarm_cs", diag.get("loyo_mean")), 4)
        )
        self.values["loyo_sd"].set(
            _fmt_number(diag.get("loyo_sd_wmarm_cs", diag.get("loyo_sd")), 4)
        )
        self.values["loyo_max_deviation"].set(
            _fmt_number(
                diag.get("loyo_max_deviation_wmarm_cs", diag.get("loyo_max_deviation")), 4
            )
        )
        context = []
        if self.model.null_cutoff is not None:
            context.append(f"Null cutoff WMARMcs: {_fmt_number(self.model.null_cutoff, 4)}")
        if self.model.surface_coherence is not None:
            context.append(f"Surface coherence: {_fmt_number(self.model.surface_coherence, 3)}")
        if self.model.p_max is not None:
            context.append(f"Search-adjusted p: {_fmt_number(self.model.p_max, 4)}")
        self.context.set("\n".join(context))


def _build_surface_3d_mesh(rows, metric: str, diagnostics=None):
    """Return a rectangular surface mesh for the dependency-free 3D renderer.

    The mesh uses the actual threshold and lag coordinates as grid keys while
    leaving the screen projection to :class:`Surface3DPreview`. Missing cells do
    not create synthetic values; polygons are emitted only when all four corner
    cells exist and have finite values.
    """
    diagnostics = diagnostics or {}
    rows = list(rows or [])
    qs = sorted({_float(row.get("q")) for row in rows if _float(row.get("q")) is not None})
    lags = sorted({_int(row.get("lag")) for row in rows if _int(row.get("lag")) is not None})
    by_key = {(_float(row.get("q")), _int(row.get("lag"))): row for row in rows}
    values = {}
    for key, row in by_key.items():
        if metric == "persistence":
            value = _float((diagnostics.get(key) or {}).get("persistence"))
        else:
            value = _float(row.get(metric))
        if value is not None:
            values[key] = value
    finite = list(values.values())
    minimum, maximum = (min(finite), max(finite)) if finite else (0.0, 0.0)
    quads = []
    for i in range(max(0, len(qs) - 1)):
        for j in range(max(0, len(lags) - 1)):
            corners = (
                (qs[i], lags[j]),
                (qs[i], lags[j + 1]),
                (qs[i + 1], lags[j + 1]),
                (qs[i + 1], lags[j]),
            )
            if all(key in values for key in corners):
                quads.append(corners)
    return {
        "qs": qs,
        "lags": lags,
        "by_key": by_key,
        "values": values,
        "minimum": minimum,
        "maximum": maximum,
        "quads": quads,
    }


class Surface3DPreview(ttk.Frame):
    """Dependency-free rotated 3D threshold-lag surface renderer.

    Height and fill encode the selected metric. The camera is oriented like a
    conventional scientific surface plot: cumulative lag runs across the
    foreground with lag zero on the right, threshold increases away along the
    left side, and the metric axis is vertical on the right. The renderer uses
    the same exported surface rows as the 2D heatmap and never interpolates
    missing analytical cells. Clicking a rendered grid point selects that exact
    threshold-lag configuration in the shared Results Workspace state.
    """

    def __init__(
        self,
        parent,
        title: str,
        metric: str,
        on_select=None,
        value_format: str = "number",
        show_overlay_legend: bool = False,
    ):
        super().__init__(parent, padding=8, relief="ridge")
        self.metric = metric
        self.title = tk.StringVar(value=title)
        self.on_select = on_select
        self.value_format = value_format
        self.rows = []
        self.selected = None
        self.diagnostics = {}
        self.overlays = {}
        self.representative = None
        self.robust_keys = set()
        self.calibration_boundary = None
        self._point_hits = []
        ttk.Label(self, textvariable=self.title, style="Overview.SectionTitle.TLabel").pack(
            anchor="w"
        )
        self.canvas = tk.Canvas(
            self,
            width=390,
            height=300,
            highlightthickness=0,
            background="white",
        )
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Button-1>", self._click)
        ttk.Label(
            self,
            text="Rotated 3D view: lag runs across the foreground, threshold recedes along the left axis, and height/color encode the metric.",
            style="Overview.CardSub.TLabel",
            wraplength=390,
        ).pack(anchor="w", pady=(4, 0))
        if show_overlay_legend:
            self.overlay_legend = tk.StringVar(
                value="★ representative   R robust   O near-optimal   N null-supported   T LOYO-stable"
            )
            ttk.Label(
                self,
                textvariable=self.overlay_legend,
                style="Overview.CardSub.TLabel",
            ).pack(anchor="w", pady=(2, 0))
        else:
            self.overlay_legend = None

    def set_metric(self, metric: str, title: str | None = None, value_format: str | None = None):
        self.metric = metric
        if title is not None:
            self.title.set(title)
        if value_format is not None:
            self.value_format = value_format
        self.redraw()

    def set_data(
        self,
        rows,
        selected=None,
        diagnostics=None,
        overlays=None,
        representative=None,
        robust_keys=None,
        calibration_boundary=None,
    ):
        self.rows = list(rows)
        self.selected = selected
        self.diagnostics = diagnostics or {}
        self.overlays = dict(overlays or {})
        self.representative = representative
        self.robust_keys = set(robust_keys or ())
        self.calibration_boundary = _float(calibration_boundary)
        if self.overlay_legend is not None:
            text = "★ representative   R robust   O near-optimal   N null-supported   T LOYO-stable"
            if self.calibration_boundary is not None and self.metric == "wmarm_cs":
                text += "   C above negligible q97.5"
            self.overlay_legend.set(text)
        self.redraw()

    def _display_value(self, value):
        if value is None:
            return "—"
        if self.value_format == "percent":
            return f"{100.0 * value:.0f}%"
        if abs(value) < 100:
            return f"{value:.3f}"
        return f"{value:.1f}"

    @staticmethod
    def _normalise(value, minimum, maximum):
        if value is None:
            return 0.0
        if maximum <= minimum + ZERO_TOLERANCE:
            return 0.0 if abs(maximum) <= ZERO_TOLERANCE else 0.55
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    @staticmethod
    def _project(xn, yn, zn, width, height):
        """Project normalized coordinates using the rotated scientific-view camera.

        ``xn`` increases with cumulative lag, so lag 0 is placed on the right
        and larger lags extend toward the left foreground. ``yn`` increases
        with threshold and therefore recedes upward along the left side. ``zn``
        is drawn vertically. The layout deliberately resembles a standard
        Matplotlib-style 3D surface while remaining a native Tkinter renderer.
        """
        base_x = 0.78 * width - xn * 0.58 * width - yn * 0.06 * width
        base_y = 0.72 * height + xn * 0.08 * height - yn * 0.28 * height
        return (base_x, base_y - zn * 0.34 * height)

    def _overlay_badges(self, key, diagnostic, value):
        badges = []
        if self.overlays.get("representative") and key == self.representative:
            badges.append("★")
        if self.overlays.get("robust") and key in self.robust_keys:
            badges.append("R")
        if self.overlays.get("near_optimal") and _bool((diagnostic or {}).get("near_optimal")):
            badges.append("O")
        if self.overlays.get("null_supported") and _bool((diagnostic or {}).get("null_supported")):
            badges.append("N")
        if self.overlays.get("stable") and _bool((diagnostic or {}).get("stable")):
            badges.append("T")
        if (
            self.overlays.get("calibration_relevance")
            and self.metric == "wmarm_cs"
            and self.calibration_boundary is not None
            and value is not None
            and value > self.calibration_boundary
        ):
            badges.append("C")
        return " ".join(badges)

    def _click(self, event):
        if not self.on_select or not self._point_hits:
            return
        best = min(
            self._point_hits,
            key=lambda item: (item[0] - event.x) ** 2 + (item[1] - event.y) ** 2,
        )
        distance2 = (best[0] - event.x) ** 2 + (best[1] - event.y) ** 2
        if distance2 <= 16 ** 2:
            self.on_select(best[2], best[3])

    def redraw(self):
        c = self.canvas
        c.delete("all")
        self._point_hits = []
        mesh = _build_surface_3d_mesh(self.rows, self.metric, self.diagnostics)
        qs, lags = mesh["qs"], mesh["lags"]
        values = mesh["values"]
        minimum, maximum = mesh["minimum"], mesh["maximum"]
        if not qs or not lags or not values:
            c.create_text(15, 20, anchor="nw", text="No surface data", fill="#666")
            return
        w = max(c.winfo_width(), 340)
        h = max(c.winfo_height(), 250)
        q_index = {q: i for i, q in enumerate(qs)}
        lag_index = {lag: j for j, lag in enumerate(lags)}

        def coords(key, *, base=False):
            q, lag = key
            xn = 0.0 if len(lags) <= 1 else lag_index[lag] / (len(lags) - 1)
            yn = 0.0 if len(qs) <= 1 else q_index[q] / (len(qs) - 1)
            zn = 0.0 if base else self._normalise(values.get(key), minimum, maximum)
            return self._project(xn, yn, zn, w, h)

        # Base-plane axes and light grid. The camera mirrors a conventional
        # scientific 3D surface: lag 0 at right, larger lags toward left,
        # threshold increasing away on the left, and metric vertical at right.
        lag_zero = self._project(0.0, 0.0, 0.0, w, h)
        lag_max = self._project(1.0, 0.0, 0.0, w, h)
        q_max_left = self._project(1.0, 1.0, 0.0, w, h)
        q_max_right = self._project(0.0, 1.0, 0.0, w, h)
        z_top = self._project(0.0, 0.0, 1.0, w, h)

        # Base perimeter.
        c.create_line(*lag_max, *lag_zero, fill="#666", width=2)
        c.create_line(*lag_max, *q_max_left, fill="#666", width=2)
        c.create_line(*q_max_left, *q_max_right, fill="#aeb7bd", width=1)
        c.create_line(*q_max_right, *lag_zero, fill="#aeb7bd", width=1)
        # Vertical metric axis on the right.
        c.create_line(*lag_zero, *z_top, fill="#555", width=2)

        for q in qs:
            y = 0.0 if len(qs) <= 1 else q_index[q] / (len(qs) - 1)
            p0 = self._project(0.0, y, 0.0, w, h)
            p1 = self._project(1.0, y, 0.0, w, h)
            c.create_line(*p0, *p1, fill="#e0e4e7")
        for lag in lags:
            x = 0.0 if len(lags) <= 1 else lag_index[lag] / (len(lags) - 1)
            p0 = self._project(x, 0.0, 0.0, w, h)
            p1 = self._project(x, 1.0, 0.0, w, h)
            c.create_line(*p0, *p1, fill="#e0e4e7")

        # Filled surface patches, back-to-front by projected grid depth.
        # Higher-threshold rows are farther away; at equal threshold the
        # smaller-lag/right side is drawn first.
        quads = sorted(
            mesh["quads"],
            key=lambda quad: (
                sum(q_index[k[0]] for k in quad) / len(quad),
                -sum(lag_index[k[1]] for k in quad) / len(quad),
            ),
            reverse=True,
        )
        for quad in quads:
            pts = []
            vals = []
            for key in quad:
                pts.extend(coords(key))
                vals.append(values[key])
            avg = sum(vals) / len(vals)
            c.create_polygon(
                *pts,
                fill=HeatmapPreview._color(avg, minimum, maximum),
                outline="#66757f",
                width=1,
            )

        # Surface grid lines.
        for q in qs:
            keys = [(q, lag) for lag in lags if (q, lag) in values]
            for first, second in zip(keys, keys[1:]):
                c.create_line(*coords(first), *coords(second), fill="#43535d", width=1)

        # Descriptive calibration reference plane. This is intentionally not
        # used by the optimization logic; it only visualizes the upper 97.5th
        # percentile of the synthetic negligible WMARMcs distribution.
        if (
            self.overlays.get("calibration_relevance")
            and self.metric == "wmarm_cs"
            and self.calibration_boundary is not None
            and minimum <= self.calibration_boundary <= maximum
        ):
            zn = self._normalise(self.calibration_boundary, minimum, maximum)
            plane = [
                self._project(0.0, 0.0, zn, w, h),
                self._project(1.0, 0.0, zn, w, h),
                self._project(1.0, 1.0, zn, w, h),
                self._project(0.0, 1.0, zn, w, h),
            ]
            flat = [coord for point in plane for coord in point]
            c.create_polygon(*flat, fill="", outline="#6a3d9a", width=2, dash=(5, 3))
            c.create_text(
                plane[0][0] + 6,
                plane[0][1] - 5,
                anchor="sw",
                text=f"negligible q97.5 {self._display_value(self.calibration_boundary)}",
                fill="#5a2f7a",
                font=("Segoe UI", 8, "bold"),
            )
        for lag in lags:
            keys = [(q, lag) for q in qs if (q, lag) in values]
            for first, second in zip(keys, keys[1:]):
                c.create_line(*coords(first), *coords(second), fill="#43535d", width=1)

        # Points, selection and optimization badges.
        for key, value in values.items():
            x, y = coords(key)
            self._point_hits.append((x, y, key[0], key[1]))
            selected = key == self.selected
            robust = self.overlays.get("robust") and key in self.robust_keys
            radius = 6 if selected else 4
            outline = "#111" if selected else ("#187d3c" if robust else "#34454f")
            width = 3 if selected or robust else 1
            c.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white" if selected else HeatmapPreview._color(value, minimum, maximum),
                outline=outline,
                width=width,
            )
            badges = self._overlay_badges(key, self.diagnostics.get(key) or {}, value)
            if badges:
                c.create_text(x + 7, y - 7, anchor="sw", text=badges, fill="#5b1a1a", font=("Segoe UI", 8, "bold"))

        # Axis labels and ticks.
        lag_mid = ((lag_max[0] + lag_zero[0]) / 2.0, (lag_max[1] + lag_zero[1]) / 2.0)
        q_mid = ((lag_max[0] + q_max_left[0]) / 2.0, (lag_max[1] + q_max_left[1]) / 2.0)
        c.create_text(lag_mid[0], lag_mid[1] + 30, anchor="n", text="Cumulative lag", fill="#444")
        try:
            c.create_text(q_mid[0] - 25, q_mid[1], text="Threshold q", fill="#444", angle=62)
        except tk.TclError:
            c.create_text(q_mid[0] - 28, q_mid[1], text="Threshold q", fill="#444")

        # Lag ticks: ascending data values are intentionally displayed from
        # right (lag 0) to left (larger lags), matching the reference view.
        for lag in lags:
            xnorm = 0.0 if len(lags) <= 1 else lag_index[lag] / (len(lags) - 1)
            x, y = self._project(xnorm, 0.0, 0.0, w, h)
            c.create_text(x, y + 9, anchor="n", text=str(lag), fill="#555", font=("Segoe UI", 8))

        # Threshold ticks are drawn on the left edge of the base plane.
        for q in qs:
            ynorm = 0.0 if len(qs) <= 1 else q_index[q] / (len(qs) - 1)
            x, y = self._project(1.0, ynorm, 0.0, w, h)
            c.create_text(x - 7, y, anchor="e", text=f"q{q:g}", fill="#555", font=("Segoe UI", 8))

        # Metric ticks on the right vertical axis.
        for zn, value in ((0.0, minimum), (0.5, (minimum + maximum) / 2.0), (1.0, maximum)):
            x, y = self._project(0.0, 0.0, zn, w, h)
            c.create_line(x, y, x + 5, y, fill="#555")
            c.create_text(x + 8, y, anchor="w", text=self._display_value(value), fill="#555", font=("Segoe UI", 8))
        c.create_text(z_top[0] + 35, (z_top[1] + lag_zero[1]) / 2.0, text="Metric", fill="#444", angle=90)

        c.create_text(
            w - 8,
            8,
            anchor="ne",
            text=f"range {self._display_value(minimum)} – {self._display_value(maximum)}",
            fill="#555",
            font=("Segoe UI", 8),
        )


class SurfacePanel(ttk.Frame):
    """Detailed threshold-lag surface explorer."""

    def __init__(self, parent, model: OverviewModel, on_select=None, calibration_model: CalibrationModel | None = None):
        super().__init__(parent, padding=14)
        self.model = model
        self.calibration_model = calibration_model
        self.calibration_boundary = (
            calibration_model.negligible_reference_boundary()
            if calibration_model is not None else None
        )
        self.on_select = on_select
        self.selected = model.default_cell_key
        self.secondary_metric_name = tk.StringVar(value="Diagnostic WMARM")
        self.view_mode = tk.StringVar(value="2D")
        self.overlay_vars = {
            "robust": tk.BooleanVar(value=True),
            "representative": tk.BooleanVar(value=True),
            "near_optimal": tk.BooleanVar(value=False),
            "null_supported": tk.BooleanVar(value=True),
            "stable": tk.BooleanVar(value=False),
            "calibration_relevance": tk.BooleanVar(value=False),
        }
        self.columnconfigure(0, weight=1)
        self.control_row = 2 if model.all_zero_supported else 1
        self.body_row = self.control_row + 1
        self.rowconfigure(self.body_row, weight=1)
        self._build_header()
        self._build_controls()
        self._build_body()
        self.set_selected_cell(*(self.selected or (None, None)))

    @property
    def diagnostic_lookup(self):
        result = {}
        for row in (self.model.optimization or {}).get("diagnostics") or []:
            key = self.model._key_from_dict(row)
            if key:
                result[key] = row
        return result

    def _build_header(self):
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Threshold-lag surface", font=("Segoe UI", 16, "bold")).pack(
            side="left"
        )
        summary = []
        if self.model.null_cutoff is not None:
            summary.append(f"null cutoff {_fmt_number(self.model.null_cutoff, 4)}")
        if self.model.p_max is not None:
            summary.append(f"pmax {_fmt_number(self.model.p_max, 4)}")
        if self.model.surface_coherence is not None:
            summary.append(f"LOYO coherence {_fmt_number(self.model.surface_coherence, 3)}")
        if self.calibration_boundary is not None:
            summary.append(f"negligible q97.5 {_fmt_number(self.calibration_boundary, 4)}")
        if summary:
            ttk.Label(header, text=" · ".join(summary)).pack(side="right")
        if self.model.all_zero_supported:
            note = ttk.Label(
                self,
                text=(
                    "WMARMcs is zero across the admissible surface. Near-optimal flags are therefore "
                    "degenerate and are hidden by default; diagnostic WMARM can still be explored."
                ),
                wraplength=1050,
                justify="left",
            )
            note.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def _build_controls(self):
        controls = ttk.Frame(self, padding=(0, 8))
        controls.grid(row=self.control_row, column=0, sticky="ew")
        ttk.Label(controls, text="View:").pack(side="left")
        for value, label in (("2D", "2D heatmaps"), ("3D", "3D surfaces")):
            ttk.Radiobutton(
                controls,
                text=label,
                value=value,
                variable=self.view_mode,
                command=self._apply_view_mode,
            ).pack(side="left", padx=(6 if value == "2D" else 2, 4))
        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(controls, text="Secondary surface:").pack(side="left")
        combo = ttk.Combobox(
            controls,
            textvariable=self.secondary_metric_name,
            values=list(SECONDARY_SURFACE_METRICS),
            state="readonly",
            width=24,
        )
        combo.pack(side="left", padx=(6, 18))
        combo.bind("<<ComboboxSelected>>", lambda event: self._refresh_heatmaps())
        ttk.Label(controls, text="Overlays:").pack(side="left")
        labels = (
            ("robust", "robust region"),
            ("representative", "representative"),
            ("near_optimal", "near-optimal"),
            ("null_supported", "null-supported"),
            ("stable", "LOYO-stable"),
        )
        for key, label in labels:
            ttk.Checkbutton(
                controls,
                text=label,
                variable=self.overlay_vars[key],
                command=self._refresh_heatmaps,
            ).pack(side="left", padx=(0, 8))
        if self.calibration_boundary is not None:
            ttk.Checkbutton(
                controls,
                text="calibration relevance",
                variable=self.overlay_vars["calibration_relevance"],
                command=self._refresh_heatmaps,
            ).pack(side="left", padx=(0, 8))
            ttk.Label(
                controls,
                text="C = observed WMARMcs above negligible q97.5, descriptive only",
                style="Overview.CardSub.TLabel",
            ).pack(side="left", padx=(4, 0))

    def _build_body(self):
        body = ttk.Frame(self)
        body.grid(row=self.body_row, column=0, sticky="nsew", pady=(6, 0))
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=4)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)
        self.reference_heatmap = HeatmapPreview(
            body,
            "Reference WMARMcs",
            "wmarm_cs",
            self._user_select,
            value_format="number",
            show_overlay_legend=True,
        )
        self.reference_heatmap.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.reference_3d = Surface3DPreview(
            body,
            "Reference WMARMcs",
            "wmarm_cs",
            self._user_select,
            value_format="number",
            show_overlay_legend=True,
        )
        self.reference_3d.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        metric, title, value_format = SECONDARY_SURFACE_METRICS[self.secondary_metric_name.get()]
        self.secondary_heatmap = HeatmapPreview(
            body,
            title,
            metric,
            self._user_select,
            value_format=value_format,
            show_overlay_legend=True,
        )
        self.secondary_heatmap.grid(row=0, column=1, sticky="nsew", padx=6)
        self.secondary_3d = Surface3DPreview(
            body,
            title,
            metric,
            self._user_select,
            value_format=value_format,
            show_overlay_legend=True,
        )
        self.secondary_3d.grid(row=0, column=1, sticky="nsew", padx=6)
        self.detail = SurfaceDetailPanel(body, self.model)
        self.detail.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self._apply_view_mode()

    def _overlay_state(self):
        return {key: var.get() for key, var in self.overlay_vars.items()}

    def _user_select(self, q, lag):
        if self.on_select:
            self.on_select(q, lag)
        else:
            self.set_selected_cell(q, lag)

    def _apply_view_mode(self):
        if not hasattr(self, "reference_heatmap"):
            return
        is_3d = self.view_mode.get() == "3D"
        if is_3d:
            self.reference_heatmap.grid_remove()
            self.secondary_heatmap.grid_remove()
            self.reference_3d.grid()
            self.secondary_3d.grid()
        else:
            self.reference_3d.grid_remove()
            self.secondary_3d.grid_remove()
            self.reference_heatmap.grid()
            self.secondary_heatmap.grid()
        self._refresh_heatmaps()

    def _refresh_heatmaps(self):
        """Refresh both 2D and 3D surfaces from the same read-only result state."""
        metric, title, value_format = SECONDARY_SURFACE_METRICS[self.secondary_metric_name.get()]
        self.secondary_heatmap.set_metric(metric, title, value_format)
        self.secondary_3d.set_metric(metric, title, value_format)
        diagnostics = self.diagnostic_lookup
        common = dict(
            selected=self.selected,
            diagnostics=diagnostics,
            overlays=self._overlay_state(),
            representative=self.model.representative_key,
            robust_keys=self.model.robust_keys,
            calibration_boundary=self.calibration_boundary,
        )
        self.reference_heatmap.set_data(self.model.surface, **common)
        self.secondary_heatmap.set_data(self.model.surface, **common)
        self.reference_3d.set_data(self.model.surface, **common)
        self.secondary_3d.set_data(self.model.surface, **common)

    def set_selected_cell(self, q, lag):
        if q is None or lag is None:
            self.selected = None
            row = None
            diagnostic = None
        else:
            key = (_float(q), _int(lag))
            row = self.model.cell(*key)
            if row is None:
                return
            self.selected = key
            diagnostic = self.model.optimization_cell(*key)
        self._refresh_heatmaps()
        self.detail.set_cell(row, diagnostic)

class CalibrationDistributionChart(ttk.Frame):
    """Dependency-free box/whisker summary for calibration effect classes."""

    def __init__(self, parent, model: CalibrationModel):
        super().__init__(parent, padding=8, relief="ridge")
        self.model = model
        self.metric = "wmarm_cs"
        self.value_format = "number"
        self.title = tk.StringVar(value="WMARMcs distribution by synthetic effect class")
        ttk.Label(self, textvariable=self.title, style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.canvas = tk.Canvas(self, height=310, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        ttk.Label(
            self,
            text="Whiskers: 2.5th–97.5th percentiles · box: 25th–75th · line: median · dot: mean",
            style="Overview.CardSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

    def set_metric(self, metric: str, title: str, value_format: str):
        self.metric = metric
        self.value_format = value_format
        self.title.set(f"{title} distribution by synthetic effect class")
        self.redraw()

    def _fmt_axis(self, value):
        if self.value_format == "percent":
            return f"{100 * value:.0f}%"
        return f"{value:.2f}"

    def redraw(self):
        c = self.canvas
        c.delete("all")
        classes = self.model.effect_classes
        rows = [(name, self.model.territory_distribution(name, self.metric)) for name in classes]
        rows = [(name, row) for name, row in rows if row]
        if not rows:
            c.create_text(15, 20, anchor="nw", text="No territorial calibration distribution for this metric", fill="#666")
            return
        values = []
        for _, row in rows:
            for key in ("q025", "q25", "median", "q75", "q975", "mean"):
                value = _float(row.get(key))
                if value is not None:
                    values.append(value)
        if not values:
            c.create_text(15, 20, anchor="nw", text="No finite calibration values", fill="#666")
            return
        minimum, maximum = min(values), max(values)
        if self.metric in {"wmarm_cs", "wmarm", "confidence_support_retention_ratio", "ci_share"}:
            minimum = min(0.0, minimum)
        if abs(maximum - minimum) <= ZERO_TOLERANCE:
            pad = 1.0 if abs(maximum) <= ZERO_TOLERANCE else abs(maximum) * 0.2
        else:
            pad = (maximum - minimum) * 0.12
        ymin, ymax = minimum - pad, maximum + pad
        w, h = max(c.winfo_width(), 700), max(c.winfo_height(), 280)
        left, right, top, bottom = 62, 20, 18, 72
        plot_w, plot_h = w - left - right, h - top - bottom

        def yy(value):
            return top + (ymax - value) / (ymax - ymin) * plot_h

        # Reference line at zero when visible.
        if ymin <= 0 <= ymax:
            y0 = yy(0)
            c.create_line(left, y0, w - right, y0, fill="#c8cdd1", dash=(3, 3))
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = yy(value)
            c.create_line(left - 4, y, left, y, fill="#666")
            c.create_text(left - 8, y, anchor="e", text=self._fmt_axis(value), fill="#555", font=("Segoe UI", 8))
        step = plot_w / max(1, len(rows))
        for i, (name, row) in enumerate(rows):
            x = left + (i + 0.5) * step
            q025, q25 = _float(row.get("q025")), _float(row.get("q25"))
            median, q75 = _float(row.get("median")), _float(row.get("q75"))
            q975, mean = _float(row.get("q975")), _float(row.get("mean"))
            if None not in (q025, q25, median, q75, q975):
                c.create_line(x, yy(q025), x, yy(q975), fill="#315d78", width=2)
                c.create_line(x - 8, yy(q025), x + 8, yy(q025), fill="#315d78", width=2)
                c.create_line(x - 8, yy(q975), x + 8, yy(q975), fill="#315d78", width=2)
                bw = min(48, step * 0.48)
                c.create_rectangle(x - bw / 2, yy(q75), x + bw / 2, yy(q25), fill="#dceaf2", outline="#315d78", width=2)
                c.create_line(x - bw / 2, yy(median), x + bw / 2, yy(median), fill="#152f40", width=2)
            if mean is not None:
                c.create_oval(x - 4, yy(mean) - 4, x + 4, yy(mean) + 4, fill="#152f40", outline="")
            label = _human_class(name).replace(" ", "\n", 1) if len(_human_class(name)) > 12 else _human_class(name)
            c.create_text(x, h - bottom + 12, anchor="n", text=label, fill="#333", font=("Segoe UI", 8), justify="center")
            valid_count = _int(row.get("valid_count"))
            availability = "n=0, unavailable" if valid_count == 0 else (f"n={valid_count}" if valid_count is not None else "n=—")
            c.create_text(x, h - 15, anchor="s", text=availability, fill="#666", font=("Segoe UI", 8), justify="center")


class CalibrationEffectPanel(ttk.Frame):
    def __init__(self, parent, model: CalibrationModel):
        super().__init__(parent, padding=10)
        self.model = model
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)
        self.metric_name = tk.StringVar(value="WMARMcs")
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Territorial metric:").pack(side="left")
        combo = ttk.Combobox(
            controls,
            textvariable=self.metric_name,
            values=list(CALIBRATION_METRICS),
            state="readonly",
            width=27,
        )
        combo.pack(side="left", padx=(6, 14))
        combo.bind("<<ComboboxSelected>>", lambda event: self._refresh_chart())
        ttk.Label(
            controls,
            text="Synthetic effect classes are defined by the configured outcome-percentile intervals.",
            style="Overview.CardSub.TLabel",
        ).pack(side="left")

        self.chart = CalibrationDistributionChart(self, model)
        self.chart.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Operating characteristics", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        columns = ("class", "error_power", "sign", "coverage")
        self.operating_tree = ttk.Treeview(right, columns=columns, show="headings", height=8)
        headings = {
            "class": "Class",
            "error_power": "Type I / Power",
            "sign": "Sign recovery",
            "coverage": "Ref. coverage",
        }
        widths = {"class": 175, "error_power": 110, "sign": 105, "coverage": 105}
        for key in columns:
            self.operating_tree.heading(key, text=headings[key])
            self.operating_tree.column(key, width=widths[key], anchor="center" if key != "class" else "w")
        self.operating_tree.grid(row=1, column=0, sticky="nsew", pady=(5, 10))

        ttk.Label(right, text="Magnitude-ordering diagnostics", style="Overview.SectionTitle.TLabel").grid(row=2, column=0, sticky="w")
        order_columns = ("direction", "cs", "point", "support", "n")
        self.order_tree = ttk.Treeview(right, columns=order_columns, show="headings", height=3)
        for key, heading, width in (
            ("direction", "Direction", 85), ("cs", "WMARMcs", 85),
            ("point", "WMARM", 85), ("support", "CI support", 85), ("n", "Matched n", 75),
        ):
            self.order_tree.heading(key, text=heading)
            self.order_tree.column(key, width=width, anchor="center")
        self.order_tree.grid(row=3, column=0, sticky="ew", pady=(5, 0))

        note = ttk.Label(
            right,
            text="WMARMcs ordering is the reference coherence diagnostic. Standard WMARM ordering remains descriptive.",
            wraplength=470,
            justify="left",
            style="Overview.CardSub.TLabel",
        )
        note.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self._populate_tables()
        self._refresh_chart()

    def _refresh_chart(self):
        metric, title, value_format = CALIBRATION_METRICS[self.metric_name.get()]
        self.chart.set_metric(metric, title, value_format)

    def _populate_tables(self):
        for item in self.operating_tree.get_children():
            self.operating_tree.delete(item)
        for name in self.model.effect_classes:
            row = self.model.operating_row(name) or {}
            type_i = _float(row.get("type_I_error"))
            power = _float(row.get("power"))
            error_power = (f"Type I {_fmt_percent(type_i)}" if type_i is not None else (f"Power {_fmt_percent(power)}" if power is not None else "—"))
            self.operating_tree.insert("", "end", values=(
                _human_class(name),
                error_power,
                _fmt_percent(row.get("sign_recovery")),
                _fmt_percent(row.get("empirical_reference_coverage")),
            ))
        for item in self.order_tree.get_children():
            self.order_tree.delete(item)
        for direction in ("harmful", "protective"):
            row = self.model.ordering_row(direction) or {}
            self.order_tree.insert("", "end", values=(
                direction.title(),
                _fmt_percent(row.get("wmarm_cs_strong_gt_moderate_gt_negligible_probability")),
                _fmt_percent(row.get("diagnostic_wmarm_strong_gt_moderate_gt_negligible_probability")),
                _fmt_percent(row.get("strict_CI_support_ordering_probability")),
                _int(row.get("valid_matched_replicates")) if row else "—",
            ))


class SyntheticSignalChart(ttk.Frame):
    """Observed versus synthetic daily-incidence signal."""

    def __init__(self, parent):
        super().__init__(parent, padding=8, relief="ridge")
        ttk.Label(self, text="Signal comparison", style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.canvas = tk.Canvas(self, height=280, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.dates = []
        self.observed = []
        self.synthetic = []
        self.subtitle = tk.StringVar(value="Select a synthetic realization and reconstruct it for inspection.")
        ttk.Label(
            self,
            textvariable=self.subtitle,
            style="Overview.CardSub.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(4, 0))

    def set_data(self, dates, observed, synthetic, subtitle=""):
        self.dates = [str(v)[:10] for v in dates]
        self.observed = list(observed)
        self.synthetic = list(synthetic)
        self.subtitle.set(subtitle)
        self.redraw()

    @staticmethod
    def _finite(value):
        value = _float(value)
        return value

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if not self.dates or not self.observed or not self.synthetic:
            c.create_text(15, 20, anchor="nw", text="No reconstructed signal loaded", fill="#666")
            return
        finite = [
            value
            for series in (self.observed, self.synthetic)
            for raw in series
            for value in [self._finite(raw)]
            if value is not None
        ]
        if not finite:
            c.create_text(15, 20, anchor="nw", text="No finite signal values", fill="#666")
            return
        ymin, ymax = min(finite), max(finite)
        if ymax <= ymin + ZERO_TOLERANCE:
            pad = 1.0 if abs(ymax) <= ZERO_TOLERANCE else abs(ymax) * 0.1
        else:
            pad = 0.08 * (ymax - ymin)
        ymin, ymax = max(0.0, ymin - pad), ymax + pad
        if ymax <= ymin:
            ymax = ymin + 1.0
        w, h = max(c.winfo_width(), 560), max(c.winfo_height(), 240)
        left, right, top, bottom = 58, 18, 20, 42
        plot_w, plot_h = w - left - right, h - top - bottom

        def xx(index):
            return left if len(self.dates) <= 1 else left + index * plot_w / (len(self.dates) - 1)

        def yy(value):
            return top + (ymax - value) * plot_h / (ymax - ymin)

        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = yy(value)
            c.create_line(left, y, w - right, y, fill="#edf0f2")
            c.create_text(left - 7, y, anchor="e", text=f"{value:.2f}", fill="#555", font=("Segoe UI", 8))
        c.create_text(6, top + plot_h / 2, text="Daily incidence", angle=90, fill="#555")

        def draw_series(values, color, width):
            segment = []
            for i, raw in enumerate(values):
                value = self._finite(raw)
                if value is None:
                    if len(segment) >= 4:
                        c.create_line(*segment, fill=color, width=width)
                    segment = []
                    continue
                segment.extend((xx(i), yy(value)))
            if len(segment) >= 4:
                c.create_line(*segment, fill=color, width=width)
            elif len(segment) == 2:
                x, y = segment
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")

        draw_series(self.observed, "#2b5d7e", 2)
        draw_series(self.synthetic, "#a14f2a", 2)
        for index in sorted({0, max(0, len(self.dates) // 2), len(self.dates) - 1}):
            c.create_text(xx(index), h - bottom + 8, anchor="n", text=self.dates[index], fill="#555", font=("Segoe UI", 8))
        c.create_line(w - 205, 10, w - 180, 10, fill="#2b5d7e", width=2)
        c.create_text(w - 175, 10, anchor="w", text="Observed", fill="#333", font=("Segoe UI", 8))
        c.create_line(w - 105, 10, w - 80, 10, fill="#a14f2a", width=2)
        c.create_text(w - 75, 10, anchor="w", text="Synthetic", fill="#333", font=("Segoe UI", 8))


class SyntheticDistributionChart(ttk.Frame):
    """Overlaid empirical CDFs for observed and synthetic incidence."""

    def __init__(self, parent):
        super().__init__(parent, padding=8, relief="ridge")
        ttk.Label(self, text="Distribution comparison", style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.canvas = tk.Canvas(self, height=280, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.observed = []
        self.synthetic = []
        ttk.Label(
            self,
            text="Empirical cumulative distributions, same daily-incidence scale as the synthetic generator.",
            style="Overview.CardSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

    def set_data(self, observed, synthetic):
        self.observed = list(observed)
        self.synthetic = list(synthetic)
        self.redraw()

    def redraw(self):
        import numpy as np

        c = self.canvas
        c.delete("all")
        a = np.sort(np.asarray(self.observed, dtype=float)[np.isfinite(self.observed)]) if self.observed else np.array([])
        b = np.sort(np.asarray(self.synthetic, dtype=float)[np.isfinite(self.synthetic)]) if self.synthetic else np.array([])
        if not a.size or not b.size:
            c.create_text(15, 20, anchor="nw", text="No finite distribution data", fill="#666")
            return
        xmin, xmax = min(float(a[0]), float(b[0])), max(float(a[-1]), float(b[-1]))
        if xmax <= xmin + ZERO_TOLERANCE:
            xmax = xmin + 1.0
        w, h = max(c.winfo_width(), 420), max(c.winfo_height(), 240)
        left, right, top, bottom = 52, 18, 20, 42
        plot_w, plot_h = w - left - right, h - top - bottom

        def xx(value):
            return left + (value - xmin) * plot_w / (xmax - xmin)

        def yy(probability):
            return top + (1.0 - probability) * plot_h

        for tick in range(5):
            p = tick / 4
            y = yy(p)
            c.create_line(left, y, w - right, y, fill="#edf0f2")
            c.create_text(left - 7, y, anchor="e", text=f"{p:.2f}", fill="#555", font=("Segoe UI", 8))
        for tick in range(5):
            value = xmin + (xmax - xmin) * tick / 4
            x = xx(value)
            c.create_text(x, h - bottom + 8, anchor="n", text=f"{value:.2f}", fill="#555", font=("Segoe UI", 8))

        def draw_ecdf(values, color):
            n = len(values)
            points = []
            for i, value in enumerate(values, 1):
                points.extend((xx(float(value)), yy(i / n)))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2)

        draw_ecdf(a, "#2b5d7e")
        draw_ecdf(b, "#a14f2a")
        c.create_text(left + plot_w / 2, h - 4, anchor="s", text="Daily incidence", fill="#555")
        c.create_text(7, top + plot_h / 2, text="Empirical CDF", angle=90, fill="#555")
        c.create_line(w - 205, 10, w - 180, 10, fill="#2b5d7e", width=2)
        c.create_text(w - 175, 10, anchor="w", text="Observed", fill="#333", font=("Segoe UI", 8))
        c.create_line(w - 105, 10, w - 80, 10, fill="#a14f2a", width=2)
        c.create_text(w - 75, 10, anchor="w", text="Synthetic", fill="#333", font=("Segoe UI", 8))


class CalibrationSyntheticPanel(ttk.Frame):
    """Read-only reconstruction and comparison of calibration outcomes."""

    TERRITORY = "Territory, population weighted"

    def __init__(self, parent, model: CalibrationModel):
        super().__init__(parent, padding=10)
        self.model = model
        self._data = None
        self._config = None
        self._observed_incidence = None
        self._synthetic = None
        self._loaded_run = None
        self._loaded_sha = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._load_selector_metadata()
        self._build_controls()
        self._build_diagnostics()
        self._build_body()

    def _load_selector_metadata(self):
        meta_path = self.model.project_dir / "canonical" / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.bsa_ids = [str(v) for v in meta.get("bsa_ids") or []]
            self.years = [str(v) for v in meta.get("years") or []]
        else:
            self.bsa_ids, self.years = [], []
        self.run_rows = self.model.synthetic_runs
        self.run_labels = {}
        for row in self.run_rows:
            run = str(row.get("run"))
            class_name = _human_class(row.get("class"))
            lag = _int(row.get("lag_generating"))
            if lag is None:
                description = f"RC, {class_name}"
            else:
                description = (
                    f"{_human_mode(row.get('threshold_lag_response_mode'))}, {class_name}, "
                    f"{_fmt_q(row.get('q_generating'))}, lag {lag}"
                )
            self.run_labels[f"{description} | {run}"] = run

    def _build_controls(self):
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Synthetic realization:").pack(side="left")
        self.run_name = tk.StringVar(value=next(iter(self.run_labels), ""))
        run_combo = ttk.Combobox(
            controls,
            textvariable=self.run_name,
            values=list(self.run_labels),
            state="readonly" if self.run_labels else "disabled",
            width=58,
        )
        run_combo.pack(side="left", padx=(6, 12))
        ttk.Label(controls, text="Spatial unit:").pack(side="left")
        self.bsa_name = tk.StringVar(value=self.TERRITORY if self.bsa_ids else "")
        bsa_combo = ttk.Combobox(
            controls,
            textvariable=self.bsa_name,
            values=[self.TERRITORY] + self.bsa_ids if self.bsa_ids else [],
            state="readonly" if self.bsa_ids else "disabled",
            width=28,
        )
        bsa_combo.pack(side="left", padx=(6, 12))
        ttk.Label(controls, text="Period:").pack(side="left")
        self.year_name = tk.StringVar(value=self.years[-1] if self.years else "All dates")
        year_combo = ttk.Combobox(
            controls,
            textvariable=self.year_name,
            values=["All dates"] + self.years,
            state="readonly" if self.years else "disabled",
            width=10,
        )
        year_combo.pack(side="left", padx=(6, 12))
        self.load_button = ttk.Button(controls, text="Reconstruct and compare", command=self.reconstruct)
        self.load_button.pack(side="left")
        bsa_combo.bind("<<ComboboxSelected>>", lambda event: self._render_cached())
        year_combo.bind("<<ComboboxSelected>>", lambda event: self._render_cached())
        run_combo.bind("<<ComboboxSelected>>", lambda event: self._run_changed())
        self.status = tk.StringVar(
            value=(
                "Select a calibration realization. Reconstruction is isolated and does not modify the project."
                if self.run_labels else
                "No synthetic generation metadata were found in this project."
            )
        )
        ttk.Label(
            self,
            textvariable=self.status,
            style="Overview.CardSub.TLabel",
            wraplength=1180,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 8))

    def _build_diagnostics(self):
        self.diag_frame = ttk.Frame(self)
        self.diag_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.diag_cards = {}
        specs = (
            ("paired", "Paired days"),
            ("correlation", "Signal correlation"),
            ("mae", "Mean absolute difference"),
            ("rmse", "Root mean square difference"),
            ("ecdf", "ECDF maximum gap"),
        )
        for i, (key, title) in enumerate(specs):
            self.diag_frame.columnconfigure(i, weight=1, uniform="synthetic-diagnostic")
            card = MetricCard(self.diag_frame, title, "—", "descriptive comparison")
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 5, 0))
            self.diag_cards[key] = card

    def _build_body(self):
        body = ttk.Frame(self)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        self.signal_chart = SyntheticSignalChart(body)
        self.signal_chart.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        self.distribution_chart = SyntheticDistributionChart(body)
        self.distribution_chart.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        summary = ttk.Frame(body, padding=8, relief="ridge")
        summary.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))
        summary.columnconfigure(0, weight=1)
        summary.rowconfigure(1, weight=1)
        ttk.Label(summary, text="Distribution indicators", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        columns = ("statistic", "observed", "synthetic", "difference")
        self.summary_tree = ttk.Treeview(summary, columns=columns, show="headings", height=11)
        for key, heading, width in (
            ("statistic", "Indicator", 140),
            ("observed", "Observed", 105),
            ("synthetic", "Synthetic", 105),
            ("difference", "Synthetic minus observed", 145),
        ):
            self.summary_tree.heading(key, text=heading)
            self.summary_tree.column(key, width=width, anchor="e" if key != "statistic" else "w")
        self.summary_tree.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _run_changed(self):
        selected = self.run_labels.get(self.run_name.get())
        if selected != self._loaded_run:
            self.status.set("Selected realization has not yet been reconstructed in this Results Workspace session.")

    def _ensure_context(self):
        if self._data is not None:
            return
        import numpy as np

        from . import numerics as nm
        from .data import Data

        config_path = self.model.project_dir / "resolved_config.json"
        if not config_path.is_file():
            raise FileNotFoundError("resolved_config.json is required for synthetic comparison")
        self._config = json.loads(config_path.read_text(encoding="utf-8"))
        self._data = Data(self.model.project_dir / "canonical")
        scaling = float(self._config["core"]["incidence"]["scaling_multiplier"])
        observed = np.full((self._data.B, self._data.n), np.nan, dtype=float)
        for b in range(self._data.B):
            observed[b] = nm.cumulative_incidence(
                self._data.series("outcome", b),
                self._data.daily_population(b),
                0,
                scaling,
                "EVENT_COUNTS",
                self._data.segment_index,
            )
        self._observed_incidence = observed

    def reconstruct(self):
        run = self.run_labels.get(self.run_name.get())
        metadata = self.model.synthetic_run(run) if run else None
        if metadata is None:
            messagebox.showwarning("Synthetic comparison", "No synthetic realization is selected.", parent=self)
            return
        self.load_button.configure(state="disabled")
        self.status.set(f"Reconstructing {run} from retained calibration metadata...")
        self.update_idletasks()
        try:
            self._ensure_context()
            synthetic, actual_sha = reconstruct_synthetic_outcome(self.model.project_dir, metadata)
            self._synthetic = synthetic
            self._loaded_run = run
            self._loaded_sha = actual_sha
            self._render_cached()
        except Exception as exc:
            self._synthetic = None
            self._loaded_run = None
            self._loaded_sha = None
            self.status.set(f"Synthetic reconstruction failed: {exc}")
            messagebox.showerror("Synthetic comparison", f"Could not reconstruct the selected realization:\n{exc}", parent=self)
        finally:
            self.load_button.configure(state="normal")

    @staticmethod
    def _weighted_territory(matrix, population):
        import numpy as np

        matrix = np.asarray(matrix, dtype=float)
        population = np.asarray(population, dtype=float)
        valid = np.isfinite(matrix) & np.isfinite(population) & (population > 0)
        numerator = np.sum(np.where(valid, matrix * population, 0.0), axis=0)
        denominator = np.sum(np.where(valid, population, 0.0), axis=0)
        result = np.full(matrix.shape[1], np.nan, dtype=float)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    def _series_for_selection(self):
        import numpy as np

        if self._data is None or self._synthetic is None:
            return None
        spatial = self.bsa_name.get()
        if spatial == self.TERRITORY:
            population = self._data.population[:, self._data.year_index]
            observed = self._weighted_territory(self._observed_incidence, population)
            synthetic = self._weighted_territory(self._synthetic, population)
        else:
            try:
                b = self._data.ids.index(spatial)
            except ValueError:
                return None
            observed = self._observed_incidence[b]
            synthetic = self._synthetic[b]
        dates = self._data.dates
        year = self.year_name.get()
        if year and year != "All dates":
            mask = self._data.year_index == self._data.years.index(int(year))
            return dates[mask], observed[mask], synthetic[mask]
        return dates, observed, synthetic

    def _render_cached(self):
        selected_run = self.run_labels.get(self.run_name.get())
        if self._synthetic is None or selected_run != self._loaded_run:
            return
        selected = self._series_for_selection()
        if selected is None:
            return
        dates, observed, synthetic = selected
        obs_summary = distribution_summary(observed)
        syn_summary = distribution_summary(synthetic)
        diagnostics = signal_comparison_diagnostics(observed, synthetic)
        location = self.bsa_name.get() or "—"
        period = self.year_name.get() or "All dates"
        subtitle = (
            f"{location}, {period}. Observed event counts are converted with the same zero-lag incidence rule used by calibration."
        )
        self.signal_chart.set_data(dates, observed, synthetic, subtitle)
        self.distribution_chart.set_data(observed, synthetic)
        self._populate_summary(obs_summary, syn_summary)
        self.diag_cards["paired"].value.set(str(diagnostics["paired_count"]))
        self.diag_cards["correlation"].value.set(_fmt_number(diagnostics["correlation"], 3))
        self.diag_cards["mae"].value.set(_fmt_number(diagnostics["mae"], 3))
        self.diag_cards["rmse"].value.set(_fmt_number(diagnostics["rmse"], 3))
        self.diag_cards["ecdf"].value.set(_fmt_number(diagnostics["ecdf_max_gap"], 3))
        metadata = self.model.synthetic_run(self._loaded_run) or {}
        class_name = _human_class(metadata.get("class"))
        percentile_draw = _float(metadata.get("percentile_draw"))
        draw_text = "exact null" if percentile_draw is None else f"percentile draw {percentile_draw:.4f}"
        self.status.set(
            f"Verified reconstruction: {self._loaded_run}, {class_name}, {draw_text}, SHA-256 {self._loaded_sha}. "
            "Comparison is descriptive and does not change calibration or optimization."
        )

    def _populate_summary(self, observed: dict, synthetic: dict):
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        rows = (
            ("Valid days", "valid_count", True),
            ("Missing days", "missing_count", True),
            ("Mean", "mean", False),
            ("SD", "sd", False),
            ("Minimum", "minimum", False),
            ("2.5th percentile", "q025", False),
            ("25th percentile", "q25", False),
            ("Median", "median", False),
            ("75th percentile", "q75", False),
            ("97.5th percentile", "q975", False),
            ("Maximum", "maximum", False),
            ("IQR", "iqr", False),
        )
        for label, key, integer in rows:
            ov, sv = observed.get(key), synthetic.get(key)
            if integer:
                ofmt = "—" if ov is None else str(int(ov))
                sfmt = "—" if sv is None else str(int(sv))
                diff = None if ov is None or sv is None else int(sv) - int(ov)
                dfmt = "—" if diff is None else f"{diff:+d}"
            else:
                ofmt = _fmt_number(ov, 3)
                sfmt = _fmt_number(sv, 3)
                diff = None if _float(ov) is None or _float(sv) is None else float(sv) - float(ov)
                dfmt = "—" if diff is None else f"{diff:+.3f}"
            self.summary_tree.insert("", "end", values=(label, ofmt, sfmt, dfmt))


class CalibrationCellDetailPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10, relief="ridge")
        ttk.Label(self, text="Selected candidate cell", style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.title = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.title, style="Overview.CardValue.TLabel").pack(anchor="w", pady=(4, 8))
        self.values = {}
        for key, label in (
            ("cs_mean", "Mean WMARMcs"), ("cs_median", "Median WMARMcs"),
            ("cs_interval", "2.5–97.5% range"), ("cs_sd", "SD WMARMcs"),
            ("wmarm_mean", "Mean diagnostic WMARM"), ("valid", "Valid realizations"),
        ):
            row = ttk.Frame(self); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            var = tk.StringVar(value="—"); ttk.Label(row, textvariable=var).pack(side="right")
            self.values[key] = var
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(self, text="Generating-scenario recovery", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for key, label in (
            ("valid_recovery", "Valid for recovery"),
            ("regions", "Nonempty robust regions"), ("threshold", "Threshold recovery"),
            ("lag", "Lag recovery"), ("joint", "Joint recovery"),
            ("threshold_cond", "Threshold recovery | nonempty"),
            ("lag_cond", "Lag recovery | nonempty"), ("joint_cond", "Joint recovery | nonempty"),
        ):
            row = ttk.Frame(self); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            var = tk.StringVar(value="—"); ttk.Label(row, textvariable=var).pack(side="right")
            self.values[key] = var
        self.generated_note = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.generated_note, wraplength=300, justify="left", style="Overview.CardSub.TLabel").pack(anchor="w", pady=(8, 0))

    def set_data(self, cell: dict | None, recovery: dict | None, generating_pair):
        qg, lg = generating_pair if generating_pair else (None, None)
        if cell:
            q, lag = _float(cell.get("q")), _int(cell.get("lag"))
            self.title.set(f"{_fmt_q(q)}, lag {lag}")
            self.values["cs_mean"].set(_fmt_number(cell.get("wmarm_cs_mean", cell.get("mean")), 4))
            self.values["cs_median"].set(_fmt_number(cell.get("wmarm_cs_median", cell.get("median")), 4))
            self.values["cs_interval"].set(
                f"{_fmt_number(cell.get('wmarm_cs_q025', cell.get('q025')), 4)} – {_fmt_number(cell.get('wmarm_cs_q975', cell.get('q975')), 4)}"
            )
            self.values["cs_sd"].set(_fmt_number(cell.get("wmarm_cs_sd", cell.get("sd")), 4))
            self.values["wmarm_mean"].set(_fmt_number(cell.get("diagnostic_wmarm_mean"), 4))
            self.values["valid"].set(str(_int(cell.get("wmarm_cs_valid_count", cell.get("valid_count"))) or "—"))
            self.generated_note.set(
                "This is the generating cell." if q == _float(qg) and lag == _int(lg)
                else f"Generating cell: {_fmt_q(qg)}, lag {_int(lg)}."
            )
        else:
            self.title.set("—")
            for key in ("cs_mean", "cs_median", "cs_interval", "cs_sd", "wmarm_mean", "valid"):
                self.values[key].set("—")
            self.generated_note.set("")
        r = recovery or {}
        total, valid = _int(r.get("total_realizations")), _int(r.get("valid_realizations"))
        nonempty = _int(r.get("nonempty_robust_regions"))
        self.values["valid_recovery"].set("—" if total is None else f"{valid or 0} / {total}")
        self.values["regions"].set("—" if valid is None else f"{nonempty or 0} / {valid}")
        self.values["threshold"].set(_fmt_percent(r.get("threshold_recovered_unconditional")))
        self.values["lag"].set(_fmt_percent(r.get("lag_recovered_unconditional")))
        self.values["joint"].set(_fmt_percent(r.get("joint_recovered_unconditional")))
        self.values["threshold_cond"].set(_fmt_percent(r.get("threshold_recovered_conditional_nonempty")))
        self.values["lag_cond"].set(_fmt_percent(r.get("lag_recovered_conditional_nonempty")))
        self.values["joint_cond"].set(_fmt_percent(r.get("joint_recovered_conditional_nonempty")))


class CalibrationThresholdLagPanel(ttk.Frame):
    def __init__(self, parent, model: CalibrationModel):
        super().__init__(parent, padding=10)
        self.model = model
        self.mode = tk.StringVar(value=model.primary_mode or (model.response_modes[0] if model.response_modes else ""))
        pairs = model.generating_pairs
        self.pair_labels = {f"{_fmt_q(q)}, lag {lag}": (q, lag) for q, lag in pairs}
        first_pair = next(iter(self.pair_labels), "")
        self.pair_name = tk.StringVar(value=first_pair)
        self.selected_candidate = self.pair_labels.get(first_pair)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_controls()
        self._build_summary_cards()
        self._build_mode_comparison()
        self._build_body()
        self.refresh_view()

    @property
    def generating_pair(self):
        return self.pair_labels.get(self.pair_name.get())

    def _build_controls(self):
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Response mode:").pack(side="left")
        mode_combo = ttk.Combobox(controls, textvariable=self.mode, values=self.model.response_modes, state="readonly", width=18)
        mode_combo.pack(side="left", padx=(6, 18))
        mode_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh_view(reset_candidate=True))
        ttk.Label(controls, text="Generating pair:").pack(side="left")
        pair_combo = ttk.Combobox(controls, textvariable=self.pair_name, values=list(self.pair_labels), state="readonly", width=18)
        pair_combo.pack(side="left", padx=(6, 18))
        pair_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh_view(reset_candidate=True))
        self.role_text = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.role_text, style="Overview.CardSub.TLabel").pack(side="left")

    def _build_summary_cards(self):
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.cards = {}
        specs = (
            ("n", "Valid realizations", "valid for recovery / completed"),
            ("robust", "Robust-region rate", "nonempty robust region"),
            ("threshold", "Threshold recovery", "unconditional"),
            ("lag", "Lag recovery", "unconditional"),
            ("joint", "Joint recovery", "unconditional"),
        )
        for i, (key, title, subtitle) in enumerate(specs):
            frame.columnconfigure(i, weight=1, uniform="calibration-card")
            card = MetricCard(frame, title, "—", subtitle)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 5, 0))
            self.cards[key] = card

    def _build_mode_comparison(self):
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(frame, text="Response-mode recovery summary", style="Overview.SectionTitle.TLabel").pack(side="left", padx=(0, 12))
        columns = ("mode", "role", "valid", "robust", "threshold", "lag", "joint")
        self.mode_tree = ttk.Treeview(frame, columns=columns, show="headings", height=max(2, len(self.model.response_modes)))
        for key, title, width in (
            ("mode", "Mode", 110), ("role", "Role", 210), ("valid", "Valid / total", 85),
            ("robust", "Robust region", 95), ("threshold", "Threshold", 85),
            ("lag", "Lag", 75), ("joint", "Joint", 75),
        ):
            self.mode_tree.heading(key, text=title)
            self.mode_tree.column(key, width=width, anchor="center" if key not in ("mode", "role") else "w")
        self.mode_tree.pack(side="left", fill="x", expand=True)
        self._populate_mode_tree()

    def _build_body(self):
        body = ttk.Frame(self)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=4); body.columnconfigure(1, weight=4); body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)
        self.cs_heatmap = HeatmapPreview(body, "Mean WMARMcs", "wmarm_cs_mean", self.select_candidate)
        self.cs_heatmap.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.wmarm_heatmap = HeatmapPreview(body, "Mean diagnostic WMARM", "diagnostic_wmarm_mean", self.select_candidate)
        self.wmarm_heatmap.grid(row=0, column=1, sticky="nsew", padx=6)
        self.detail = CalibrationCellDetailPanel(body)
        self.detail.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

    def _populate_mode_tree(self):
        for item in self.mode_tree.get_children():
            self.mode_tree.delete(item)
        for mode in self.model.response_modes:
            s = self.model.mode_summary(mode)
            self.mode_tree.insert("", "end", values=(
                _human_mode(mode),
                str(s.get("response_role") or "—").replace("_", " ").title(),
                f"{s.get('valid_realizations') or 0} / {s.get('total_realizations') or 0}",
                _fmt_percent(s.get("robust_region_rate")),
                _fmt_percent(s.get("threshold_recovered_unconditional")),
                _fmt_percent(s.get("lag_recovered_unconditional")),
                _fmt_percent(s.get("joint_recovered_unconditional")),
            ))

    def select_candidate(self, q, lag):
        self.selected_candidate = (_float(q), _int(lag))
        self._refresh_surfaces()

    def refresh_view(self, reset_candidate=False):
        mode = self.mode.get()
        pair = self.generating_pair
        if not mode or not pair:
            return
        if reset_candidate or self.selected_candidate is None:
            self.selected_candidate = pair
        summary = self.model.mode_summary(mode)
        self.role_text.set(str(summary.get("response_role") or "").replace("_", " ").title())
        self.cards["n"].value.set(
            f"{summary.get('valid_realizations') or 0} / {summary.get('total_realizations') or 0}"
        )
        self.cards["robust"].value.set(_fmt_percent(summary.get("robust_region_rate")))
        self.cards["threshold"].value.set(_fmt_percent(summary.get("threshold_recovered_unconditional")))
        self.cards["lag"].value.set(_fmt_percent(summary.get("lag_recovered_unconditional")))
        self.cards["joint"].value.set(_fmt_percent(summary.get("joint_recovered_unconditional")))
        rows = self.model.mean_surface_rows(mode, *pair)
        keys = {(_float(r.get("q")), _int(r.get("lag"))) for r in rows}
        if self.selected_candidate not in keys:
            self.selected_candidate = pair if pair in keys else (next(iter(keys)) if keys else None)
        self._refresh_surfaces()

    def _refresh_surfaces(self):
        mode, pair = self.mode.get(), self.generating_pair
        if not mode or not pair:
            self.cs_heatmap.set_data([]); self.wmarm_heatmap.set_data([])
            self.detail.set_data(None, None, None)
            return
        rows = self.model.mean_surface_rows(mode, *pair)
        self.cs_heatmap.set_data(rows, selected=self.selected_candidate)
        self.wmarm_heatmap.set_data(rows, selected=self.selected_candidate)
        cell = self.model.mean_surface_cell(mode, *pair, *(self.selected_candidate or (None, None))) if self.selected_candidate else None
        recovery = self.model.recovery_row(mode, *pair)
        self.detail.set_data(cell, recovery, pair)


class CalibrationPanel(ttk.Frame):
    """Read-only calibration results page."""

    def __init__(self, parent, model: CalibrationModel):
        super().__init__(parent, padding=14)
        self.model = model
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_header()
        if not model.available:
            ttk.Label(
                self,
                text="No completed calibration outputs were found for this project. Run calibration and refresh the Results Workspace.",
                wraplength=900,
                justify="left",
            ).grid(row=1, column=0, sticky="nw", pady=(18, 0))
            return
        self._build_cards()
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.effect_page = CalibrationEffectPanel(self.notebook, model)
        self.threshold_lag_page = CalibrationThresholdLagPanel(self.notebook, model)
        self.synthetic_page = CalibrationSyntheticPanel(self.notebook, model)
        self.notebook.add(self.effect_page, text="Effect magnitude")
        self.notebook.add(self.threshold_lag_page, text="Threshold and lag")
        self.notebook.add(self.synthetic_page, text="Synthetic vs observed")

    def _build_header(self):
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Calibration", font=("Segoe UI", 16, "bold")).pack(side="left")
        if self.model.available:
            ttk.Label(
                header,
                text="Synthetic outcomes evaluated through the same APHREH-ADSMap estimator",
                style="Overview.CardSub.TLabel",
            ).pack(side="left", padx=(14, 0))

    def _build_cards(self):
        cards = ttk.Frame(self)
        cards.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        specs = (
            ("Status", self.model.status, "calibration workflow"),
            ("Integrity", self.model.integrity_status, "automated calibration QC"),
            ("Reference metric", "WMARMcs", "territorial relevance"),
            ("Calibration threshold", _fmt_q(self.model.calibration_threshold), "RC magnitude calibration"),
            ("Realizations / scenario", str(self.model.realizations_per_scenario or "—"), "Monte Carlo ensemble"),
            ("Primary response", _human_mode(self.model.primary_mode), "threshold-lag calibration"),
        )
        for i, (title, value, subtitle) in enumerate(specs):
            cards.columnconfigure(i, weight=1, uniform="calibration-header-card")
            MetricCard(cards, title, value, subtitle).grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 5, 0))
        if self.model.concerns:
            codes = []
            for concern in self.model.concerns:
                code = str(concern.get("code") or "CALIBRATION_WARNING")
                if code not in codes:
                    codes.append(code)
            message = "Calibration QC requires inspection: " + ", ".join(codes[:5])
            if len(codes) > 5:
                message += f" (+{len(codes)-5} more)"
            ttk.Label(
                cards, text=message, wraplength=1450, justify="left",
                style="Overview.CardSub.TLabel",
            ).grid(row=1, column=0, columnspan=len(specs), sticky="w", pady=(8, 0))


class DiagnosticsSupportPanel(ttk.Frame):
    def __init__(self, parent, overview: OverviewModel, diagnostics: DiagnosticsModel):
        super().__init__(parent, padding=10)
        self.overview = overview
        self.diagnostics = diagnostics
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        summary = ttk.Frame(self, padding=10, relief="ridge")
        summary.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(summary, text="Selected-cell support", style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.values = {}
        rows = (
            ("eligible_bsas", "Eligible BSAs"),
            ("represented_population", "Represented population"),
            ("contributing_years", "Contributing years min / mean / max"),
            ("mean_i2", "Mean I²"),
            ("bootstrap_validity", "Bootstrap validity"),
            ("local_concordance_exposed", "Local concordance, exposed"),
            ("local_concordance_nonexposed", "Local concordance, non-exposed"),
            ("relative_precision", "Relative precision"),
            ("directional_stability", "Directional stability"),
            ("sign_concordance", "Year-pooled sign concordance"),
            ("spatial_coherence_cs", "Spatial directional coherence, CS"),
            ("spatial_dispersion_cs", "Spatial magnitude dispersion, CS"),
        )
        for key, label in rows:
            line = ttk.Frame(summary)
            line.pack(fill="x", pady=2)
            ttk.Label(line, text=label).pack(side="left")
            var = tk.StringVar(value="—")
            ttk.Label(line, textvariable=var).pack(side="right")
            self.values[key] = var

        table_box = ttk.Frame(self, padding=10, relief="ridge")
        table_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        table_box.columnconfigure(0, weight=1)
        table_box.rowconfigure(1, weight=1)
        ttk.Label(table_box, text="Yearly estimator support", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        cols = ("year", "eligible", "bootstrap", "exposed", "effective", "donors", "miss_inc", "miss_base")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=12)
        labels = {
            "year":"Year", "eligible":"Eligible BSA", "bootstrap":"Bootstrap valid", "exposed":"Mean exposed",
            "effective":"Effective exposed", "donors":"Baseline donors", "miss_inc":"Missing incidence", "miss_base":"Missing baseline",
        }
        widths = {"year":70,"eligible":90,"bootstrap":100,"exposed":90,"effective":100,"donors":100,"miss_inc":105,"miss_base":105}
        for col in cols:
            self.tree.heading(col, text=labels[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.tree.configure(yscrollcommand=scroll.set)

    def set_cell(self, q, lag):
        row = self.overview.cell(q, lag)
        if not row:
            return
        eligible, total = _int(row.get("eligible_bsas")), _int(row.get("total_bsas"))
        self.values["eligible_bsas"].set("—" if eligible is None else f"{eligible} / {total if total is not None else '—'}")
        represented = _float(row.get("represented_population")); reference = _float(row.get("total_reference_population"))
        self.values["represented_population"].set(
            "—" if represented is None else f"{represented:,.0f} / {reference:,.0f}" if reference is not None else f"{represented:,.0f}"
        )
        ymin, ymean, ymax = _int(row.get("contributing_years_min")), _float(row.get("contributing_years_mean")), _int(row.get("contributing_years_max"))
        self.values["contributing_years"].set("—" if ymean is None else f"{ymin if ymin is not None else '—'} / {ymean:.1f} / {ymax if ymax is not None else '—'}")
        self.values["mean_i2"].set("—" if _float(row.get("mean_i_squared_percent")) is None else f"{_float(row.get('mean_i_squared_percent')):.1f}%")
        self.values["bootstrap_validity"].set(_fmt_percent(row.get("bootstrap_validity")))
        self.values["local_concordance_exposed"].set(_fmt_percent(row.get("local_concordance_exposed")))
        self.values["local_concordance_nonexposed"].set(_fmt_percent(row.get("local_concordance_nonexposed")))
        self.values["relative_precision"].set(_fmt_number(row.get("relative_precision"), 3))
        self.values["directional_stability"].set(_fmt_number(row.get("directional_stability"), 3))
        self.values["sign_concordance"].set(_fmt_number(row.get("year_pooled_sign_concordance"), 3))
        self.values["spatial_coherence_cs"].set(_fmt_number(row.get("spatial_directional_coherence_cs"), 3))
        self.values["spatial_dispersion_cs"].set(_fmt_number(row.get("spatial_magnitude_dispersion_cs"), 4))
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.diagnostics.year_summaries(q, lag):
            self.tree.insert("", "end", values=(
                item["year"], f"{item['eligible']} / {item['total']}", _fmt_percent(item["bootstrap_validity"]),
                _fmt_number(item["n_exposed"], 1), _fmt_number(item["effective_exposed"], 1),
                _fmt_number(item["baseline_donors"], 1), item["missing_incidence"], item["missing_baseline"],
            ))


class DiagnosticsBsaPanel(ttk.Frame):
    def __init__(self, parent, diagnostics: DiagnosticsModel):
        super().__init__(parent, padding=10)
        self.diagnostics = diagnostics
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.summary = tk.StringVar(value="")
        ttk.Label(self, text="Pooled BSA effects and heterogeneity", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.summary, style="Overview.CardSub.TLabel").grid(row=0, column=0, sticky="e")
        cols = ("bsa", "rc", "ci", "support", "years", "tau2", "i2", "eligible", "reason")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        labels = {"bsa":"BSA", "rc":"Pooled RC", "ci":"95% CI", "support":"S_b", "years":"Years", "tau2":"Tau²", "i2":"I²", "eligible":"Eligible", "reason":"Reason"}
        widths = {"bsa":80,"rc":90,"ci":190,"support":80,"years":70,"tau2":85,"i2":75,"eligible":70,"reason":220}
        for col in cols:
            self.tree.heading(col, text=labels[col]); self.tree.column(col, width=widths[col], anchor="center" if col != "reason" else "w")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(8, 0)); self.tree.configure(yscrollcommand=scroll.set)

    def set_cell(self, q, lag):
        for item in self.tree.get_children(): self.tree.delete(item)
        rows = self.diagnostics.pooled_rows(q, lag)
        supported, eligible = self.diagnostics.pooled_supported_count(q, lag)
        self.summary.set(f"CI-supported BSAs: {supported} / {eligible}")
        for row in rows:
            lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
            ci = "—" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"
            i2 = _float(row.get("i_squared_percent"))
            self.tree.insert("", "end", values=(
                row.get("bsa_id", "—"), _fmt_number(row.get("pooled_rc"), 4), ci,
                _fmt_number(self.diagnostics.supported_magnitude(row), 4), _int(row.get("contributing_years")) or "—",
                _fmt_number(row.get("tau_squared"), 4), "—" if i2 is None else f"{i2:.1f}%",
                _fmt_bool(row.get("eligible")), row.get("reason") or "",
            ))


class DiagnosticsLoyoPanel(ttk.Frame):
    def __init__(self, parent, overview: OverviewModel, diagnostics: DiagnosticsModel):
        super().__init__(parent, padding=10)
        self.overview = overview; self.diagnostics = diagnostics
        self.columnconfigure(0, weight=1); self.rowconfigure(1, weight=1)
        self.context = tk.StringVar(value="")
        ttk.Label(self, text="Leave-one-year-out diagnostics", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.context, style="Overview.CardSub.TLabel").grid(row=0, column=0, sticky="e")
        cols=("year","cs","wmarm","kcs","ci","bsa","pop")
        self.tree=ttk.Treeview(self,columns=cols,show="headings")
        labels={"year":"Excluded year","cs":"WMARMcs","wmarm":"WMARM","kcs":"Kcs","ci":"CI-supported pop.","bsa":"BSA coverage","pop":"Pop. coverage"}
        widths={"year":100,"cs":90,"wmarm":90,"kcs":80,"ci":120,"bsa":105,"pop":105}
        for col in cols:
            self.tree.heading(col,text=labels[col]); self.tree.column(col,width=widths[col],anchor="center")
        self.tree.grid(row=1,column=0,sticky="nsew",pady=(8,0))
        scroll=ttk.Scrollbar(self,orient="vertical",command=self.tree.yview); scroll.grid(row=1,column=1,sticky="ns",pady=(8,0)); self.tree.configure(yscrollcommand=scroll.set)

    def set_cell(self,q,lag):
        for item in self.tree.get_children(): self.tree.delete(item)
        diag=self.overview.optimization_cell(q,lag) or {}
        pieces=[]
        if diag:
            pieces.append(f"persistence {_fmt_percent(diag.get('persistence'))}")
            pieces.append(f"stable {_fmt_bool(diag.get('stable'))}")
        if self.overview.surface_coherence is not None:
            pieces.append(f"surface coherence {_fmt_number(self.overview.surface_coherence,3)}")
        self.context.set(" · ".join(pieces))
        for row in self.diagnostics.loyo_rows(q,lag):
            self.tree.insert("","end",values=(
                _int(row.get("excluded_year")) or "—", _fmt_number(row.get("wmarm_cs"),4), _fmt_number(row.get("wmarm"),4),
                _fmt_number(row.get("confidence_support_retention_ratio"),3), _fmt_percent(row.get("ci_excluding_zero_population_share")),
                _fmt_percent(row.get("bsa_coverage")), _fmt_percent(row.get("population_coverage")),
            ))


class DiagnosticsExclusionsPanel(ttk.Frame):
    def __init__(self,parent,diagnostics:DiagnosticsModel):
        super().__init__(parent,padding=10); self.diagnostics=diagnostics
        self.columnconfigure(0,weight=2); self.columnconfigure(1,weight=1); self.rowconfigure(0,weight=1)
        box=ttk.Frame(self,padding=10,relief="ridge"); box.grid(row=0,column=0,sticky="nsew",padx=(0,6)); box.columnconfigure(0,weight=1); box.rowconfigure(1,weight=1)
        ttk.Label(box,text="Estimator exclusions",style="Overview.SectionTitle.TLabel").grid(row=0,column=0,sticky="w")
        self.tree=ttk.Treeview(box,columns=("stage","reason","count"),show="headings")
        for col,label,width in (("stage","Stage",120),("reason","Reason",360),("count","Count",70)):
            self.tree.heading(col,text=label); self.tree.column(col,width=width,anchor="w" if col!="count" else "center")
        self.tree.grid(row=1,column=0,sticky="nsew",pady=(8,0))
        side=ttk.Frame(self,padding=10,relief="ridge"); side.grid(row=0,column=1,sticky="nsew",padx=(6,0))
        ttk.Label(side,text="Data-quality totals",style="Overview.SectionTitle.TLabel").pack(anchor="w")
        self.values={}
        for key,label in (
            ("missing_incidence_days","Missing incidence days"),("missing_local_hazard_days","Missing local-hazard days"),
            ("unavailable_baseline_days","Unavailable baseline days"),("zero_weight_valid_days","Zero-weight valid days"),
            ("global_redraws","Bootstrap global redraws"),("mean_invalid_bootstrap_proportion","Mean invalid bootstrap"),
            ("max_invalid_bootstrap_proportion","Max invalid bootstrap"),
        ):
            line=ttk.Frame(side); line.pack(fill="x",pady=3); ttk.Label(line,text=label).pack(side="left")
            var=tk.StringVar(value="—"); ttk.Label(line,textvariable=var).pack(side="right"); self.values[key]=var
        ttk.Label(side,text="Counts summarize the selected threshold-lag cell only. Calendar-edge exclusions are expected when lag or baseline windows cannot be formed within retained contiguous date segments.",wraplength=300,justify="left",style="Overview.CardSub.TLabel").pack(anchor="w",pady=(14,0))

    def set_cell(self,q,lag):
        for item in self.tree.get_children(): self.tree.delete(item)
        counts=self.diagnostics.exclusion_counts(q,lag)
        if counts:
            for stage,reason,count in counts: self.tree.insert("","end",values=(stage,reason,count))
        else:
            self.tree.insert("","end",values=("—","No ineligible yearly or pooled records",0))
        totals=self.diagnostics.data_quality_totals(q,lag)
        for key,var in self.values.items():
            value=totals.get(key)
            if key.endswith("proportion"):
                var.set(_fmt_percent(value))
            else:
                var.set("—" if value is None else str(value))


class DiagnosticsPanel(ttk.Frame):
    """Read-only analytical support and exclusion diagnostics."""
    def __init__(self,parent,overview:OverviewModel,diagnostics:DiagnosticsModel):
        super().__init__(parent,padding=14); self.overview=overview; self.diagnostics=diagnostics
        self.columnconfigure(0,weight=1); self.rowconfigure(2,weight=1); self.selected=overview.default_cell_key
        header=ttk.Frame(self); header.grid(row=0,column=0,sticky="ew")
        ttk.Label(header,text="Diagnostics",font=("Segoe UI",16,"bold")).pack(side="left")
        self.selected_text=tk.StringVar(value="")
        ttk.Label(header,textvariable=self.selected_text,style="Overview.CardSub.TLabel").pack(side="right")
        self.cards_frame=ttk.Frame(self); self.cards_frame.grid(row=1,column=0,sticky="ew",pady=(12,0))
        self.cards={}
        for i,(key,title,subtitle) in enumerate((
            ("bsa","BSA coverage","eligible territorial units"),("pop","Population coverage","represented population"),
            ("ci","CI-supported population","pooled CI excludes zero"),("kcs","Kcs","confidence-scaled / point magnitude"),
            ("i2","Mean I²","pooled heterogeneity"),("bootstrap","Bootstrap validity","valid requested replicates"),
        )):
            self.cards_frame.columnconfigure(i,weight=1,uniform="diag-card")
            card=MetricCard(self.cards_frame,title,"—",subtitle); card.grid(row=0,column=i,sticky="nsew",padx=(0 if i==0 else 5,0)); self.cards[key]=card
        self.notebook=ttk.Notebook(self); self.notebook.grid(row=2,column=0,sticky="nsew",pady=(12,0))
        self.support=DiagnosticsSupportPanel(self.notebook,overview,diagnostics)
        self.bsa=DiagnosticsBsaPanel(self.notebook,diagnostics)
        self.loyo=DiagnosticsLoyoPanel(self.notebook,overview,diagnostics)
        self.exclusions=DiagnosticsExclusionsPanel(self.notebook,diagnostics)
        self.notebook.add(self.support,text="Support"); self.notebook.add(self.bsa,text="BSA heterogeneity"); self.notebook.add(self.loyo,text="LOYO"); self.notebook.add(self.exclusions,text="Exclusions")
        if self.selected: self.set_selected_cell(*self.selected)

    def set_selected_cell(self,q,lag):
        row=self.overview.cell(q,lag)
        if row is None: return
        self.selected=(_float(q),_int(lag)); self.selected_text.set(f"Selected: {_fmt_q(q)}, lag {_int(lag)}")
        self.cards["bsa"].value.set(_fmt_percent(row.get("bsa_coverage")))
        self.cards["pop"].value.set(_fmt_percent(row.get("population_coverage")))
        self.cards["ci"].value.set(_fmt_percent(row.get("ci_excluding_zero_population_share")))
        self.cards["kcs"].value.set(_fmt_number(row.get("confidence_support_retention_ratio"),3))
        i2=_float(row.get("mean_i_squared_percent")); self.cards["i2"].value.set("—" if i2 is None else f"{i2:.1f}%")
        self.cards["bootstrap"].value.set(_fmt_percent(row.get("bootstrap_validity")))
        self.support.set_cell(q,lag); self.bsa.set_cell(q,lag); self.loyo.set_cell(q,lag); self.exclusions.set_cell(q,lag)



class RobustnessPanel(ttk.Frame):
    """Read-only robustness/sensitivity realization explorer."""

    def __init__(self, parent, model: RobustnessModel):
        super().__init__(parent, padding=14)
        self.model = model
        self.selected_rid = model.default_realization_id
        self.selected_surface_cell = None
        self.metric_var = tk.StringVar(value="WMARMcs")
        self.selector_var = tk.StringVar()
        self.display_to_rid = {}
        self.cards = {}
        self._build_header()
        self._build_cards()
        self._build_notebook()
        self._populate_selector()
        self.set_realization(self.selected_rid)

    def _build_header(self):
        top = ttk.Frame(self); top.pack(fill="x")
        ttk.Label(top, text="Robustness / sensitivity", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(top, text="Realization").pack(side="left", padx=(22,6))
        self.selector = ttk.Combobox(top, textvariable=self.selector_var, state="readonly", width=62)
        self.selector.pack(side="left", fill="x", expand=True)
        self.selector.bind("<<ComboboxSelected>>", lambda e: self._selector_changed())
        ttk.Label(self, text="Reference-versus-sensitivity results are read from persisted outputs only; navigating this page never reruns the model.", wraplength=1050, justify="left").pack(anchor="w", pady=(4,8))

    def _build_cards(self):
        frame = ttk.Frame(self); frame.pack(fill="x", pady=(0,10))
        for i in range(3): frame.columnconfigure(i, weight=1, uniform="rs-card")
        specs=(
            ("analysis","Analysis","—","realization family"),
            ("mode","Comparison mode","—","fixed, reoptimized, or both"),
            ("delta_cs","Δ WMARMcs","—","fixed-point sensitivity minus reference"),
            ("rmse","Surface RMSEcs","—","common admissible domain"),
            ("jaccard","Robust-region Jaccard","—","reoptimized overlap"),
            ("representative","Sensitivity representative","—","reoptimized q / lag"),
        )
        for i,(key,title,value,sub) in enumerate(specs):
            row,col=divmod(i,3)
            card=MetricCard(frame,title,value,sub); card.grid(row=row,column=col,sticky="nsew",padx=(0 if col==0 else 5,0),pady=(0 if row==0 else 5,0)); self.cards[key]=card

    def _build_notebook(self):
        self.notebook=ttk.Notebook(self); self.notebook.pack(fill="both",expand=True)
        self.overview_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.overview_page,text="Overview")
        self.fixed_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.fixed_page,text="Fixed-point BSA")
        self.surface_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.surface_page,text="Surfaces")
        self.fingerprint_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.fingerprint_page,text="Fingerprint")
        self.trajectory_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.trajectory_page,text="Trajectory")
        self.region_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.region_page,text="Region stability")
        self._build_overview_page(); self._build_fixed_page(); self._build_surface_page(); self._build_fingerprint_page(); self._build_trajectory_page(); self._build_region_page()

    def _build_overview_page(self):
        left=ttk.LabelFrame(self.overview_page,text="Realization definition",padding=10); left.pack(side="left",fill="both",expand=True,padx=(0,6))
        right=ttk.LabelFrame(self.overview_page,text="Comparison results",padding=10); right.pack(side="left",fill="both",expand=True,padx=(6,0))
        left.columnconfigure(1,weight=1); right.columnconfigure(1,weight=1)
        self.meta_vars={}
        for i,(key,label) in enumerate((("scenario","Scenario"),("step","Step"),("coordinate","Trajectory coordinate"),("invalidates","Invalidates from"),("factors","Realized factors"))):
            ttk.Label(left,text=label).grid(row=i,column=0,sticky="nw",padx=(0,12),pady=3)
            var=tk.StringVar(value="—"); ttk.Label(left,textvariable=var,wraplength=360,justify="right",anchor="e").grid(row=i,column=1,sticky="ew",pady=3); self.meta_vars[key]=var
        self.result_vars={}
        for i,(key,label) in enumerate((("fixed_status","Fixed-point status"),("delta_wmarm","Δ diagnostic WMARM"),("delta_k","Δ Kcs"),("delta_rci","Δ CI-supported population"),("spearman","BSA pooled-RC Spearman"),("reopt_status","Reoptimization state"),("pmax","Reoptimized pmax"),("null_cutoff","Null cutoff WMARMcs"),("surface_state","Surface-state transition"),("rho","Surface rank concordance"),("delta_q","Representative Δq"),("delta_lag","Representative Δlag"))):
            ttk.Label(right,text=label).grid(row=i,column=0,sticky="nw",padx=(0,12),pady=2)
            var=tk.StringVar(value="—"); ttk.Label(right,textvariable=var,wraplength=360,justify="right",anchor="e").grid(row=i,column=1,sticky="ew",pady=2); self.result_vars[key]=var

    def _tree(self,parent,columns):
        tree=ttk.Treeview(parent,columns=[c[0] for c in columns],show="headings",selectmode="browse")
        for key,label,width in columns:
            tree.heading(key,text=label); tree.column(key,width=width,anchor="w")
        scroll=ttk.Scrollbar(parent,orient="vertical",command=tree.yview); tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left",fill="both",expand=True); scroll.pack(side="right",fill="y")
        return tree

    @staticmethod
    def _clear(tree):
        for item in tree.get_children(): tree.delete(item)

    def _build_fixed_page(self):
        self.fixed_tree=self._tree(self.fixed_page,(
            ("bsa","BSA",80),("ref_rc","Reference RC",105),("sen_rc","Sensitivity RC",110),("drc","Δ RC",90),
            ("ref_s","Reference S",95),("sen_s","Sensitivity S",95),("ds","Δ S",90),("transition","Support transition",190),("sign","Same sign",80)))

    def _build_surface_page(self):
        controls=ttk.Frame(self.surface_page); controls.pack(fill="x",pady=(0,8))
        ttk.Label(controls,text="Metric").pack(side="left")
        metric=ttk.Combobox(controls,textvariable=self.metric_var,values=("WMARMcs","Diagnostic WMARM"),state="readonly",width=22); metric.pack(side="left",padx=(6,0)); metric.bind("<<ComboboxSelected>>",lambda e:self._refresh_surface())
        self.surface_note=tk.StringVar(value=""); ttk.Label(controls,textvariable=self.surface_note,wraplength=700,justify="left").pack(side="left",padx=(16,0))
        body=ttk.Frame(self.surface_page); body.pack(fill="both",expand=True); body.columnconfigure(0,weight=1); body.columnconfigure(1,weight=1); body.rowconfigure(0,weight=1)
        self.ref_heat=HeatmapPreview(body,"Reference", "value", self._select_surface_cell); self.ref_heat.grid(row=0,column=0,sticky="nsew",padx=(0,6))
        self.sen_heat=HeatmapPreview(body,"Sensitivity", "value", self._select_surface_cell); self.sen_heat.grid(row=0,column=1,sticky="nsew",padx=(6,0))

    def _build_fingerprint_page(self):
        self.fp_tree=self._tree(self.fingerprint_page,(
            ("domain","Domain",95),("component","Component",190),("role","Role",155),("ref","Reference",100),("sen","Sensitivity",100),("delta","Δ",100),("status","Status",190)))

    def _build_trajectory_page(self):
        self.traj_tree=self._tree(self.trajectory_page,(
            ("step","Step",60),("coord","Coordinate",90),("mode","Mode",120),("factors","Realized factors",560),("rid","Realization ID",310)))

    def _build_region_page(self):
        top=ttk.Frame(self.region_page); top.pack(fill="x",pady=(0,8))
        status=(self.model.region_summary or {}).get("status") if self.model.region_summary else "NOT_AVAILABLE"
        ttk.Label(top,text=f"Reference robust-region stability: {status}",font=("Segoe UI",10,"bold")).pack(side="left")
        holder=ttk.Frame(self.region_page); holder.pack(fill="both",expand=True)
        self.region_tree=self._tree(holder,(
            ("bsa","BSA",80),("valid","Valid region",90),("rc","Median RC",90),("rc_iqr","RC IQR",85),("s","Median S",85),("s_iqr","S IQR",85),("support","CS persistence",100),("sign","Sign consistency",100),("rank","Rank IQR",85),("state","Interpretation",130)))
        for row in self.model.region_bsa:
            if str(row.get("region_scope")) != "ALL": continue
            self.region_tree.insert("","end",values=(row.get("bsa_id"),_fmt_percent(row.get("valid_cell_proportion")),_fmt_number(row.get("median_pooled_rc"),4),_fmt_number(row.get("pooled_rc_iqr"),4),_fmt_number(row.get("median_supported_magnitude"),4),_fmt_number(row.get("supported_magnitude_iqr"),4),_fmt_percent(row.get("confidence_support_persistence")),_fmt_percent(row.get("sign_consistency")),_fmt_number(row.get("rank_iqr"),2),"Interpretable" if _bool(row.get("stability_interpretable")) else str(row.get("support_reason") or "Unavailable")))

    def _populate_selector(self):
        self.display_to_rid={}
        values=[]
        for row in self.model.realizations:
            rid=row.get("realization_id"); display=self.model.label(rid)
            # keep display unique if labels repeat
            if display in self.display_to_rid: display=f"{display} · {rid[-8:]}"
            self.display_to_rid[display]=rid; values.append(display)
        self.selector.configure(values=values)
        if self.selected_rid:
            for display,rid in self.display_to_rid.items():
                if rid==self.selected_rid: self.selector_var.set(display); break
        elif not values:
            self.selector_var.set("No robustness realizations")

    def _selector_changed(self):
        self.set_realization(self.display_to_rid.get(self.selector_var.get()))

    def set_realization(self,rid):
        self.selected_rid=rid
        row=self.model.realization(rid) or {}
        analysis=ROBUSTNESS_ANALYSIS_LABELS.get(row.get("analysis_id"),row.get("analysis_id") or "—")
        fixed=self.model.fixed_row(rid) or {}; reopt=self.model.reoptimization_row(rid) or {}; surface=self.model.surface_row(rid) or {}
        self.cards["analysis"].value.set(analysis)
        self.cards["mode"].value.set(str(row.get("comparison_mode") or "—"))
        self.cards["delta_cs"].value.set(_fmt_number(fixed.get("delta_wmarm_cs"),4))
        self.cards["rmse"].value.set(_fmt_number(surface.get("rmse_cs"),4))
        self.cards["jaccard"].value.set(_fmt_number(surface.get("robust_region_jaccard"),3))
        rq=reopt.get("representative_q"); rl=reopt.get("representative_lag")
        self.cards["representative"].value.set("—" if _float(rq) is None or _int(rl) is None else f"{_fmt_q(rq)}, lag {_int(rl)}")
        self.meta_vars["scenario"].set(str(row.get("scenario_id") or "—"))
        self.meta_vars["step"].set("—" if row.get("step_index") is None else str(int(row.get("step_index"))+1))
        self.meta_vars["coordinate"].set(_fmt_number(row.get("trajectory_coordinate"),3))
        self.meta_vars["invalidates"].set(str(row.get("invalidates_from") or "—"))
        self.meta_vars["factors"].set(self.model.factor_text(rid))
        self.result_vars["fixed_status"].set(str(fixed.get("status") or "—"))
        self.result_vars["delta_wmarm"].set(_fmt_number(fixed.get("delta_diagnostic_wmarm"),4))
        self.result_vars["delta_k"].set(_fmt_number(fixed.get("delta_k_cs"),4))
        self.result_vars["delta_rci"].set(_fmt_number(fixed.get("delta_r_ci"),4))
        self.result_vars["spearman"].set(_fmt_number(fixed.get("pooled_rc_spearman"),3))
        self.result_vars["reopt_status"].set(str(reopt.get("status") or reopt.get("optimizer_status") or "—"))
        self.result_vars["pmax"].set(_fmt_number(reopt.get("p_max"),4))
        self.result_vars["null_cutoff"].set(_fmt_number(reopt.get("null_cutoff_wmarm_cs"),4))
        self.result_vars["surface_state"].set(str(surface.get("surface_state_transition") or "—"))
        self.result_vars["rho"].set(_fmt_number(surface.get("rho_cs"),3))
        self.result_vars["delta_q"].set(_fmt_number(surface.get("delta_representative_q"),2))
        self.result_vars["delta_lag"].set(_fmt_number(surface.get("delta_representative_lag"),0))
        self._refresh_fixed(); self._refresh_surface(); self._refresh_fingerprint(); self._refresh_trajectory()

    def _refresh_fixed(self):
        self._clear(self.fixed_tree)
        for row in self.model.bsa_rows(self.selected_rid):
            self.fixed_tree.insert("","end",values=(row.get("bsa_id"),_fmt_number(row.get("reference_pooled_rc"),4),_fmt_number(row.get("sensitivity_pooled_rc"),4),_fmt_number(row.get("delta_pooled_rc"),4),_fmt_number(row.get("reference_supported_magnitude"),4),_fmt_number(row.get("sensitivity_supported_magnitude"),4),_fmt_number(row.get("delta_supported_magnitude"),4),row.get("confidence_support_transition") or "—",_fmt_bool(row.get("sign_agreement"))))

    def _select_surface_cell(self,q,lag):
        self.selected_surface_cell=(_float(q),_int(lag)); self._refresh_surface()

    def _refresh_surface(self):
        rows=self.model.comparison_surface_rows(self.selected_rid)
        if not rows:
            self.surface_note.set("Complete-surface comparison is unavailable for this realization. Select a REOPTIMIZED or BOTH realization with completed Step 10 outputs.")
            self.ref_heat.set_data([]); self.sen_heat.set_data([]); return
        metric=self.metric_var.get(); suffix="wmarm_cs" if metric=="WMARMcs" else "wmarm"
        ref=[]; sen=[]
        for row in rows:
            q=_float(row.get("q")); lag=_int(row.get("lag"))
            ref.append({"q":q,"lag":lag,"value":_float(row.get(f"reference_{suffix}"))})
            sen.append({"q":q,"lag":lag,"value":_float(row.get(f"sensitivity_{suffix}"))})
        self.surface_note.set("Primary comparison uses the common admissible threshold-lag domain; grey cells may be outside that domain or unavailable.")
        self.ref_heat.set_metric("value",f"Reference {metric}"); self.sen_heat.set_metric("value",f"Sensitivity {metric}")
        self.ref_heat.set_data(ref,self.selected_surface_cell); self.sen_heat.set_data(sen,self.selected_surface_cell)

    def _refresh_fingerprint(self):
        self._clear(self.fp_tree)
        for row in self.model.fingerprint_rows(self.selected_rid):
            self.fp_tree.insert("","end",values=(row.get("fingerprint_domain"),row.get("component"),row.get("metric_role"),_fmt_number(row.get("reference_value"),4),_fmt_number(row.get("sensitivity_value"),4),_fmt_number(row.get("delta"),4),row.get("status") or "—"))

    def _refresh_trajectory(self):
        self._clear(self.traj_tree)
        for row in self.model.trajectory_rows(self.selected_rid):
            factors="; ".join(f"{f.get('name')}={f.get('value')}" for f in (row.get("factors") or []))
            self.traj_tree.insert("","end",values=("—" if row.get("step_index") is None else int(row.get("step_index"))+1,_fmt_number(row.get("trajectory_coordinate"),3),row.get("comparison_mode") or "—",factors,row.get("realization_id") or "—"))


class ConfounderPanel(ttk.Frame):
    """Read-only explorer for persisted candidate-confounder outputs."""

    def __init__(self, parent, model: ConfounderModel):
        super().__init__(parent, padding=14)
        self.model = model
        self.cards = {}
        self._build_header()
        self._build_cards()
        self._build_notebook()
        self.refresh_content()

    def _build_header(self):
        ttk.Label(self, text="Candidate confounder", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(self, text="Cross-fitted candidate calibration, fixed-parameter adjusted inference, and optional complete adjusted reoptimization. The fixed confounder reference is the supported representative when available, otherwise the strongest admissible observed WMARMcs cell is shown explicitly as exploratory. All values are read from persisted results.", wraplength=1080, justify="left").pack(anchor="w", pady=(4,8))

    def _build_cards(self):
        frame=ttk.Frame(self); frame.pack(fill="x", pady=(0,10))
        for i in range(4): frame.columnconfigure(i, weight=1, uniform="ca-card")
        specs=(
            ("candidate","Candidate","—","configured candidate ID"),
            ("delta_cs","Δ WMARMcs","—","adjusted minus reference at fixed q / lag"),
            ("sign","BSA sign agreement","—","comparable pooled BSAs"),
            ("spearman","Pooled RC Spearman","—","reference versus adjusted"),
            ("status","CA8 status","—","adjusted optimization inference"),
            ("region","Adjusted robust region","—","number of supported cells"),
            ("representative","Adjusted representative","—","q / lag when defined"),
            ("jaccard","Region Jaccard","—","reference versus adjusted"),
        )
        for i,(key,title,value,sub) in enumerate(specs):
            row,col=divmod(i,4)
            card=MetricCard(frame,title,value,sub); card.grid(row=row,column=col,sticky="nsew",padx=(0 if col==0 else 5,0),pady=(0 if row==0 else 5,0)); self.cards[key]=card

    def _tree(self,parent,columns):
        holder=ttk.Frame(parent); holder.pack(fill="both",expand=True)
        tree=ttk.Treeview(holder,columns=[c[0] for c in columns],show="headings",selectmode="browse")
        for key,label,width in columns:
            tree.heading(key,text=label); tree.column(key,width=width,anchor="w")
        scroll=ttk.Scrollbar(holder,orient="vertical",command=tree.yview); tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left",fill="both",expand=True); scroll.pack(side="right",fill="y")
        return tree

    @staticmethod
    def _clear(tree):
        for item in tree.get_children(): tree.delete(item)

    def _build_notebook(self):
        self.notebook=ttk.Notebook(self); self.notebook.pack(fill="both",expand=True)
        self.overview_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.overview_page,text="Overview")
        self.calibration_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.calibration_page,text="Calibration")
        self.fixed_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.fixed_page,text="Fixed-point BSA")
        self.surface_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.surface_page,text="Adjusted surface")
        self.optimization_page=ttk.Frame(self.notebook,padding=10); self.notebook.add(self.optimization_page,text="Optimization")
        self.overview_vars={}
        for i,(key,label) in enumerate((
            ("run","Analysis run"),("ref_config","Confounder reference configuration"),("ref_basis","Reference configuration status"),
            ("fixed","Fixed comparison status"),("ref_cs","Reference WMARMcs"),("adj_cs","Adjusted WMARMcs"),
            ("ref_cov","Reference population coverage"),("adj_cov","Adjusted population coverage"),("ci","Adjusted CI-supported population"),
            ("reopt","Complete reoptimization requested"),("reason","Optimization reason"))):
            ttk.Label(self.overview_page,text=label).grid(row=i,column=0,sticky="w",padx=(0,15),pady=3)
            var=tk.StringVar(value="—"); ttk.Label(self.overview_page,textvariable=var,wraplength=720,justify="left").grid(row=i,column=1,sticky="w",pady=3); self.overview_vars[key]=var
        self.calibration_tree=self._tree(self.calibration_page,(("year","Held-out year",95),("status","Status",150),("caliper","Selected caliper",110),("amplitude","Relevance amplitude",125),("residual","Residual relevance",125),("support","Retained support",110),("reason","Exclusion",250)))
        self.fixed_tree=self._tree(self.fixed_page,(("bsa","BSA",85),("ref","Reference RC",105),("adj","Adjusted RC",105),("delta","Δ RC",90),("ref_s","Reference S",95),("adj_s","Adjusted S",95),("transition","CI-support transition",190),("sign","Same sign",80)))
        self.surface_tree=self._tree(self.surface_page,(("q","q",65),("lag","Lag",60),("cs","WMARMcs",95),("wmarm","WMARM",95),("k","Kcs",85),("bsa","BSA coverage",100),("pop","Population coverage",115),("valid","Valid",70)))
        self.optimization_vars={}
        for i,(key,label) in enumerate((("status","Adjusted region status"),("rep","Adjusted representative"),("pmax","Adjusted pmax"),("cutoff","Adjusted null cutoff WMARMcs"),("coherence","Adjusted surface coherence"),("rmse","WMARMcs surface RMSE"),("rho","WMARMcs surface Spearman"),("jaccard","Robust-region Jaccard"),("delta","Representative displacement"))):
            ttk.Label(self.optimization_page,text=label).grid(row=i,column=0,sticky="w",padx=(0,15),pady=3)
            var=tk.StringVar(value="—"); ttk.Label(self.optimization_page,textvariable=var,wraplength=700,justify="left").grid(row=i,column=1,sticky="w",pady=3); self.optimization_vars[key]=var

    def refresh_content(self):
        m=self.model; fixed=m.fixed_summary or {}; summary=m.summary or {}; opt=m.optimization or {}; comp=m.surface_comparison or {}
        self.cards["candidate"].value.set(m.candidate_id)
        self.cards["delta_cs"].value.set(_fmt_number(fixed.get("delta_wmarm_cs"),4))
        self.cards["sign"].value.set(_fmt_percent(fixed.get("bsa_sign_agreement")))
        self.cards["spearman"].value.set(_fmt_number(fixed.get("pooled_rc_spearman"),3))
        self.cards["status"].value.set(str(summary.get("robust_region_status") or summary.get("status") or "—"))
        region=m.robust_region_count; self.cards["region"].value.set("—" if region is None else str(region))
        rep=m.representative; self.cards["representative"].value.set("—" if not rep else f"{_fmt_q(rep.get('q'))}, lag {_int(rep.get('lag'))}")
        self.cards["jaccard"].value.set(_fmt_number(comp.get("robust_region_jaccard"),3))
        self.overview_vars["run"].set(m.analysis_run_id or "—")
        reference_configuration = fixed.get("reference_configuration") or summary.get("reference_configuration") or fixed.get("reference_representative") or summary.get("reference_representative")
        if isinstance(reference_configuration, dict) and reference_configuration.get("q") is not None and reference_configuration.get("lag") is not None:
            self.overview_vars["ref_config"].set(f"{_fmt_q(reference_configuration.get('q'))}, lag {_int(reference_configuration.get('lag'))}")
        else:
            self.overview_vars["ref_config"].set("—")
        reference_status = fixed.get("reference_configuration_inference_status") or summary.get("reference_configuration_inference_status")
        reference_source = fixed.get("reference_configuration_source") or summary.get("reference_configuration_source")
        if reference_status and reference_source:
            self.overview_vars["ref_basis"].set(f"{reference_status} ({reference_source})")
        else:
            self.overview_vars["ref_basis"].set(str(reference_status or reference_source or "—"))
        self.overview_vars["fixed"].set(str(fixed.get("status") or "—"))
        self.overview_vars["ref_cs"].set(_fmt_number(fixed.get("reference_wmarm_cs"),5))
        self.overview_vars["adj_cs"].set(_fmt_number(fixed.get("adjusted_wmarm_cs"),5))
        self.overview_vars["ref_cov"].set(_fmt_percent(fixed.get("reference_population_coverage")))
        self.overview_vars["adj_cov"].set(_fmt_percent(fixed.get("adjusted_population_coverage")))
        self.overview_vars["ci"].set(_fmt_percent(fixed.get("adjusted_ci_excluding_zero_population_share")))
        self.overview_vars["reopt"].set(_fmt_bool(summary.get("complete_adjusted_surface_requested")))
        self.overview_vars["reason"].set(str(summary.get("reason") or "—"))
        self._clear(self.calibration_tree)
        for row in m.calibration_rows:
            self.calibration_tree.insert("","end",values=(row.get("held_out_year"),row.get("status"),_fmt_number(row.get("selected_caliper"),3),_fmt_number(row.get("relevance_amplitude"),4),_fmt_number(row.get("selected_residual_relevance"),4),_fmt_number(row.get("selected_retained_support"),4),row.get("exclusion_reason") or "—"))
        self._clear(self.fixed_tree)
        for row in m.fixed_bsa_rows:
            self.fixed_tree.insert("","end",values=(row.get("bsa_id"),_fmt_number(row.get("reference_pooled_rc"),4),_fmt_number(row.get("adjusted_pooled_rc"),4),_fmt_number(row.get("delta_pooled_rc"),4),_fmt_number(row.get("reference_confidence_scaled_magnitude"),4),_fmt_number(row.get("adjusted_confidence_scaled_magnitude"),4),row.get("confidence_support_transition") or "—",_fmt_bool(row.get("sign_agreement"))))
        self._clear(self.surface_tree)
        for row in m.surface_rows:
            self.surface_tree.insert("","end",values=(_fmt_q(row.get("q")),_int(row.get("lag")),_fmt_number(row.get("wmarm_cs"),5),_fmt_number(row.get("wmarm"),5),_fmt_number(row.get("confidence_support_retention_ratio"),3),_fmt_percent(row.get("bsa_coverage")),_fmt_percent(row.get("population_coverage")),_fmt_bool(row.get("valid"))))
        self.optimization_vars["status"].set(str(opt.get("status") or summary.get("robust_region_status") or "—"))
        self.optimization_vars["rep"].set("—" if not rep else f"{_fmt_q(rep.get('q'))}, lag {_int(rep.get('lag'))}")
        self.optimization_vars["pmax"].set(_fmt_number(opt.get("p_max"),4))
        self.optimization_vars["cutoff"].set(_fmt_number(opt.get("null_cutoff_wmarm_cs"),5))
        self.optimization_vars["coherence"].set(_fmt_number(opt.get("surface_coherence"),4))
        self.optimization_vars["rmse"].set(_fmt_number(comp.get("wmarm_cs_surface_rmse"),5))
        self.optimization_vars["rho"].set(_fmt_number(comp.get("wmarm_cs_surface_spearman"),3))
        self.optimization_vars["jaccard"].set(_fmt_number(comp.get("robust_region_jaccard"),3))
        dq,dl=comp.get("representative_delta_q"),comp.get("representative_delta_lag")
        self.optimization_vars["delta"].set("—" if dq is None or dl is None else f"Δq={_fmt_number(dq,3)}, Δlag={_int(dl)}")


class OutputsPanel(ttk.Frame):
    """Read-only artifact browser for completed APHREH project outputs."""

    def __init__(self, parent, model: OutputsModel):
        super().__init__(parent, padding=14)
        self.model = model
        self.selected_artifact: OutputArtifact | None = None
        self.item_artifacts: dict[str, OutputArtifact] = {}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_header()
        self._build_cards()
        self._build_controls()
        self._build_browser()
        self.refresh_list()

    def _build_header(self):
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Outputs", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(
            header,
            text="Browse exported analysis artifacts without recomputing the project.",
            style="Overview.CardSub.TLabel",
        ).pack(side="left", padx=(14, 0))

    def _build_cards(self):
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        cards = (
            ("all", "Artifacts", str(len(self.model.artifacts)), "discoverable project outputs"),
            ("observed", "Observed", str(self.model.count("Observed analysis")), "reference-analysis artifacts"),
            ("calibration", "Calibration", str(self.model.count("Calibration")), "calibration artifacts"),
            ("size", "Total size", _format_file_size(self.model.total_size_bytes), "listed artifacts only"),
        )
        self.cards = {}
        for i, (key, title, value, subtitle) in enumerate(cards):
            frame.columnconfigure(i, weight=1, uniform="outputs-card")
            card = MetricCard(frame, title, value, subtitle)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 5, 0))
            self.cards[key] = card

    def _build_controls(self):
        controls = ttk.Frame(self, padding=(0, 10, 0, 0))
        controls.grid(row=2, column=0, sticky="ew")
        ttk.Label(controls, text="Group").pack(side="left")
        self.group_var = tk.StringVar(value="All")
        self.group_combo = ttk.Combobox(
            controls,
            textvariable=self.group_var,
            values=["All", *self.model.groups],
            state="readonly",
            width=24,
        )
        self.group_combo.pack(side="left", padx=(6, 12))
        self.group_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh_list())
        ttk.Label(controls, text="Search").pack(side="left")
        self.search_var = tk.StringVar(value="")
        search = ttk.Entry(controls, textvariable=self.search_var, width=28)
        search.pack(side="left", padx=(6, 12))
        search.bind("<KeyRelease>", lambda event: self.refresh_list())
        ttk.Button(controls, text="Refresh list", command=self.reload).pack(side="left", padx=3)
        ttk.Button(controls, text="Open selected", command=self.open_selected).pack(side="right", padx=3)
        ttk.Button(controls, text="Save copy…", command=self.save_copy).pack(side="right", padx=3)
        ttk.Button(controls, text="Copy path", command=self.copy_path).pack(side="right", padx=3)
        ttk.Button(controls, text="Open folder", command=self.open_selected_folder).pack(side="right", padx=3)

    def _build_browser(self):
        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        left = ttk.Frame(pane, padding=8, relief="ridge")
        right = ttk.Frame(pane, padding=10, relief="ridge")
        pane.add(left, weight=3)
        pane.add(right, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        cols = ("group", "file", "type", "size", "modified")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        widths = {"group": 170, "file": 330, "type": 115, "size": 85, "modified": 140}
        labels = {"group": "Group", "file": "Artifact", "type": "Type", "size": "Size", "modified": "Modified"}
        for col in cols:
            self.tree.heading(col, text=labels[col])
            self.tree.column(col, width=widths[col], anchor="w" if col in {"group", "file", "type"} else "center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Double-1>", lambda event: self.open_selected())

        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)
        ttk.Label(right, text="Selected artifact", style="Overview.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.selected_name = tk.StringVar(value="Select an artifact")
        ttk.Label(right, textvariable=self.selected_name, style="Overview.CardValue.TLabel", wraplength=430).grid(row=1, column=0, sticky="w", pady=(5, 3))
        self.selected_meta = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.selected_meta, style="Overview.CardSub.TLabel", wraplength=430, justify="left").grid(row=2, column=0, sticky="w")
        self.selected_description = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.selected_description, wraplength=430, justify="left").grid(row=3, column=0, sticky="w", pady=(8, 8))
        ttk.Label(right, text="Preview", style="Overview.SectionTitle.TLabel").grid(row=4, column=0, sticky="w")
        preview_frame = ttk.Frame(right)
        preview_frame.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview = tk.Text(preview_frame, wrap="none", height=20, font=("TkFixedFont", 9))
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.configure(state="disabled")
        p_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview.yview)
        p_y.grid(row=0, column=1, sticky="ns")
        p_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview.xview)
        p_x.grid(row=1, column=0, sticky="ew")
        self.preview.configure(yscrollcommand=p_y.set, xscrollcommand=p_x.set)

    def reload(self):
        self.model = load_outputs(self.model.project_dir)
        self.group_combo.configure(values=["All", *self.model.groups])
        if self.group_var.get() not in {"All", *self.model.groups}:
            self.group_var.set("All")
        self.cards["all"].value.set(str(len(self.model.artifacts)))
        self.cards["observed"].value.set(str(self.model.count("Observed analysis")))
        self.cards["calibration"].value.set(str(self.model.count("Calibration")))
        self.cards["size"].value.set(_format_file_size(self.model.total_size_bytes))
        self.refresh_list()

    def refresh_list(self):
        previous = self.selected_artifact.path if self.selected_artifact else None
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.item_artifacts.clear()
        rows = self.model.filtered(self.group_var.get(), self.search_var.get())
        selected_item = None
        for artifact in rows:
            item = self.tree.insert("", "end", values=(
                artifact.group,
                artifact.relative_path,
                artifact.type_label,
                _format_file_size(artifact.size_bytes),
                _format_modified(artifact.modified_timestamp),
            ))
            self.item_artifacts[item] = artifact
            if previous and artifact.path == previous:
                selected_item = item
        if selected_item is None and rows:
            selected_item = next(iter(self.item_artifacts))
        if selected_item:
            self.tree.selection_set(selected_item)
            self.tree.focus(selected_item)
            self.tree.see(selected_item)
            self._show_artifact(self.item_artifacts[selected_item])
        else:
            self._show_artifact(None)

    def _selection_changed(self, event=None):
        selection = self.tree.selection()
        if selection:
            self._show_artifact(self.item_artifacts.get(selection[0]))

    def _show_artifact(self, artifact: OutputArtifact | None):
        self.selected_artifact = artifact
        if artifact is None:
            self.selected_name.set("No matching artifact")
            self.selected_meta.set("")
            self.selected_description.set("")
            preview = ""
        else:
            self.selected_name.set(artifact.name)
            self.selected_meta.set(
                f"{artifact.group} · {artifact.type_label} · {_format_file_size(artifact.size_bytes)}\n"
                f"{artifact.relative_path}\nModified {_format_modified(artifact.modified_timestamp)}"
            )
            self.selected_description.set(artifact.description)
            preview = _preview_artifact(artifact)
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", preview)
        self.preview.configure(state="disabled")

    def open_selected(self):
        artifact = self.selected_artifact
        if artifact is None:
            return
        try:
            _open_external(artifact.path, force_browser=artifact.suffix == ".html")
        except Exception as exc:
            messagebox.showerror("Open output", f"Could not open {artifact.name}:\n{exc}", parent=self)

    def open_selected_folder(self):
        artifact = self.selected_artifact
        target = artifact.path.parent if artifact else self.model.project_dir / "results"
        try:
            _open_external(target)
        except Exception as exc:
            messagebox.showerror("Open folder", f"Could not open {target}:\n{exc}", parent=self)

    def copy_path(self):
        artifact = self.selected_artifact
        if artifact is None:
            return
        self.clipboard_clear()
        self.clipboard_append(str(artifact.path))
        self.update_idletasks()

    def save_copy(self):
        artifact = self.selected_artifact
        if artifact is None:
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Save output copy",
            initialfile=artifact.name,
            defaultextension=artifact.suffix,
        )
        if not destination:
            return
        try:
            shutil.copy2(artifact.path, destination)
        except OSError as exc:
            messagebox.showerror("Save output copy", f"Could not save the file:\n{exc}", parent=self)


class ResultsWorkspace(tk.Toplevel):
    """Read-only results workspace for observed, calibration, and robustness outputs."""

    def __init__(self, parent, project_dir: str | Path):
        self.model = load_overview(project_dir)
        self.calibration_model = load_calibration(project_dir)
        self.diagnostics_model = load_diagnostics(project_dir)
        self.spatial_model = load_spatial(project_dir)
        self.bsa_detail_model = load_bsa_detail(project_dir)
        self.robustness_model = load_robustness(project_dir)
        self.confounder_model = load_confounder(project_dir)
        self.outputs_model = load_outputs(project_dir)
        super().__init__(parent)
        self.title(f"APHREH-ADSMap | Results | {self.model.project_dir.name}")
        self.geometry("1400x850")
        self.minsize(1080, 700)
        self.transient(parent)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self.selected_cell = self.model.default_cell_key
        self.selected_bsa = self.bsa_detail_model.default_bsa(*self.selected_cell) if self.selected_cell else None
        self.active_page = "Overview"
        self._build_header()
        self._build_navigation()
        self.content = ttk.Frame(self)
        self.content.grid(row=1, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self._build_pages()
        self._build_statusbar()
        self.show_page("Overview")
        if self.selected_cell:
            self.select_cell(*self.selected_cell)

    def _build_header(self):
        header = ttk.Frame(self, padding=(14, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="APHREH-ADSMap Results", font=("Segoe UI", 17, "bold")).pack(
            side="left"
        )
        ttk.Label(header, text=self.model.project_dir.name).pack(side="left", padx=(14, 0))
        buttons = ttk.Frame(header)
        buttons.pack(side="right")
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(buttons, text="Open static HTML report", command=self.open_report).pack(side="left", padx=4)
        ttk.Button(buttons, text="Open results folder", command=self.open_folder).pack(side="left", padx=4)

    def _build_navigation(self):
        nav = ttk.Frame(self, padding=(10, 12), relief="ridge")
        nav.grid(row=1, column=0, sticky="ns")
        ttk.Label(nav, text="RESULTS", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        self.nav_buttons = {}
        for label in ("Overview", "Surface", "Spatial", "BSA detail", "Calibration", "Diagnostics", "Robustness / sensitivity", "Candidate confounder", "Outputs"):
            button = ttk.Button(nav, text=label, width=24, command=lambda name=label: self.show_page(name))
            button.pack(fill="x", pady=2)
            self.nav_buttons[label] = button

    def _build_pages(self):
        self.overview = OverviewPanel(
            self.content,
            self.model,
            self.select_cell,
            calibration_model=self.calibration_model,
        )
        self.surface_page = SurfacePanel(
            self.content,
            self.model,
            self.select_cell,
            calibration_model=self.calibration_model,
        )
        self.spatial_page = SpatialPanel(self.content, self.model, self.spatial_model, self.select_bsa)
        self.bsa_detail_page = BsaDetailPanel(
            self.content, self.model, self.bsa_detail_model, self.select_bsa, spatial_model=self.spatial_model
        )
        self.calibration_page = CalibrationPanel(self.content, self.calibration_model)
        self.diagnostics_page = DiagnosticsPanel(self.content, self.model, self.diagnostics_model)
        self.robustness_page = RobustnessPanel(self.content, self.robustness_model)
        self.confounder_page = ConfounderPanel(self.content, self.confounder_model)
        self.outputs_page = OutputsPanel(self.content, self.outputs_model)
        self.pages = {
            "Overview": self.overview,
            "Surface": self.surface_page,
            "Spatial": self.spatial_page,
            "BSA detail": self.bsa_detail_page,
            "Calibration": self.calibration_page,
            "Diagnostics": self.diagnostics_page,
            "Robustness / sensitivity": self.robustness_page,
            "Candidate confounder": self.confounder_page,
            "Outputs": self.outputs_page,
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(12, 5), relief="ridge")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        manifest_status = (self.model.manifest or {}).get("status", "unknown")
        ttk.Label(bar, text=f"Project: {manifest_status}").pack(side="left")
        ttk.Label(bar, text="Reference metric: WMARMcs").pack(side="left", padx=(18, 0))
        self.selection_status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.selection_status).pack(side="left", padx=(18, 0))
        ttk.Label(bar, text=self.model.status_title).pack(side="right")

    def show_page(self, name: str):
        page = self.pages[name]
        page.tkraise()
        self.active_page = name
        for label, button in self.nav_buttons.items():
            button.configure(state="disabled" if label == name else "normal")

    def select_cell(self, q, lag):
        key = (_float(q), _int(lag))
        if key[0] is None or key[1] is None or self.model.cell(*key) is None:
            return
        self.selected_cell = key
        self.overview.set_selected_cell(*key)
        self.surface_page.set_selected_cell(*key)
        self.spatial_page.set_selected_cell(*key)
        self.bsa_detail_page.set_selected_cell(*key)
        self.diagnostics_page.set_selected_cell(*key)
        ids=self.bsa_detail_model.bsa_ids(*key)
        if self.selected_bsa not in ids:
            self.selected_bsa=self.bsa_detail_model.default_bsa(*key)
        if self.selected_bsa:
            self.spatial_page.set_selected_bsa(self.selected_bsa)
            self.bsa_detail_page.set_selected_bsa(self.selected_bsa)
        self._update_selection_status()

    def select_bsa(self,bsa_id):
        if not self.selected_cell:
            return
        bid=_join_key(bsa_id)
        if self.bsa_detail_model.pooled_row(*self.selected_cell,bid) is None:
            return
        self.selected_bsa=bid
        self.spatial_page.set_selected_bsa(bid)
        self.bsa_detail_page.set_selected_bsa(bid)
        self._update_selection_status()

    def _update_selection_status(self):
        if not self.selected_cell:
            self.selection_status.set(""); return
        q,lag=self.selected_cell
        suffix=f" · BSA {self.selected_bsa}" if self.selected_bsa else ""
        self.selection_status.set(f"Selected: {_fmt_q(q)}, lag {lag}{suffix}")

    def refresh(self):
        model = load_overview(self.model.project_dir)
        calibration_model = load_calibration(self.model.project_dir)
        diagnostics_model = load_diagnostics(self.model.project_dir)
        spatial_model = load_spatial(self.model.project_dir)
        bsa_detail_model = load_bsa_detail(self.model.project_dir)
        robustness_model = load_robustness(self.model.project_dir)
        confounder_model = load_confounder(self.model.project_dir)
        outputs_model = load_outputs(self.model.project_dir)
        old_selected = self.selected_cell
        old_bsa = self.selected_bsa
        old_page = self.active_page
        self.model = model
        self.calibration_model = calibration_model
        self.diagnostics_model = diagnostics_model
        self.spatial_model = spatial_model
        self.bsa_detail_model = bsa_detail_model
        self.robustness_model = robustness_model
        self.confounder_model = confounder_model
        self.outputs_model = outputs_model
        for page in self.pages.values():
            page.destroy()
        self._build_pages()
        if old_selected and model.cell(*old_selected):
            self.selected_cell = old_selected
        else:
            self.selected_cell = model.default_cell_key
        if self.selected_cell:
            ids=self.bsa_detail_model.bsa_ids(*self.selected_cell)
            self.selected_bsa=old_bsa if old_bsa in ids else self.bsa_detail_model.default_bsa(*self.selected_cell)
            self.select_cell(*self.selected_cell)
        self.show_page(old_page if old_page in self.pages else "Overview")

    def open_report(self):
        path = self.model.results_dir / "report.html"
        if path.is_file():
            webbrowser.open(path.resolve().as_uri())

    def open_folder(self):
        webbrowser.open(self.model.results_dir.resolve().as_uri())

"""CA7 complete confounder-adjusted threshold-lag surface generation.

The complete adjusted surface reuses the authoritative FixedParametersEngine and
SurfaceRunner. Cross-fitted CA4 compatibility contexts remain frozen across the
entire target-hazard threshold-lag domain. CA7 deliberately does not execute
adjusted LOYO persistence, null calibration, robust-region resolution, or
representative selection; those are CA8 responsibilities.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ..engine import FixedParametersEngine, SurfaceRunner, cell_key
from ..storage import Store
from .contracts import (
    ConfounderCalibrationResult,
    ConfounderCompatibilityContext,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
)
from .inference import adjusted_fixed_run_id
from .persistence import ConfounderCheckpointState, ConfounderPersistence


SURFACE_STAGE = "CA7_COMPLETE_ADJUSTED_SURFACE"
SURFACE_SCHEMA = "APHREH_CONFOUNDER_ADJUSTED_SURFACE_CA7_V1"


class ConfounderAdjustedSurfaceError(ValueError):
    pass


def adjusted_surface_run_id(analysis_run_id: str) -> str:
    """Run identity shared with CA6 so the representative cell is reused.

    CA6 initially materializes one adjusted fixed-parameter cell. CA7 extends the
    same adjusted realization over the full threshold-lag domain, so a second run
    identity would duplicate the representative cell and weaken provenance.
    """
    return adjusted_fixed_run_id(analysis_run_id)


def _project_array(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfounderAdjustedSurfaceError(
            "candidate percentile path escapes the project root"
        ) from exc
    if not path.is_file():
        raise ConfounderAdjustedSurfaceError(
            f"candidate percentile panel is unavailable: {path}"
        )
    return path


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class ConfounderAdjustedSurfaceRunner:
    """Extend the CA6 adjusted realization over the complete target-hazard grid."""

    def __init__(
        self,
        data,
        config: dict,
        store: Store,
        project_root: str | Path,
        persistence: ConfounderPersistence,
        analysis_run_id: str,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.store = store
        self.root = Path(project_root).expanduser().resolve()
        self.persistence = persistence
        self.analysis_run_id = analysis_run_id
        self.preprocessing = preprocessing
        self.log = log
        self.engine = FixedParametersEngine(data, config, store, self.root, log)
        self.surface = SurfaceRunner(self.engine)

    def _contexts(self) -> dict[int, ConfounderCompatibilityContext]:
        return {
            int(raw["held_out_year"]): ConfounderCompatibilityContext.from_dict(raw)
            for raw in self.store.rows(
                "confounder_compatibility_context", self.analysis_run_id
            )
        }

    def _calibration_exclusions(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for raw in self.store.rows("confounder_calibration", self.analysis_run_id):
            calibration = ConfounderCalibrationResult.from_dict(raw)
            if calibration.exclusion_reason is not None:
                out[int(calibration.held_out_year)] = calibration.exclusion_reason.value
        return out

    def _persist_surface_cells(
        self, adjusted_run: str, cells: list[Mapping[str, Any]]
    ) -> None:
        for cell in cells:
            key = cell_key(float(cell["q"]), int(cell["lag"]))
            row = {
                "schema_version": SURFACE_SCHEMA,
                "analysis_run_id": self.analysis_run_id,
                "adjusted_run_id": adjusted_run,
                **dict(cell),
            }
            self.store.put("confounder_adjusted_surface", self.analysis_run_id, key, row)
            self.persistence.put_checkpoint(
                self.analysis_run_id,
                SURFACE_STAGE,
                "cell",
                key,
                ConfounderCheckpointState.COMPLETED,
                payload={
                    "valid": bool(cell.get("valid")),
                    "wmarm_cs": cell.get("wmarm_cs"),
                    "reason": cell.get("reason"),
                },
            )
        self.store.commit()

    @staticmethod
    def _surface_summary(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
        valid = [c for c in cells if c.get("valid") and _finite(c.get("wmarm_cs")) is not None]
        wmarm_values = [float(c["wmarm_cs"]) for c in valid]
        maximum = max(wmarm_values) if wmarm_values else None
        maxima = []
        if maximum is not None:
            maxima = [
                {"q": float(c["q"]), "lag": int(c["lag"]), "wmarm_cs": float(c["wmarm_cs"])}
                for c in valid
                if math.isclose(float(c["wmarm_cs"]), maximum, rel_tol=0.0, abs_tol=1e-15)
            ]
        return {
            "surface_cell_count": len(cells),
            "valid_surface_cell_count": len(valid),
            "invalid_surface_cell_count": len(cells) - len(valid),
            "descriptive_surface_maximum_wmarm_cs": maximum,
            "descriptive_surface_maximum_cells": maxima,
            "all_valid_wmarm_cs_zero": bool(valid) and all(v == 0.0 for v in wmarm_values),
        }

    def run(self, *, requested: bool) -> dict[str, Any]:
        if not requested:
            summary = {
                "schema_version": SURFACE_SCHEMA,
                "status": "NOT_REQUESTED",
                "stage": SURFACE_STAGE,
                "complete_adjusted_surface_requested": False,
                "complete_adjusted_surface_enabled": True,
                "adjusted_reoptimization_inference_enabled": False,
                "surface_cell_count": 0,
                "valid_surface_cell_count": 0,
                "invalid_surface_cell_count": 0,
            }
            self.store.put("workflow", self.analysis_run_id, "adjusted_surface_result", summary)
            self.store.commit()
            return summary

        contexts = self._contexts()
        if not contexts:
            summary = {
                "schema_version": SURFACE_SCHEMA,
                "status": "UNAVAILABLE",
                "stage": SURFACE_STAGE,
                "reason": ConfounderExclusionReason.REFERENCE_CONFIGURATION_UNAVAILABLE.value,
                "complete_adjusted_surface_requested": True,
                "complete_adjusted_surface_enabled": True,
                "adjusted_reoptimization_inference_enabled": False,
                "surface_cell_count": 0,
                "valid_surface_cell_count": 0,
                "invalid_surface_cell_count": 0,
            }
            self.store.put("workflow", self.analysis_run_id, "adjusted_surface_result", summary)
            self.store.commit()
            return summary

        exclusions = self._calibration_exclusions()
        percentile_path = _project_array(
            self.root, self.preprocessing.percentile_panel.array_path
        )
        adjusted_run = adjusted_surface_run_id(self.analysis_run_id)
        expected = len(self.c["thresholds"]) * len(self.c["lags"])
        self.log(
            f"Confounder CA7: complete adjusted surface; cells={expected}; "
            f"contexts={len(contexts)}/{len(self.data.years)}"
        )
        cells = self.surface.run(
            adjusted_run,
            loyo=False,
            retain=False,
            rng_identity_run="REFERENCE",
            candidate_percentile_path=percentile_path,
            compatibility_contexts=contexts,
            compatibility_exclusion_reasons=exclusions,
        )
        self._persist_surface_cells(adjusted_run, cells)
        stats = self._surface_summary(cells)
        summary = {
            "schema_version": SURFACE_SCHEMA,
            "status": "COMPLETED" if stats["valid_surface_cell_count"] else "UNAVAILABLE",
            "stage": SURFACE_STAGE,
            "reason": None if stats["valid_surface_cell_count"] else ConfounderExclusionReason.NO_ELIGIBLE_ADJUSTED_POOLED_BSAS.value,
            "candidate_id": self.preprocessing.candidate_id,
            "analysis_run_id": self.analysis_run_id,
            "adjusted_run_id": adjusted_run,
            "reference_metric": "WMARM_CS",
            "complete_adjusted_surface_requested": True,
            "complete_adjusted_surface_enabled": True,
            "adjusted_reoptimization_inference_enabled": False,
            "loyo_executed": False,
            "null_search_executed": False,
            "robust_region_resolved": False,
            "representative_selected": False,
            "available_compatibility_contexts": len(contexts),
            "unavailable_compatibility_years": len(self.data.years) - len(contexts),
            "standard_reoptimization_year_requirement_met": len(self.data.years)
            >= int(self.c["confounder"]["minimum_years_reoptimization"]),
            **stats,
        }
        self.store.put("workflow", self.analysis_run_id, "adjusted_surface_result", summary)
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            SURFACE_STAGE,
            "surface",
            "complete",
            ConfounderCheckpointState.COMPLETED,
            payload=summary,
        )
        self.log(
            f"Confounder CA7 complete: valid={stats['valid_surface_cell_count']}/{stats['surface_cell_count']}; "
            f"descriptive max WMARMcs={stats['descriptive_surface_maximum_wmarm_cs']}"
        )
        return summary

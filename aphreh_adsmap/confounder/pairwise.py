"""CA4 bounded-memory target-hazard pair-sample extraction.

The extractor reconstructs the original-sample differential outcome, target
hazard relevance weight, and exposure class at the reference representative
configuration for one cross-fit fold. It deliberately stops before bootstrap
inference. Candidate percentiles are referenced from the CA2 disk-backed panel.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .. import numerics as nm
from ..data import Data
from ..storage import mapped
from ..support import active_support_policy, apply_baseline_support_filter
from .contracts import ConfounderPreprocessingResult, PairwiseSampleBundle
from .execution import CrossfitOutcomeData, ConfounderExecutionError


class PairwiseSampleExtractionError(ValueError):
    pass


def _inside_project(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise PairwiseSampleExtractionError("pairwise input path escapes the project root") from exc
    if not path.is_file():
        raise PairwiseSampleExtractionError(f"required pairwise input is unavailable: {path}")
    return path


class PairwiseSampleExtractor:
    """Extract one CA9.2 original target-hazard sample panel per held-out year."""

    def __init__(
        self,
        data: Data,
        config: dict,
        project_root: str | Path,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.root = Path(project_root).expanduser().resolve()
        self.preprocessing = preprocessing
        self.log = log

    def extract(
        self,
        *,
        analysis_run_id: str,
        held_out_year: int,
        training_years: Iterable[int],
        target_threshold: float,
        target_lag: int,
    ) -> PairwiseSampleBundle:
        training = tuple(sorted(set(int(y) for y in training_years)))
        if held_out_year in training:
            raise PairwiseSampleExtractionError("held-out year cannot occur in training_years")
        filtered = CrossfitOutcomeData(self.data, training_years=training)
        d, c = filtered, self.c
        if tuple(self.preprocessing.years) != tuple(d.years):
            raise PairwiseSampleExtractionError("candidate preprocessing years do not match canonical data")
        if self.preprocessing.bsa_count != d.B or self.preprocessing.date_count != d.n:
            raise PairwiseSampleExtractionError("candidate preprocessing dimensions do not match canonical data")

        percentile_path = _inside_project(
            self.root, self.preprocessing.percentile_panel.array_path
        )
        with mapped(percentile_path) as percentiles:
            if percentiles.shape != (d.B, d.n):
                raise PairwiseSampleExtractionError("candidate percentile panel shape is inconsistent")

        exposure_mode = c["core"]["exposure"].get(
            "mode", "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE"
        )
        local_exposure = exposure_mode == "BSA_SPECIFIC"
        if exposure_mode not in (
            "TERRITORY_WIDE_WITH_LOCAL_CONCORDANCE", "BSA_SPECIFIC"
        ):
            raise PairwiseSampleExtractionError(f"unsupported exposure mode: {exposure_mode}")
        support_policy = active_support_policy(c)
        if local_exposure and support_policy.get("min_local_concordance") is not None:
            raise PairwiseSampleExtractionError(
                "min_local_concordance is not applicable under BSA_SPECIFIC exposure classification"
            )

        common_labels = None
        if not local_exposure:
            territory = d.series("territory")
            common_labels, _ = nm.exposure(territory, d.year_index, float(target_threshold))

        base_dir = self.root / "confounder_runs" / analysis_run_id / "pairwise"
        base_dir.mkdir(parents=True, exist_ok=True)
        final_dir = base_dir / f"heldout_{int(held_out_year)}"
        staging = Path(
            tempfile.mkdtemp(
                prefix=f"heldout_{int(held_out_year)}.", suffix=".tmp", dir=base_dir
            )
        )
        valid_count = 0
        positive_weight_count = 0
        try:
            differential_path = staging / "differential.npy"
            raw_weight_path = staging / "raw_weight.npy"
            label_path = staging / "exposure_label.npy"
            with mapped(differential_path, shape=(d.B, d.n)) as differentials, \
                 mapped(raw_weight_path, shape=(d.B, d.n)) as raw_weights, \
                 mapped(label_path, shape=(d.B, d.n), dtype="int8") as labels_panel:
                differentials[:] = np.nan
                raw_weights[:] = 0.0
                labels_panel[:] = -1
                for b, bid in enumerate(d.ids):
                    source = d.series("outcome", b)
                    incidence = nm.cumulative_incidence(
                        source,
                        d.daily_population(b),
                        int(target_lag),
                        c["core"]["incidence"]["scaling_multiplier"],
                        "EVENT_COUNTS",
                        d.segment_index,
                    )
                    local = d.series("hazard", b)
                    if local_exposure:
                        labels_b, thresholds = nm.exposure(
                            local, d.year_index, float(target_threshold)
                        )
                    else:
                        labels_b = common_labels
                        thresholds = None
                    baseline, _, donors = nm.baseline(
                        incidence,
                        labels_b,
                        c["core"]["baseline"],
                        d.segment_index,
                        d.weekday_index,
                        d.holiday_state,
                    )
                    x_unfiltered = incidence - baseline
                    x, _ = apply_baseline_support_filter(
                        x_unfiltered,
                        donors,
                        support_policy.get("min_baseline_support"),
                    )
                    w, thresholds_check = nm.relevance(
                        local, labels_b, d.year_index, float(target_threshold)
                    )
                    if local_exposure and not np.allclose(
                        thresholds, thresholds_check, equal_nan=True
                    ):
                        raise ArithmeticError("local exposure and relevance thresholds diverged")
                    differentials[b] = x
                    raw_weights[b] = w
                    labels_panel[b] = labels_b
                    valid = np.isfinite(x) & (labels_b >= 0)
                    valid_count += int(np.sum(valid))
                    positive_weight_count += int(np.sum(valid & (w > 0)))

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging, final_dir)
            relative = final_dir.relative_to(self.root)
            bundle = PairwiseSampleBundle(
                candidate_id=self.preprocessing.candidate_id,
                held_out_year=int(held_out_year),
                training_years=training,
                target_threshold=float(target_threshold),
                target_lag=int(target_lag),
                differential_array_path=(relative / "differential.npy").as_posix(),
                raw_weight_array_path=(relative / "raw_weight.npy").as_posix(),
                exposure_label_array_path=(relative / "exposure_label.npy").as_posix(),
                candidate_percentile_array_path=self.preprocessing.percentile_panel.array_path,
                bsa_count=d.B,
                date_count=d.n,
                valid_differential_count=valid_count,
                positive_weight_count=positive_weight_count,
            )
            self.log(
                f"Confounder CA4: pairwise sample held-out={held_out_year}; "
                f"q={float(target_threshold):g}; lag={int(target_lag)}; "
                f"valid={valid_count}; positive-weight={positive_weight_count}"
            )
            return bundle
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

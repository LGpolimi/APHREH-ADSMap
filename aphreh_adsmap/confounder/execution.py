"""Cross-fitted data bindings for candidate-confounder work.

CA3 uses :class:`ConfounderProfilingData` to bind the candidate as the APHREH
exposure. CA4 additionally uses :class:`CrossfitOutcomeData` to retain the
ordinary target hazard while masking the held-out health-outcome year before
cumulative incidence, baseline construction, and dependence estimation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..data import Data
from ..storage import mapped
from .contracts import ConfounderPreprocessingResult


class ConfounderExecutionError(ValueError):
    pass


def crossfit_training_years(years: Iterable[int], held_out_year: int) -> tuple[int, ...]:
    values = tuple(sorted(set(int(y) for y in years)))
    if held_out_year not in values:
        raise ConfounderExecutionError(f"held-out year {held_out_year} is not in the analytical years")
    training = tuple(y for y in values if y != held_out_year)
    if not training:
        raise ConfounderExecutionError("cross-fitted profiling requires at least one training year")
    return training


def _project_array(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ConfounderExecutionError("confounder array path escapes the project root") from exc
    if not path.is_file():
        raise ConfounderExecutionError(f"required confounder array is unavailable: {path}")
    return path


class CrossfitOutcomeData:
    """Read-only Data-compatible view with selected outcome years available.

    The target hazard remains the canonical reference hazard. Outcome and
    population support outside ``training_years`` are removed before any core
    temporal operation. Segment identifiers are split wherever the availability
    state changes, so cumulative windows and local baselines cannot cross a
    held-out-year boundary. This is the CA5.1 cross-fitting rule reused by CA4
    target-hazard pair-sample extraction.
    """

    def __init__(self, base: Data, *, training_years: Iterable[int]):
        self.base = base
        self.root = base.root
        self.ids = list(base.ids)
        self.years = list(base.years)
        self.dates = np.array(base.dates, copy=True)
        self.year_index = np.array(base.year_index, copy=True)
        self.n, self.B = base.n, base.B
        self.weekday_index = np.array(base.weekday_index, copy=True)
        self.holiday_state = np.array(base.holiday_state, copy=True)
        self.meta = dict(base.meta)

        training = tuple(sorted(set(int(y) for y in training_years)))
        if not training:
            raise ConfounderExecutionError("training_years cannot be empty")
        unknown = [y for y in training if y not in self.years]
        if unknown:
            raise ConfounderExecutionError(f"training years are absent from canonical data: {unknown}")
        self.training_years = training
        active_year_indices = {self.years.index(y) for y in training}
        self.training_mask = np.array(
            [int(iy) in active_year_indices for iy in self.year_index], dtype=bool
        )
        if not np.any(self.training_mask):
            raise ConfounderExecutionError("training-year mask contains no canonical dates")

        self.population = np.array(base.population, copy=True)
        self.population_source_year = np.array(base.population_source_year, copy=True)
        for iy, year in enumerate(self.years):
            if year not in training:
                self.population[:, iy] = np.nan
                self.population_source_year[:, iy] = -1

        self.segment_index = np.zeros(self.n, dtype=np.int32)
        if self.n > 1:
            breaks = (
                (base.segment_index[1:] != base.segment_index[:-1])
                | (self.training_mask[1:] != self.training_mask[:-1])
            )
            self.segment_index[1:] = np.cumsum(breaks, dtype=np.int32)
        self.segment_starts = np.empty(self.n, dtype=np.int32)
        self.segment_stops = np.empty(self.n, dtype=np.int32)
        for sid in np.unique(self.segment_index):
            ix = np.flatnonzero(self.segment_index == sid)
            self.segment_starts[ix] = ix[0]
            self.segment_stops[ix] = ix[-1] + 1

    def series(self, name: str, b: int | None = None, path: Path | None = None) -> np.ndarray:
        result = self.base.series(name, b, path=path)
        if name == "outcome":
            result[~self.training_mask] = np.nan
        return result

    def daily_population(self, b: int) -> np.ndarray:
        return self.population[b, self.year_index]


class ConfounderProfilingData(CrossfitOutcomeData):
    """Cross-fitted data view that binds the candidate as APHREH exposure."""

    def __init__(
        self,
        base: Data,
        *,
        project_root: str | Path,
        preprocessing: ConfounderPreprocessingResult,
        training_years: Iterable[int],
    ):
        super().__init__(base, training_years=training_years)
        self.project_root = Path(project_root).expanduser().resolve()
        self.preprocessing = preprocessing
        self._territory_path = _project_array(
            self.project_root, preprocessing.territory_array_path
        )
        self._bsa_path = _project_array(self.project_root, preprocessing.bsa_array_path)
        with mapped(self._territory_path) as territory:
            if territory.shape != (self.n,):
                raise ConfounderExecutionError("candidate territory array shape does not match canonical dates")
        with mapped(self._bsa_path) as local:
            if local.shape != (self.B, self.n):
                raise ConfounderExecutionError("candidate BSA array shape does not match canonical panel")
        if preprocessing.bsa_count != self.B or preprocessing.date_count != self.n:
            raise ConfounderExecutionError("candidate preprocessing dimensions do not match canonical data")
        if tuple(preprocessing.years) != tuple(self.years):
            raise ConfounderExecutionError("candidate preprocessing years do not match canonical data")

    def series(self, name: str, b: int | None = None, path: Path | None = None) -> np.ndarray:
        if name == "territory":
            if path is not None or b is not None:
                raise ConfounderExecutionError("candidate territory binding does not accept path/BSA overrides")
            with mapped(self._territory_path) as a:
                return np.array(a)
        if name == "hazard":
            if path is not None or b is None:
                raise ConfounderExecutionError("candidate local hazard binding requires one BSA and no path override")
            with mapped(self._bsa_path) as a:
                return np.array(a[b])
        return super().series(name, b, path=path)

"""CA3 cross-fitted candidate-confounder profiling orchestration."""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from ..data import Data
from ..engine import Cancelled, FixedParametersEngine, SurfaceRunner, cell_key
from ..performance import MIB, bounded_parallel_plan, configure_process_worker, process_tree_rss_bytes
from ..storage import Store
from .configuration import ConfounderAdjustmentSettings
from .contracts import (
    ConfounderCandidateDefinition,
    ConfounderExclusionReason,
    ConfounderPreprocessingResult,
    ConfounderProfilingResult,
    ConfounderProfilingStatus,
)
from .execution import ConfounderProfilingData, crossfit_training_years
from .identity import profile_identity, profile_run_id
from .persistence import (
    ConfounderCheckpointState,
    ConfounderPersistence,
    ConfounderRunState,
)


PROFILE_STAGE = "CROSS_FITTED_PROFILE"
_PROFILE_WORKER_CONTEXT: dict = {}


def _finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def confounder_profile_parallel_plan(task_count: int) -> dict:
    """Bounded concurrency for independent held-out-year profile surfaces."""
    return bounded_parallel_plan(
        task_count,
        workers_env="APHREH_ADSMAP_CONFOUNDER_WORKERS",
        worker_rss_env="APHREH_ADSMAP_CONFOUNDER_WORKER_RSS_MB",
        parent_reserve_env="APHREH_ADSMAP_CONFOUNDER_PARENT_RESERVE_MB",
    )


def _profile_worker_initializer(
    config: dict,
    canonical_root: str,
    project_root: str,
    worker_base: str,
    parent_db: str,
    preprocessing: dict,
    plan: dict,
) -> None:
    configure_process_worker(plan)
    _PROFILE_WORKER_CONTEXT.clear()
    _PROFILE_WORKER_CONTEXT.update({
        "config": config,
        "data": Data(Path(canonical_root)),
        "project_root": Path(project_root),
        "worker_base": Path(worker_base),
        "parent_db": Path(parent_db),
        "preprocessing": ConfounderPreprocessingResult.from_dict(preprocessing),
    })


def _profile_worker(task: dict) -> dict:
    """Compute one CA3 held-out-year surface in an isolated Store."""
    ctx = _PROFILE_WORKER_CONTEXT
    held_out_year = int(task["held_out_year"])
    surface_run = str(task["surface_run"])
    root = ctx["worker_base"] / f"y{held_out_year}"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / "results.sqlite")
    error = None
    try:
        # Import any committed partial cells from an interrupted serial/parallel
        # attempt so the ordinary FixedParametersEngine resumes them verbatim.
        store.import_run_from(ctx["parent_db"], surface_run)
        profile_data = ConfounderProfilingData(
            ctx["data"],
            project_root=ctx["project_root"],
            preprocessing=ctx["preprocessing"],
            training_years=tuple(int(y) for y in task["training_years"]),
        )
        engine = FixedParametersEngine(
            profile_data,
            ctx["config"],
            store,
            root,
            log=lambda *_: None,
            cancellation_root=ctx["project_root"],
        )
        SurfaceRunner(engine).run(
            surface_run,
            loyo=False,
            retain=False,
            rng_identity_run=surface_run,
        )
        store.commit()
    except BaseException as exc:
        try:
            store.db.rollback()
            store.commit()
        except Exception:
            pass
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        store.close()
    return {
        "ok": error is None,
        "held_out_year": held_out_year,
        "surface_run": surface_run,
        "error": error,
        "worker_root": str(root),
        "database": str(root / "results.sqlite"),
    }


def _terminate_pool(executor: ProcessPoolExecutor) -> None:
    for process in list(getattr(executor, "_processes", {}).values()):
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            pass
    for process in list(getattr(executor, "_processes", {}).values()):
        try:
            process.join(timeout=2)
        except Exception:
            pass


class ConfounderProfilingRunner:
    """Run one candidate as the exposure in leave-one-year-out training datasets.

    CA5.1-CA6: for each held-out target year, all health-outcome observations in
    that year are unavailable before cumulative incidence and baseline
    construction. The candidate-only surface is then evaluated by the ordinary
    ``FixedParametersEngine`` and ``SurfaceRunner`` with LOYO disabled, because
    cross-fitting itself already defines the training dataset.

    Parallel execution is orchestration-only: each held-out year is still
    evaluated by the same engine, with the same cross-fit mask and RNG identity.
    Workers write isolated SQLite stores and the parent merges them serially.
    """

    def __init__(
        self,
        data: Data,
        config: dict,
        store: Store,
        root: str | Path,
        persistence: ConfounderPersistence,
        analysis_run_id: str,
        reference_identity: str,
        candidate: ConfounderCandidateDefinition,
        preprocessing: ConfounderPreprocessingResult,
        log=print,
    ):
        self.data = data
        self.c = config
        self.store = store
        self.root = Path(root).expanduser().resolve()
        self.persistence = persistence
        self.analysis_run_id = analysis_run_id
        self.reference_identity = reference_identity
        self.candidate = candidate
        self.preprocessing = preprocessing
        self.log = log
        self.settings = ConfounderAdjustmentSettings.from_dict(config["confounder"])
        self.thresholds = tuple(float(q) for q in config["thresholds"])
        self.lags = tuple(int(lag) for lag in config["lags"])

    def _existing_result(self, held_out_year: int) -> ConfounderProfilingResult | None:
        raw = self.store.get(
            "confounder_profile", self.analysis_run_id, str(int(held_out_year))
        )
        return None if raw is None else ConfounderProfilingResult.from_dict(raw)

    def _profile_definition(self, held_out_year: int) -> tuple[tuple[int, ...], str, str]:
        training_years = crossfit_training_years(self.data.years, held_out_year)
        identity = profile_identity(
            reference_identity=self.reference_identity,
            candidate=self.candidate,
            preprocessing=self.preprocessing,
            held_out_year=held_out_year,
            training_years=training_years,
            thresholds=self.thresholds,
            lags=self.lags,
        )
        return training_years, identity, profile_run_id(identity, held_out_year)

    def _reusable_result(
        self, held_out_year: int, surface_run: str
    ) -> ConfounderProfilingResult | None:
        checkpoint_key = str(int(held_out_year))
        checkpoint = self.persistence.checkpoint(
            self.analysis_run_id, PROFILE_STAGE, "held_out_year", checkpoint_key
        )
        existing = self._existing_result(held_out_year)
        if (
            checkpoint is not None
            and checkpoint["state"] == ConfounderCheckpointState.COMPLETED.value
            and existing is not None
            and existing.surface_run_id == surface_run
        ):
            self.log(
                f"Confounder CA3: reuse held-out {held_out_year} candidate profile "
                f"({existing.valid_cell_count}/{existing.cell_count} valid cells)"
            )
            return existing
        return None

    def _mark_running(
        self,
        held_out_year: int,
        training_years: tuple[int, ...],
        identity: str,
        surface_run: str,
    ) -> None:
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            PROFILE_STAGE,
            "held_out_year",
            str(int(held_out_year)),
            ConfounderCheckpointState.RUNNING,
            payload={
                "candidate_id": self.candidate.candidate_id,
                "held_out_year": held_out_year,
                "training_years": list(training_years),
                "surface_run_id": surface_run,
                "profile_identity": identity,
            },
        )
        self.log(
            f"Confounder CA3: profile {self.candidate.candidate_id}; held-out={held_out_year}; "
            f"training={list(training_years)}"
        )

    def _mark_failed(self, held_out_year: int, exc: BaseException) -> None:
        state = (
            ConfounderCheckpointState.CANCELLED
            if isinstance(exc, (Cancelled, KeyboardInterrupt))
            else ConfounderCheckpointState.FAILED
        )
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            PROFILE_STAGE,
            "held_out_year",
            str(int(held_out_year)),
            state,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _write_profile_cells(
        self,
        held_out_year: int,
        training_years: tuple[int, ...],
        surface_run_id: str,
        cells: list[dict],
    ) -> None:
        for cell in cells:
            row = {
                "analysis_run_id": self.analysis_run_id,
                "candidate_id": self.candidate.candidate_id,
                "held_out_year": int(held_out_year),
                "training_years": list(training_years),
                "surface_run_id": surface_run_id,
                "reference_metric": "WMARM_CS",
                "q": float(cell["q"]),
                "lag": int(cell["lag"]),
                "valid": bool(cell.get("valid")),
                "reason": cell.get("reason"),
                "wmarm_cs": cell.get("wmarm_cs"),
                "diagnostic_wmarm": cell.get("wmarm"),
                "marm": cell.get("marm"),
                "bsa_coverage": cell.get("bsa_coverage"),
                "population_coverage": cell.get("population_coverage"),
                "eligible_bsas": cell.get("eligible_bsas"),
                "represented_population": cell.get("represented_population"),
                "confidence_support_retention_ratio": cell.get(
                    "confidence_support_retention_ratio"
                ),
            }
            self.store.put(
                "confounder_profile_cell",
                self.analysis_run_id,
                f"{int(held_out_year)}/{cell_key(float(cell['q']), int(cell['lag']))}",
                row,
            )

    def _surface_cells(self, surface_run: str) -> list[dict]:
        cells: list[dict] = []
        for q in self.thresholds:
            for lag in self.lags:
                cell = self.store.get("cell", surface_run, cell_key(q, lag))
                if cell is None:
                    raise RuntimeError(
                        f"Confounder CA3 surface {surface_run} is missing q={q}, lag={lag}"
                    )
                cells.append(cell)
        return cells

    def _finalize_profile(
        self,
        held_out_year: int,
        training_years: tuple[int, ...],
        surface_run: str,
    ) -> ConfounderProfilingResult:
        cells = self._surface_cells(surface_run)
        self._write_profile_cells(held_out_year, training_years, surface_run, cells)
        valid = [
            cell for cell in cells
            if cell.get("valid") and cell.get("wmarm_cs") is not None
            and math.isfinite(float(cell["wmarm_cs"]))
        ]
        coverages = _finite(cell.get("population_coverage") for cell in valid)
        if valid:
            result = ConfounderProfilingResult(
                candidate_id=self.candidate.candidate_id,
                held_out_year=held_out_year,
                training_years=training_years,
                surface_run_id=surface_run,
                thresholds=self.thresholds,
                lags=self.lags,
                cell_count=len(cells),
                valid_cell_count=len(valid),
                status=ConfounderProfilingStatus.AVAILABLE,
                wmarm_cs_max=max(float(cell["wmarm_cs"]) for cell in valid),
                population_coverage_min=min(coverages) if coverages else None,
                population_coverage_max=max(coverages) if coverages else None,
            )
        else:
            result = ConfounderProfilingResult(
                candidate_id=self.candidate.candidate_id,
                held_out_year=held_out_year,
                training_years=training_years,
                surface_run_id=surface_run,
                thresholds=self.thresholds,
                lags=self.lags,
                cell_count=len(cells),
                valid_cell_count=0,
                status=ConfounderProfilingStatus.UNAVAILABLE,
                exclusion_reason=ConfounderExclusionReason.PROFILING_WMARM_CS_SURFACE_UNAVAILABLE,
            )
        checkpoint_key = str(int(held_out_year))
        self.store.put(
            "confounder_profile", self.analysis_run_id, checkpoint_key, result.to_dict()
        )
        self.store.commit()
        self.persistence.put_checkpoint(
            self.analysis_run_id,
            PROFILE_STAGE,
            "held_out_year",
            checkpoint_key,
            ConfounderCheckpointState.COMPLETED,
            payload=result.to_dict(),
        )
        return result

    def _profile_one(self, held_out_year: int) -> ConfounderProfilingResult:
        training_years, identity, surface_run = self._profile_definition(held_out_year)
        existing = self._reusable_result(held_out_year, surface_run)
        if existing is not None:
            return existing
        self._mark_running(held_out_year, training_years, identity, surface_run)
        try:
            profile_data = ConfounderProfilingData(
                self.data,
                project_root=self.root,
                preprocessing=self.preprocessing,
                training_years=training_years,
            )
            engine = FixedParametersEngine(
                profile_data, self.c, self.store, self.root, self.log
            )
            SurfaceRunner(engine).run(
                surface_run,
                loyo=False,
                retain=False,
                rng_identity_run=surface_run,
            )
            return self._finalize_profile(held_out_year, training_years, surface_run)
        except BaseException as exc:
            self._mark_failed(held_out_year, exc)
            raise

    def _worker_base(self) -> Path:
        token = hashlib.sha256(self.analysis_run_id.encode("utf-8")).hexdigest()[:16]
        return self.root / "scratch" / "parallel_confounder_profiles" / token

    def _recover_worker_checkpoints(self, worker_base: Path) -> int:
        recovered = 0
        if not worker_base.is_dir():
            return recovered
        for worker_root in sorted(worker_base.iterdir()):
            database = worker_root / "results.sqlite"
            if not worker_root.is_dir() or not database.is_file():
                continue
            try:
                self.store.merge_from(database)
                self.store.commit()
                recovered += 1
                shutil.rmtree(worker_root, ignore_errors=True)
            except Exception as exc:
                self.log(
                    f"Confounder CA3 checkpoint recovery skipped {worker_root.name}: {exc}"
                )
        if recovered:
            self.log(
                f"Confounder CA3 checkpoint recovery: merged {recovered} isolated worker stores"
            )
        return recovered

    def _parallel_profiles(
        self,
        definitions: list[tuple[int, tuple[int, ...], str, str]],
        plan: dict,
        worker_base: Path,
    ) -> dict[int, ConfounderProfilingResult]:
        worker_base.mkdir(parents=True, exist_ok=True)
        for held_out_year, training_years, identity, surface_run in definitions:
            self._mark_running(held_out_year, training_years, identity, surface_run)

        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=int(plan["worker_count"]),
            mp_context=ctx,
            initializer=_profile_worker_initializer,
            initargs=(
                self.c,
                str(self.data.root),
                str(self.root),
                str(worker_base),
                str(self.store.path),
                self.preprocessing.to_dict(),
                plan,
            ),
        )
        tasks = [
            {
                "held_out_year": year,
                "training_years": list(training),
                "surface_run": surface_run,
            }
            for year, training, _identity, surface_run in definitions
        ]
        definition_by_year = {d[0]: d for d in definitions}
        futures = {executor.submit(_profile_worker, task): task for task in tasks}
        pending = set(futures)
        results: dict[int, ConfounderProfilingResult] = {}
        completed = 0
        failure: BaseException | None = None
        failed_year: int | None = None
        try:
            while pending:
                if (self.root / "CANCEL").exists():
                    failure = Cancelled("Cancellation requested; committed profiles are resumable")
                    _terminate_pool(executor)
                    break
                if process_tree_rss_bytes() >= int(plan["aggregate_rss_cap_mib"]) * MIB:
                    failure = MemoryError(
                        "Analytical process-tree RSS reached the configured confounder-profile cap; "
                        "committed profile checkpoints remain resumable"
                    )
                    _terminate_pool(executor)
                    break
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures[future]
                    held_out_year = int(task["held_out_year"])
                    try:
                        payload = future.result()
                    except BaseException as exc:
                        failed_year = held_out_year
                        failure = RuntimeError(
                            f"Confounder CA3 worker process failed for held-out {held_out_year}: {exc}"
                        )
                        _terminate_pool(executor)
                        break
                    worker_root = Path(payload["worker_root"])
                    database = Path(payload["database"])
                    if database.is_file():
                        self.store.merge_from(database)
                        self.store.commit()
                    if not payload.get("ok"):
                        error = payload.get("error") or {}
                        etype = str(error.get("type") or "RuntimeError")
                        message = str(error.get("message") or "unknown worker failure")
                        failed_year = held_out_year
                        if etype == "Cancelled":
                            failure = Cancelled(message)
                        elif etype == "MemoryError":
                            failure = MemoryError(message)
                        else:
                            failure = RuntimeError(
                                f"Confounder CA3 worker failed for held-out {held_out_year}: "
                                f"{etype}: {message}\n{error.get('traceback', '')}"
                            )
                        _terminate_pool(executor)
                        break
                    shutil.rmtree(worker_root, ignore_errors=True)
                    _, training_years, _identity, surface_run = definition_by_year[held_out_year]
                    results[held_out_year] = self._finalize_profile(
                        held_out_year, training_years, surface_run
                    )
                    completed += 1
                    self.log(
                        f"Confounder CA3: held-out profile {completed}/{len(definitions)} complete "
                        f"({plan['worker_count']} workers)"
                    )
                if failure is not None:
                    break
        finally:
            if failure is not None:
                for future in pending:
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if worker_base.is_dir() and not any(worker_base.iterdir()):
                worker_base.rmdir()
                parent = worker_base.parent
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
        if failure is not None:
            if failed_year is not None:
                self._mark_failed(failed_year, failure)
            raise failure
        return results

    def run(self) -> dict[str, Any]:
        years = tuple(int(y) for y in self.data.years)
        if not self.settings.supports_primary_adjustment(len(years)):
            summary = {
                "status": "UNAVAILABLE",
                "stage": "CA3_CROSS_FITTED_PROFILING",
                "candidate_id": self.candidate.candidate_id,
                "analysis_run_id": self.analysis_run_id,
                "reference_metric": "WMARM_CS",
                "available_years": list(years),
                "minimum_years_primary": self.settings.minimum_years_primary,
                "exclusion_reason": ConfounderExclusionReason.INSUFFICIENT_YEARS_PRIMARY.value,
                "profiles": [],
            }
            self.store.put("workflow", self.analysis_run_id, "result", summary)
            self.store.commit()
            self.persistence.set_run_state(
                self.analysis_run_id, ConfounderRunState.COMPLETED
            )
            return summary

        self.persistence.set_run_state(
            self.analysis_run_id, ConfounderRunState.RUNNING
        )
        try:
            result_by_year: dict[int, ConfounderProfilingResult] = {}
            worker_base = self._worker_base()
            self._recover_worker_checkpoints(worker_base)
            pending: list[tuple[int, tuple[int, ...], str, str]] = []
            for year in years:
                training_years, identity, surface_run = self._profile_definition(year)
                existing = self._reusable_result(year, surface_run)
                if existing is not None:
                    result_by_year[year] = existing
                else:
                    pending.append((year, training_years, identity, surface_run))

            plan = confounder_profile_parallel_plan(len(pending))
            if (
                len(pending) > 1
                and int(plan["worker_count"]) > 1
                and mp.current_process().name == "MainProcess"
            ):
                self.log(
                    f"Confounder CA3 parallel profiling: {len(pending)} held-out years across "
                    f"{plan['worker_count']}/{plan['physical_cores']} physical-core workers"
                )
                result_by_year.update(self._parallel_profiles(pending, plan, worker_base))
            else:
                for year, _training, _identity, _surface_run in pending:
                    result_by_year[year] = self._profile_one(year)

            results = [result_by_year[year] for year in years]
            available = [r for r in results if r.status is ConfounderProfilingStatus.AVAILABLE]
            summary = {
                "status": "COMPLETED" if len(available) == len(results) else "COMPLETED_WITH_EXCLUSIONS",
                "stage": "CA3_CROSS_FITTED_PROFILING",
                "candidate_id": self.candidate.candidate_id,
                "analysis_run_id": self.analysis_run_id,
                "reference_metric": "WMARM_CS",
                "profile_count": len(results),
                "available_profile_count": len(available),
                "held_out_years": list(years),
                "profiles": [r.to_dict() for r in results],
            }
            self.store.put("workflow", self.analysis_run_id, "result", summary)
            self.store.commit()
            self.persistence.set_run_state(
                self.analysis_run_id, ConfounderRunState.COMPLETED
            )
            return summary
        except BaseException as exc:
            self.persistence.set_run_state(
                self.analysis_run_id,
                ConfounderRunState.CANCELLED
                if isinstance(exc, (Cancelled, KeyboardInterrupt))
                else ConfounderRunState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

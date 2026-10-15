"""CM4-27: empirical null scaffold, percentile effects, and threshold/lag recovery.

Only a single synthetic daily-incidence file is live at a time. Every generation
has recorded seeds, percentile draws, support diagnostics and an SHA-256 hash.
Scaffold data are retained on disk; completed outcomes can be reconstructed.
"""
from __future__ import annotations
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import numpy as np
import psutil
from .randomness import rng_for_identity
from . import numerics as nm
from .engine import SurfaceRunner, cell_scope, finite_mean
from .optimization import Optimizer, admissible
from .storage import Store, mapped, atomic_json, sha256_file
from .performance import MIB, memory_budget


MODEL_THRESHOLD_RESPONSE = "LINEAR_RAMP"
DIAGNOSTIC_STEP_RESPONSE = "STEP"
SUPPORTED_THRESHOLD_RESPONSES = (MODEL_THRESHOLD_RESPONSE, DIAGNOSTIC_STEP_RESPONSE)


def threshold_response_from_percentiles(percentiles: np.ndarray, q: float, mode: str) -> np.ndarray:
    """Return the generating threshold response for calibration.

    STEP is the primary onset-threshold recovery calibration: the imposed effect
    is constant once the empirical hazard percentile reaches q. LINEAR_RAMP is
    retained as a secondary severity-gradient calibration in which the imposed
    effect increases with exposure extremeness above q.
    """
    values = np.asarray(percentiles, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    cutoff = float(q) / 100.0
    if mode == MODEL_THRESHOLD_RESPONSE:
        out[valid] = np.maximum((values[valid] - cutoff) / (1.0 - cutoff), 0.0)
    elif mode == DIAGNOSTIC_STEP_RESPONSE:
        out[valid] = (values[valid] >= cutoff).astype(float)
    else:
        raise ValueError(f"Unsupported calibration threshold response mode: {mode}")
    return out


def generating_pairs(c: dict) -> list[tuple[float,int]]:
    requested = c["calibration"].get("generating_threshold_lag_pairs")
    if requested:
        return list(dict.fromkeys((float(q),int(l)) for q,l in requested))
    qs,ls = c["thresholds"],[l for l in c["lags"] if l>0]
    qmid = qs[(len(qs)-1)//2]
    pairs = [(qmid,0)]
    def nearest_lower_tie(a):
        pos = .75*(len(a)-1)
        low = int(math.floor(pos))
        return a[low if pos-low<=.5 else low+1]
    if ls:
        pairs.extend([(qmid,ls[(len(ls)-1)//2]),(nearest_lower_tie(qs),nearest_lower_tie(ls))])
    return list(dict.fromkeys(pairs))


def class_sign(item: dict) -> int | None:
    if item["exact_null"]:
        return 0
    if item["lower_percentile"]>=.5:
        return 1
    if item["upper_percentile"]<=.5:
        return -1
    return None


def distribution(values) -> dict:
    a = np.array([v for v in values if v is not None and math.isfinite(v)],dtype=float)
    return {"valid_count":len(a),"median":nm.finite_quantile(a,.5),"q25":nm.finite_quantile(a,.25),
            "q75":nm.finite_quantile(a,.75),"q025":nm.finite_quantile(a,.025),"q975":nm.finite_quantile(a,.975),
            "mean":finite_mean(a),"sd":float(np.std(a,ddof=1)) if len(a)>=2 else None}


def empirical_donor_sample(observed: np.ndarray, labels: np.ndarray,
                           widths: np.ndarray, segment_starts: np.ndarray,
                           segment_stops: np.ndarray, rng) -> np.ndarray:
    """CM4.6: exact chronological donor draws via compact interval indices.

    Binary searches replace repeated per-date window scans. Self-exclusion,
    segment boundaries, ascending donor order and every randrange call are
    identical to the original local empirical regeneration.
    """
    n = len(observed)
    days = np.arange(n)
    donor_mask = np.isfinite(observed) & (labels == 0)
    donors = np.flatnonzero(donor_mask)
    lower = np.maximum(segment_starts, days - widths)
    upper = np.minimum(segment_stops, days + widths + 1)
    left = np.searchsorted(donors, lower, side="left")
    right = np.searchsorted(donors, upper, side="left")
    self_position = np.searchsorted(donors, days, side="left")
    exclude_self = donor_mask & (lower <= days) & (days < upper)
    counts = right - left - exclude_self
    valid = np.flatnonzero((widths >= 0) & (counts > 0))
    result = np.full(n, np.nan)
    draw = rng.randrange
    offsets = np.fromiter((draw(int(counts[day])) for day in valid),
                          dtype=np.intp, count=len(valid))
    positions = left[valid] + offsets
    positions += exclude_self[valid] & (positions >= self_position[valid])
    result[valid] = observed[donors[positions]]
    return result


_CAL_WORKER_CONTEXT: dict = {}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def calibration_parallel_plan(task_count: int) -> dict:
    """Choose safe realization-level calibration concurrency.

    Scientific RNG identities remain run-keyed, so task scheduling cannot alter
    draws.  The default plan uses physical cores but clamps concurrency with a
    conservative aggregate RSS budget.  Environment variables allow validation
    or deployment overrides without changing the scientific project identity.
    """
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    physical = max(1, int(physical))
    requested = _env_int("APHREH_ADSMAP_CALIBRATION_WORKERS", physical, 1, 64)
    total_cap_mib = _env_int("APHREH_ADSMAP_TOTAL_RSS_MB", 1792, 512, 1792)
    worker_guard_mib = _env_int("APHREH_ADSMAP_CAL_WORKER_RSS_MB", 256, 256, 768)
    parent_reserve_mib = _env_int("APHREH_ADSMAP_CAL_PARENT_RESERVE_MB", 256, 128, 768)
    system_reserve_mib = _env_int("APHREH_ADSMAP_SYSTEM_FREE_RESERVE_MB", 512, 128, 4096)
    by_app_memory = max(1, (total_cap_mib - parent_reserve_mib) // worker_guard_mib)
    available_mib = int(psutil.virtual_memory().available // MIB)
    by_system_memory = max(1, (max(0, available_mib - system_reserve_mib)) // worker_guard_mib)
    workers = max(1, min(int(task_count) if task_count else 1, physical, requested, by_app_memory, by_system_memory))
    return {
        "worker_count": workers,
        "requested_workers": requested,
        "physical_cores": physical,
        "task_count": int(task_count),
        "aggregate_rss_cap_mib": total_cap_mib,
        "worker_rss_guard_mib": worker_guard_mib,
        "parent_reserve_mib": parent_reserve_mib,
        "system_available_mib_at_start": available_mib,
        "system_free_reserve_mib": system_reserve_mib,
        "worker_cache_mib": 48,
        "worker_workspace_mib": 48,
        "worker_rss_headroom_mib": 64,
    }


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def synthetic_eligibility_path(outcome_path: Path) -> Path:
    """Sidecar BSA-year eligibility mask for percentage-based calibration."""
    outcome_path = Path(outcome_path)
    return outcome_path.with_name(outcome_path.stem + "_bsa_year_eligibility.npy")


def _cleanup_synthetic_files(outcome_path: Path) -> None:
    for path in (Path(outcome_path), synthetic_eligibility_path(outcome_path)):
        path.unlink(missing_ok=True)


def _calibration_worker_initializer(config: dict, canonical_root: str, project_root: str,
                                    scaffold_root: str, worker_base: str, parent_db: str,
                                    memory_settings: dict, progress_queue=None) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["APHREH_ADSMAP_CACHE_MB"] = str(memory_settings["cache_mib"])
    os.environ["APHREH_ADSMAP_WORKSPACE_MB"] = str(memory_settings["workspace_mib"])
    os.environ["APHREH_ADSMAP_RSS_GUARD_MB"] = str(memory_settings["rss_guard_mib"])
    os.environ["APHREH_ADSMAP_RSS_HEADROOM_MB"] = str(memory_settings["rss_headroom_mib"])
    from .data import Data
    _CAL_WORKER_CONTEXT.clear()
    _CAL_WORKER_CONTEXT.update({
        "config": config,
        "data": Data(Path(canonical_root)),
        "project_root": Path(project_root),
        "scaffold_root": Path(scaffold_root),
        "worker_base": Path(worker_base),
        "parent_db": Path(parent_db),
        "progress_queue": progress_queue,
        "active_run": None,
    })


def _worker_log(message: str) -> None:
    """Best-effort worker progress relay; computation never blocks on GUI logging."""
    ctx = _CAL_WORKER_CONTEXT
    progress_queue = ctx.get("progress_queue")
    if progress_queue is None:
        return
    payload = {
        "run": ctx.get("active_run"),
        "message": str(message),
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    try:
        progress_queue.put_nowait(payload)
    except (queue_module.Full, BrokenPipeError, EOFError, OSError):
        pass


def _calibration_worker(task: dict) -> dict:
    """Execute one independent calibration realization in an isolated Store."""
    from .engine import FixedParametersEngine
    ctx = _CAL_WORKER_CONTEXT
    run = task["run"]
    task_identity = f"{task['kind']}|{run}|{task.get('replicate','')}"
    token = hashlib.sha256(task_identity.encode("utf-8")).hexdigest()[:20]
    root = ctx["worker_base"] / token
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / "results.sqlite")
    started = time.perf_counter()
    message = None
    error = None
    try:
        ctx["active_run"] = run
        if task["kind"] == "NULL":
            # Reuse only the partial surface for this replicate, not every null
            # record belonging to the parent TL run. This keeps resume I/O small.
            store.import_run_from(
                ctx["parent_db"], f"{run}__NULL_{int(task['replicate']):05d}"
            )
        else:
            store.import_run_from(ctx["parent_db"], run)
            for resume_run in task.get("resume_runs", []):
                store.import_run_from(ctx["parent_db"], resume_run)
        engine = FixedParametersEngine(
            ctx["data"], ctx["config"], store, root, log=_worker_log,
            cancellation_root=ctx["project_root"],
        )
        runner = SurfaceRunner(engine)
        if task["kind"] == "NULL":
            from .optimization import run_null_replicate
            replicate = int(task["replicate"])
            prior = run_null_replicate(
                runner, run, replicate, outcome_path=Path(task["outcome_path"]),
                outcome_semantics=str(task.get("outcome_semantics", "DAILY_INCIDENCE")),
                bsa_year_eligibility_path=Path(task["bsa_year_eligibility_path"])
                if task.get("bsa_year_eligibility_path") else None,
            )
            message = f"{run}: null replicate {replicate+1}/{ctx['config']['optimization']['null_realizations_R']} completed"
        else:
            calibration = Calibration(runner, Optimizer(runner), scaffold_root=ctx["scaffold_root"])
            if task["kind"] == "RC":
                message = calibration._run_rc_instance(task["item"], int(task["replicate"]))
            elif task["kind"] == "TL":
                message = calibration._run_tl_instance(
                    task["item"], float(task["q"]), int(task["lag"]),
                    str(task["response_mode"]), int(task["replicate"]),
                )
            else:
                raise ValueError(f"Unknown calibration task kind: {task['kind']}")
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
        ctx["active_run"] = None
        store.close()
    perf = memory_budget().snapshot()
    return {
        "ok": error is None,
        "run": run,
        "kind": task["kind"],
        "message": message,
        "error": error,
        "worker_root": str(root),
        "database": str(root / "results.sqlite"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_mib": perf.get("peak_process_rss_mib"),
        "performance": perf,
    }


def _process_tree_rss_bytes() -> int:
    process = psutil.Process()
    total = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class Calibration:
    def __init__(self, runner: SurfaceRunner, optimizer: Optimizer, *, scaffold_root: Path | None = None):
        self.runner,self.optimizer = runner,optimizer
        self.e = runner.engine
        self.c,self.d,self.s = self.e.c,self.e.data,self.e.store
        self.root = Path(scaffold_root) if scaffold_root is not None else self.e.root/"calibration_scaffolds"
        self.root.mkdir(parents=True,exist_ok=True)
        qs = self.c["thresholds"]
        self.qcal = self.c["calibration"].get("calibration_threshold") or qs[(len(qs)-1)//2]
        cal = self.c["calibration"]
        self.threshold_response_modes = list(cal["threshold_lag_response_modes"])
        self.primary_threshold_response_mode = cal["primary_threshold_lag_response_mode"]

    def scaffold(self,q: float) -> Path:
        root = self.root/f"q{q:g}"
        if (root/"complete.json").exists():
            return root
        root.mkdir(parents=True,exist_ok=True)
        labels,_ = nm.exposure(self.d.series("territory"),self.d.year_index,q)
        with mapped(root/"baseline.npy",shape=(self.d.B,self.d.n)) as base, mapped(root/"half_width.npy",shape=(self.d.B,self.d.n),dtype="int32") as width, mapped(root/"donors.npy",shape=(self.d.B,self.d.n),dtype="int32") as donors:
            for b in range(self.d.B):
                observed = nm.cumulative_incidence(self.d.series("outcome",b),self.d.daily_population(b),0,
                                                   self.c["core"]["incidence"]["scaling_multiplier"],"EVENT_COUNTS",
                                                   self.d.segment_index)
                bv,wv,nv = nm.baseline(observed,labels,self.c["core"]["baseline"],self.d.segment_index,self.d.weekday_index,self.d.holiday_state)
                base[b],width[b],donors[b] = bv,wv,nv
        atomic_json(root/"complete.json",{"q":q,"lag":0,"rule":"CM4 local empirical non-exposed zero-lag incidence donors",
                                          "bsa_ids":self.d.ids,"dates_ref":"../canonical/metadata.json"})
        return root

    def response(self,q: float,lag: int,response_mode: str) -> tuple[np.ndarray,np.ndarray]:
        """CM13-15: right-continuous empirical hazard CDF and normalized past lags."""
        hazard = self.d.series("territory")
        p = np.full(self.d.n,np.nan)
        for iy in range(len(self.d.years)):
            ix = np.flatnonzero((self.d.year_index==iy)&np.isfinite(hazard))
            if len(ix):
                values = np.sort(hazard[ix])
                p[ix] = np.searchsorted(values,hazard[ix],side="right")/len(values)
        threshold = threshold_response_from_percentiles(p, q, response_mode)
        weights = 1-np.arange(lag+1)/(lag+1)
        weights = weights/weights.sum()
        response = np.zeros(self.d.n)
        valid = np.ones(self.d.n,dtype=bool)
        for k,w in enumerate(weights):
            shifted = nm.shift_within_segments(threshold,k,self.d.segment_index)
            valid &= np.isfinite(shifted)
            response += w*np.where(np.isfinite(shifted),shifted,0.0)
        response[~valid] = np.nan
        return threshold,response

    def _response_paths(self, q: float, lag: int, response_mode: str) -> tuple[Path, Path]:
        scaffold = self.scaffold(q)
        tag = response_mode.lower()
        return (scaffold/f"threshold_response_{tag}_lag{lag}.npy",
                scaffold/f"temporal_response_{tag}_lag{lag}.npy")

    def prepare_response(self, q: float, lag: int, response_mode: str) -> tuple[Path, Path]:
        """Prepare immutable CM13-15 response arrays once for all worker processes."""
        if response_mode not in SUPPORTED_THRESHOLD_RESPONSES:
            raise ValueError(f"Unsupported calibration threshold response mode: {response_mode}")
        threshold_path,response_path = self._response_paths(q,lag,response_mode)
        if threshold_path.is_file() and response_path.is_file():
            return threshold_path,response_path
        threshold,response = self.response(q,lag,response_mode)
        if not threshold_path.is_file():
            _atomic_save_npy(threshold_path,threshold)
        if not response_path.is_file():
            _atomic_save_npy(response_path,response)
        return threshold_path,response_path

    def generate(self,run: str,item: dict,q: float,lag: int | None, response_mode: str | None = None, rng_identity: str | None = None) -> Path:
        """CM4.6,7.4,16.3: generate DAILY_INCIDENCE, never synthetic integer counts.

        CM5.2 exclusions are persisted in a compact BSA-year eligibility sidecar.
        The synthetic outcome itself keeps NaN for excluded BSA-years, ensuring
        they remain locally ineligible. The core engine uses the sidecar only to
        omit those structural exclusions from the whole-territory dependence sum.
        """
        if lag is not None:
            if response_mode is None:
                response_mode = self.primary_threshold_response_mode
            if response_mode not in SUPPORTED_THRESHOLD_RESPONSES:
                raise ValueError(f"Unsupported calibration threshold response mode: {response_mode}")
        rng_identity = rng_identity or run
        scaffold = self.scaffold(q)
        labels,_ = nm.exposure(self.d.series("territory"),self.d.year_index,q)
        rngp = rng_for_identity(self.c["seed"],namespace="DISK_DEV_CAL_PERCENTILE_V1",identity={"run":rng_identity})
        p = None if item["exact_null"] else rngp.uniform(item["lower_percentile"],item["upper_percentile"])
        if lag is None:
            factor = np.where(labels>=0,(labels==1).astype(float),np.nan)
        else:
            _,response_path = self.prepare_response(q,lag,response_mode)
            factor = np.load(response_path,allow_pickle=False)
        destination = self.e.scratch/"active_synthetic.npy"
        eligibility_path = synthetic_eligibility_path(destination)
        excluded_bsa_years = 0
        with mapped(scaffold/"baseline.npy") as base, mapped(scaffold/"half_width.npy") as widths, \
             mapped(destination,shape=(self.d.B,self.d.n)) as outcome, \
             mapped(eligibility_path,shape=(self.d.B,len(self.d.years)),dtype="bool") as eligibility:
            eligibility[:] = True
            for b,bid in enumerate(self.d.ids):
                self.e.check()
                observed = nm.cumulative_incidence(self.d.series("outcome",b),self.d.daily_population(b),0,
                                                   self.c["core"]["incidence"]["scaling_multiplier"],"EVENT_COUNTS",
                                                   self.d.segment_index)
                rng = rng_for_identity(self.c["seed"],namespace="DISK_DEV_CAL_DONORS_V1",identity={"run":rng_identity,"bsa_id":bid})
                synthetic = empirical_donor_sample(
                    observed, labels, np.asarray(widths[b]),
                    self.d.segment_starts, self.d.segment_stops, rng,
                )
                for iy,year in enumerate(self.d.years):
                    ix = self.d.year_index==iy
                    reference = observed[ix&np.isfinite(observed)]
                    median = nm.finite_quantile(reference,.5)
                    reason,resolution = None,None
                    if item["exact_null"]:
                        delta = 0.0
                    elif not np.isfinite(median) or median<=0:
                        delta = None
                        reason = "ZERO_OR_UNAVAILABLE_MEDIAN_REFERENCE"
                        eligibility[b,iy] = False
                        excluded_bsa_years += 1
                        synthetic[ix] = np.nan
                    else:
                        delta = (nm.finite_quantile(reference,p)-median)/median
                        lower,upper = (nm.finite_quantile(reference,item[v]) for v in ("lower_percentile","upper_percentile"))
                        if lower==upper:
                            resolution = "INSUFFICIENT_PERCENTILE_RESOLUTION"
                    clipped = 0
                    if delta is not None:
                        candidate = synthetic[ix]+delta*np.asarray(base[b,ix])*factor[ix]
                        clipped = int(np.sum(candidate<0))
                        synthetic[ix] = np.maximum(candidate,0.0)
                    valid = synthetic[ix][np.isfinite(synthetic[ix])]
                    self.s.put("synthetic_support",run,f"{bid}/{year}",
                               {"run":run,"class":item["name"],"q_generating":q,"lag_generating":lag,
                                "bsa_id":bid,"year":year,"percentile_draw":p,"median_reference":median,
                                "percentage_alteration":delta,"reason":reason,"resolution_flag":resolution,
                                "calibration_bsa_year_eligible":bool(eligibility[b,iy]),
                                "valid_days":len(valid),"missing_days":int(np.sum(ix))-len(valid),
                                "truncated_negative_days":clipped,"synthetic_mean":finite_mean(valid),
                                "synthetic_min":float(np.min(valid)) if len(valid) else None,
                                "synthetic_max":float(np.max(valid)) if len(valid) else None})
                outcome[b] = synthetic
        total_bsa_years = self.d.B*len(self.d.years)
        self.s.put("synthetic_generation",run,"metadata",{"run":run,"class":item["name"],"class_specification":item,
                    "q_generating":q,"lag_generating":lag,"percentile_draw":p,"seed":self.c["seed"],
                    "paired_rng_identity":rng_identity,
                    "threshold_lag_response_mode":response_mode if lag is not None else "NOT_APPLICABLE",
                    "threshold_response_rule":(
                        "NOT_APPLICABLE_RC_MAGNITUDE_CALIBRATION" if lag is None else
                        ("CM14_LINEAR_ABOVE_Q0_NORMALIZED_PERCENTILE_V1" if response_mode == MODEL_THRESHOLD_RESPONSE
                         else "STEP_AT_OR_ABOVE_Q0_V1")),
                    "rng_family":"PYTHON_RANDOM_MT19937_V1","subseed_derivation":"APHREH_SHA256_SUBSEED_V1",
                    "namespace_donors":"DISK_DEV_CAL_DONORS_V1","namespace_percentile":"DISK_DEV_CAL_PERCENTILE_V1",
                    "outcome_semantics":"DAILY_INCIDENCE","synthetic_file_sha256":sha256_file(destination),
                    "bsa_year_eligibility_file_sha256":sha256_file(eligibility_path),
                    "bsa_year_eligibility_rule":"CM5.2_ZERO_OR_UNAVAILABLE_MEDIAN_EXCLUDED_FROM_LOCAL_ESTIMATION_AND_TERRITORY_SUM",
                    "eligible_bsa_years":total_bsa_years-excluded_bsa_years,
                    "excluded_bsa_years":excluded_bsa_years,"total_bsa_years":total_bsa_years,
                    "retention":"seed-reconstructible; active numerical file and eligibility sidecar released after this realization"})
        self.s.commit()
        return destination

    def _run_rc_instance(self, item: dict, m: int) -> str:
        c = self.c["calibration"]
        run = f"CAL_RC_{item['name']}_{m:04d}"
        if self.s.get("calibration_instance","RC",run) is not None:
            return f"{run}: already completed"
        path = self.generate(run,item,self.qcal,None)
        try:
            eligibility_path = synthetic_eligibility_path(path)
            cell = self.e.run(
                run,self.qcal,0,outcome_path=path,outcome_semantics="DAILY_INCIDENCE",loyo=False,
                bsa_year_eligibility_path=eligibility_path
            )
            if item["exact_null"] and c["run_exact_null_surface_check"]:
                fullrun = f"CAL_EXACT_NULL_SURFACE_{m:04d}"
                cells = self.runner.run(
                    fullrun,outcome_path=path,outcome_semantics="DAILY_INCIDENCE",
                    bsa_year_eligibility_path=eligibility_path
                )
                self.optimizer.run(
                    fullrun,cells,outcome_path=path,outcome_semantics="DAILY_INCIDENCE",
                    bsa_year_eligibility_path=eligibility_path
                )
            self.s.put("calibration_instance","RC",run,{"run":run,"kind":"RC","class":item["name"],"replicate":m,
                       "q_generating":self.qcal,"lag_generating":0,
                       "wmarm_cs":cell["wmarm_cs"],"wmarm":cell["wmarm"],
                       "confidence_support_retention_ratio":cell["confidence_support_retention_ratio"],
                       "directional_balance_cs":cell["directional_balance_cs"],
                       "directional_balance":cell["directional_balance"],
                       "ci_share":cell["ci_excluding_zero_population_share"],"valid":cell["valid"]})
            self.s.commit()
            return f"{run}: completed"
        finally:
            _cleanup_synthetic_files(path)

    def _run_observed_tl_surface(self, run: str, path: Path, eligibility_path: Path) -> list[dict]:
        """Run the observed TL surface with bounded, non-scientific progress reporting."""
        total = len(self.c["thresholds"]) * len(self.c["lags"])
        cells: list[dict] = []
        self.e.log(f"{run}: observed surface started ({total} cells)")
        report_every = max(1, total // 8)
        completed = 0
        for candidate_q in self.c["thresholds"]:
            for candidate_lag in self.c["lags"]:
                cell = self.e.run(
                    run, candidate_q, candidate_lag, outcome_path=path,
                    outcome_semantics="DAILY_INCIDENCE",
                    bsa_year_eligibility_path=eligibility_path,
                )
                cells.append(cell)
                completed += 1
                if completed == 1 or completed % report_every == 0 or completed == total:
                    self.e.log(f"{run}: observed surface {completed}/{total} cells complete")
        return cells

    def _persist_tl_instance(self, run: str, item: dict, q: float, lag: int,
                             response_mode: str, m: int, opt: dict) -> str:
        region = opt["robust_region"]
        representative = opt["representative"]
        surface_valid = bool(opt.get("surface_valid", True))
        admissible_cells = int(opt.get("admissible_observed_cells") or 0)
        loyo_complete_cells = int(opt.get("loyo_complete_admissible_cells") or 0)
        null_complete = bool(opt.get("null_complete"))
        valid_for_recovery = bool(
            surface_valid and admissible_cells>0 and null_complete and loyo_complete_cells>0
        )
        failure_reason = opt.get("failure_reason")
        if failure_reason is None and not valid_for_recovery:
            if not surface_valid:
                failure_reason = "NO_VALID_SYNTHETIC_REFERENCE_SURFACE"
            elif admissible_cells<=0:
                failure_reason = "NO_ADMISSIBLE_SYNTHETIC_REFERENCE_SURFACE"
            elif not null_complete:
                failure_reason = "INCOMPLETE_SYNTHETIC_OPTIMIZATION_NULL"
            elif loyo_complete_cells<=0:
                failure_reason = "NO_LOYO_COMPLETE_ADMISSIBLE_SYNTHETIC_CELL"
        self.s.put("calibration_instance","TL",run,{"run":run,"kind":"TL","class":item["name"],"replicate":m,
                   "threshold_lag_response_mode":response_mode,
                   "response_role":"PRIMARY_ONSET_THRESHOLD_RECOVERY" if response_mode==self.primary_threshold_response_mode else "SECONDARY_SEVERITY_GRADIENT",
                   "q_generating":q,"lag_generating":lag,"region_nonempty":bool(region),
                   "surface_valid":surface_valid,
                   "admissible_observed_cells":admissible_cells,
                   "loyo_complete_admissible_cells":loyo_complete_cells,
                   "calibration_valid_for_recovery":valid_for_recovery,
                   # Backward-compatible alias: from 2.3.21 this means valid for
                   # threshold/lag recovery, not merely numerically computable.
                   "calibration_valid":valid_for_recovery,
                   "calibration_failure_reason":failure_reason,
                   "optimization_null_complete":null_complete,"p_max":opt["p_max"],
                   "threshold_recovered":any(v["q"]==q for v in region),
                   "lag_recovered":any(v["lag"]==lag for v in region),
                   "joint_recovered":any(v["q"]==q and v["lag"]==lag for v in region),
                   "representative":representative,
                   "representative_q_displacement":representative["q"]-q if representative else None,
                   "representative_lag_displacement":representative["lag"]-lag if representative else None})
        self.s.commit()
        validity = "valid for recovery" if valid_for_recovery else f"invalid for recovery ({failure_reason})"
        return f"{run}: completed; {validity}; robust cells={len(region)}"

    @staticmethod
    def _surface_has_reference_support(cells: list[dict]) -> bool:
        return any(
            bool(cell.get("valid")) and cell.get("wmarm_cs") is not None
            and math.isfinite(float(cell["wmarm_cs"]))
            for cell in cells
        )

    def _surface_admissible_count(self, cells: list[dict]) -> int:
        return sum(admissible(cell,self.c["optimization"]) for cell in cells)

    @staticmethod
    def _invalid_tl_optimization(reason: str, *, surface_valid: bool, admissible_cells: int = 0) -> dict:
        return {
            "robust_region": [], "representative": None, "null_complete": False,
            "p_max": None, "surface_valid": bool(surface_valid),
            "admissible_observed_cells": int(admissible_cells),
            "loyo_complete_admissible_cells": 0,
            "calibration_valid_for_recovery": False,
            "calibration_valid": False, "failure_reason": reason,
        }

    def _run_tl_instance(self, item: dict, q: float, lag: int, response_mode: str, m: int) -> str:
        paired_rng_identity = f"CAL_TL_q{q:g}_lag{lag}_{m:04d}"
        run = f"CAL_TL_{response_mode}_q{q:g}_lag{lag}_{m:04d}"
        if self.s.get("calibration_instance","TL",run) is not None:
            return f"{run}: already completed"
        self.e.log(f"{run}: synthetic outcome preparation started")
        path = self.generate(run,item,q,lag,response_mode=response_mode,rng_identity=paired_rng_identity)
        eligibility_path = synthetic_eligibility_path(path)
        try:
            cells = self._run_observed_tl_surface(run,path,eligibility_path)
            surface_valid = self._surface_has_reference_support(cells)
            admissible_cells = self._surface_admissible_count(cells)
            if not surface_valid:
                reason = "NO_VALID_SYNTHETIC_REFERENCE_SURFACE"
                self.e.log(f"{run}: fail-fast; {reason}; optimization null search skipped")
                return self._persist_tl_instance(
                    run,item,q,lag,response_mode,m,
                    self._invalid_tl_optimization(reason,surface_valid=False)
                )
            if admissible_cells<=0:
                reason = "NO_ADMISSIBLE_SYNTHETIC_REFERENCE_SURFACE"
                self.e.log(f"{run}: fail-fast; {reason}; optimization null search skipped")
                return self._persist_tl_instance(
                    run,item,q,lag,response_mode,m,
                    self._invalid_tl_optimization(reason,surface_valid=True,admissible_cells=0)
                )
            self.e.log(
                f"{run}: {admissible_cells}/{len(cells)} observed cells admissible under "
                "calibration-conditional support; null search started "
                f"({self.c['optimization']['null_realizations_R']} realizations)"
            )
            opt = self.optimizer.run(
                run,cells,outcome_path=path,outcome_semantics="DAILY_INCIDENCE",
                bsa_year_eligibility_path=eligibility_path
            )
            return self._persist_tl_instance(run,item,q,lag,response_mode,m,opt)
        finally:
            _cleanup_synthetic_files(path)

    def _build_tasks(self, count: int, pairs: list[tuple[float,int]], modes: list[str]) -> list[dict]:
        c = self.c["calibration"]
        tasks: list[dict] = []
        for item in c["effect_classes"]:
            for m in range(count):
                run = f"CAL_RC_{item['name']}_{m:04d}"
                if self.s.get("calibration_instance","RC",run) is not None:
                    continue
                resume_runs = []
                if item["exact_null"] and c["run_exact_null_surface_check"]:
                    resume_runs.append(f"CAL_EXACT_NULL_SURFACE_{m:04d}")
                tasks.append({"kind":"RC","run":run,"item":item,"replicate":m,"resume_runs":resume_runs})
        amp = next(v for v in c["effect_classes"] if v["name"]==c["threshold_lag_amplitude_class"])
        for response_mode in modes:
            for q,lag in pairs:
                for m in range(count):
                    run = f"CAL_TL_{response_mode}_q{q:g}_lag{lag}_{m:04d}"
                    if self.s.get("calibration_instance","TL",run) is not None:
                        continue
                    tasks.append({"kind":"TL","run":run,"item":amp,"q":q,"lag":lag,
                                  "response_mode":response_mode,"replicate":m,"resume_runs":[]})
        return tasks

    def _prepare_parallel_inputs(self, pairs: list[tuple[float,int]], modes: list[str]) -> None:
        for q in dict.fromkeys([self.qcal,*[pair[0] for pair in pairs]]):
            self.scaffold(float(q))
        for response_mode in modes:
            for q,lag in pairs:
                self.prepare_response(q,lag,response_mode)
        # Parent cache is not useful while realization workers are active and
        # would consume part of the aggregate RSS allowance.
        memory_budget().clear()

    def _worker_base(self) -> Path:
        override = os.environ.get("APHREH_ADSMAP_WORKER_TEMP_DIR")
        base = Path(override).expanduser().resolve() if override else Path(tempfile.gettempdir())
        project_token = hashlib.sha256(str(self.e.root.resolve()).encode("utf-8")).hexdigest()[:12]
        identity = "unknown"
        manifest_path = self.e.root/"manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = str(manifest.get("reference_identity") or manifest.get("identity") or "unknown")[:16]
        except Exception:
            pass
        base = base/"aphreh_adsmap_calibration_workers"/f"{project_token}_{identity}"
        base.mkdir(parents=True,exist_ok=True)
        minimum = self.c["runtime"]["minimum_free_disk_mb"]*MIB
        if shutil.disk_usage(base).free < minimum:
            fallback = self.e.root/"scratch"/"calibration_workers"/identity
            fallback.mkdir(parents=True,exist_ok=True)
            base = fallback
        return base

    def _recover_worker_checkpoints(self, worker_base: Path) -> int:
        """Merge committed records left by workers after an interrupted parent run."""
        recovered = 0
        if not worker_base.is_dir():
            return recovered
        for worker_root in sorted(worker_base.iterdir()):
            database = worker_root/"results.sqlite"
            if not worker_root.is_dir() or not database.is_file():
                continue
            try:
                self._merge_worker_artifacts(worker_root)
                self.s.merge_from(database)
                self.s.commit()
                recovered += 1
                shutil.rmtree(worker_root,ignore_errors=True)
            except Exception as exc:
                self.e.log(f"Calibration checkpoint recovery skipped {worker_root.name}: {exc}")
        if recovered:
            self.e.log(f"Calibration checkpoint recovery: merged {recovered} isolated worker stores")
        return recovered

    def _merge_worker_artifacts(self, worker_root: Path) -> None:
        for name in ("intermediates","bootstrap"):
            source = worker_root/name
            if not source.is_dir():
                continue
            target_root = self.e.root/name
            target_root.mkdir(parents=True,exist_ok=True)
            for child in source.iterdir():
                target = target_root/child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child),str(target))

    def _merge_worker_result(self, result: dict) -> None:
        worker_root = Path(result["worker_root"])
        try:
            self._merge_worker_artifacts(worker_root)
            self.s.merge_from(Path(result["database"]))
            self.s.commit()
        finally:
            if worker_root.exists():
                shutil.rmtree(worker_root,ignore_errors=True)

    @staticmethod
    def _parallel_conflict(tasks: list[dict]) -> bool:
        owners: set[str] = set()
        for task in tasks:
            for name in [task["run"],*task.get("resume_runs",[])]:
                if name in owners:
                    return True
                owners.add(name)
        return False

    def _execute_serial_tasks(self, tasks: list[dict]) -> None:
        for task in tasks:
            if task["kind"] == "RC":
                message = self._run_rc_instance(task["item"],int(task["replicate"]))
            else:
                message = self._run_tl_instance(task["item"],float(task["q"]),int(task["lag"]),
                                                str(task["response_mode"]),int(task["replicate"]))
            self.e.log(message)

    def _drain_worker_progress(self, progress_queue) -> int:
        drained = 0
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue_module.Empty:
                break
            except (BrokenPipeError, EOFError, OSError):
                break
            message = event.get("message") if isinstance(event,dict) else str(event)
            if message:
                self.e.log(message)
                drained += 1
        return drained

    @staticmethod
    def _pool_settings(plan: dict) -> dict:
        return {
            "cache_mib":plan["worker_cache_mib"],
            "workspace_mib":plan["worker_workspace_mib"],
            "rss_guard_mib":plan["worker_rss_guard_mib"],
            "rss_headroom_mib":plan["worker_rss_headroom_mib"],
        }

    def _new_pool(self, plan: dict, worker_base: Path):
        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue(maxsize=4096)
        executor = ProcessPoolExecutor(
            max_workers=int(plan["worker_count"]),
            mp_context=ctx,
            initializer=_calibration_worker_initializer,
            initargs=(
                self.c,str(self.d.root),str(self.e.root),str(self.root),str(worker_base),
                str(self.s.path),self._pool_settings(plan),progress_queue,
            ),
        )
        return executor,progress_queue

    @staticmethod
    def _terminate_pool(executor: ProcessPoolExecutor) -> None:
        for process in list(getattr(executor,"_processes",{}).values()):
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in list(getattr(executor,"_processes",{}).values()):
            try:
                process.join(timeout=2)
            except Exception:
                pass

    def _worker_failure(self, task: dict, result: dict) -> BaseException | None:
        if result.get("ok"):
            return None
        error = result.get("error") or {}
        etype = error.get("type")
        message = error.get("message") or "unknown worker failure"
        if etype == "Cancelled":
            from .engine import Cancelled
            return Cancelled(message)
        if etype == "MemoryError":
            return MemoryError(message)
        return RuntimeError(
            f"Calibration worker failed for {task['run']}: {etype}: {message}\n"
            f"{error.get('traceback','')}"
        )

    def _execute_parallel_tasks(self, tasks: list[dict], plan: dict, *, phase: str) -> dict:
        """Execute independent complete realizations in the shared worker pool."""
        if not tasks:
            return {**plan,"phase":phase,"completed_parallel_tasks":0}
        worker_base = self._worker_base()
        workers = int(plan["worker_count"])
        cap_bytes = int(plan["aggregate_rss_cap_mib"])*MIB
        phase_label = "RC" if phase == "RC_REALIZATIONS" else "threshold/lag"
        self.e.log(
            f"Calibration {phase_label} parallel runtime: workers={workers}/{plan['physical_cores']} physical cores; "
            f"aggregate RSS cap={plan['aggregate_rss_cap_mib']} MiB; worker guard={plan['worker_rss_guard_mib']} MiB"
        )
        executor,progress_queue = self._new_pool(plan,worker_base)
        futures = {executor.submit(_calibration_worker,task):task for task in tasks}
        pending = set(futures)
        peak_aggregate = _process_tree_rss_bytes()
        worker_peaks: list[float] = []
        completed = 0
        failure: BaseException | None = None
        last_heartbeat = time.monotonic()
        try:
            while pending and failure is None:
                self._drain_worker_progress(progress_queue)
                done,pending = wait(pending,timeout=0.10,return_when=FIRST_COMPLETED)
                aggregate = _process_tree_rss_bytes()
                peak_aggregate = max(peak_aggregate,aggregate)
                if aggregate >= cap_bytes:
                    failure = MemoryError(
                        f"Calibration process tree reached {aggregate/MIB:.1f} MiB, above the "
                        f"{plan['aggregate_rss_cap_mib']} MiB safety cap. Completed checkpoints are preserved."
                    )
                    break
                now = time.monotonic()
                if now-last_heartbeat >= 30 and pending:
                    self.e.log(
                        f"Calibration {phase_label} workers active: {completed}/{len(tasks)} realizations complete; "
                        f"{len(pending)} pending; process-tree RSS={aggregate/MIB:.1f} MiB"
                    )
                    last_heartbeat = now
                for future in done:
                    task = futures[future]
                    try:
                        result = future.result()
                    except BaseException as exc:
                        failure = RuntimeError(f"Calibration worker process failed for {task['run']}: {exc}")
                        break
                    self._merge_worker_result(result)
                    peak = result.get("peak_process_rss_mib")
                    if peak is not None:
                        worker_peaks.append(float(peak))
                    worker_error = self._worker_failure(task,result)
                    if worker_error is not None:
                        failure = worker_error
                        break
                    completed += 1
                    self.e.log(result.get("message") or f"{task['run']}: completed")
            self._drain_worker_progress(progress_queue)
            if failure is not None:
                for future in pending:
                    future.cancel()
                self._terminate_pool(executor)
                raise failure
        finally:
            executor.shutdown(wait=failure is None,cancel_futures=failure is not None)
            try:
                progress_queue.close(); progress_queue.join_thread()
            except Exception:
                pass
            try:
                if worker_base.is_dir() and not any(worker_base.iterdir()):
                    worker_base.rmdir()
            except OSError:
                pass
        return {
            **plan,"phase":phase,"completed_parallel_tasks":completed,
            "peak_observed_process_tree_rss_mib":peak_aggregate/MIB,
            "maximum_observed_worker_rss_mib":max(worker_peaks) if worker_peaks else None,
            "scratch_root":str(worker_base),
        }

    def _run_tl_parallel_nulls(self, task: dict, executor: ProcessPoolExecutor, progress_queue,
                               plan: dict, stats: dict) -> None:
        """Run one TL realization, distributing its independent null surfaces."""
        item = task["item"]
        q = float(task["q"]); lag = int(task["lag"])
        response_mode = str(task["response_mode"]); m = int(task["replicate"])
        run = task["run"]
        if self.s.get("calibration_instance","TL",run) is not None:
            self.e.log(f"{run}: already completed")
            return
        paired_rng_identity = f"CAL_TL_q{q:g}_lag{lag}_{m:04d}"
        self.e.log(f"{run}: synthetic outcome preparation started")
        path = self.generate(run,item,q,lag,response_mode=response_mode,rng_identity=paired_rng_identity)
        eligibility_path = synthetic_eligibility_path(path)
        try:
            cells = self._run_observed_tl_surface(run,path,eligibility_path)
            surface_valid = self._surface_has_reference_support(cells)
            admissible_cells = self._surface_admissible_count(cells)
            if not surface_valid:
                reason = "NO_VALID_SYNTHETIC_REFERENCE_SURFACE"
                self.e.log(f"{run}: fail-fast; {reason}; optimization null search skipped")
                self.e.log(self._persist_tl_instance(
                    run,item,q,lag,response_mode,m,
                    self._invalid_tl_optimization(reason,surface_valid=False)
                ))
                return
            if admissible_cells<=0:
                reason = "NO_ADMISSIBLE_SYNTHETIC_REFERENCE_SURFACE"
                self.e.log(f"{run}: fail-fast; {reason}; optimization null search skipped")
                self.e.log(self._persist_tl_instance(
                    run,item,q,lag,response_mode,m,
                    self._invalid_tl_optimization(reason,surface_valid=True,admissible_cells=0)
                ))
                return
            self.e.log(
                f"{run}: {admissible_cells}/{len(cells)} observed cells admissible under "
                "calibration-conditional support"
            )
            R = int(self.c["optimization"]["null_realizations_R"])
            missing = [r for r in range(R) if self.s.get("null_maximum",run,f"{r:05d}") is None]
            completed_count = R-len(missing)
            if missing:
                self.e.log(
                    f"{run}: parallel null search started; {completed_count}/{R} reusable, "
                    f"{len(missing)} pending across {plan['worker_count']} workers"
                )
                null_tasks = [
                    {"kind":"NULL","run":run,"replicate":r,"outcome_path":str(path),
                     "bsa_year_eligibility_path":str(eligibility_path),
                     "outcome_semantics":"DAILY_INCIDENCE","resume_runs":[]}
                    for r in missing
                ]
                futures = {executor.submit(_calibration_worker,null_task):null_task for null_task in null_tasks}
                pending = set(futures)
                last_heartbeat = time.monotonic()
                while pending:
                    self.e.check()
                    self._drain_worker_progress(progress_queue)
                    done,pending = wait(pending,timeout=0.10,return_when=FIRST_COMPLETED)
                    aggregate = _process_tree_rss_bytes()
                    stats["peak_observed_process_tree_rss_mib"] = max(
                        float(stats.get("peak_observed_process_tree_rss_mib",0.0)), aggregate/MIB
                    )
                    if aggregate >= int(plan["aggregate_rss_cap_mib"])*MIB:
                        for future in pending:
                            future.cancel()
                        raise MemoryError(
                            f"Calibration process tree reached {aggregate/MIB:.1f} MiB, above the "
                            f"{plan['aggregate_rss_cap_mib']} MiB safety cap. Completed null checkpoints are preserved."
                        )
                    now = time.monotonic()
                    if now-last_heartbeat >= 30 and pending:
                        self.e.log(
                            f"{run}: null search active; {completed_count}/{R} complete; "
                            f"{len(pending)} replicates pending; process-tree RSS={aggregate/MIB:.1f} MiB"
                        )
                        last_heartbeat = now
                    for future in done:
                        null_task = futures[future]
                        try:
                            result = future.result()
                        except BaseException as exc:
                            for other in pending:
                                other.cancel()
                            raise RuntimeError(
                                f"Calibration null worker process failed for {run}, replicate {null_task['replicate']}: {exc}"
                            ) from exc
                        self._merge_worker_result(result)
                        peak = result.get("peak_process_rss_mib")
                        if peak is not None:
                            stats["maximum_observed_worker_rss_mib"] = max(
                                float(stats.get("maximum_observed_worker_rss_mib") or 0.0), float(peak)
                            )
                        worker_error = self._worker_failure(null_task,result)
                        if worker_error is not None:
                            for other in pending:
                                other.cancel()
                            raise worker_error
                        stats["completed_parallel_tasks"] = int(stats.get("completed_parallel_tasks",0))+1
                        completed_count += 1
                        if completed_count == 1 or completed_count % 5 == 0 or completed_count == R:
                            self.e.log(
                                f"{run}: null search {completed_count}/{R} complete "
                                f"(parallel, {plan['worker_count']} workers)"
                            )
                self._drain_worker_progress(progress_queue)
            else:
                self.e.log(f"{run}: null search {R}/{R} already checkpointed")
            self.e.log(f"{run}: final optimization resolution started")
            opt = self.optimizer.run(
                run,cells,outcome_path=path,outcome_semantics="DAILY_INCIDENCE",
                bsa_year_eligibility_path=eligibility_path
            )
            self.e.log(self._persist_tl_instance(run,item,q,lag,response_mode,m,opt))
        finally:
            try:
                _cleanup_synthetic_files(path)
            except PermissionError:
                # A failing Windows worker can still hold a read-only mapping
                # until the outer pool shutdown terminates it. The deterministic
                # synthetic file is safe to overwrite or remove on resume.
                self.e.log(f"{run}: synthetic cleanup deferred until worker shutdown")

    def _execute_parallel_tl_tasks(self, tasks: list[dict], plan: dict) -> dict:
        """Process TL realizations sequentially while parallelizing each null search.

        One synthetic outcome is live at a time, preserving the bounded-disk and
        bounded-memory design. The expensive null surfaces are independent and
        are distributed across physical-core workers with deterministic run and
        replicate RNG identities.
        """
        if not tasks:
            return {**plan,"phase":"TL_NULL_SURFACES","completed_parallel_tasks":0}
        worker_base = self._worker_base()
        self.e.log(
            f"Threshold/lag calibration runtime: one active synthetic realization at a time; "
            f"null surfaces distributed across {plan['worker_count']}/{plan['physical_cores']} physical-core workers"
        )
        executor,progress_queue = self._new_pool(plan,worker_base)
        failure: BaseException | None = None
        stats = {
            **plan,"phase":"TL_NULL_SURFACES","completed_parallel_tasks":0,
            "peak_observed_process_tree_rss_mib":_process_tree_rss_bytes()/MIB,
            "maximum_observed_worker_rss_mib":None,"scratch_root":str(worker_base),
            "tl_realizations_completed":0,
        }
        try:
            for index,task in enumerate(tasks,1):
                self.e.check()
                self.e.log(f"Threshold/lag calibration realization {index}/{len(tasks)}: {task['run']}")
                self._run_tl_parallel_nulls(task,executor,progress_queue,plan,stats)
                stats["tl_realizations_completed"] = index
        except BaseException as exc:
            failure = exc
            self._terminate_pool(executor)
            raise
        finally:
            executor.shutdown(wait=failure is None,cancel_futures=failure is not None)
            try:
                progress_queue.close(); progress_queue.join_thread()
            except Exception:
                pass
            try:
                if worker_base.is_dir() and not any(worker_base.iterdir()):
                    worker_base.rmdir()
            except OSError:
                pass
        return stats

    def run(self) -> dict:
        c = self.c["calibration"]
        count = c["number_of_realizations"]
        pairs = generating_pairs(self.c)
        modes = self.threshold_response_modes
        self.e.log(f"Calibration: {len(c['effect_classes'])} RC classes x {count} realizations at ({self.qcal:g},0)")
        self.e.log(
            f"Threshold/lag calibration: generating pairs {pairs}, {count} realizations each, "
            f"response modes={modes}; primary={self.primary_threshold_response_mode}"
        )
        worker_base = self._worker_base()
        self._recover_worker_checkpoints(worker_base)
        tasks = self._build_tasks(count,pairs,modes)
        if tasks:
            rc_tasks = [task for task in tasks if task["kind"] == "RC"]
            tl_tasks = [task for task in tasks if task["kind"] == "TL"]
            null_realizations = int(self.c["optimization"]["null_realizations_R"])
            pool_task_count = max(len(rc_tasks), null_realizations if tl_tasks else 0, 1)
            plan = calibration_parallel_plan(pool_task_count)
            bootstrap_iterations = int(self.c["core"]["bootstrap"]["iterations"])
            grid_cells = len(self.c["thresholds"])*len(self.c["lags"])
            work_units = sum(
                bootstrap_iterations if task["kind"] == "RC" else
                bootstrap_iterations*grid_cells*(1+null_realizations)
                for task in tasks
            )
            minimum_parallel_work = _env_int("APHREH_ADSMAP_PARALLEL_MIN_WORK_UNITS",1500,0,10**9)
            plan["estimated_work_units"] = work_units
            plan["parallel_minimum_work_units"] = minimum_parallel_work
            if work_units < minimum_parallel_work:
                plan["worker_count"] = 1
                plan["serial_reason"] = "workload below process-pool break-even threshold"
            if self._parallel_conflict(tasks):
                plan["worker_count"] = 1
                plan["serial_reason"] = "overlapping calibration run namespaces"
            if plan["worker_count"] > 1:
                self._prepare_parallel_inputs(pairs,modes)
                phase_stats = []
                if rc_tasks:
                    phase_stats.append(self._execute_parallel_tasks(rc_tasks,plan,phase="RC_REALIZATIONS"))
                if tl_tasks:
                    workers = int(plan["worker_count"])
                    # Use complete-realization parallelism whenever enough TL
                    # realizations exist to occupy all workers.  This minimizes
                    # process/task overhead on production calibrations.  Only a
                    # small tail that cannot keep at least half the workers busy
                    # is decomposed into independent null surfaces, which keeps
                    # the physical cores occupied without changing RNG identities.
                    bulk_count = (len(tl_tasks)//workers)*workers
                    bulk = tl_tasks[:bulk_count]
                    tail = tl_tasks[bulk_count:]
                    if bulk:
                        phase_stats.append(self._execute_parallel_tasks(
                            bulk,plan,phase="TL_REALIZATIONS"
                        ))
                    if tail:
                        if len(tail) <= max(1,workers//2):
                            phase_stats.append(self._execute_parallel_tl_tasks(tail,plan))
                        else:
                            phase_stats.append(self._execute_parallel_tasks(
                                tail,plan,phase="TL_REALIZATIONS_TAIL"
                            ))
                combined = dict(plan)
                combined.update({
                    "execution_strategy":"HYBRID_RC_AND_TL_REALIZATION_POOL_WITH_NULL_SURFACE_TAIL",
                    "completed_parallel_tasks":sum(int(v.get("completed_parallel_tasks",0)) for v in phase_stats),
                    "peak_observed_process_tree_rss_mib":max(
                        (float(v.get("peak_observed_process_tree_rss_mib",0.0)) for v in phase_stats),default=0.0
                    ),
                    "maximum_observed_worker_rss_mib":max(
                        (float(v.get("maximum_observed_worker_rss_mib") or 0.0) for v in phase_stats),default=0.0
                    ) or None,
                    "phases":phase_stats,
                })
                memory_budget().record_parallel_calibration(combined)
            else:
                reason = plan.get("serial_reason")
                suffix = f" ({reason})" if reason else ""
                self.e.log(f"Calibration runtime: one worker selected by CPU/memory safety plan{suffix}")
                self._execute_serial_tasks(tasks)
                memory_budget().record_parallel_calibration({
                    **plan,"execution_strategy":"SERIAL","completed_parallel_tasks":0
                })
        else:
            memory_budget().record_parallel_calibration({
                **calibration_parallel_plan(0),"worker_count":0,
                "execution_strategy":"NO_PENDING_CALIBRATION_TASKS","completed_parallel_tasks":0
            })
        self.summarize()
        result = {"status":"COMPLETED","mode":c["mode"],"realizations_per_scenario":count,
                  "rc_class_count":len(c["effect_classes"]),"rc_realizations":len(c["effect_classes"])*count,
                  "generating_pairs":[list(p) for p in pairs],
                  "threshold_lag_response_modes":modes,
                  "primary_threshold_lag_response_mode":self.primary_threshold_response_mode,
                  "tl_realizations_per_response_mode":len(pairs)*count,
                  "tl_realizations":len(modes)*len(pairs)*count,
                  "calibration_threshold":self.qcal,"reference_territory_metric":"WMARM_CS",
                  "diagnostic_territory_metric":"WMARM",
                  "interpretation":"WMARM_CS is the reference territorial relevance metric for calibration and optimization. STEP is the primary onset-threshold recovery calibration; LINEAR_RAMP is retained as a secondary severity-gradient calibration. Standard WMARM is diagnostic only. Execution completion is not a statistical validation pass."}
        self.s.put("workflow","CALIBRATION","result",result)
        self.s.commit()
        return result

    def summarize(self) -> None:
        """CM8-10,19-20: retained empirical distributions and transparent denominators."""
        instances = list(self.s.rows("calibration_instance","RC"))
        for item in self.c["calibration"]["effect_classes"]:
            rows = [v for v in instances if v["class"]==item["name"]]
            for metric in ("wmarm_cs","wmarm","confidence_support_retention_ratio","directional_balance_cs","directional_balance","ci_share"):
                self.s.put("calibration_distribution","RC_TERRITORY",f"{item['name']}/{metric}",
                           {"class":item["name"],"metric":metric,
                            "metric_role":"REFERENCE" if metric in ("wmarm_cs","directional_balance_cs") else "DIAGNOSTIC",
                            **distribution(v.get(metric) for v in rows)})
            sign = class_sign(item)
            significance,correct_sign,power,reference_coverage,bias = [],[],[],[],[]
            for bid in self.d.ids:
                pooled = []
                for r in rows:
                    value = self.s.get("pooled",cell_scope(r["run"],self.qcal,0),bid)
                    if value and value["eligible"]:
                        pooled.append(value)
                self.s.put("calibration_distribution","RC_BSA",f"{item['name']}/{bid}",
                           {"class":item["name"],"bsa_id":bid,"metric":"pooled_rc",**distribution(v["pooled_rc"] for v in pooled)})
                for year in self.d.years:
                    annual = [self.s.get("yearly",cell_scope(r["run"],self.qcal,0),f"{bid}/{year}") for r in rows]
                    self.s.put("calibration_distribution","RC_BSA_YEAR",f"{item['name']}/{bid}/{year}",
                               {"class":item["name"],"bsa_id":bid,"year":year,"metric":"yearly_rc",
                                **distribution(v["rc"] for v in annual if v and v["rc"] is not None)})
                for j,v in enumerate(pooled):
                    nonzero = v["ci_low"]>0 or v["ci_high"]<0
                    significance.append(float(nonzero))
                    if sign in (-1,1):
                        correct = np.sign(v["pooled_rc"])==sign
                        correct_sign.append(float(correct));power.append(float(correct and nonzero))
                    others = [r["pooled_rc"] for k,r in enumerate(pooled) if k!=j]
                    if others:
                        reference = float(np.median(others))
                        reference_coverage.append(float(v["ci_low"]<=reference<=v["ci_high"]))
                        bias.append(v["pooled_rc"]-reference)
            self.s.put("calibration_operating_characteristics","RC",item["name"],
                       {"class":item["name"],"eligible_bsa_realizations":len(significance),
                        "type_I_error":finite_mean(significance) if item["exact_null"] else None,
                        "power":finite_mean(power),"sign_recovery":finite_mean(correct_sign),
                        "empirical_reference_coverage":finite_mean(reference_coverage),
                        "empirical_reference_coverage_n":len(reference_coverage),
                        "calibration_bias_against_leave_one_realization_out_median":finite_mean(bias),
                        "interpretation":"dataset-specific empirical RC reference, not imposed percentage as an RC truth"})
        for direction in ("harmful","protective"):
            lookup = {(r["class"],r["replicate"]):r for r in instances}
            magnitude_cs,magnitude_point,support = [],[],[]
            for m in range(self.c["calibration"]["number_of_realizations"]):
                names = ["negligible",f"moderate_{direction}",f"strong_{direction}"]
                rr = [lookup.get((name,m)) for name in names]
                if all(v is not None and v.get("wmarm_cs") is not None for v in rr):
                    magnitude_cs.append(float(rr[2]["wmarm_cs"]>rr[1]["wmarm_cs"]>rr[0]["wmarm_cs"]))
                if all(v is not None and v.get("wmarm") is not None for v in rr):
                    magnitude_point.append(float(rr[2]["wmarm"]>rr[1]["wmarm"]>rr[0]["wmarm"]))
                if all(v is not None and v["ci_share"] is not None for v in rr):
                    support.append(float(rr[2]["ci_share"]>rr[1]["ci_share"]>rr[0]["ci_share"]))
            self.s.put("calibration_ordering","RC",direction,{"direction":direction,"reference_metric":"WMARM_CS",
                       "wmarm_cs_strong_gt_moderate_gt_negligible_probability":finite_mean(magnitude_cs),
                       "strong_gt_moderate_gt_negligible_probability":finite_mean(magnitude_cs),
                       "valid_matched_replicates":len(magnitude_cs),
                       "diagnostic_wmarm_strong_gt_moderate_gt_negligible_probability":finite_mean(magnitude_point),
                       "valid_diagnostic_wmarm_matched_replicates":len(magnitude_point),
                       "strict_CI_support_ordering_probability":finite_mean(support),"valid_CI_matched_replicates":len(support),
                       "interpretation":"WMARM_CS ordering is the reference coherence diagnostic; standard WMARM ordering is retained descriptively."})
        tl = list(self.s.rows("calibration_instance","TL"))
        for response_mode in self.threshold_response_modes:
            role = "PRIMARY_ONSET_THRESHOLD_RECOVERY" if response_mode==self.primary_threshold_response_mode else "SECONDARY_SEVERITY_GRADIENT"
            for q,lag in generating_pairs(self.c):
                runs = [v for v in tl if v.get("threshold_lag_response_mode")==response_mode and v["q_generating"]==q and v["lag_generating"]==lag]
                valid_runs = [
                    v for v in runs
                    if v.get("calibration_valid_for_recovery",v.get("calibration_valid",True))
                ]
                supported = [v for v in valid_runs if v["region_nonempty"]]
                self.s.put("calibration_recovery",f"TL_{response_mode}",f"q{q:g}_lag{lag}",
                           {"threshold_lag_response_mode":response_mode,"response_role":role,
                            "q_generating":q,"lag_generating":lag,"total_realizations":len(runs),
                            "valid_realizations":len(valid_runs),"invalid_realizations":len(runs)-len(valid_runs),
                            "recovery_rate_denominator":"VALID_REALIZATIONS",
                            "nonempty_robust_regions":len(supported),"empty_regions":len(valid_runs)-len(supported),
                            "null_incomplete_count":sum(not v["optimization_null_complete"] for v in runs),
                            **{f"{name}_conditional_nonempty":finite_mean(float(v[name]) for v in supported)
                               for name in ("threshold_recovered","lag_recovered","joint_recovered")},
                            **{f"{name}_unconditional":finite_mean(float(v[name]) for v in valid_runs)
                               for name in ("threshold_recovered","lag_recovered","joint_recovered")}})
                for qc in self.c["thresholds"]:
                    for lc in self.c["lags"]:
                        from .engine import cell_key
                        values = [self.s.get("cell",v["run"],cell_key(qc,lc)) for v in runs]
                        cs_dist = distribution(v.get("wmarm_cs") for v in values if v)
                        point_dist = distribution(v.get("wmarm") for v in values if v)
                        payload = {"threshold_lag_response_mode":response_mode,"response_role":role,
                                   "reference_metric":"WMARM_CS",
                                   "q_generating":q,"lag_generating":lag,"q":qc,"lag":lc,
                                   **cs_dist}
                        payload.update({f"wmarm_cs_{k}":v for k,v in cs_dist.items()})
                        payload.update({f"diagnostic_wmarm_{k}":v for k,v in point_dist.items()})
                        self.s.put("calibration_mean_surface",f"{response_mode}/q{q:g}_lag{lag}",cell_key(qc,lc),payload)
        self.s.commit()

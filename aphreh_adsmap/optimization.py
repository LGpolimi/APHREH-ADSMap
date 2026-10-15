"""PO: full-surface LOYO, synchronized null searches, and unforced robust regions.

WMARMcs is the reference optimization metric. Standard WMARM remains a
diagnostic point-estimate surface and never determines near-optimality, null
support, persistence, or representative selection.
"""
from __future__ import annotations
import hashlib
import math
import multiprocessing as mp
import os
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import numpy as np
from .randomness import rng_for_identity
from .engine import Cancelled, FixedParametersEngine, SurfaceRunner, cell_key, cell_scope, finite_mean
from .numerics import finite_quantile
from .performance import MIB, bounded_parallel_plan, configure_process_worker, process_tree_rss_bytes
from .storage import Store

REFERENCE_METRIC = "wmarm_cs"
DIAGNOSTIC_METRIC = "wmarm"


_NULL_WORKER_CONTEXT: dict = {}


def null_parallel_plan(task_count: int) -> dict:
    """Bounded concurrency for independent production null replicates.

    This changes orchestration only. Each replicate keeps the same run/replicate
    RNG identity and executes the same :func:`run_null_replicate` primitive as
    the serial path.
    """
    return bounded_parallel_plan(
        task_count,
        workers_env="APHREH_ADSMAP_NULL_WORKERS",
        worker_rss_env="APHREH_ADSMAP_NULL_WORKER_RSS_MB",
        parent_reserve_env="APHREH_ADSMAP_NULL_PARENT_RESERVE_MB",
    )


def _null_worker_initializer(
    config: dict,
    canonical_root: str,
    project_root: str,
    worker_base: str,
    parent_db: str,
    plan: dict,
) -> None:
    configure_process_worker(plan)
    from .data import Data
    _NULL_WORKER_CONTEXT.clear()
    _NULL_WORKER_CONTEXT.update({
        "config": config,
        "data": Data(Path(canonical_root)),
        "project_root": Path(project_root),
        "worker_base": Path(worker_base),
        "parent_db": Path(parent_db),
    })


def _null_worker(task: dict) -> dict:
    """Execute one null replicate in an isolated SQLite store."""
    ctx = _NULL_WORKER_CONTEXT
    run = str(task["run"])
    replicate = int(task["replicate"])
    token = f"r{replicate:05d}"
    root = ctx["worker_base"] / token
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / "results.sqlite")
    error = None
    prior = None
    try:
        nullrun = f"{run}__NULL_{replicate:05d}"
        # Preserve partial-cell resume exactly as the serial Store would.
        store.import_run_from(ctx["parent_db"], nullrun)
        engine = FixedParametersEngine(
            ctx["data"], ctx["config"], store, root, log=lambda *_: None,
            cancellation_root=ctx["project_root"],
        )
        runner = SurfaceRunner(engine)
        prior = run_null_replicate(
            runner,
            run,
            replicate,
            outcome_path=Path(task["outcome_path"]) if task.get("outcome_path") else None,
            outcome_semantics=str(task.get("outcome_semantics", "EVENT_COUNTS")),
            bsa_year_eligibility_path=(
                Path(task["bsa_year_eligibility_path"])
                if task.get("bsa_year_eligibility_path") else None
            ),
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
        "run": run,
        "replicate": replicate,
        "prior": prior,
        "error": error,
        "worker_root": str(root),
        "database": str(root / "results.sqlite"),
    }


def _null_worker_base(engine: FixedParametersEngine, run: str) -> Path:
    token = hashlib.sha256(str(run).encode("utf-8")).hexdigest()[:16]
    return engine.root / "scratch" / "parallel_null_workers" / token


def _recover_null_worker_checkpoints(
    engine: FixedParametersEngine, worker_base: Path
) -> int:
    recovered = 0
    if not worker_base.is_dir():
        return recovered
    for worker_root in sorted(worker_base.iterdir()):
        database = worker_root / "results.sqlite"
        if not worker_root.is_dir() or not database.is_file():
            continue
        try:
            engine.store.merge_from(database)
            engine.store.commit()
            recovered += 1
            shutil.rmtree(worker_root, ignore_errors=True)
        except Exception as exc:
            engine.log(
                f"Null-search checkpoint recovery skipped {worker_root.name}: {exc}"
            )
    if recovered:
        engine.log(
            f"Null-search checkpoint recovery: merged {recovered} isolated worker stores"
        )
    return recovered


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


def _parallel_null_search(
    runner: SurfaceRunner,
    run: str,
    missing: list[int],
    *,
    outcome_path: Path | None,
    outcome_semantics: str,
    bsa_year_eligibility_path: Path | None,
    worker_base: Path,
    plan: dict,
) -> None:
    e = runner.engine
    worker_base.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=int(plan["worker_count"]),
        mp_context=ctx,
        initializer=_null_worker_initializer,
        initargs=(
            e.c,
            str(e.data.root),
            str(e.root),
            str(worker_base),
            str(e.store.path),
            plan,
        ),
    )
    tasks = [
        {
            "run": run,
            "replicate": r,
            "outcome_path": str(outcome_path) if outcome_path is not None else None,
            "outcome_semantics": outcome_semantics,
            "bsa_year_eligibility_path": (
                str(bsa_year_eligibility_path)
                if bsa_year_eligibility_path is not None else None
            ),
        }
        for r in missing
    ]
    futures = {executor.submit(_null_worker, task): task for task in tasks}
    pending = set(futures)
    completed = 0
    failure: BaseException | None = None
    try:
        while pending:
            e.check()
            if process_tree_rss_bytes() >= int(plan["aggregate_rss_cap_mib"]) * MIB:
                failure = MemoryError(
                    "Analytical process-tree RSS reached the configured parallel null-search cap; "
                    "committed replicate checkpoints remain resumable"
                )
                _terminate_pool(executor)
                break
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    failure = RuntimeError(
                        f"Null worker process failed for {run}, replicate {task['replicate']}: {exc}"
                    )
                    _terminate_pool(executor)
                    break
                worker_root = Path(result["worker_root"])
                database = Path(result["database"])
                if database.is_file():
                    e.store.merge_from(database)
                    e.store.commit()
                if result.get("ok"):
                    shutil.rmtree(worker_root, ignore_errors=True)
                    completed += 1
                    if completed == 1 or completed % 5 == 0 or completed == len(tasks):
                        e.log(
                            f"{run}: parallel null search {completed}/{len(tasks)} pending replicates complete "
                            f"({plan['worker_count']} workers)"
                        )
                else:
                    error = result.get("error") or {}
                    etype = str(error.get("type") or "RuntimeError")
                    message = str(error.get("message") or "unknown worker failure")
                    if etype == "Cancelled":
                        failure = Cancelled(message)
                    elif etype == "MemoryError":
                        failure = MemoryError(message)
                    else:
                        failure = RuntimeError(
                            f"Null worker failed for {run}, replicate {task['replicate']}: "
                            f"{etype}: {message}\n{error.get('traceback', '')}"
                        )
                    _terminate_pool(executor)
                    break
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
        raise failure


def admissibility_coverages(cell: dict) -> tuple[float | None, float | None, str]:
    """Return the coverage pair that defines optimization admissibility.

    Ordinary observed analyses use full-territory coverage. Synthetic
    calibration cells produced with a CM5.2 structural-eligibility sidecar use
    calibration-conditional coverage, while retaining the ordinary coverage
    fields as diagnostics. This prevents methodologically excluded BSA-years
    from being counted again as optimization support failures.
    """
    bsa = cell.get("admissibility_bsa_coverage")
    pop = cell.get("admissibility_population_coverage")
    basis = cell.get("admissibility_coverage_basis")
    if bsa is None and pop is None:
        return cell.get("bsa_coverage"), cell.get("population_coverage"), "FULL_TERRITORY"
    return bsa, pop, str(basis or "CALIBRATION_CONDITIONAL_STRUCTURAL_ELIGIBILITY")


def admissible(cell: dict, opt: dict) -> bool:
    bsa_coverage, population_coverage, _ = admissibility_coverages(cell)
    return bool(cell.get("valid") and cell.get(REFERENCE_METRIC) is not None
                and bsa_coverage is not None
                and bsa_coverage>=opt["minimum_bsa_coverage"]
                and population_coverage is not None
                and population_coverage>=opt["minimum_population_coverage"])


def components(region: list[dict], qs: list[float], lags: list[int]) -> list[dict]:
    remaining = {(qs.index(r["q"]),lags.index(r["lag"])) for r in region}
    result = []
    while remaining:
        start = min(remaining); remaining.remove(start)
        todo, group = [start],[start]
        while todo:
            i,j = todo.pop()
            for neighbour in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if neighbour in remaining:
                    remaining.remove(neighbour);todo.append(neighbour);group.append(neighbour)
        pairs = [{"q":qs[i],"lag":lags[j]} for i,j in sorted(group)]
        result.append({"cells":pairs,"q_min":min(v["q"] for v in pairs),"q_max":max(v["q"] for v in pairs),
                       "lag_min":min(v["lag"] for v in pairs),"lag_max":max(v["lag"] for v in pairs)})
    return result


def resolve_region(cells: list[dict], loyo: list[dict], maxima: list[float | None],
                   years: list[int], config: dict) -> dict:
    """Pure PO4-7 decision layer using WMARMcs exclusively as reference."""
    opt = config["optimization"]
    eligible = [r for r in cells if admissible(r,opt)]
    observed = max((r[REFERENCE_METRIC] for r in eligible),default=None)
    epsilon = opt["near_optimal_relative_tolerance_epsilon"]
    foldmax = {y:max((r[REFERENCE_METRIC] for r in loyo if r["excluded_year"]==y and admissible(r,opt)),default=None) for y in years}
    null_complete = len(maxima)==opt["null_realizations_R"] and all(m is not None and math.isfinite(m) for m in maxima)
    cutoff = finite_quantile(np.array(maxima,dtype=float),1-opt["search_adjusted_alpha"]) if null_complete else None
    pmax = (1+sum(m>=observed for m in maxima))/(len(maxima)+1) if null_complete and observed is not None else None
    diagnostics,region = [],[]
    for cell in cells:
        folds = [next((r for r in loyo if r["q"]==cell["q"] and r["lag"]==cell["lag"] and r["excluded_year"]==y),None) for y in years]
        complete = len(years)>0 and all(r is not None and admissible(r,opt) and foldmax[y] is not None for r,y in zip(folds,years))
        vals = [r[REFERENCE_METRIC] for r in folds] if complete else []
        persistence = sum(r[REFERENCE_METRIC] >= (1-epsilon)*foldmax[y]-1e-12 for r,y in zip(folds,years))/len(years) if complete else None
        support = admissible(cell,opt)
        near = support and observed is not None and cell[REFERENCE_METRIC] >= (1-epsilon)*observed-1e-12
        stable = persistence is not None and persistence >= opt["minimum_persistence_pi"]
        unusual = support and cutoff is not None and cell[REFERENCE_METRIC]>cutoff
        effective_bsa_coverage, effective_population_coverage, coverage_basis = admissibility_coverages(cell)
        diag = {"q":cell["q"],"lag":cell["lag"],"reference_metric":"WMARM_CS",
                "wmarm_cs":cell.get("wmarm_cs"),"wmarm":cell.get("wmarm"),
                "confidence_support_retention_ratio":cell.get("confidence_support_retention_ratio"),
                "admissible_support":support,"near_optimal":near,"stable":stable,"null_supported":unusual,
                "loyo_complete":complete,"loyo_mean_wmarm_cs":finite_mean(vals),
                "loyo_sd_wmarm_cs":float(np.std(vals,ddof=1)) if len(vals)>=2 else None,
                "loyo_max_deviation_wmarm_cs":max(abs(v-cell[REFERENCE_METRIC]) for v in vals) if vals and cell.get(REFERENCE_METRIC) is not None else None,
                "persistence":persistence,"robust":support and near and stable and unusual,
                "bsa_coverage":cell["bsa_coverage"],"population_coverage":cell["population_coverage"],
                "calibration_bsa_coverage":cell.get("calibration_bsa_coverage"),
                "calibration_population_coverage":cell.get("calibration_population_coverage"),
                "admissibility_bsa_coverage":effective_bsa_coverage,
                "admissibility_population_coverage":effective_population_coverage,
                "admissibility_coverage_basis":coverage_basis}
        # Backward-compatible aliases, explicitly tied to the reference metric.
        diag["loyo_mean"] = diag["loyo_mean_wmarm_cs"]
        diag["loyo_sd"] = diag["loyo_sd_wmarm_cs"]
        diag["loyo_max_deviation"] = diag["loyo_max_deviation_wmarm_cs"]
        diagnostics.append(diag)
        if diag["robust"]: region.append(diag)
    representative = None
    if region:
        maxw = max(r[REFERENCE_METRIC] for r in region)
        tied = [r for r in region if abs(r[REFERENCE_METRIC]-maxw)<=1e-12]
        best = min(tied,key=lambda r:(
            -r["persistence"],
            -(r.get("admissibility_population_coverage") if r.get("admissibility_population_coverage") is not None else -1.0),
            -(r.get("admissibility_bsa_coverage") if r.get("admissibility_bsa_coverage") is not None else -1.0),
            r["lag"],r["q"]
        ))
        representative = {"q":best["q"],"lag":best["lag"],"wmarm_cs":best["wmarm_cs"],"wmarm":best.get("wmarm")}
    peak = min(eligible,key=lambda r:(-r[REFERENCE_METRIC],r["q"],r["lag"])) if eligible else None
    peak_payload = ({"q":peak["q"],"lag":peak["lag"],"wmarm_cs":peak["wmarm_cs"],"wmarm":peak.get("wmarm")} if peak else None)
    return {"status":"ROBUST_REGION" if region else "NO_ROBUST_REGION",
            "reference_metric":"WMARM_CS",
            "observed_admissible_maximum_wmarm_cs":observed,
            "observed_admissible_maximum":observed,
            "observed_peak":peak_payload,
            "p_max":pmax,"null_cutoff_wmarm_cs":cutoff,"null_cutoff":cutoff,
            "null_realizations_requested":opt["null_realizations_R"],
            "null_realizations_valid":sum(m is not None and math.isfinite(m) for m in maxima),
            "null_complete":null_complete,"monte_carlo_minimum_p":1/(opt["null_realizations_R"]+1),
            "admissible_observed_cells":len(eligible),
            "loyo_complete_admissible_cells":sum(
                bool(v.get("admissible_support")) and bool(v.get("loyo_complete")) for v in diagnostics
            ),
            "loyo_years":years,"diagnostics":diagnostics,"robust_region":region,
            "components":components(region,config["thresholds"],config["lags"]),
            "representative":representative,
            "representative_tie_rule":"MAX_WMARM_CS; ABS_1E-12; greater persistence, population and BSA coverage; earlier lag; lower q"}


def fingerprint(cells: list[dict], optimization: dict, config: dict) -> dict:
    """MF6: WMARMcs reference fingerprint plus explicit point-estimate diagnostics."""
    domain = [r for r in cells if admissible(r,config["optimization"])]
    def mean(key): return finite_mean(r.get(key) for r in domain)
    region = optimization["robust_region"]
    return {"domain":"admissible WMARMcs full-grid cells; equal cell weights",
            "reference_metric":"WMARM_CS","domain_cell_count":len(domain),
            "effect":{"E_CS_max":max((r[REFERENCE_METRIC] for r in domain),default=None),
                      "E_CS_mean":mean(REFERENCE_METRIC),"E_CS_region":finite_mean(r[REFERENCE_METRIC] for r in region),
                      "A_region":len(region)/len(domain) if domain and region else None,
                      "D_CS_signed_balance":mean("directional_balance_cs"),
                      "K_CS_mean":mean("confidence_support_retention_ratio"),
                      "diagnostic_WMARM_max":max((r[DIAGNOSTIC_METRIC] for r in domain if r.get(DIAGNOSTIC_METRIC) is not None),default=None),
                      "diagnostic_WMARM_mean":mean(DIAGNOSTIC_METRIC),
                      "diagnostic_D_signed_balance":mean("directional_balance")},
            "statistical":{"R_CI":mean("ci_excluding_zero_population_share"),"R_DS":mean("directional_stability"),
                           "R_precision":mean("relative_precision"),"C_BSA":mean("bsa_coverage"),
                           "C_POP":mean("population_coverage"),"R_validity":mean("bootstrap_validity"),"p_max":optimization["p_max"]},
            "temporal":{"T_sign":mean("year_pooled_sign_concordance"),"I_squared_percent":mean("mean_i_squared_percent"),
                        "T_surface_CS":optimization.get("surface_coherence"),
                        "T_optimal":finite_mean(r["persistence"] for r in region)},
            "spatial":{"S_direction_CS":mean("spatial_directional_coherence_cs"),
                       "S_dispersion_CS":mean("spatial_magnitude_dispersion_cs"),
                       "diagnostic_S_direction":mean("spatial_directional_coherence"),
                       "diagnostic_S_dispersion":mean("spatial_magnitude_dispersion"),
                       "S_concordance_E":mean("local_concordance_exposed"),"S_concordance_N":mean("local_concordance_nonexposed"),
                       "C_BSA":mean("bsa_coverage"),"C_POP":mean("population_coverage")},
            "unavailable_policy":"null with underlying exclusion/support records; no combined score"}


def run_null_replicate(runner: SurfaceRunner, run: str, replicate: int, *,
                       outcome_path: Path | None = None,
                       outcome_semantics: str = "EVENT_COUNTS",
                       bsa_year_eligibility_path: Path | None = None) -> dict:
    """Execute one deterministic synchronized maximum-statistic null replicate.

    The replicate identity, circular shifts, estimator, admissibility rule and
    persisted null-cell payload are exactly the same as the production null
    search.  Exposing the replicate as an orchestration primitive allows the
    calibration layer to distribute independent null surfaces across isolated
    worker processes without duplicating or changing the statistical method.
    Existing committed replicates are reused verbatim.
    """
    e = runner.engine
    c,d,s = e.c,e.data,e.store
    r = int(replicate)
    R = int(c["optimization"]["null_realizations_R"])
    if not 0 <= r < R:
        raise ValueError(f"Null replicate {r} is outside configured range 0..{R-1}")
    prior = s.get("null_maximum",run,f"{r:05d}")
    if prior is not None:
        return prior
    shiftmin = c["optimization"]["null_randomization_rule"]["parameters"]["minimum_shift_days"]
    rng = rng_for_identity(c["seed"],namespace="DISK_DEV_NULL_V1",identity={"run":run,"replicate":r})
    permutation = np.arange(d.n); shifts = {}
    for iy,y in enumerate(d.years):
        ix = np.flatnonzero(d.year_index==iy)
        if len(ix)<2*shiftmin:
            raise ValueError(f"Year {y}: record too short for the specified minimum circular shift")
        shift = rng.randint(shiftmin,len(ix)-shiftmin); permutation[ix] = np.roll(ix,shift); shifts[str(y)] = shift
    nullrun = f"{run}__NULL_{r:05d}"
    nullcells = runner.run(
        nullrun,outcome_path=outcome_path,outcome_semantics=outcome_semantics,
        bsa_year_eligibility_path=bsa_year_eligibility_path,
        permutation=permutation,loyo=False,retain=False
    )
    admissible_cells = [v for v in nullcells if admissible(v,c["optimization"])]
    maximum_cs = max((v[REFERENCE_METRIC] for v in admissible_cells),default=None)
    maximum_point = max((v[DIAGNOSTIC_METRIC] for v in admissible_cells if v.get(DIAGNOSTIC_METRIC) is not None),default=None)
    prior = {"run":run,"replicate":r,"reference_metric":"WMARM_CS",
             "maximum_wmarm_cs":maximum_cs,"maximum_wmarm":maximum_point,"shifts":shifts,
             "valid_cells":len(admissible_cells),"complete_grid_cells":len(nullcells),
             "reason":None if maximum_cs is not None else "NULL_SURFACE_HAS_NO_ADMISSIBLE_WMARM_CS_CELL"}
    s.put("null_maximum",run,f"{r:05d}",prior)
    for v in nullcells:
        s.put("null_cell",run,f"{r:05d}/{cell_key(v['q'],v['lag'])}",
              {"run":run,"replicate":r,**{k:v.get(k) for k in (
                  "q","lag","wmarm_cs","wmarm","marm","confidence_support_retention_ratio",
                  "bsa_coverage","population_coverage","eligible_bsas","valid",
                  "calibration_eligible_bsas","calibration_represented_bsas","calibration_bsa_coverage",
                  "calibration_population_coverage","admissibility_bsa_coverage",
                  "admissibility_population_coverage","admissibility_coverage_basis"
              )}})
        s.delete_scope(cell_scope(nullrun,v["q"],v["lag"]))
    s.delete_scope(nullrun); s.commit()
    return prior


def ensure_null_search(runner: SurfaceRunner, run: str, *, outcome_path: Path | None = None,
                       outcome_semantics: str = "EVENT_COUNTS",
                       bsa_year_eligibility_path: Path | None = None,
                       replicate_callback=None) -> list[float | None]:
    """Ensure the production synchronized maximum-statistic null search exists.

    Replicates are scientifically identical to the serial implementation: each
    one still enters :func:`run_null_replicate`, with the same run/replicate RNG
    identity and the same estimator. Only independent replicate scheduling is
    parallelized. Worker SQLite stores are isolated and merged by the parent.
    Calibration-prefixed runs retain their existing calibration scheduler.
    """
    e = runner.engine
    c, s = e.c, e.store
    R = int(c["optimization"]["null_realizations_R"])

    main_process = mp.current_process().name == "MainProcess"
    production_parallel_scope = main_process and not str(run).startswith("CAL_")
    worker_base = None
    if production_parallel_scope:
        worker_base = _null_worker_base(e, run)
        _recover_null_worker_checkpoints(e, worker_base)

    missing = [r for r in range(R) if s.get("null_maximum", run, f"{r:05d}") is None]
    if missing:
        plan = null_parallel_plan(len(missing))
        if production_parallel_scope and len(missing) > 1 and int(plan["worker_count"]) > 1:
            e.log(
                f"{run}: parallel null search started; {R-len(missing)}/{R} reusable, "
                f"{len(missing)} pending across {plan['worker_count']}/{plan['physical_cores']} physical-core workers"
            )
            _parallel_null_search(
                runner,
                run,
                missing,
                outcome_path=outcome_path,
                outcome_semantics=outcome_semantics,
                bsa_year_eligibility_path=bsa_year_eligibility_path,
                worker_base=worker_base,
                plan=plan,
            )
        else:
            for r in missing:
                e.check()
                prior = run_null_replicate(
                    runner,
                    run,
                    r,
                    outcome_path=outcome_path,
                    outcome_semantics=outcome_semantics,
                    bsa_year_eligibility_path=bsa_year_eligibility_path,
                )
                if r == 0 or (r + 1) % 5 == 0 or r + 1 == R:
                    e.log(f"{run}: null search {r+1}/{R} complete")

    maxima: list[float | None] = []
    for r in range(R):
        e.check()
        prior = s.get("null_maximum", run, f"{r:05d}")
        # Defensive fallback: this can only occur after an external worker loss;
        # it uses the same serial primitive and therefore cannot change the model.
        if prior is None:
            prior = run_null_replicate(
                runner,
                run,
                r,
                outcome_path=outcome_path,
                outcome_semantics=outcome_semantics,
                bsa_year_eligibility_path=bsa_year_eligibility_path,
            )
        maxima.append(prior.get("maximum_wmarm_cs"))
        if replicate_callback is not None:
            replicate_callback(r, prior)
    return maxima


class Optimizer:
    def __init__(self, runner: SurfaceRunner):
        self.runner = runner; self.e = runner.engine

    def run(self, run: str, cells: list[dict], *, outcome_path: Path | None = None,
            outcome_semantics: str = "EVENT_COUNTS",
            bsa_year_eligibility_path: Path | None = None) -> dict:
        previous = self.e.store.get("optimization",run,"result")
        if previous is not None: return previous
        c,d,s = self.e.c,self.e.data,self.e.store
        maxima = ensure_null_search(
            self.runner, run, outcome_path=outcome_path, outcome_semantics=outcome_semantics,
            bsa_year_eligibility_path=bsa_year_eligibility_path
        )
        loyo = list(s.rows("loyo",run))
        result = resolve_region(cells,loyo,maxima,d.years,c)
        absdev = []
        for diag in result["diagnostics"]:
            if diag["admissible_support"] and diag["loyo_complete"]:
                for v in loyo:
                    if v["q"]==diag["q"] and v["lag"]==diag["lag"]:
                        absdev.append(abs(v[REFERENCE_METRIC]-diag[REFERENCE_METRIC]))
        result["surface_coherence"] = 1-float(np.mean(absdev)) if absdev else None
        result["run"] = run
        s.put("optimization",run,"result",result)
        for diag in result["diagnostics"]:
            s.put("optimization_cell",run,cell_key(diag["q"],diag["lag"]),{"run":run,**diag})
        s.put("fingerprint",run,"result",fingerprint(cells,result,c)); s.commit(); return result

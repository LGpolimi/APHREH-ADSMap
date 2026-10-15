"""Bounded, process-local acceleration without changing scientific settings.

Caches are disposable. Persistent arrays and SQLite checkpoints retain their
original formats and durability. No process pool or whole-surface draw cache is
created. All caps use MiB (2**20 bytes), not decimal megabytes.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import psutil

MIB = 1024 ** 2


def _env_limit(name: str, default: int, maximum: int, minimum: int = 0) -> int:
    text = os.environ.get(name)
    try:
        value = default if text is None else int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer MiB limit") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} MiB")
    return value * MIB


def _array_bytes(value) -> int:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot enter the numerical cache")
        return int(value.nbytes)
    if isinstance(value, tuple):
        return sum(_array_bytes(v) for v in value)
    if value is None:
        return 0
    raise TypeError("The bounded numerical cache accepts only arrays/tuples")


def _freeze(value) -> None:
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, tuple):
        for item in value:
            _freeze(item)


def file_signature(path: Path) -> tuple:
    """Invalidate cached inputs after replacement or in-place file updates."""
    path = Path(path).resolve()
    st = path.stat()
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class MemoryBudget:
    """One shared LRU and temporary-array budget for every engine in a process.

    The guard is a cooperative process-RSS safeguard, not an OS-wide memory
    quota. It leaves headroom below 2 GB for native temporaries and GUI overhead.
    Allocation admission uses current RSS, including disk-backed resident pages.
    """
    def __init__(self, *, cache_bytes: int | None = None,
                 workspace_bytes: int | None = None, rss_guard_bytes: int | None = None):
        self.cache_limit = (_env_limit("APHREH_ADSMAP_CACHE_MB", 256, 512)
                            if cache_bytes is None else int(cache_bytes))
        self.workspace_limit = (_env_limit("APHREH_ADSMAP_WORKSPACE_MB", 128, 256)
                                if workspace_bytes is None else int(workspace_bytes))
        self.rss_guard = (_env_limit("APHREH_ADSMAP_RSS_GUARD_MB", 1536, 1792, 256)
                          if rss_guard_bytes is None else int(rss_guard_bytes))
        self.rss_headroom = _env_limit("APHREH_ADSMAP_RSS_HEADROOM_MB", 256, 512, 64)
        if min(self.cache_limit, self.workspace_limit) < 0 or self.rss_guard <= 0:
            raise ValueError("Invalid memory budget")
        if self.rss_headroom >= self.rss_guard:
            raise ValueError("APHREH_ADSMAP_RSS_HEADROOM_MB must be smaller than the RSS guard")
        self.rss_target = min(1024 * MIB, max(0, self.rss_guard - self.rss_headroom))
        self.cache = OrderedDict()
        self.cache_bytes = self.workspace_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.ram_workspaces = self.disk_workspaces = 0
        self.parallel_calibration = None
        self.peak_rss_bytes = 0
        self.process = psutil.Process()
        self.lock = threading.RLock()
        self._last_check = 0.0

    def rss(self) -> int:
        value = int(self.process.memory_info().rss)
        if value > self.peak_rss_bytes:
            self.peak_rss_bytes = value
        return value

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.cache_bytes = 0

    def check(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_check < 0.1:
            return
        self._last_check = now
        resident = self.rss()
        if resident > self.rss_target:
            self.clear()
            resident = self.rss()
        if resident >= self.rss_guard:
            raise MemoryError(
                f"Process RSS reached the {self.rss_guard / MIB:g} MiB safety guard. "
                "The active cell is not committed; completed checkpoints remain resumable. "
                "Close other views or reduce cache/workspace limits before resuming."
            )

    def get(self, key):
        with self.lock:
            item = self.cache.pop(key, None)
            if item is None:
                self.misses += 1
                return None
            self.cache[key] = item
            self.hits += 1
            return item[0]

    def can_cache(self, size: int) -> bool:
        # A single file cannot monopolize the entire numerical cache.
        return (0 < size <= min(self.cache_limit, 64 * MIB)
                and self.rss() + size <= self.rss_target)

    def put(self, key, value) -> bool:
        size = _array_bytes(value)
        if size > self.cache_limit or self.cache_limit == 0 or self.rss() > self.rss_target:
            return False
        with self.lock:
            old = self.cache.pop(key, None)
            if old is not None:
                self.cache_bytes -= old[1]
            while self.cache and (self.cache_bytes + size > self.cache_limit or len(self.cache) >= 1024):
                _, (_, nbytes) = self.cache.popitem(last=False)
                self.cache_bytes -= nbytes
                self.evictions += 1
            _freeze(value)
            self.cache[key] = (value, size)
            self.cache_bytes += size
        return True

    def reserve_workspace(self, size: int) -> bool:
        self.check(force=True)
        with self.lock:
            if size > self.workspace_limit - self.workspace_bytes or self.rss() + size > self.rss_target:
                self.disk_workspaces += 1
                return False
            self.workspace_bytes += size
            self.ram_workspaces += 1
            return True

    def release_workspace(self, size: int) -> None:
        with self.lock:
            self.workspace_bytes -= size

    def record_parallel_calibration(self, stats: dict) -> None:
        self.parallel_calibration = dict(stats)

    def snapshot(self) -> dict:
        result = {
            "cache_limit_mib": self.cache_limit / MIB,
            "workspace_limit_mib": self.workspace_limit / MIB,
            "rss_target_mib": self.rss_target / MIB,
            "rss_guard_mib": self.rss_guard / MIB,
            "rss_headroom_mib": self.rss_headroom / MIB,
            "peak_process_rss_mib": max(self.peak_rss_bytes, self.rss()) / MIB,
            "cache_live_mib": self.cache_bytes / MIB,
            "workspace_live_mib": self.workspace_bytes / MIB,
            "cache_hits": self.hits, "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "ram_workspaces": self.ram_workspaces, "disk_workspaces": self.disk_workspaces,
        }
        if self.parallel_calibration is not None:
            result["calibration_parallel"] = dict(self.parallel_calibration)
        return result


_budget: MemoryBudget | None = None


def memory_budget() -> MemoryBudget:
    global _budget
    if _budget is None:
        _budget = MemoryBudget()
    return _budget


@contextlib.contextmanager
def working_array(path: Path, *, shape: tuple[int, ...], dtype: str = "float64"):
    """Disposable array in bounded RAM, falling back to the original memmap."""
    from .storage import mapped
    budget = memory_budget()
    size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
    if budget.reserve_workspace(size):
        try:
            try:
                array = np.empty(shape, dtype=dtype)
            except MemoryError:
                # The allocator can still refuse a cooperatively admitted size.
                with mapped(path, shape=shape, dtype=dtype) as array:
                    yield array
            else:
                yield array
        finally:
            budget.release_workspace(size)
    else:
        with mapped(path, shape=shape, dtype=dtype) as array:
            yield array


def _parallel_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def bounded_parallel_plan(
    task_count: int,
    *,
    workers_env: str,
    worker_rss_env: str,
    parent_reserve_env: str,
    default_worker_rss_mib: int = 256,
    default_parent_reserve_mib: int = 256,
) -> dict:
    """Choose bounded process-level concurrency for independent work units.

    This is orchestration only. Scientific random-number identities, estimator
    calls, support rules, and reduction order inside each work unit are not
    changed. Physical cores and both application/system memory headroom cap the
    worker count, matching the safety principles used by calibration.
    """
    task_count = max(0, int(task_count))
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    physical = max(1, int(physical))
    requested = _parallel_env_int(workers_env, physical, 1, 64)
    total_cap_mib = _parallel_env_int("APHREH_ADSMAP_TOTAL_RSS_MB", 1792, 512, 1792)
    worker_guard_mib = _parallel_env_int(
        worker_rss_env, default_worker_rss_mib, 192, 768
    )
    parent_reserve_mib = _parallel_env_int(
        parent_reserve_env, default_parent_reserve_mib, 128, 768
    )
    system_reserve_mib = _parallel_env_int(
        "APHREH_ADSMAP_SYSTEM_FREE_RESERVE_MB", 512, 128, 4096
    )
    by_app_memory = max(1, (total_cap_mib - parent_reserve_mib) // worker_guard_mib)
    available_mib = int(psutil.virtual_memory().available // MIB)
    by_system_memory = max(
        1, (max(0, available_mib - system_reserve_mib)) // worker_guard_mib
    )
    workers = max(
        1,
        min(
            task_count if task_count else 1,
            physical,
            requested,
            by_app_memory,
            by_system_memory,
        ),
    )
    return {
        "worker_count": workers,
        "requested_workers": requested,
        "physical_cores": physical,
        "task_count": task_count,
        "aggregate_rss_cap_mib": total_cap_mib,
        "worker_rss_guard_mib": worker_guard_mib,
        "parent_reserve_mib": parent_reserve_mib,
        "system_available_mib_at_start": available_mib,
        "system_free_reserve_mib": system_reserve_mib,
        "worker_cache_mib": 48,
        "worker_workspace_mib": 48,
        "worker_rss_headroom_mib": 64,
    }


def configure_process_worker(plan: dict) -> None:
    """Keep each spawned worker single-threaded and inside its RSS budget."""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["APHREH_ADSMAP_CACHE_MB"] = str(int(plan["worker_cache_mib"]))
    os.environ["APHREH_ADSMAP_WORKSPACE_MB"] = str(int(plan["worker_workspace_mib"]))
    os.environ["APHREH_ADSMAP_RSS_GUARD_MB"] = str(int(plan["worker_rss_guard_mib"]))
    os.environ["APHREH_ADSMAP_RSS_HEADROOM_MB"] = str(int(plan["worker_rss_headroom_mib"]))


def process_tree_rss_bytes() -> int:
    """Resident memory of the current process plus all live descendants."""
    process = psutil.Process()
    total = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total

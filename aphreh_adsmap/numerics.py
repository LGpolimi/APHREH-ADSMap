"""Formula-bearing, bounded-workspace APHREH kernels. References: MF, PO and CM.

Floating arithmetic is float64. Quantiles use HF7 linear interpolation. Exact
outcome ties contribute zero. Missing support never receives uniform weights,
a fabricated variance, a default independence assumption, or a forced optimum.
"""
from __future__ import annotations
import warnings
from functools import lru_cache
from math import ceil
from random import Random
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from scipy.special import stdtrit


def finite_quantile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p, method="linear")) if x.size else float("nan")


def edge_fill(x: np.ndarray) -> np.ndarray:
    """Only edge gaps are filled; internal unavailable stretches remain unavailable."""
    result = x.copy()
    ix = np.flatnonzero(np.isfinite(result))
    if ix.size:
        result[:ix[0]] = result[ix[0]]
        result[ix[-1]+1:] = result[ix[-1]]
    return result


def _row_nanmedian(values: np.ndarray) -> np.ndarray:
    """Exact row medians for bounded float64 windows, ignoring only NaNs.

    MF2.2/MF3.2: select the same middle order statistics as nanmedian; no
    interpolation, approximation, or change to donor support is introduced.
    The caller's overlapping window view is never modified.
    """
    # NumPy uses a different 1-D selection path for unusually wide windows.
    # Retain it verbatim, including extreme-value overflow conventions.
    if values.shape[1] >= 600:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(values, axis=1)
    ordered = np.sort(values, axis=1)
    count = np.sum(~np.isnan(ordered), axis=1)
    last = np.maximum(count - 1, 0)
    rows = np.arange(len(ordered))
    with np.errstate(invalid="ignore", over="ignore"):
        median = (ordered[rows, last // 2] + ordered[rows, count // 2]) / 2.0
    median[count == 0] = np.nan
    return median


def rolling_median(x: np.ndarray, width: int, minimum: int) -> np.ndarray:
    """Calendar-centered median in bounded 1024-row batches; even extra day is right."""
    left, right = (width-1)//2, width//2
    padded = np.pad(x, (left,right), constant_values=np.nan)
    windows = sliding_window_view(padded, width)
    result = np.full(len(x), np.nan)
    batch_rows = max(1, min(1024, (16 * 1024**2) // max(1, 32 * width)))
    for start in range(0, len(x), batch_rows):
        w = windows[start:start+batch_rows]
        med = _row_nanmedian(w)
        med[np.sum(np.isfinite(w),axis=1) < minimum] = np.nan
        result[start:start+len(w)] = med
    return result


def segment_slices(segment_index: np.ndarray | None, n: int):
    """Yield half-open slices for contiguous calendar segments."""
    if segment_index is None:
        yield 0, n
        return
    seg = np.asarray(segment_index)
    if len(seg) != n:
        raise ValueError("segment_index length must match the series length")
    if n == 0:
        return
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    stops = np.r_[starts[1:], n]
    for a, b in zip(starts, stops):
        yield int(a), int(b)


def max_contiguous_length(segment_index: np.ndarray | None, n: int | None = None) -> int:
    if segment_index is None:
        return int(n or 0)
    seg = np.asarray(segment_index)
    if not len(seg):
        return 0
    return max(b-a for a,b in segment_slices(seg,len(seg)))


def shift_within_segments(values: np.ndarray, lag: int, segment_index: np.ndarray | None = None) -> np.ndarray:
    """Shift values from t-lag to t without crossing an unavailable calendar gap."""
    x = np.asarray(values)
    result = np.full(x.shape, np.nan, dtype=float)
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    for a,b in segment_slices(segment_index,len(x)):
        if lag == 0:
            result[a:b] = x[a:b]
        elif b-a > lag:
            result[a+lag:b] = x[a:b-lag]
    return result


def cumulative_incidence(outcome: np.ndarray, population: np.ndarray,
                         lag: int, scaling: float, semantics: str,
                         segment_index: np.ndarray | None = None) -> np.ndarray:
    """MF2.1 / CM17 with exact calendar-gap handling.

    A cumulative window is available only when every date t..t+lag belongs to
    the same contiguous retained-date segment. This prevents an omitted year or
    missing matching day from being compressed into an adjacent observation.
    """
    outcome = np.asarray(outcome)
    population = np.asarray(population)
    n = len(outcome)
    result = np.full(n,np.nan)
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    for a,b in segment_slices(segment_index,n):
        length = b-a
        if lag >= length:
            continue
        m = length-lag
        total = np.zeros(m)
        complete = np.ones(m,dtype=bool)
        for k in range(lag+1):
            values = outcome[a+k:a+k+m]
            complete &= np.isfinite(values)
            total += np.where(np.isfinite(values),values,0.0)
        if semantics == "EVENT_COUNTS":
            target_population = population[a:a+m]
            complete &= np.isfinite(target_population) & (target_population > 0)
            np.divide(total*scaling,target_population,out=total,where=complete)
        elif semantics != "DAILY_INCIDENCE":
            raise ValueError(f"Unsupported outcome semantics: {semantics}")
        result[a:a+m] = np.where(complete,total,np.nan)
    return result

def exposure(hazard: np.ndarray, year_index: np.ndarray, q: float) -> tuple[np.ndarray,np.ndarray]:
    """MF1.2-1.3: each candidate date retains its own calendar-year classification."""
    labels = np.full(len(hazard),-1,dtype=np.int8)
    thresholds = np.full(int(year_index.max())+1,np.nan)
    for iy in range(len(thresholds)):
        mask = year_index == iy
        threshold = finite_quantile(hazard[mask],q/100)
        thresholds[iy] = threshold
        valid = mask & np.isfinite(hazard) & np.isfinite(threshold)
        labels[valid] = (hazard[valid] >= threshold).astype(np.int8)
    return labels,thresholds


def _baseline_contiguous(incidence: np.ndarray, labels: np.ndarray, config: dict,
                         weekday_index: np.ndarray | None = None,
                         holiday_state: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """MF2.2 / RS8: local non-exposed baseline within one contiguous segment.

    The smallest admissible half-width is selected subject to the active
    matching rules, minimum donor count, and the requirement for at least one
    eligible donor before and after the target day.  With dynamic enlargement
    disabled, only ``initial_half_width`` is tested.
    """
    incidence=np.asarray(incidence,dtype=float); labels=np.asarray(labels)
    n=len(incidence); result=np.full(n,np.nan); widths=np.full(n,-1,dtype=np.int32); counts=np.zeros(n,dtype=np.int32)
    if n==0: return result,widths,counts
    initial=int(config["initial_half_width"]); maximum=int(config["maximum_half_width"] if config["dynamic_enlargement"] else initial)
    required=int(config.get("minimum_eligible_dates") or 0)
    dow_match=bool(config.get("day_of_week_matching",False)); hol_match=bool(config.get("holiday_matching",False))
    # Preserve the original vectorized kernel for the reference baseline.
    # The generalized matching/support path below is used only when Analysis 4
    # activates a criterion that actually changes donor eligibility.
    if not dow_match and not hol_match and config.get("minimum_eligible_dates") is None:
        valid=np.isfinite(incidence)&(labels==0)
        idx=np.arange(n)
        prev=np.maximum.accumulate(np.where(valid,idx,-n-1)); prev=np.r_[-n-1,prev[:-1]]
        nxt=np.minimum.accumulate(np.where(valid,idx,2*n+1)[::-1])[::-1]; nxt=np.r_[nxt[1:],2*n+1]
        maximum_fast=int(config["maximum_half_width"] if config["dynamic_enlargement"] else initial)
        widths_fast=np.maximum(initial,np.maximum(idx-prev,nxt-idx))
        widths_fast[(widths_fast>maximum_fast)|~np.isfinite(incidence)|(labels<0)]=-1
        donors_arr=np.where(valid,incidence,np.nan)
        for w in np.unique(widths_fast[widths_fast>=0]):
            w=int(w); rows=np.flatnonzero(widths_fast==w)
            padded=np.pad(donors_arr,(w,w),constant_values=np.nan)
            all_windows=sliding_window_view(padded,2*w+1)
            batch_rows = max(1, min(1024, (16 * 1024**2) // max(1, 32 * (2*w+1))))
            for start in range(0,len(rows),batch_rows):
                ix=rows[start:start+batch_rows]; a=all_windows[ix].copy(); a[:,w]=np.nan
                counts[ix]=np.sum(np.isfinite(a),axis=1); result[ix]=_row_nanmedian(a)
        return result,widths_fast.astype(np.int32),counts
    if dow_match:
        if weekday_index is None or len(weekday_index)!=n: raise ValueError("weekday_index is required for day-of-week baseline matching")
        weekday_index=np.asarray(weekday_index,dtype=np.int8)
    else: weekday_index=np.zeros(n,dtype=np.int8)
    if hol_match:
        if holiday_state is None or len(holiday_state)!=n: raise ValueError("holiday_state is required for holiday baseline matching")
        holiday_state=np.asarray(holiday_state,dtype=np.uint8)
    else: holiday_state=np.zeros(n,dtype=np.uint8)
    target_valid=np.isfinite(incidence)&(labels>=0)
    base_donor=np.isfinite(incidence)&(labels==0)
    idx=np.arange(n)
    # Group by the active matching key so donor eligibility is target-specific
    # without a quadratic target-by-donor mask.
    keys=np.zeros(n,dtype=np.int16)
    if dow_match: keys += weekday_index.astype(np.int16)
    if hol_match: keys += 8*holiday_state.astype(np.int16)
    for key in np.unique(keys[target_valid]):
        targets=np.flatnonzero(target_valid&(keys==key))
        if targets.size==0: continue
        donors=base_donor&(keys==key)
        prefix=np.r_[0,np.cumsum(donors.astype(np.int64))]
        pending=np.ones(targets.size,dtype=bool)
        for w in range(initial,maximum+1):
            if not np.any(pending): break
            t=targets[pending]
            left=np.maximum(0,t-w); right=np.minimum(n,t+w+1)
            total=prefix[right]-prefix[left]-donors[t].astype(np.int64)
            before=(prefix[t]-prefix[left])>0
            after=(prefix[right]-prefix[t+1])>0
            ok=before&after&(total>=required)
            if not np.any(ok): continue
            chosen=t[ok]
            widths[chosen]=w; counts[chosen]=total[ok].astype(np.int32)
            # Compute medians only for newly accepted targets.  Windows are
            # small (baseline half-widths) and this avoids materializing an
            # n-by-window array when matching varies by target.
            for i in chosen:
                lo=max(0,i-w); hi=min(n,i+w+1)
                mask=donors[lo:hi].copy()
                rel=i-lo
                if 0<=rel<len(mask): mask[rel]=False
                vals=incidence[lo:hi][mask]
                result[i]=float(np.median(vals)) if vals.size else np.nan
            pos=np.flatnonzero(pending); pending[pos[ok]]=False
    return result,widths,counts


def baseline(incidence: np.ndarray, labels: np.ndarray, config: dict,
             segment_index: np.ndarray | None = None,
             weekday_index: np.ndarray | None = None,
             holiday_state: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """MF2.2 / RS8 baseline with strict calendar-gap boundaries."""
    n=len(incidence)
    if weekday_index is not None and len(weekday_index)!=n: raise ValueError("weekday_index length mismatch")
    if holiday_state is not None and len(holiday_state)!=n: raise ValueError("holiday_state length mismatch")
    if segment_index is None:
        return _baseline_contiguous(incidence,labels,config,weekday_index,holiday_state)
    result=np.full(n,np.nan); widths=np.full(n,-1,dtype=np.int32); counts=np.zeros(n,dtype=np.int32)
    for a,b in segment_slices(segment_index,n):
        wd=None if weekday_index is None else np.asarray(weekday_index[a:b])
        hs=None if holiday_state is None else np.asarray(holiday_state[a:b])
        rb,wb,cb=_baseline_contiguous(np.asarray(incidence[a:b]),np.asarray(labels[a:b]),config,wd,hs)
        result[a:b],widths[a:b],counts[a:b]=rb,wb,cb
    return result,widths,counts


def relevance(local: np.ndarray, labels: np.ndarray, year_index: np.ndarray,
              q: float) -> tuple[np.ndarray,np.ndarray]:
    """MF2.6: zero weight for discordance, missing hazard, and local threshold ties."""
    _, thresholds = exposure(local,year_index,q)
    margin = np.where(labels == 1,1.0,-1.0)*(local-thresholds[year_index])
    raw = np.maximum(margin,0.0)
    raw[~np.isfinite(raw) | (labels<0)] = 0.0
    return raw,thresholds


def _prefix(x: np.ndarray) -> np.ndarray:
    # Identical cumulative-sum order, without np.r_'s indexing machinery.
    result = np.empty(len(x) + 1, dtype=np.float64)
    result[0] = 0.0
    np.cumsum(x, dtype=np.float64, out=result[1:])
    return result


def _dependence_contiguous(values: np.ndarray, c: dict) -> np.ndarray:
    """MF3.2: exact local pairwise Pearson sums in O(Kmax * days), not O(K*W*days).

    Prefix sufficient statistics produce every shifted local ACF interval.
    The estimator, detrending, minimum support and persistence rule are unchanged.
    """
    n = len(values)
    trend = edge_fill(rolling_median(values,c["trend_window_days"],20))
    residual = values-trend
    valid = np.isfinite(residual)
    if not np.any(valid):
        return np.full(n,np.nan)
    # A common translation leaves every Pearson correlation unchanged and
    # reduces cancellation in the pairwise moment computations.
    residual = residual-float(np.mean(residual[valid]))
    width = min(c["local_acf_interval_days"],n)
    starts = np.clip(np.arange(n)-(width-1)//2,0,n-width)
    ends = starts+width
    pv = _prefix(valid)
    count = pv[ends]-pv[starts]
    eligible = count >= c["minimum_local_observations"]
    if not np.any(eligible):
        return np.full(n, np.nan)
    critical = np.zeros(n)
    critical[eligible] = 2*np.sqrt(np.log10(count[eligible])/count[eligible])
    K,h = c["maximum_acf_lag_days"],c["negligible_lag_persistence"]
    raw = np.full(n,float(K))
    streak = np.zeros(n,dtype=np.int16)
    found, any_correlation = np.zeros(n,dtype=bool),np.zeros(n,dtype=bool)
    for k in range(1,K+1):
        if k >= width:
            break
        x,y = residual[:-k],residual[k:]
        pair = np.isfinite(x)&np.isfinite(y)
        x,y = np.where(pair,x,0.0),np.where(pair,y,0.0)
        def local_sum(a):
            p = _prefix(a)
            return p[ends-k]-p[starts]
        m,sx,sy,sxx,syy,sxy = (local_sum(a) for a in (pair,x,y,x*x,y*y,x*y))
        safe = np.maximum(m,1)
        vx,vy = sxx-sx*sx/safe,syy-sy*sy/safe
        ok = eligible & (m>=2) & (vx>0) & (vy>0)
        corr = np.full(n,np.nan)
        corr[ok] = np.clip((sxy[ok]-sx[ok]*sy[ok]/m[ok])/np.sqrt(vx[ok]*vy[ok]),-1,1)
        any_correlation |= ok
        negligible = ok & (np.abs(corr)<=critical)
        streak = np.where(negligible,streak+1,0)
        new = (streak>=h)&~found
        raw[new] = k-h
        found |= new
        # MF3.2: only the FIRST persistent negligible-lag run defines a radius.
        # Once every eligible target has that run, later lags cannot change it.
        if np.all(found | ~eligible):
            break
    raw[~eligible | ~any_correlation] = np.nan
    smooth = edge_fill(rolling_median(raw,c["smoothing_window_days"],5))
    return np.clip(np.floor(smooth+0.5),0,K)  # frozen round-half-up convention



def dependence(values: np.ndarray, c: dict, segment_index: np.ndarray | None = None) -> np.ndarray:
    """MF3.2 dependence field, evaluated independently across calendar gaps."""
    if segment_index is None:
        return _dependence_contiguous(values,c)
    result = np.full(len(values),np.nan)
    for a,b in segment_slices(segment_index,len(values)):
        result[a:b] = _dependence_contiguous(np.asarray(values[a:b]),c)
    return result

def histogram_quantile(counts: np.ndarray, p: float) -> float:
    """HF7 on integer radii without accumulating BSA x date arrays in RAM."""
    n = int(np.sum(counts))
    if n == 0:
        return float("nan")
    h = (n-1)*p
    low, high = int(np.floor(h)),int(np.ceil(h))
    cum = np.cumsum(counts)
    a,b = np.searchsorted(cum,[low+1,high+1])
    return float(a+(h-low)*(b-a))


def episode_lengths(labels: np.ndarray, segment_index: np.ndarray | None = None) -> np.ndarray:
    """Lengths of exposed episodes without crossing retained-calendar gaps."""
    lengths: list[int] = []
    for a,b in segment_slices(segment_index,len(labels)):
        e = np.r_[False,np.asarray(labels[a:b])==1,False].astype(np.int8)
        edges = np.diff(e)
        lengths.extend((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).tolist())
    return np.asarray(lengths,dtype=float)


def episode_requirement(labels: np.ndarray, quantile: float, segment_index: np.ndarray | None = None) -> int:
    lengths = episode_lengths(labels, segment_index)
    return int(ceil(finite_quantile(lengths,quantile))) if len(lengths) else 1


def pooled_episode_requirement(label_series: list[np.ndarray], quantile: float,
                               segment_index: np.ndarray | None = None) -> int:
    """RS7.6: conservative episode requirement pooled across eligible BSAs."""
    chunks = [episode_lengths(labels, segment_index) for labels in label_series]
    chunks = [x for x in chunks if len(x)]
    if not chunks:
        return 1
    return int(ceil(finite_quantile(np.concatenate(chunks), quantile)))


def block_length_requirement(lag: int, d_local: float, d_territory: float, exposure_requirement: int, multiplier: float = 1.0) -> tuple[int | None, int | None]:
    """RS9.3 / MF3.4: annual raw and multiplier-adjusted MBB requirement.

    The reference realization uses ``multiplier=1`` and therefore reproduces
    the pre-RS implementation exactly. Non-finite dependence summaries make
    the block unavailable.
    """
    if not (np.isfinite(d_local) and np.isfinite(d_territory)):
        return None, None
    raw = max(int(lag) + 1, 1 + int(np.ceil(max(float(d_local), float(d_territory)))), int(exposure_requirement))
    if not np.isfinite(multiplier) or float(multiplier) <= 0:
        raise ValueError("final block-length multiplier must be finite and > 0")
    adopted = max(1, int(np.ceil(float(multiplier) * raw)))
    return raw, adopted

def _draw_fixed_choices(rng: Random, choices: list[int], count: int) -> np.ndarray:
    """AS7/MF3.4: identical MT19937 rejection draws with a constant upper bound.

    Hoist the fixed bit length out of Python's randrange loop. Every getrandbits
    request (including rejected values and power-of-two bounds) is unchanged.
    Custom generators or overrides retain the generic public randrange path.
    """
    n = len(choices)
    native = (
        type(rng) is Random
        and not any(name in rng.__dict__ for name in ("randrange", "getrandbits", "_randbelow"))
        and getattr(rng._randbelow, "__func__", None) is Random._randbelow_with_getrandbits
    )
    if native:
        bits = n.bit_length()
        getbits = rng.getrandbits
        def values():
            for _ in range(count):
                index = getbits(bits)
                while index >= n:
                    index = getbits(bits)
                yield choices[index]
        return np.fromiter(values(), dtype=np.intp, count=count)
    draw = rng.randrange
    return np.fromiter((choices[draw(n)] for _ in range(count)), dtype=np.intp, count=count)


def _moving_block_counts(n: int, block: int, rng: Random, out: np.ndarray,
                         segment_index: np.ndarray | None,
                         labels: np.ndarray | None) -> int:
    """MF3.4/RS7.7: batch MBB proposals while preserving every RNG call.

    Proposals are generated in exactly the original row/block order using
    Random.randrange. At most the number of still-needed draws is proposed,
    so rejecting proposals never consumes surplus random numbers at the end.
    """
    if not (0 < block < n):
        raise ValueError("Bootstrap block must be shorter than annual record")
    if labels is not None and (not np.any(labels == 1) or not np.any(labels == 0)):
        raise ValueError("Annual record must contain both exposure classes")
    starts = np.arange(n - block + 1, dtype=int)
    if segment_index is not None:
        seg = np.asarray(segment_index)
        if len(seg) != n:
            raise ValueError("segment_index length must match annual record")
        starts = starts[seg[starts] == seg[starts + block - 1]]
    if not len(starts):
        raise ValueError("No moving bootstrap block fits within a contiguous retained-date segment")
    # Python integer lookup avoids a NumPy scalar conversion for every block.
    valid_starts = starts.tolist()
    nblocks = ceil(n / block)
    offsets = np.arange(block)
    retries = written = invalid_streak = 0
    # The proposal workspace stays below approximately 8 MiB even for a
    # non-annual direct caller. Normal annual records use 64 proposals.
    batch_limit = max(1, min(64, (8 * 1024**2) // max(1, 32 * n)))
    while written < len(out):
        size = min(batch_limit, len(out) - written, 100000 - invalid_streak)
        chosen = _draw_fixed_choices(rng, valid_starts, size * nblocks).reshape(size, nblocks)
        indices = (chosen[:, :, None] + offsets).reshape(size, -1)[:, :n]
        flat = indices + (np.arange(size) * n)[:, None]
        counts = np.bincount(flat.ravel(), minlength=size * n).reshape(size, n)
        if labels is None:
            accepted = np.ones(size, dtype=bool)
        else:
            accepted = ((counts @ (labels == 1)) > 0) & ((counts @ (labels == 0)) > 0)
        # Keep the original consecutive-failure guard, not a batch-level rule.
        for good in accepted:
            invalid_streak = 0 if good else invalid_streak + 1
            if invalid_streak >= 100000:
                raise RuntimeError("100000 globally invalid MBB draws; cannot generate requested bootstrap")
        valid_count = int(np.sum(accepted))
        out[written:written + valid_count] = counts[accepted]
        written += valid_count
        retries += size - valid_count
    return retries


def draw_synchronized_counts(n: int, block: int, rng: Random, out: np.ndarray,
                             segment_index: np.ndarray | None = None) -> int:
    """RS7.7: synchronized chronological MBB with local validity downstream."""
    return _moving_block_counts(n, block, rng, out, segment_index, None)


def draw_counts(labels: np.ndarray, block: int, rng: Random, out: np.ndarray,
                segment_index: np.ndarray | None = None) -> int:
    """MF3.4: identical ordered accepted MBB draws and global redraw count."""
    return _moving_block_counts(len(labels), block, rng, out, segment_index, labels)


def prepare_weighted_delta(values: np.ndarray, raw: np.ndarray, labels: np.ndarray):
    """MF3.1: invariant stable sorting and tie bounds for one BSA/year."""
    keep = np.flatnonzero(np.isfinite(values)&np.isfinite(raw)&(raw>0)&(labels>=0))
    if not len(keep):
        return None
    keep = keep[np.argsort(values[keep],kind="stable")]
    x,w,e = values[keep],raw[keep],labels[keep]==1
    starts,ends = np.searchsorted(x,x,side="left"),np.searchsorted(x,x,side="right")
    separation = 0
    if np.any(e) and np.any(~e):
        if np.min(x[e]) > np.max(x[~e]):
            separation = 1
        elif np.max(x[e]) < np.min(x[~e]):
            separation = -1
    return keep, w, e, starts, ends, separation


def weighted_delta_prepared(plan, counts: np.ndarray) -> np.ndarray:
    """MF3.4: evaluate the original cumulative-weight formula on a plan."""
    if plan is None:
        return np.full(len(counts), np.nan)
    keep, w, e, starts, ends, separation = plan
    weights = counts[:,keep]*w[None,:]
    u,v = weights*e[None,:],weights*(~e)[None,:]
    su = u.sum(axis=1)
    # Match NumPy's original layout as well as summation order. Column-indexed
    # bootstrap weights are normally Fortran-contiguous, including their pad.
    order = "F" if v.flags.f_contiguous else "C"
    prefix = np.empty((len(counts), v.shape[1] + 1), dtype=v.dtype, order=order)
    prefix[:, 0] = 0.0
    np.cumsum(v, axis=1, out=prefix[:, 1:])
    sv = prefix[:,-1]  # consistent prefix total preserves exact all-tie zeros
    less = prefix[:,starts]
    greater = np.maximum(sv[:,None]-prefix[:,ends],0.0)
    positive = np.sum(u*less,axis=1)
    negative = np.sum(u*greater,axis=1)
    denominator = su*sv
    result = np.full(len(counts),np.nan)
    ok = (su>0)&(sv>0)
    result[ok] = np.clip((positive[ok]-negative[ok])/denominator[ok],-1,1)
    # Preserve exact algebraic endpoints, including bootstrap invalidity.
    if separation:
        result[ok] = float(separation)
    return result


def weighted_delta_batch(values: np.ndarray, raw: np.ndarray, labels: np.ndarray,
                         counts: np.ndarray) -> np.ndarray:
    """MF3.1/MF3.4: public one-shot interface to the same prepared kernel."""
    return weighted_delta_prepared(prepare_weighted_delta(values, raw, labels), counts)


def bootstrap_summary(original: float, draws: np.ndarray, requested: int,
                      maximum_invalid: float) -> dict:
    valid = draws[np.isfinite(draws)]
    invalid = 1-len(valid)/requested
    reason = None
    if not np.isfinite(original):
        reason = "NO_LOCALLY_CONCORDANT_CONTRAST"
    elif invalid > maximum_invalid+1e-12:
        reason = "EXCESSIVE_INVALID_BOOTSTRAP"
    elif len(valid)<2:
        reason = "INSUFFICIENT_VALID_BOOTSTRAP"
    variance = float(np.var(valid,ddof=1)) if len(valid)>=2 else None
    if len(valid) and np.all(valid==valid[0]):
        variance = 0.0 if len(valid)>=2 else None
    if reason is None and variance == 0:
        reason = "DEGENERATE_BOOTSTRAP_VARIANCE"
    return {"rc": original,"bootstrap_valid":len(valid),"bootstrap_requested":requested,
            "invalid_proportion":invalid,"variance":variance,
            "ci_low":finite_quantile(valid,.025),"ci_high":finite_quantile(valid,.975),
            "directional_stability":float(np.mean(np.sign(valid)==np.sign(original))) if len(valid) and np.isfinite(original) and original!=0 else None,
            "bootstrap_positive_share":float(np.mean(valid>0)) if len(valid) else None,
            "bootstrap_negative_share":float(np.mean(valid<0)) if len(valid) else None,
            "bootstrap_zero_share":float(np.mean(valid==0)) if len(valid) else None,
            "eligible":reason is None,"reason":reason}


@lru_cache(maxsize=128)
def t975(df: int) -> float:
    return float(stdtrit(df,.975))


def pool(rows: list[dict]) -> dict:
    """MF4.1-4.2, MF5.7-5.8: REML + modified Hartung-Knapp; no variance floor."""
    rows = [r for r in rows if r.get("eligible") and r.get("variance") is not None and r["variance"]>0]
    K = len(rows)
    base = {"contributing_years":K,"year_ids":[r["year"] for r in rows],"eligible":K>=2,
            "single_year_rc":rows[0]["rc"] if K==1 else None,
            "reason":None if K>=2 else "INSUFFICIENT_VALID_YEARS"}
    if K<2:
        return {**base,"pooled_rc":None,"ci_low":None,"ci_high":None,"tau_squared":None,
                "i_squared_percent":None,"prediction_low":None,"prediction_high":None}
    effects = np.array([r["rc"] for r in rows])
    variances = np.array([r["variance"] for r in rows])
    scale = max(float(np.var(effects,ddof=1)),float(variances.max()),np.finfo(float).tiny)
    normalized_variances = variances/scale
    normalized_effects = (effects-float(np.mean(effects)))/np.sqrt(scale)
    def objective(z):
        s = normalized_variances+z
        w = 1/s
        mu = np.dot(w,normalized_effects)/w.sum()
        return float(np.log(s).sum()+np.log(w.sum())+np.dot(w,(normalized_effects-mu)**2))
    # Coarse logarithmic bracketing explicitly checks the zero boundary and
    # all detected local minima rather than presuming global unimodality.
    grid = np.r_[0.0,np.geomspace(max(1e-18,float(normalized_variances.min())*1e-8),16.0,65)]
    s = normalized_variances[None,:]+grid[:,None]
    w = 1/s
    mu = (w*normalized_effects).sum(axis=1)/w.sum(axis=1)
    objectives = np.log(s).sum(axis=1)+np.log(w.sum(axis=1))+(w*(normalized_effects-mu[:,None])**2).sum(axis=1)
    candidates = [(float(objectives[0]),0.0)]
    for j in range(1,len(grid)-1):
        if objectives[j]<=objectives[j-1] and objectives[j]<=objectives[j+1]:
            result = minimize_scalar(objective,bounds=(float(grid[j-1]),float(grid[j+1])),method="bounded",options={"xatol":1e-13})
            if not result.success:
                raise ArithmeticError("REML minimization failed")
            candidates.append((float(result.fun),float(result.x)))
    _,tau = min(candidates)
    tau *= scale
    weights = 1/(variances+tau)
    rc = float(np.dot(weights,effects)/weights.sum())
    hk_q = float(np.dot(weights,(effects-rc)**2)/(K-1))
    se = float(np.sqrt(max(1.0,hk_q)/weights.sum()))
    half = t975(K-1)*se
    fixedw = 1/variances
    fixed = float(np.dot(fixedw,effects)/fixedw.sum())
    Q = float(np.dot(fixedw,(effects-fixed)**2))
    i2 = max(0.0,(Q-(K-1))/Q)*100 if Q>0 else 0.0
    phalf = t975(K-2)*np.sqrt(tau+se*se) if K>=3 else None
    return {**base,"pooled_rc":rc,"ci_low":rc-half,"ci_high":rc+half,"tau_squared":tau,
            "i_squared_percent":i2,"cochran_q":Q,"hk_q":hk_q,"hk_standard_error":se,
            "prediction_low":rc-phalf if phalf is not None else None,
            "prediction_high":rc+phalf if phalf is not None else None}

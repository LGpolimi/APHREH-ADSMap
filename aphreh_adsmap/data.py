"""Streaming source ingestion and intersection-area-weighted canonical data (MF1-2).

The canonical time axis is the exact intersection of hazard and outcome dates.
It may contain non-contiguous years or shorter gaps. Calendar-sensitive kernels
receive contiguous-segment identifiers so a missing date is never silently
compressed into an adjacent day.
"""
from __future__ import annotations
import csv
import json
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator
import numpy as np
from .storage import atomic_json, mapped, sha256_file
from .performance import memory_budget, file_signature


class SourceError(ValueError):
    pass


def parse_date(text: str) -> date:
    text = text.strip()
    # Explicit DATE_STR rule retained for compatibility: six digits are 20yy-mm-dd.
    if len(text) == 6 and text.isdigit():
        return date(2000 + int(text[:2]), int(text[2:4]), int(text[4:]))
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    return date.fromisoformat(text)


def csv_headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.reader(f), None)
    if not row or len(set(row)) != len(row):
        raise SourceError(f"Missing or duplicate headers: {path}")
    return row


def csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise SourceError(f"Missing or duplicate headers: {path}")
        for n, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise SourceError(f"Malformed CSV row {n}: {path}")
            yield row


def number(value: str, *, nonnegative: bool = False) -> float:
    if value.strip().lower() in ("", "na", "nan", "null", "none"):
        return float("nan")
    x = float(value)
    if not math.isfinite(x) or (nonnegative and x < 0):
        raise SourceError(f"Invalid numeric source value {value!r}")
    return x


def binary_state(value: str) -> int:
    """Parse a date-level binary state used by optional holiday matching."""
    text = value.strip().lower()
    if text in ("1", "true", "yes", "y", "holiday"):
        return 1
    if text in ("0", "false", "no", "n", "nonholiday", "non-holiday"):
        return 0
    raise SourceError(f"Invalid binary calendar state {value!r}")


def inspect_dates(path: Path, field: str) -> set[date]:
    dates: set[date] = set()
    for row in csv_rows(path):
        if field not in row:
            raise SourceError(f"Missing date column {field!r} in {path}")
        d = parse_date(row[field])
        if d in dates:
            raise SourceError(f"Duplicate source date {d}: {path}")
        dates.add(d)
    if not dates:
        raise SourceError(f"No dates: {path}")
    return dates


def source_path(data_dir: str | Path, configured: str) -> Path:
    p = Path(configured).expanduser()
    return p.resolve() if p.is_absolute() else (Path(data_dir).expanduser().resolve() / p).resolve()


def contiguous_segments(dates: list[date]) -> tuple[np.ndarray, list[dict]]:
    if not dates:
        return np.array([], dtype=np.int32), []
    ids = np.zeros(len(dates), dtype=np.int32)
    segment = 0
    start = 0
    records = []
    for i in range(1, len(dates)):
        if dates[i] != dates[i - 1] + timedelta(days=1):
            records.append({
                "segment_id": segment,
                "start": dates[start].isoformat(),
                "end": dates[i - 1].isoformat(),
                "date_count": i - start,
            })
            segment += 1
            start = i
        ids[i] = segment
    records.append({
        "segment_id": segment,
        "start": dates[start].isoformat(),
        "end": dates[-1].isoformat(),
        "date_count": len(dates) - start,
    })
    return ids, records


def _require_columns(path: Path, headers: list[str], fields: list[str]) -> None:
    missing = [v for v in fields if v not in headers]
    if missing:
        raise SourceError(f"Missing configured columns in {path.name}: {', '.join(missing)}")


def _population_value(row: dict[str, str], field: str) -> float:
    if field not in row:
        return float("nan")
    return number(row[field], nonnegative=True)


def _select_population(row: dict[str, str], mapping: dict[str, str], target_year: int, policy: dict) -> tuple[float, int | None]:
    mode = policy["mode"]
    mapped = sorted((int(y), field) for y, field in mapping.items())
    if mode == "STATIC_EXPLICIT":
        source_year = int(policy["source_year"])
        field = dict(mapped).get(source_year)
        value = _population_value(row, field) if field else float("nan")
        return value, source_year if math.isfinite(value) else None
    if mode == "ANNUAL_STRICT":
        field = dict(mapped).get(target_year)
        value = _population_value(row, field) if field else float("nan")
        return value, target_year if math.isfinite(value) else None
    # ANNUAL_NEAREST: choose among finite positive values for this BSA. Ties in
    # absolute distance use the more recent source year (larger calendar year).
    candidates = []
    for source_year, field in mapped:
        value = _population_value(row, field)
        if math.isfinite(value) and value > 0:
            candidates.append((abs(source_year - target_year), -source_year, source_year, value))
    if not candidates:
        return float("nan"), None
    _, _, source_year, value = min(candidates)
    return value, source_year




def project_hazard_field(
    *,
    source_file: str | Path,
    crosswalk_file: str | Path,
    date_field: str,
    crosswalk_bsa_field: str,
    grid_id_field: str,
    intersection_area_field: str,
    dates: list[date],
    bsa_ids: list[str],
    raw_output: str | Path,
    bsa_output: str | Path,
    territory_output: str | Path,
    spatial_db_output: str | Path | None = None,
) -> dict:
    """Project a standard APHREH hazard-field CSV to BSA and territory scales.

    This is the authoritative spatial projection used by both the primary hazard
    and candidate-confounder hazard fields. The source format is the standard
    APHREH wide hazard table: one date column plus one column per source grid
    cell. Projection uses actual intersection areas, sums duplicate BSA-grid
    intersections, and applies strict contributor completeness independently at
    BSA and territory scales.
    """
    source_file = Path(source_file)
    crosswalk_file = Path(crosswalk_file)
    raw_output = Path(raw_output)
    bsa_output = Path(bsa_output)
    territory_output = Path(territory_output)
    headers = csv_headers(source_file)
    _require_columns(source_file, headers, [date_field])
    cross_headers = csv_headers(crosswalk_file)
    _require_columns(
        crosswalk_file, cross_headers,
        [crosswalk_bsa_field, grid_id_field, intersection_area_field],
    )
    # Inspect the complete source, not only canonical dates, so duplicate dates
    # fail exactly as they do for the primary hazard.
    inspect_dates(source_file, date_field)

    grids = [v for v in headers if v != date_field]
    if not grids:
        raise SourceError("Hazard table has no grid-value columns")
    bi = {v: i for i, v in enumerate(bsa_ids)}
    gi = {v: i for i, v in enumerate(grids)}
    date_index = {d: i for i, d in enumerate(dates)}
    B, G, n = len(bsa_ids), len(grids), len(dates)

    pair_areas: dict[tuple[int, int], float] = {}
    areas = np.zeros(B, dtype=float)
    territory_weights = np.zeros(G, dtype=float)
    n_intersections = 0
    raw_rows: list[tuple[str, str, float]] = []
    for row in csv_rows(crosswalk_file):
        b = row[crosswalk_bsa_field].strip()
        g = row[grid_id_field].strip()
        if b not in bi:
            continue
        if g not in gi:
            raise SourceError(f"Crosswalk grid ID {g!r} is absent from hazard data")
        a = number(row[intersection_area_field], nonnegative=True)
        if not math.isfinite(a) or a <= 0:
            raise SourceError("Intersection areas must be finite and strictly positive")
        ib, ig = bi[b], gi[g]
        pair_areas[(ib, ig)] = pair_areas.get((ib, ig), 0.0) + float(a)
        areas[ib] += float(a)
        territory_weights[ig] += float(a)
        raw_rows.append((b, g, float(a)))
        n_intersections += 1
    if np.any(areas <= 0):
        missing = [bsa_ids[i] for i in np.flatnonzero(areas <= 0)]
        raise SourceError("Every BSA must have positive intersection area; missing: " + ", ".join(missing[:20]))

    if spatial_db_output is not None:
        spatial_db_output = Path(spatial_db_output)
        spatial = sqlite3.connect(spatial_db_output)
        try:
            spatial.execute("DROP TABLE IF EXISTS intersections")
            spatial.execute("CREATE TABLE intersections(b TEXT,g TEXT,a REAL)")
            spatial.executemany("INSERT INTO intersections VALUES(?,?,?)", raw_rows)
            spatial.execute("CREATE INDEX IF NOT EXISTS ix_b ON intersections(b)")
            spatial.commit()
        finally:
            spatial.close()

    with mapped(raw_output, shape=(G, n)) as raw:
        raw[:] = np.nan
        for row in csv_rows(source_file):
            d = parse_date(row[date_field])
            idx = date_index.get(d)
            if idx is not None:
                raw[:, idx] = [number(row[g]) for g in grids]

    def aggregate(raw: np.ndarray, parts):
        numerator = np.zeros(n, dtype=float)
        complete = np.ones(n, dtype=bool)
        denominator = 0.0
        for ig, area in parts:
            x = np.asarray(raw[ig], dtype=float)
            complete &= np.isfinite(x)
            numerator += np.where(np.isfinite(x), x, 0.0) * float(area)
            denominator += float(area)
        if denominator <= 0:
            raise SourceError("No positive spatial aggregation weight")
        result = numerator / denominator
        result[~complete] = np.nan
        return result

    by_bsa: list[list[tuple[int, float]]] = [[] for _ in range(B)]
    for (ib, ig), area in sorted(pair_areas.items()):
        by_bsa[ib].append((ig, area))
    with mapped(raw_output) as raw, \
         mapped(bsa_output, shape=(B, n)) as local, \
         mapped(territory_output, shape=(n,)) as territory:
        for ib in range(B):
            local[ib] = aggregate(raw, by_bsa[ib])
        territory[:] = aggregate(
            raw, ((ig, float(a)) for ig, a in enumerate(territory_weights) if a > 0)
        )

    return {
        "grid_ids": grids,
        "grid_count": G,
        "intersection_count": n_intersections,
        "bsa_intersection_areas": areas,
        "territory_grid_weights": territory_weights,
        "missing_contributor_policy": "REQUIRE_ALL_CONTRIBUTORS",
        "projection_rule": "INTERSECTION_AREA_WEIGHTED_MEAN_ACTUAL_AREAS",
    }


def build_canonical(c: dict, root: Path, log=print) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    f = c["sources"]
    paths = {k: source_path(c["data_dir"], f[k]) for k in ("hazard_file", "outcome_file", "population_file", "crosswalk_file")}
    if f.get("holiday_calendar_file"):
        paths["holiday_calendar_file"] = source_path(c["data_dir"], f["holiday_calendar_file"])
    for path in paths.values():
        if not path.is_file():
            raise SourceError(f"Required input file does not exist: {path}")

    hheaders = csv_headers(paths["hazard_file"])
    oheaders = csv_headers(paths["outcome_file"])
    pheaders = csv_headers(paths["population_file"])
    cheaders = csv_headers(paths["crosswalk_file"])
    holiday_headers = csv_headers(paths["holiday_calendar_file"]) if "holiday_calendar_file" in paths else None
    _require_columns(paths["hazard_file"], hheaders, [f["hazard_date_field"]])
    _require_columns(paths["outcome_file"], oheaders, [f["outcome_date_field"]])
    _require_columns(paths["population_file"], pheaders,
                     [f["bsa_id_field"], f["bsa_area_field"], *f["population_fields"].values()])
    _require_columns(paths["crosswalk_file"], cheaders,
                     [f["crosswalk_bsa_field"], f["grid_id_field"], f["intersection_area_field"]])
    if holiday_headers is not None:
        holiday_required = [f["holiday_date_field"]]
        if f.get("holiday_state_field"):
            holiday_required.append(f["holiday_state_field"])
        _require_columns(paths["holiday_calendar_file"], holiday_headers, holiday_required)

    hd = inspect_dates(paths["hazard_file"], f["hazard_date_field"])
    od = inspect_dates(paths["outcome_file"], f["outcome_date_field"])
    dates = sorted(hd & od)
    if not dates:
        raise SourceError("Hazard and outcome sources have no matching dates")
    date_index = {d: i for i, d in enumerate(dates)}
    years = sorted({d.year for d in dates})
    segment_index, segment_records = contiguous_segments(dates)

    # Optional date-level holiday calendar. When no state column is configured,
    # every listed date is a holiday and all other canonical dates are ordinary
    # days. With an explicit state column, listed rows define 0/1 states and
    # unlisted dates remain non-holidays.
    holiday_state = np.zeros(len(dates), dtype=np.uint8)
    if "holiday_calendar_file" in paths:
        seen_holidays: set[date] = set()
        for row in csv_rows(paths["holiday_calendar_file"]):
            d = parse_date(row[f["holiday_date_field"]])
            if d in seen_holidays:
                raise SourceError(f"Duplicate holiday-calendar date {d}: {paths['holiday_calendar_file']}")
            seen_holidays.add(d)
            if d in date_index:
                holiday_state[date_index[d]] = (
                    binary_state(row[f["holiday_state_field"]]) if f.get("holiday_state_field") else 1
                )
    np.save(root / "holiday.npy", holiday_state, allow_pickle=False)

    bsas: dict[str, dict] = {}
    for row in csv_rows(paths["population_file"]):
        b = row[f["bsa_id_field"]].strip()
        if not b or b in bsas:
            raise SourceError(f"Missing/duplicate BSA identifier {b!r}")
        bsas[b] = row
    ids = sorted(bsas, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    if not ids:
        raise SourceError("Population table has no BSAs")

    grids = [v for v in hheaders if v != f["hazard_date_field"]]
    outcome_columns = [v for v in oheaders if v != f["outcome_date_field"]]
    if not grids:
        raise SourceError("Hazard table has no grid-value columns")
    if any(b not in outcome_columns for b in ids):
        missing = [b for b in ids if b not in outcome_columns]
        raise SourceError("Outcome columns must contain every population-table BSA ID; missing: " + ", ".join(missing[:20]))

    n, B = len(dates), len(ids)
    projection = project_hazard_field(
        source_file=paths["hazard_file"],
        crosswalk_file=paths["crosswalk_file"],
        date_field=f["hazard_date_field"],
        crosswalk_bsa_field=f["crosswalk_bsa_field"],
        grid_id_field=f["grid_id_field"],
        intersection_area_field=f["intersection_area_field"],
        dates=dates,
        bsa_ids=ids,
        raw_output=root / "grid_hazard.npy",
        bsa_output=root / "hazard.npy",
        territory_output=root / "territory.npy",
        spatial_db_output=root / "spatial.sqlite",
    )
    grids = list(projection["grid_ids"])
    G = int(projection["grid_count"])
    areas = np.asarray(projection["bsa_intersection_areas"], dtype=float)
    n_intersections = int(projection["intersection_count"])

    with mapped(root / "outcome.npy", shape=(B, n)) as out:
        out[:] = np.nan
        for row in csv_rows(paths["outcome_file"]):
            d = parse_date(row[f["outcome_date_field"]])
            if d not in date_index:
                continue
            vals = [number(row[b], nonnegative=True) for b in ids]
            if any(math.isfinite(v) and not v.is_integer() for v in vals):
                raise SourceError("EVENT_COUNTS source contains non-integer events")
            out[:, date_index[d]] = vals

    pop_source_summary: dict[str, set[int]] = {str(y): set() for y in years}
    with mapped(root / "population.npy", shape=(B, len(years))) as pop, \
         mapped(root / "population_source_year.npy", shape=(B, len(years)), dtype="int32") as pop_year:
        pop[:] = np.nan
        pop_year[:] = -1
        for ib, b in enumerate(ids):
            for iy, y in enumerate(years):
                p, source_year = _select_population(bsas[b], f["population_fields"], y, c["population_policy"])
                if math.isfinite(p) and p <= 0:
                    raise SourceError(f"Non-positive population: BSA {b}, analysis year {y}")
                pop[ib, iy] = p
                if source_year is not None:
                    pop_year[ib, iy] = int(source_year)
                    pop_source_summary[str(y)].add(int(source_year))
            if not np.any(np.isfinite(pop[ib]) & (pop[ib] > 0)):
                raise SourceError(f"No valid mapped population value is available for BSA {b}")

    coverage = []
    for i, b in enumerate(ids):
        b_area = number(bsas[b][f["bsa_area_field"]], nonnegative=True)
        coverage.append({"bsa_id": b, "intersection_area": float(areas[i]), "bsa_area": b_area,
                         "coverage_ratio": float(areas[i] / b_area) if b_area > 0 else None})

    meta = {
        "bsa_ids": ids,
        "years": years,
        "dates": [d.isoformat() for d in dates],
        "actual_start": dates[0].isoformat(),
        "actual_end": dates[-1].isoformat(),
        "analysis_period": {"mode": "COMMON_DATES"},
        "date_count": n,
        "bsa_count": B,
        "grid_count": G,
        "intersection_count": n_intersections,
        "contiguous_segment_count": len(segment_records),
        "contiguous_segments": segment_records,
        "hazard_only_date_count": len(hd - od),
        "outcome_only_date_count": len(od - hd),
        "included_years": years,
        "population_policy": c["population_policy"],
        "population_source_years_by_analysis_year": {k: sorted(v) for k, v in pop_source_summary.items()},
        "spatial_coverage": coverage,
        "source_hashes": {k: sha256_file(p) for k, p in paths.items()},
        "source_fields": {
            "hazard_date_field": f["hazard_date_field"],
            "outcome_date_field": f["outcome_date_field"],
            "bsa_id_field": f["bsa_id_field"],
            "bsa_area_field": f["bsa_area_field"],
            "population_fields": f["population_fields"],
            "crosswalk_bsa_field": f["crosswalk_bsa_field"],
            "grid_id_field": f["grid_id_field"],
            "intersection_area_field": f["intersection_area_field"],
            "holiday_date_field": f.get("holiday_date_field"),
            "holiday_state_field": f.get("holiday_state_field"),
        },
        "holiday_calendar_configured": bool(f.get("holiday_calendar_file")),
        "holiday_date_count": int(np.sum(holiday_state == 1)),
        "source_date_format": "AUTO: ISO YYYY-MM-DD, YYYYMMDD, or YYMMDD interpreted as 20YYMMDD",
        "missing_grid_policy": "REQUIRE_ALL_CONTRIBUTORS",
        "outcome_semantics": "EVENT_COUNTS",
        "gap_policy": "CALENDAR_GAPS_SPLIT_CONTIGUOUS_SEGMENTS; EDGE_WINDOWS_REQUIRING_MISSING_DATES_ARE_UNAVAILABLE",
    }
    atomic_json(root / "metadata.json", meta)
    log(f"Canonical inputs: {B} BSAs, {G} hazard cells, {n} matching days, years {years}, contiguous segments={len(segment_records)}")
    return meta


class Data:
    def __init__(self, root: Path):
        self.root = root
        self.meta = json.loads((root / "metadata.json").read_text())
        self.ids = self.meta["bsa_ids"]
        self.years = self.meta["years"]
        self.dates = np.array(self.meta["dates"], dtype="datetime64[D]")
        self.year_index = np.array([self.years.index(int(d[:4])) for d in self.meta["dates"]])
        self.n, self.B = len(self.dates), len(self.ids)
        # Monday=0 ... Sunday=6, matching datetime.date.weekday().
        epoch_weekday = 3  # 1970-01-01 was Thursday.
        self.weekday_index = (self.dates.astype("datetime64[D]").astype(np.int64) + epoch_weekday) % 7
        holiday_path = root / "holiday.npy"
        if holiday_path.exists():
            with mapped(holiday_path) as h:
                self.holiday_state = np.array(h, dtype=np.uint8)
        else:
            self.holiday_state = np.zeros(self.n, dtype=np.uint8)
        self.segment_index = np.zeros(self.n, dtype=np.int32)
        if self.n:
            self.segment_index[1:] = np.cumsum(np.diff(self.dates).astype("timedelta64[D]").astype(int) != 1)
        self.segment_starts = np.empty(self.n, dtype=np.int32)
        self.segment_stops = np.empty(self.n, dtype=np.int32)
        for sid in np.unique(self.segment_index):
            ix = np.flatnonzero(self.segment_index == sid)
            self.segment_starts[ix] = ix[0]
            self.segment_stops[ix] = ix[-1] + 1
        with mapped(root / "population.npy") as p:
            self.population = np.array(p)
        source_year_path = root / "population_source_year.npy"
        if source_year_path.exists():
            with mapped(source_year_path) as py:
                self.population_source_year = np.array(py, dtype=int)
        else:
            self.population_source_year = np.full(self.population.shape, -1, dtype=int)

    def series(self, name: str, b: int | None = None, path: Path | None = None) -> np.ndarray:
        source = path or self.root / f"{name}.npy"
        if path is None:
            budget = memory_budget()
            signature = file_signature(source)
            key = ("canonical_array", signature)
            a = budget.get(key)
            if a is None and budget.can_cache(signature[3]):
                a = np.load(source, allow_pickle=False)
                budget.put(key, a)
            if a is not None:
                # Preserve the original API: callers own a writable copy.
                return np.array(a if b is None else a[b])
        with mapped(source) as a:
            return np.array(a if b is None else a[b])

    def daily_population(self, b: int) -> np.ndarray:
        return self.population[b, self.year_index]

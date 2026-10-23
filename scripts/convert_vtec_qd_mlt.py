#!/usr/bin/env python3
"""Convert quiet monthly median VTEC pickles from geographic/LT to QD/MLT.

Each input pickle is an ``atec`` object with geographic latitude/longitude and
local-time bins. This script:

1. Maps the geographic grid to quasi-dipole (QD) latitude/longitude (Apex, 350 km)
2. Converts each (glon, LT) sample to magnetic local time (MLT) via UT = LT - glon/15
3. Rebins VTEC onto a regular QD-latitude × MLT grid

Representative mid-month dates are used for the Apex/IGRF epoch.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import apexpy
import numpy as np


class atec:
    """Minimal stand-in so pickles saved from ``__main__.atec`` can load."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _AtecUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "atec":
            return atec
        return super().find_class(module, name)


def load_atec(path: Path) -> atec:
    with path.open("rb") as f:
        return _AtecUnpickler(f).load()


def parse_month_range(path: Path) -> tuple[datetime, datetime]:
    match = re.search(r"vtec_avg_(\d{8})_(\d{8})", path.name)
    if not match:
        raise ValueError(f"Cannot parse month range from {path.name}")
    start = datetime.strptime(match.group(1), "%Y%m%d")
    end = datetime.strptime(match.group(2), "%Y%m%d")
    return start, end


def mid_month_date(start: datetime, end: datetime) -> datetime:
    mid = start + (end - start) / 2
    return datetime(mid.year, mid.month, mid.day)


def hours_to_datetime(day: datetime, hours: float) -> datetime:
    hours = float(hours) % 24.0
    h = int(hours)
    m = int(round((hours - h) * 60.0))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    return datetime(day.year, day.month, day.day, h, m)


def geo_lt_to_qd_mlt_arrays(
    gdlat: np.ndarray,
    glon: np.ndarray,
    lt_hours: np.ndarray,
    apex: apexpy.Apex,
    height_km: float,
    epoch_day: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return QD lat, QD lon, and MLT arrays shaped (n_lat, n_lon, n_lt)."""
    lat_grid, lon_grid = np.meshgrid(gdlat, glon, indexing="ij")
    qdlat, qdlon = apex.geo2qd(lat_grid, lon_grid, height_km)

    n_lat, n_lon = qdlat.shape
    n_lt = lt_hours.size
    mlt = np.empty((n_lat, n_lon, n_lt), dtype=np.float64)
    day = datetime(epoch_day.year, epoch_day.month, epoch_day.day)

    for it, lt in enumerate(lt_hours):
        for ilon, lon in enumerate(glon):
            ut_hours = (float(lt) - float(lon) / 15.0) % 24.0
            dt = hours_to_datetime(day, ut_hours)
            mlt[:, ilon, it] = apex.mlon2mlt(qdlon[:, ilon], dt)

    qdlat_3d = np.broadcast_to(qdlat[:, :, None], mlt.shape).copy()
    qdlon_3d = np.broadcast_to(qdlon[:, :, None], mlt.shape).copy()
    return qdlat_3d, qdlon_3d, mlt


def rebin_2d(
    vtec: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-rebin VTEC onto 2D edges. Returns (median, count)."""
    valid = np.isfinite(vtec) & np.isfinite(y) & np.isfinite(x) & (vtec > 0)
    n_y = y_edges.size - 1
    n_x = x_edges.size - 1
    if not np.any(valid):
        return np.full((n_y, n_x), np.nan), np.zeros((n_y, n_x), dtype=np.int32)

    y_i = np.digitize(y[valid], y_edges) - 1
    x_i = np.digitize(x[valid], x_edges) - 1
    keep = (y_i >= 0) & (y_i < n_y) & (x_i >= 0) & (x_i < n_x)
    y_i = y_i[keep]
    x_i = x_i[keep]
    vals = vtec[valid][keep]

    count = np.zeros((n_y, n_x), dtype=np.int32)
    flat = y_i * n_x + x_i
    order = np.argsort(flat)
    flat_s = flat[order]
    vals_s = vals[order]
    median = np.full((n_y, n_x), np.nan, dtype=np.float64)

    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    ends = np.r_[starts[1:], flat_s.size]
    for start, end in zip(starts, ends):
        idx = int(flat_s[start])
        iy, ix = divmod(idx, n_x)
        chunk = vals_s[start:end]
        count[iy, ix] = chunk.size
        median[iy, ix] = np.median(chunk)
    return median, count


def convert_one(
    path: Path,
    out_dir: Path,
    qdlat_min: float,
    qdlat_max: float,
    qdlat_step: float,
    mlt_step: float,
    height_km: float,
    qdlon_step: float,
    target_mlt: float,
    mlt_window: float,
) -> tuple[Path, Path]:
    start, end = parse_month_range(path)
    epoch = mid_month_date(start, end)
    obj = load_atec(path)
    vtec = np.asarray(obj.vtec_all_median_lt, dtype=np.float64)
    gdlat = np.asarray(obj.gdlat, dtype=np.float64)
    glon = np.asarray(obj.glon, dtype=np.float64)
    lt_hours = np.asarray(obj.time, dtype=np.float64)

    apex = apexpy.Apex(date=epoch)
    qdlat, qdlon, mlt = geo_lt_to_qd_mlt_arrays(
        gdlat, glon, lt_hours, apex, height_km, epoch
    )

    qdlat_edges = np.arange(qdlat_min, qdlat_max + qdlat_step * 0.5, qdlat_step)
    mlt_edges = np.arange(0.0, 24.0 + mlt_step * 0.5, mlt_step)
    median, count = rebin_2d(vtec, qdlat, mlt, qdlat_edges, mlt_edges)

    qdlat_centers = 0.5 * (qdlat_edges[:-1] + qdlat_edges[1:])
    mlt_centers = 0.5 * (mlt_edges[:-1] + mlt_edges[1:])

    out_dir.mkdir(parents=True, exist_ok=True)
    month = start.strftime("%Y%m")
    out_path = out_dir / f"vtec_qd_mlt_{month}.npz"
    base_meta = {
        "source_file": path.name,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "apex_epoch": epoch.strftime("%Y-%m-%d"),
        "height_km": height_km,
        "input_field": "vtec_all_median_lt",
        "coordinate_note": (
            "QD latitude/longitude from apexpy geo2qd; "
            "MLT from mlon2mlt using UT = LT - glon/15"
        ),
    }
    meta_mlt = {
        **base_meta,
        "qdlat_min": qdlat_min,
        "qdlat_max": qdlat_max,
        "qdlat_step": qdlat_step,
        "mlt_step": mlt_step,
        "product": "qdlat_mlt",
    }
    np.savez_compressed(
        out_path,
        vtec_median=median.astype(np.float32),
        sample_count=count,
        qdlat_centers=qdlat_centers.astype(np.float32),
        mlt_centers=mlt_centers.astype(np.float32),
        qdlat_edges=qdlat_edges.astype(np.float32),
        mlt_edges=mlt_edges.astype(np.float32),
        metadata_json=np.asarray(json.dumps(meta_mlt)),
    )

    # Longitude-preserving map at fixed MLT window
    mlt_delta = np.abs(((mlt - target_mlt + 12.0) % 24.0) - 12.0)
    near = mlt_delta <= mlt_window
    vtec_mlt = np.where(near, vtec, np.nan)
    qdlon_edges = np.arange(-180.0, 180.0 + qdlon_step * 0.5, qdlon_step)
    median_lon, count_lon = rebin_2d(
        vtec_mlt, qdlat, qdlon, qdlat_edges, qdlon_edges
    )
    qdlon_centers = 0.5 * (qdlon_edges[:-1] + qdlon_edges[1:])
    out_lon = out_dir / f"vtec_qdlat_qdlon_mlt{int(target_mlt)}_{month}.npz"
    meta_lon = {
        **base_meta,
        "product": "qdlat_qdlon_fixed_mlt",
        "target_mlt": target_mlt,
        "mlt_window_h": mlt_window,
        "qdlat_step": qdlat_step,
        "qdlon_step": qdlon_step,
    }
    np.savez_compressed(
        out_lon,
        vtec_median=median_lon.astype(np.float32),
        sample_count=count_lon,
        qdlat_centers=qdlat_centers.astype(np.float32),
        qdlon_centers=qdlon_centers.astype(np.float32),
        qdlat_edges=qdlat_edges.astype(np.float32),
        qdlon_edges=qdlon_edges.astype(np.float32),
        metadata_json=np.asarray(json.dumps(meta_lon)),
    )
    return out_path, out_lon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_qd_mlt"),
    )
    parser.add_argument("--qdlat-min", type=float, default=-45.0)
    parser.add_argument("--qdlat-max", type=float, default=45.0)
    parser.add_argument("--qdlat-step", type=float, default=1.0)
    parser.add_argument("--mlt-step", type=float, default=0.25)
    parser.add_argument("--qdlon-step", type=float, default=5.0)
    parser.add_argument("--target-mlt", type=float, default=14.0)
    parser.add_argument("--mlt-window", type=float, default=0.5)
    parser.add_argument("--height-km", type=float, default=350.0)
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional YYYYMM filters, e.g. 201710",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("vtec_avg_*_kp_4_median.pkl"))
    if not files:
        raise SystemExit(f"No vtec_avg pickles in {args.input_dir}")

    if args.months:
        wanted = set(args.months)
        files = [p for p in files if parse_month_range(p)[0].strftime("%Y%m") in wanted]
        if not files:
            raise SystemExit(f"No files matched months {sorted(wanted)}")

    for path in files:
        out_mlt, out_lon = convert_one(
            path,
            args.output_dir,
            args.qdlat_min,
            args.qdlat_max,
            args.qdlat_step,
            args.mlt_step,
            args.height_km,
            args.qdlon_step,
            args.target_mlt,
            args.mlt_window,
        )
        data = np.load(out_mlt)
        data_lon = np.load(out_lon)
        print(
            f"{path.name} -> {out_mlt.name} "
            f"valid={np.isfinite(data['vtec_median']).sum()} | "
            f"{out_lon.name} "
            f"valid={np.isfinite(data_lon['vtec_median']).sum()}"
        )


if __name__ == "__main__":
    main()

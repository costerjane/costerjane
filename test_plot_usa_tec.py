#!/usr/bin/env python3
"""Smoke tests for loading CEDAR Madrigal VTEC and selecting CONUS slices."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import plot_usa_tec as tec


def test_load_and_slice(data_path: Path) -> None:
    grid = tec.load_vtec_grid(data_path)
    assert grid["tec"].ndim == 3
    assert grid["tec"].shape[0] == grid["gdlat"].size
    assert grid["tec"].shape[1] == grid["glon"].size
    assert grid["tec"].shape[2] == grid["times"].size
    assert grid["times"].size == 288  # 5-minute cadence

    when = dt.datetime(2026, 1, 19, 20, 0, tzinfo=dt.timezone.utc)
    idx = tec.nearest_time_index(grid["times"], when)
    stamp = grid["times"][idx]
    assert stamp.hour == 20
    assert stamp.minute == 0

    stats = tec._usa_stats(grid, idx)
    assert np.isfinite(stats["median"])
    assert 10.0 < stats["median"] < 50.0
    assert stats["p5"] < stats["median"] < stats["p95"]
    print(
        f"20 UT CONUS median={stats['median']:.2f} TECU  "
        f"p5={stats['p5']:.2f}  p95={stats['p95']:.2f}  max={stats['max']:.2f}"
    )

    night = tec.nearest_time_index(
        grid["times"], dt.datetime(2026, 1, 19, 6, 0, tzinfo=dt.timezone.utc)
    )
    night_stats = tec._usa_stats(grid, night)
    assert night_stats["median"] < stats["median"]
    print(f"06 UT CONUS median={night_stats['median']:.2f} TECU (nighttime lower than 20 UT)")


def test_terminator_exists_at_midnight() -> None:
    when = dt.datetime(2026, 1, 19, 0, 0, tzinfo=dt.timezone.utc)
    lat = np.linspace(24, 50, 80)
    lon = np.linspace(-125, -66, 120)
    lon2d, lat2d = np.meshgrid(lon, lat)
    sza = tec.solar_zenith_deg(lat2d, lon2d, when)
    assert sza.min() < 90 < sza.max()
    print(f"00 UT terminator crosses CONUS (SZA {sza.min():.1f}–{sza.max():.1f} deg)")


if __name__ == "__main__":
    data_dir = Path("data")
    matches = sorted(data_dir.glob("gps260119g*.hdf5"))
    if not matches:
        raise SystemExit("Missing gps260119g HDF5 in data/")
    test_load_and_slice(matches[0])
    test_terminator_exists_at_midnight()
    print("all tests passed")

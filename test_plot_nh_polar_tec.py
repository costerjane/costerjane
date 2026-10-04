#!/usr/bin/env python3
"""Smoke tests for NH polar Madrigal VTEC plotting in AACGM coordinates."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import plot_nh_polar_tec as tec


def test_load_and_magnetic_slice(data_path: Path) -> None:
    grid = tec.load_vtec_grid(data_path)
    assert grid["tec"].ndim == 3
    assert grid["tec"].shape[0] == grid["gdlat"].size
    assert grid["tec"].shape[1] == grid["glon"].size
    assert grid["tec"].shape[2] == grid["times"].size
    assert grid["times"].size == 288  # 5-minute cadence

    when = dt.datetime(2026, 1, 20, 20, 0, tzinfo=dt.timezone.utc)
    idx = tec.nearest_time_index(grid["times"], when)
    stamp = grid["times"][idx]
    assert stamp.hour == 20
    assert stamp.minute == 0

    stats = tec.nh_stats(grid, idx)
    assert stats["n"] > 1000
    assert np.isfinite(stats["median"])
    assert 0.0 < stats["median"] < 80.0
    assert stats["mlat_min"] >= tec.MLAT_OUTER - 1e-6
    assert stats["mlat_max"] <= 90.0
    assert 0.0 <= stats["mlt_min"] < 24.0
    assert 0.0 < stats["mlt_max"] <= 24.0
    print(
        f"20 UT NH mlat≥{tec.MLAT_OUTER:.0f} median={stats['median']:.2f} TECU  "
        f"n={stats['n']}  mlat {stats['mlat_min']:.1f}–{stats['mlat_max']:.1f}  "
        f"MLT {stats['mlt_min']:.2f}–{stats['mlt_max']:.2f}"
    )


def test_polar_orientation() -> None:
    theta, radius = tec.polar_coords(
        np.array([90.0, 60.0]),
        np.array([0.0, 12.0]),
    )
    assert np.isclose(radius[0], 0.0)
    assert np.isclose(radius[1], 30.0)
    assert np.isclose(theta[0], 0.0)
    assert np.isclose(theta[1], np.pi)
    print("polar orientation: pole at r=0, 12 MLT at theta=pi (top with zero@S)")


if __name__ == "__main__":
    data_dir = Path("data")
    matches = sorted(data_dir.glob("gps260120g*.hdf5"))
    if not matches:
        raise SystemExit("Missing gps260120g HDF5 in data/")
    test_load_and_magnetic_slice(matches[0])
    test_polar_orientation()
    print("all tests passed")

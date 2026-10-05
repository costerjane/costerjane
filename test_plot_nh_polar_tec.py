#!/usr/bin/env python3
"""Smoke tests for geographic NH TEC + reference-style σφ overlay."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import plot_nh_polar_tec as tec


def test_load_tec(data_path: Path) -> None:
    grid = tec.load_vtec_grid(data_path)
    assert grid["tec"].ndim == 3
    assert grid["times"].size == 288
    when = dt.datetime(2026, 1, 20, 18, 0, tzinfo=dt.timezone.utc)
    idx = tec.nearest_time_index(grid["times"], when)
    assert grid["times"][idx].hour == 18
    print(f"TEC grid OK: shape={grid['tec'].shape}")


def test_sigma_phi_bins() -> None:
    assert tec.sigma_phi_bin_style(0.05)[0] == tec.SIGMA_PHI_QUIET_FACE
    assert tec.sigma_phi_bin_style(0.15)[0] == tec.SIGMA_PHI_ACTIVE_FACE
    assert tec.sigma_phi_bin_style(0.25)[1] < tec.sigma_phi_bin_style(0.55)[1]
    assert tec.sigma_phi_bin_style(0.25)[1] < tec.sigma_phi_bin_style(0.75)[1]
    sizes = tec.sigma_phi_marker_sizes(np.array([0.05, 0.25, 0.75]))
    assert sizes[0] < sizes[1] < sizes[2]
    print("σφ bin styles OK:", list(zip([0.05, 0.25, 0.75], sizes)))


def test_mag_window_selection(scin_path: Path) -> None:
    t0 = dt.datetime(2026, 1, 20, 17, 40, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2026, 1, 20, 18, 0, tzinfo=dt.timezone.utc)
    csv_path = Path("data") / "scint_mag_test_1740-1800.csv"
    if csv_path.exists():
        csv_path.unlink()
    scin = tec.load_or_build_scintillation_mag(
        scin_path,
        csv_path=csv_path,
        elev_min=20.0,
        t0=t0,
        t1=t1,
        convert_magnetic=True,
    )
    assert len(scin) > 0
    assert {"mlat", "mlt", "sigma_phi"}.issubset(scin.columns)
    slc = tec.select_scintillation_window(scin, t0, t1, mlat_min=40.0)
    assert len(slc) > 0
    assert (slc["mlat"] >= 40.0).all()
    assert ((slc["mlt"] >= 0.0) & (slc["mlt"] < 24.0)).all()
    n_sig = int((slc["sigma_phi"] >= 0.1).sum())
    # Noon at top: MLT=12 maps to theta=π with zero at south.
    theta, radius = tec.polar_coords(np.array([70.0]), np.array([12.0]))
    assert np.isclose(theta[0], np.pi)
    assert np.isclose(radius[0], 20.0)
    print(
        f"17:40–18:00 mag scin: n={len(slc)}  σφ≥0.1={n_sig}  "
        f"mlat {slc['mlat'].min():.1f}–{slc['mlat'].max():.1f}  "
        f"MLT {slc['mlt'].min():.2f}–{slc['mlt'].max():.2f}"
    )


if __name__ == "__main__":
    data_dir = Path("data")
    tec_matches = sorted(data_dir.glob("gps260120g*.hdf5"))
    scin_matches = sorted(data_dir.glob("scin_20260120*.hdf5"))
    if not tec_matches:
        raise SystemExit("Missing gps260120g HDF5 in data/")
    if not scin_matches:
        raise SystemExit("Missing scin_20260120 HDF5 in data/")
    test_load_tec(tec_matches[0])
    test_sigma_phi_bins()
    test_mag_window_selection(scin_matches[0])
    print("all tests passed")

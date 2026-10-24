#!/usr/bin/env python3
"""Evaluate the Fourier EIA model on geographic lat/lon (and optional UT).

The fitted product is defined at MLT ≈ 14 in QD coordinates. This script maps
it back with apexpy in two modes:

1. ``mlt14_map`` — geographic lat/lon map of the MLT≈14 morphology
   (every geographic point is evaluated with coeffs(qdlon(geo)), ignoring
   the actual MLT at a single UT; this is the fixed-MLT product in geo coords).

2. ``ut_snapshot`` — at one Universal Time, evaluate only where MLT is near 14
   (a geographic strip that moves with UT).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import apexpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_eia_coeffs_qdlon import profile_model
from fit_eia_fourier_qdlon import clamp_params, fourier_design


def eval_fourier_coeffs(qdlon_deg: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """betas: (9, 1+2K) -> coeffs (..., 9) for qdlon_deg broadcastable shape."""
    phi = np.deg2rad(np.asarray(qdlon_deg, dtype=float))
    order = (betas.shape[1] - 1) // 2
    A = fourier_design(phi.ravel(), order)
    vals = A @ betas.T  # (n, 9)
    return vals.reshape(phi.shape + (9,))


def load_fourier_product(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["metrics_json"]))
    return {
        "betas": np.asarray(data["fourier_betas"], dtype=float),
        "meta": meta,
        "parameter_names": [str(x) for x in data["parameter_names"]],
    }


def evaluate_on_geographic_grid(
    betas: np.ndarray,
    apex: apexpy.Apex,
    gdlat: np.ndarray,
    glon: np.ndarray,
    height_km: float,
    ut: datetime | None = None,
    mlt_target: float = 14.0,
    mlt_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vtec, qdlat, qdlon, mlt on the geographic grid.

    If ``mlt_window`` is set and ``ut`` is given, points outside the MLT window
    are NaN. If ``mlt_window`` is None, every geographic point is filled using
    the MLT≈14 coefficient model at that point's QD longitude (fixed-MLT map).
    """
    lat_grid, lon_grid = np.meshgrid(gdlat, glon, indexing="ij")
    qdlat, qdlon = apex.geo2qd(lat_grid, lon_grid, height_km)
    coeffs = eval_fourier_coeffs(qdlon, betas)

    vtec = np.empty(lat_grid.shape, dtype=float)
    for i in range(gdlat.size):
        for j in range(glon.size):
            vtec[i, j] = float(profile_model(np.array([qdlat[i, j]]), clamp_params(coeffs[i, j]))[0])

    if ut is None:
        mlt = np.full(lat_grid.shape, np.nan)
    else:
        # mlon2mlt depends on magnetic lon and UT; qdlon varies weakly with lat
        mlt = np.empty(lat_grid.shape, dtype=float)
        for j in range(glon.size):
            mlt[:, j] = apex.mlon2mlt(qdlon[:, j], ut)
        if mlt_window is not None:
            delta = np.abs(((mlt - mlt_target + 12.0) % 24.0) - 12.0)
            vtec = np.where(delta <= mlt_window, vtec, np.nan)

    return vtec, qdlat, qdlon, mlt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fourier-file",
        type=Path,
        default=Path(
            "/workspace/data/quiet_monthly_median/2017_eia_fourier_qdlon15/"
            "eia_fourier_qdlon_mlt14_K3_201710.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_geo_from_eia"),
    )
    parser.add_argument("--epoch", type=str, default="2017-10-16")
    parser.add_argument("--height-km", type=float, default=350.0)
    parser.add_argument("--lat-step", type=float, default=2.0)
    parser.add_argument("--lon-step", type=float, default=5.0)
    parser.add_argument("--lat-min", type=float, default=-60.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument(
        "--ut",
        type=str,
        default=None,
        help="ISO UT for snapshot mode, e.g. 2017-10-16T18:00:00",
    )
    parser.add_argument("--mlt-window", type=float, default=0.75)
    parser.add_argument(
        "--mode",
        choices=["mlt14_map", "ut_snapshot", "both"],
        default="both",
    )
    args = parser.parse_args()

    product = load_fourier_product(args.fourier_file)
    epoch = datetime.fromisoformat(args.epoch)
    apex = apexpy.Apex(date=epoch)
    gdlat = np.arange(args.lat_min, args.lat_max + args.lat_step * 0.5, args.lat_step)
    glon = np.arange(-180.0, 180.0, args.lon_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.fourier_file.stem

    if args.mode in ("mlt14_map", "both"):
        vtec, qdlat, qdlon, mlt = evaluate_on_geographic_grid(
            product["betas"], apex, gdlat, glon, args.height_km, ut=None, mlt_window=None
        )
        out = args.output_dir / f"{stem}_geographic_mlt14_map.npz"
        meta = {
            "mode": "mlt14_map",
            "source": args.fourier_file.name,
            "epoch": args.epoch,
            "height_km": args.height_km,
            "note": (
                "Geographic remapping of the MLT≈14 Fourier EIA model. "
                "Each geo point uses coeffs(qdlon(geo)); not a single-UT global map."
            ),
        }
        np.savez_compressed(
            out,
            gdlat=gdlat.astype(np.float32),
            glon=glon.astype(np.float32),
            vtec=vtec.astype(np.float32),
            qdlat=qdlat.astype(np.float32),
            qdlon=qdlon.astype(np.float32),
            metadata_json=np.asarray(json.dumps(meta)),
        )
        print(f"wrote {out} shape={vtec.shape}")

    if args.mode in ("ut_snapshot", "both"):
        ut = datetime.fromisoformat(args.ut) if args.ut else datetime(epoch.year, epoch.month, epoch.day, 18, 0, 0)
        vtec, qdlat, qdlon, mlt = evaluate_on_geographic_grid(
            product["betas"],
            apex,
            gdlat,
            glon,
            args.height_km,
            ut=ut,
            mlt_target=14.0,
            mlt_window=args.mlt_window,
        )
        out = args.output_dir / f"{stem}_geographic_ut{ut.strftime('%Y%m%dT%H%M')}.npz"
        meta = {
            "mode": "ut_snapshot",
            "source": args.fourier_file.name,
            "epoch": args.epoch,
            "ut": ut.isoformat(),
            "mlt_target": 14.0,
            "mlt_window_h": args.mlt_window,
            "height_km": args.height_km,
            "note": "VTEC filled only where MLT is within the window of 14 h.",
        }
        np.savez_compressed(
            out,
            gdlat=gdlat.astype(np.float32),
            glon=glon.astype(np.float32),
            vtec=vtec.astype(np.float32),
            qdlat=qdlat.astype(np.float32),
            qdlon=qdlon.astype(np.float32),
            mlt=mlt.astype(np.float32),
            metadata_json=np.asarray(json.dumps(meta)),
        )
        n_fill = int(np.isfinite(vtec).sum())
        print(f"wrote {out} filled_points={n_fill}/{vtec.size} ut={ut.isoformat()}")


if __name__ == "__main__":
    main()

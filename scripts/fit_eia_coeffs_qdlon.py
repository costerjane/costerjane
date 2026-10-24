#!/usr/bin/env python3
"""Fit 9 EIA latent coefficients vs QD longitude at fixed ~14 MLT.

Reads monthly ``vtec_qdlat_qdlon_mlt14_YYYYMM.npz`` maps and, for each QD
longitude bin, fits

    T(λ) = c0 + c1 x + c2 (3x^2-1)/2
         + A_N exp[-((λ-μ_N)/σ_N)^2 / 2]
         + A_S exp[-((λ-μ_S)/σ_S)^2 / 2]

with x = λ/45°, λ = QD latitude (deg).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


PARAM_NAMES = ["c0", "c1", "c2", "A_N", "A_S", "mu_N", "mu_S", "sigma_N", "sigma_S"]


def profile_model(lat: np.ndarray, params: np.ndarray) -> np.ndarray:
    c0, c1, c2, a_n, a_s, mu_n, mu_s, sig_n, sig_s = params
    x = lat / 45.0
    background = c0 + c1 * x + c2 * 0.5 * (3.0 * x**2 - 1.0)
    north = a_n * np.exp(-0.5 * ((lat - mu_n) / sig_n) ** 2)
    south = a_s * np.exp(-0.5 * ((lat - mu_s) / sig_s) ** 2)
    return background + north + south


def initial_guess(lat: np.ndarray, vtec: np.ndarray) -> np.ndarray:
    """Heuristic starting vector from a latitude profile."""
    near_eq = np.isfinite(vtec) & (np.abs(lat) <= 5)
    c0 = float(np.nanmedian(vtec[near_eq])) if np.any(near_eq) else float(np.nanmedian(vtec))
    if not np.isfinite(c0):
        c0 = 10.0

    north = np.isfinite(vtec) & (lat >= 5.0) & (lat <= 30.0)
    south = np.isfinite(vtec) & (lat <= -5.0) & (lat >= -30.0)
    if np.any(north):
        i_n = int(np.nanargmax(np.where(north, vtec, -np.inf)))
        mu_n = float(lat[i_n])
        a_n = max(float(vtec[i_n] - c0), 1.0)
    else:
        mu_n, a_n = 15.0, 5.0
    if np.any(south):
        i_s = int(np.nanargmax(np.where(south, vtec, -np.inf)))
        mu_s = float(lat[i_s])
        a_s = max(float(vtec[i_s] - c0), 1.0)
    else:
        mu_s, a_s = -15.0, 5.0

    return np.array([c0, 0.0, -2.0, a_n, a_s, mu_n, mu_s, 8.0, 8.0], dtype=float)


def param_bounds(vtec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vmax = float(np.nanmax(vtec))
    if not np.isfinite(vmax):
        vmax = 50.0
    lower = np.array(
        [0.0, -40.0, -60.0, 0.0, 0.0, 5.0, -30.0, 2.5, 2.5], dtype=float
    )
    upper = np.array(
        [vmax + 20.0, 40.0, 60.0, vmax + 20.0, vmax + 20.0, 30.0, -5.0, 13.0, 13.0],
        dtype=float,
    )
    return lower, upper


def fit_profile(
    lat: np.ndarray,
    vtec: np.ndarray,
    lat_limit: float = 40.0,
    min_points: int = 15,
    min_crest_amplitude: float = 3.0,
    min_relative_crest: float = 0.04,
) -> dict:
    mask = np.isfinite(vtec) & (np.abs(lat) <= lat_limit)
    if mask.sum() < min_points:
        return {
            "params": np.full(9, np.nan),
            "rmse": np.nan,
            "success": False,
            "n_points": int(mask.sum()),
            "north_resolved": False,
            "south_resolved": False,
            "mu_N_reported": np.nan,
            "mu_S_reported": np.nan,
            "sigma_N_reported": np.nan,
            "sigma_S_reported": np.nan,
        }

    lat_f = lat[mask]
    vtec_f = vtec[mask]
    x0 = initial_guess(lat_f, vtec_f)
    lower, upper = param_bounds(vtec_f)
    x0 = np.minimum(np.maximum(x0, lower + 1e-6), upper - 1e-6)

    def residual(p):
        return profile_model(lat_f, p) - vtec_f

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=300,
        ftol=1e-8,
        xtol=1e-8,
    )
    params = result.x
    resid = residual(params)
    rmse = float(np.sqrt(np.mean(resid**2)))

    c0, _, _, a_n, a_s, mu_n, mu_s, sig_n, sig_s = params
    amp_floor = max(min_crest_amplitude, min_relative_crest * max(c0, 1.0))
    north_resolved = bool(a_n >= amp_floor and 5.0 <= mu_n <= 30.0)
    south_resolved = bool(a_s >= amp_floor and -30.0 <= mu_s <= -5.0)

    return {
        "params": params,
        "rmse": rmse,
        "success": bool(result.success),
        "n_points": int(mask.sum()),
        "north_resolved": north_resolved,
        "south_resolved": south_resolved,
        "mu_N_reported": float(mu_n) if north_resolved else np.nan,
        "mu_S_reported": float(mu_s) if south_resolved else np.nan,
        "sigma_N_reported": float(sig_n) if north_resolved else np.nan,
        "sigma_S_reported": float(sig_s) if south_resolved else np.nan,
    }


def fit_month(path: Path, out_dir: Path, lat_limit: float) -> Path:
    data = np.load(path, allow_pickle=True)
    vtec = np.asarray(data["vtec_median"], dtype=float)
    qdlat = np.asarray(data["qdlat_centers"], dtype=float)
    qdlon = np.asarray(data["qdlon_centers"], dtype=float)
    meta = json.loads(str(data["metadata_json"]))

    n_lon = qdlon.size
    coeffs = np.full((n_lon, 9), np.nan, dtype=np.float64)
    rmse = np.full(n_lon, np.nan)
    success = np.zeros(n_lon, dtype=np.uint8)
    n_points = np.zeros(n_lon, dtype=np.int32)
    north_resolved = np.zeros(n_lon, dtype=np.uint8)
    south_resolved = np.zeros(n_lon, dtype=np.uint8)
    mu_n_rep = np.full(n_lon, np.nan)
    mu_s_rep = np.full(n_lon, np.nan)
    sig_n_rep = np.full(n_lon, np.nan)
    sig_s_rep = np.full(n_lon, np.nan)

    for j in range(n_lon):
        fit = fit_profile(qdlat, vtec[:, j], lat_limit=lat_limit)
        coeffs[j] = fit["params"]
        rmse[j] = fit["rmse"]
        success[j] = 1 if fit["success"] else 0
        n_points[j] = fit["n_points"]
        north_resolved[j] = 1 if fit["north_resolved"] else 0
        south_resolved[j] = 1 if fit["south_resolved"] else 0
        mu_n_rep[j] = fit["mu_N_reported"]
        mu_s_rep[j] = fit["mu_S_reported"]
        sig_n_rep[j] = fit["sigma_N_reported"]
        sig_s_rep[j] = fit["sigma_S_reported"]

    out_dir.mkdir(parents=True, exist_ok=True)
    month = path.name.split("_")[-1].replace(".npz", "")
    out_path = out_dir / f"eia_coeffs_qdlon_mlt14_{month}.npz"
    out_meta = {
        "source_map": path.name,
        "parameter_names": PARAM_NAMES,
        "lat_limit_deg": lat_limit,
        "target_mlt": meta.get("target_mlt", 14.0),
        "mlt_window_h": meta.get("mlt_window_h", 0.5),
        "model": (
            "T(lat)=c0+c1*x+c2*(3x^2-1)/2 "
            "+ A_N*exp(-0.5*((lat-mu_N)/sigma_N)^2) "
            "+ A_S*exp(-0.5*((lat-mu_S)/sigma_S)^2); x=lat/45"
        ),
    }
    np.savez_compressed(
        out_path,
        qdlon_centers=qdlon.astype(np.float32),
        coefficients=coeffs.astype(np.float32),
        parameter_names=np.asarray(PARAM_NAMES),
        rmse=rmse.astype(np.float32),
        fit_success=success,
        n_points=n_points,
        north_resolved=north_resolved,
        south_resolved=south_resolved,
        mu_N_reported=mu_n_rep.astype(np.float32),
        mu_S_reported=mu_s_rep.astype(np.float32),
        sigma_N_reported=sig_n_rep.astype(np.float32),
        sigma_S_reported=sig_s_rep.astype(np.float32),
        metadata_json=np.asarray(json.dumps(out_meta)),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_qd_mlt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_eia_coeffs_qdlon"),
    )
    parser.add_argument("--lat-limit", type=float, default=40.0)
    parser.add_argument("--months", nargs="*", default=None)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("vtec_qdlat_qdlon_mlt14_*.npz"))
    if args.months:
        wanted = set(args.months)
        files = [p for p in files if p.stem.split("_")[-1] in wanted]
    if not files:
        raise SystemExit("No input QD-lat/QD-lon maps found")

    for path in files:
        out = fit_month(path, args.output_dir, args.lat_limit)
        data = np.load(out)
        ok = np.isfinite(data["coefficients"][:, 0]).sum()
        med_rmse = float(np.nanmedian(data["rmse"]))
        print(
            f"{path.name} -> {out.name} | "
            f"fitted_lons={ok}/{len(data['qdlon_centers'])} "
            f"median_rmse={med_rmse:.3f} TECU"
        )


if __name__ == "__main__":
    main()

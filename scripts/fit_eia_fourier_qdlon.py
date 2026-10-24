#!/usr/bin/env python3
"""Fit low-order Fourier series in QD longitude to EIA coefficients.

For each monthly ``eia_coeffs_qdlon_mlt14_*.npz`` file, fit

    c(φ) = a0 + Σ_{k=1}^K [a_k cos(kφ) + b_k sin(kφ)]

independently for each of the nine coefficients (φ in radians from QD
longitude). Then reconstruct VTEC(qdlat, qdlon) and compare against the
original ``vtec_qdlat_qdlon_mlt14_*.npz`` maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_eia_coeffs_qdlon import PARAM_NAMES, profile_model


def fourier_design(phi_rad: np.ndarray, order: int) -> np.ndarray:
    """Columns: [1, cosφ, sinφ, ..., cos(Kφ), sin(Kφ)]."""
    cols = [np.ones_like(phi_rad)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * phi_rad))
        cols.append(np.sin(k * phi_rad))
    return np.column_stack(cols)


def fit_fourier_1d(
    phi_deg: np.ndarray,
    values: np.ndarray,
    order: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coeff_vector length 1+2K, fitted values on same φ)."""
    valid = np.isfinite(values) & np.isfinite(phi_deg)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0)
    if valid.sum() < (1 + 2 * order):
        return np.full(1 + 2 * order, np.nan), np.full_like(values, np.nan)

    phi = np.deg2rad(phi_deg[valid])
    y = values[valid]
    A = fourier_design(phi, order)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights[valid], dtype=float)
        w = w / np.nanmax(w)
    Aw = A * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

    A_all = fourier_design(np.deg2rad(phi_deg), order)
    yhat = A_all @ beta
    return beta, yhat


def clamp_params(params: np.ndarray) -> np.ndarray:
    """Enforce physical bounds used in the discrete lon fits."""
    p = np.array(params, dtype=float, copy=True)
    p[0] = np.clip(p[0], 0.0, None)  # c0
    p[3] = np.clip(p[3], 0.0, None)  # A_N
    p[4] = np.clip(p[4], 0.0, None)  # A_S
    p[5] = np.clip(p[5], 5.0, 30.0)  # mu_N
    p[6] = np.clip(p[6], -30.0, -5.0)  # mu_S
    p[7] = np.clip(p[7], 2.5, 13.0)  # sigma_N
    p[8] = np.clip(p[8], 2.5, 13.0)  # sigma_S
    return p


def reconstruct_map(
    qdlat: np.ndarray, coeffs_by_lon: np.ndarray
) -> np.ndarray:
    """coeffs_by_lon: (n_lon, 9) -> VTEC (n_lat, n_lon)."""
    n_lat = qdlat.size
    n_lon = coeffs_by_lon.shape[0]
    out = np.full((n_lat, n_lon), np.nan, dtype=float)
    for j in range(n_lon):
        if np.all(np.isfinite(coeffs_by_lon[j])):
            out[:, j] = profile_model(qdlat, clamp_params(coeffs_by_lon[j]))
    return out


def compare_maps(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0)
    if not np.any(mask):
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "r2": np.nan,
        }
    o = obs[mask]
    p = pred[mask]
    resid = o - p
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "corr": float(np.corrcoef(o, p)[0, 1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def process_month(
    coeff_path: Path,
    map_path: Path,
    out_dir: Path,
    order: int,
) -> dict:
    coeff = np.load(coeff_path, allow_pickle=True)
    vmap = np.load(map_path, allow_pickle=True)
    qdlon = np.asarray(coeff["qdlon_centers"], dtype=float)
    discrete = np.asarray(coeff["coefficients"], dtype=float)
    rmse_fit = np.asarray(coeff["rmse"], dtype=float)
    qdlat = np.asarray(vmap["qdlat_centers"], dtype=float)
    obs = np.asarray(vmap["vtec_median"], dtype=float)

    # Weight by inverse RMSE (floor to avoid huge weights)
    weights = 1.0 / np.maximum(rmse_fit, 0.5)

    fourier_betas = np.full((9, 1 + 2 * order), np.nan)
    smooth = np.full_like(discrete, np.nan)
    for i in range(9):
        beta, yhat = fit_fourier_1d(qdlon, discrete[:, i], order, weights)
        fourier_betas[i] = beta
        smooth[:, i] = yhat

    pred_discrete = reconstruct_map(qdlat, discrete)
    pred_fourier = reconstruct_map(qdlat, smooth)
    metrics_discrete = compare_maps(obs, pred_discrete)
    metrics_fourier = compare_maps(obs, pred_fourier)

    # Coefficient-space fit quality
    coeff_rmse = []
    for i in range(9):
        valid = np.isfinite(discrete[:, i]) & np.isfinite(smooth[:, i])
        if valid.any():
            coeff_rmse.append(
                float(np.sqrt(np.mean((discrete[valid, i] - smooth[valid, i]) ** 2)))
            )
        else:
            coeff_rmse.append(np.nan)

    month = coeff_path.stem.split("_")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eia_fourier_qdlon_mlt14_K{order}_{month}.npz"
    meta = {
        "source_coeffs": coeff_path.name,
        "source_map": map_path.name,
        "fourier_order": order,
        "parameter_names": PARAM_NAMES,
        "beta_layout": "a0, a1, b1, ..., aK, bK",
        "metrics_discrete_vs_obs": metrics_discrete,
        "metrics_fourier_vs_obs": metrics_fourier,
        "coeff_rmse_discrete_vs_fourier": dict(zip(PARAM_NAMES, coeff_rmse)),
    }
    np.savez_compressed(
        out_path,
        qdlon_centers=qdlon.astype(np.float32),
        qdlat_centers=qdlat.astype(np.float32),
        coefficients_discrete=discrete.astype(np.float32),
        coefficients_fourier=smooth.astype(np.float32),
        fourier_betas=fourier_betas.astype(np.float32),
        parameter_names=np.asarray(PARAM_NAMES),
        vtec_obs=obs.astype(np.float32),
        vtec_from_discrete=pred_discrete.astype(np.float32),
        vtec_from_fourier=pred_fourier.astype(np.float32),
        metrics_json=np.asarray(json.dumps(meta)),
    )
    return {
        "month": month,
        "out_path": out_path,
        "metrics_discrete": metrics_discrete,
        "metrics_fourier": metrics_fourier,
        "coeff_rmse": coeff_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coeff-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_eia_coeffs_qdlon"),
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_qd_mlt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/quiet_monthly_median/2017_eia_fourier_qdlon"),
    )
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--months", nargs="*", default=None)
    args = parser.parse_args()

    coeff_files = sorted(args.coeff_dir.glob("eia_coeffs_qdlon_mlt14_*.npz"))
    if args.months:
        wanted = set(args.months)
        coeff_files = [p for p in coeff_files if p.stem.split("_")[-1] in wanted]
    if not coeff_files:
        raise SystemExit("No coefficient files found")

    rows = []
    for coeff_path in coeff_files:
        month = coeff_path.stem.split("_")[-1]
        map_path = args.map_dir / f"vtec_qdlat_qdlon_mlt14_{month}.npz"
        if not map_path.exists():
            raise SystemExit(f"Missing map {map_path}")
        result = process_month(coeff_path, map_path, args.output_dir, args.order)
        md = result["metrics_discrete"]
        mf = result["metrics_fourier"]
        print(
            f"{month} K={args.order} | "
            f"discrete RMSE={md['rmse']:.3f} R2={md['r2']:.3f} | "
            f"Fourier RMSE={mf['rmse']:.3f} R2={mf['r2']:.3f}"
        )
        rows.append(result)

    # Summary table
    summary_path = args.output_dir / f"comparison_summary_K{args.order}.json"
    summary = [
        {
            "month": r["month"],
            "discrete": r["metrics_discrete"],
            "fourier": r["metrics_fourier"],
        }
        for r in rows
    ]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

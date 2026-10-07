#!/usr/bin/env python3
"""
Quiet-time median σφ vs MLT/MLAT for October 2024.

- Drop whole UT days where any 3-hour Kp > 4 (GFZ Potsdam definitive Kp).
- Bin remaining samples in MLT × MLAT and take the median σφ per bin.
- Plot a northern-hemisphere polar map: 12 MLT at top, MLAT 90 (center) → 0 (rim).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
ARTIFACT_DIR = Path("/opt/cursor/artifacts")

MLT_BIN_H = 1.0  # hours
MLAT_BIN_DEG = 1.0  # degrees
KP_THRESHOLD = 4.0  # exclude day if any 3-h Kp > this


def fetch_quiet_october_days(year: int = 2024, month: int = 10) -> tuple[list[str], list[str]]:
    """Return (keep_dates, drop_dates) as YYYY-MM-DD strings from GFZ Kp."""
    start = f"{year}-{month:02d}-01T00:00:00Z"
    end = f"{year}-{month:02d}-31T21:00:00Z"
    url = (
        "https://kp.gfz.de/app/json/"
        f"?start={start}&end={end}&index=Kp&status=def"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)

    by_day: dict[str, list[float]] = {}
    for t, kp in zip(payload["datetime"], payload["Kp"]):
        day = t[:10]
        by_day.setdefault(day, []).append(float(kp))

    keep, drop = [], []
    for day in sorted(by_day):
        if max(by_day[day]) > KP_THRESHOLD:
            drop.append(day)
        else:
            keep.append(day)
    return keep, drop


def csv_path_for_day(day: str) -> Path | None:
    y, m, d = day.split("-")
    path = CSV_DIR / f"scint_mag_coords_{y}_{m}_{d}.csv"
    return path if path.exists() else None


def load_quiet_samples(keep_days: list[str]) -> pd.DataFrame:
    frames = []
    usecols = ["sigma_phi", "mlat", "mlt"]
    for day in keep_days:
        path = csv_path_for_day(day)
        if path is None:
            print(f"warning: missing CSV for quiet day {day}")
            continue
        df = pd.read_csv(path, usecols=usecols)
        df = df.dropna(subset=usecols)
        # Northern hemisphere map: MLAT 0 → 90
        df = df[(df["mlat"] >= 0.0) & (df["mlat"] <= 90.0)]
        df = df[(df["mlt"] >= 0.0) & (df["mlt"] < 24.0)]
        df = df[np.isfinite(df["sigma_phi"])]
        df["day"] = day
        frames.append(df)
        print(f"loaded {day}: {len(df):,} northern samples")
    if not frames:
        raise SystemExit("No quiet-day samples found.")
    return pd.concat(frames, ignore_index=True)


def median_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_H, MLT_BIN_H)
    mlat_edges = np.arange(0.0, 90.0 + MLAT_BIN_DEG, MLAT_BIN_DEG)

    mlt_idx = np.clip(
        np.floor(df["mlt"].to_numpy() / MLT_BIN_H).astype(int),
        0,
        len(mlt_edges) - 2,
    )
    mlat_idx = np.clip(
        np.floor(df["mlat"].to_numpy() / MLAT_BIN_DEG).astype(int),
        0,
        len(mlat_edges) - 2,
    )

    n_mlt = len(mlt_edges) - 1
    n_mlat = len(mlat_edges) - 1
    # Collect values per bin via pandas groupby for robust medians
    tmp = pd.DataFrame(
        {
            "mlt_i": mlt_idx,
            "mlat_i": mlat_idx,
            "sigma_phi": df["sigma_phi"].to_numpy(),
        }
    )
    med = (
        tmp.groupby(["mlat_i", "mlt_i"], sort=True)["sigma_phi"]
        .median()
        .unstack(fill_value=np.nan)
        .reindex(index=range(n_mlat), columns=range(n_mlt))
    )
    counts = (
        tmp.groupby(["mlat_i", "mlt_i"], sort=True)["sigma_phi"]
        .count()
        .unstack(fill_value=0)
        .reindex(index=range(n_mlat), columns=range(n_mlt))
    )

    median = med.to_numpy(dtype=float)
    count = counts.to_numpy(dtype=float)
    mlt_centers = 0.5 * (mlt_edges[:-1] + mlt_edges[1:])
    mlat_centers = 0.5 * (mlat_edges[:-1] + mlat_edges[1:])
    return mlt_centers, mlat_centers, median, count


def plot_polar(
    mlt_centers: np.ndarray,
    mlat_centers: np.ndarray,
    median: np.ndarray,
    keep_days: list[str],
    drop_days: list[str],
    out_path: Path,
) -> None:
    # Mesh in polar coords: theta from MLT, radius = 90 - MLAT
    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_H, MLT_BIN_H)
    mlat_edges = np.arange(0.0, 90.0 + MLAT_BIN_DEG, MLAT_BIN_DEG)
    # pcolormesh wants edges; extend median to (n_mlat, n_mlt)
    theta_edges = np.deg2rad(mlt_edges * 15.0)
    r_edges = 90.0 - mlat_edges  # 90→0 MLAT becomes 0→90 radius

    # For pcolormesh with polar: columns = theta, rows = r
    # Our median is shape (n_mlat, n_mlt) with mlat increasing 0→90 (r decreasing).
    # Flip so row 0 is pole (r near 0).
    Z = median[::-1, :]
    r_edges_plot = r_edges[::-1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection="polar")
    # 12 MLT at top, 00 at bottom; 06 at right (dawn), 18 at left (dusk)
    ax.set_theta_zero_location("S")  # MLT 0 at bottom
    ax.set_theta_direction(1)  # counterclockwise → 06 at right

    THETA, R = np.meshgrid(theta_edges, r_edges_plot)
    pcm = ax.pcolormesh(
        THETA,
        R,
        Z,
        cmap="viridis",
        shading="flat",
        vmin=0.0,
        vmax=np.nanpercentile(median, 95) if np.isfinite(median).any() else 1.0,
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label(r"median $\sigma_\phi$ (rad)")

    ax.set_ylim(0, 90)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_yticklabels(["90", "80", "70", "60", "50", "40", "30", "20", "10", "0"])
    ax.set_xticks(np.deg2rad(np.arange(0, 24, 3) * 15.0))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_title(
        "October 2024 quiet-time median "
        r"$\sigma_\phi$"
        f"\n(Kp ≤ {KP_THRESHOLD:g} all day; "
        f"{MLT_BIN_H:g} h × {MLAT_BIN_DEG:g}° bins; N={len(keep_days)} days)",
        pad=20,
    )
    # Annotate noon
    ax.text(np.pi, 95, "12 MLT", ha="center", va="bottom", fontsize=11)

    note = (
        f"Kept: {', '.join(d[-2:] for d in keep_days)}\n"
        f"Dropped (Kp>{KP_THRESHOLD:g}): {', '.join(d[-2:] for d in drop_days) or 'none'}"
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def main() -> None:
    keep, drop = fetch_quiet_october_days()
    print(f"Quiet days kept ({len(keep)}): {keep}")
    print(f"Disturbed days dropped ({len(drop)}): {drop}")

    df = load_quiet_samples(keep)
    print(f"Total samples: {len(df):,}")

    mlt_c, mlat_c, med, cnt = median_grid(df)
    filled = np.isfinite(med).sum()
    print(f"Filled bins: {filled} / {med.size}")
    print(f"Median σφ range: {np.nanmin(med):.4f} … {np.nanmax(med):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUT_DIR / "sigma_phi_median_grid_oct2024.npz"
    np.savez_compressed(
        grid_path,
        mlt_centers=mlt_c,
        mlat_centers=mlat_c,
        median_sigma_phi=med,
        counts=cnt,
        keep_days=np.array(keep),
        drop_days=np.array(drop),
    )
    print(f"wrote {grid_path}")

    plot_path = OUT_DIR / "sigma_phi_median_polar_oct2024.png"
    plot_polar(mlt_c, mlat_c, med, keep, drop, plot_path)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / "sigma_phi_median_polar_oct2024.png"
    plot_polar(mlt_c, mlat_c, med, keep, drop, artifact)

    # Also save a flat CSV summary of the grid for inspection
    rows = []
    for i, mlat in enumerate(mlat_c):
        for j, mlt in enumerate(mlt_c):
            if np.isfinite(med[i, j]):
                rows.append(
                    {
                        "mlat_center": mlat,
                        "mlt_center": mlt,
                        "median_sigma_phi": med[i, j],
                        "n": int(cnt[i, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT_DIR / "sigma_phi_median_grid_oct2024.csv", index=False)


if __name__ == "__main__":
    main()

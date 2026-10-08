#!/usr/bin/env python3
"""
Median σφ vs MLT/MLAT polar maps for October 2024.

Modes
-----
quiet:      keep days where every 3-hour Kp ≤ 4  (drop if any Kp > 4)
disturbed:  keep days where any 3-hour Kp ≥ 4

Both modes:
- Bin finite samples in 30 min MLT × 1° MLAT; median σφ per bin (NaNs excluded)
- Color scale in radians, fixed 0–1
- Northern-hemisphere polar map: 12 MLT at top, MLAT 90 (center) → 50 (rim)
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
ARTIFACT_DIR = Path("/opt/cursor/artifacts")

MLT_BIN_H = 0.5  # hours (30 min)
MLAT_BIN_DEG = 1.0  # degrees
MLAT_POLE = 90.0
MLAT_RIM = 50.0
KP_THRESHOLD = 4.0
CBAR_VMIN = 0.0
CBAR_VMAX = 1.0  # radians



def mlat_bin_edges() -> np.ndarray:
    """Edges from rim→pole in MLAT_BIN_DEG steps, closing exactly at the pole."""
    edges = np.arange(MLAT_RIM, MLAT_POLE + MLAT_BIN_DEG, MLAT_BIN_DEG)
    if edges[-1] > MLAT_POLE:
        edges[-1] = MLAT_POLE
    elif edges[-1] < MLAT_POLE:
        edges = np.append(edges, MLAT_POLE)
    return edges


def fetch_kp_by_day(year: int = 2024, month: int = 10) -> dict[str, list[float]]:
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
    return by_day


def select_days(mode: str) -> tuple[list[str], list[str], str]:
    """
    Return (keep_dates, other_dates, selection_label).

    quiet:     keep if max(Kp) ≤ 4  (i.e. no 3-h Kp > 4)
    disturbed: keep if max(Kp) ≥ 4  (i.e. any 3-h Kp ≥ 4)
    """
    by_day = fetch_kp_by_day()
    keep, other = [], []
    if mode == "quiet":
        label = f"Kp ≤ {KP_THRESHOLD:g} all day"
        for day in sorted(by_day):
            if max(by_day[day]) > KP_THRESHOLD:
                other.append(day)
            else:
                keep.append(day)
    elif mode == "disturbed":
        label = f"any Kp ≥ {KP_THRESHOLD:g}"
        for day in sorted(by_day):
            if max(by_day[day]) >= KP_THRESHOLD:
                keep.append(day)
            else:
                other.append(day)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return keep, other, label


def csv_path_for_day(day: str) -> Path | None:
    y, m, d = day.split("-")
    path = CSV_DIR / f"scint_mag_coords_{y}_{m}_{d}.csv"
    return path if path.exists() else None


def load_samples(keep_days: list[str]) -> pd.DataFrame:
    frames = []
    usecols = ["sigma_phi", "mlat", "mlt"]
    for day in keep_days:
        path = csv_path_for_day(day)
        if path is None:
            print(f"warning: missing CSV for day {day}")
            continue
        df = pd.read_csv(path, usecols=usecols)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=usecols)
        df = df[
            np.isfinite(df["sigma_phi"])
            & np.isfinite(df["mlat"])
            & np.isfinite(df["mlt"])
        ]
        df = df[(df["mlat"] >= MLAT_RIM) & (df["mlat"] <= MLAT_POLE)]
        df = df[(df["mlt"] >= 0.0) & (df["mlt"] < 24.0)]
        df["day"] = day
        frames.append(df)
        print(f"loaded {day}: {len(df):,} samples (MLAT {MLAT_RIM:g}–{MLAT_POLE:g})")
    if not frames:
        raise SystemExit("No samples found for selected days.")
    return pd.concat(frames, ignore_index=True)


def median_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sigma_phi", "mlat", "mlt"]
    )
    df = df[
        np.isfinite(df["sigma_phi"])
        & np.isfinite(df["mlat"])
        & np.isfinite(df["mlt"])
    ]

    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_H, MLT_BIN_H)
    mlat_edges = mlat_bin_edges()

    mlt_idx = np.clip(
        np.floor(df["mlt"].to_numpy() / MLT_BIN_H).astype(int),
        0,
        len(mlt_edges) - 2,
    )
    mlat_idx = np.digitize(df["mlat"].to_numpy(), mlat_edges) - 1
    mlat_idx = np.clip(mlat_idx, 0, len(mlat_edges) - 2)

    n_mlt = len(mlt_edges) - 1
    n_mlat = len(mlat_edges) - 1
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
    median: np.ndarray,
    keep_days: list[str],
    other_days: list[str],
    mode: str,
    selection_label: str,
    out_path: Path,
) -> None:
    mlt_edges = np.arange(0.0, 24.0 + MLT_BIN_H, MLT_BIN_H)
    mlat_edges = mlat_bin_edges()
    theta_edges = np.deg2rad(mlt_edges * 15.0)
    r_edges = MLAT_POLE - mlat_edges
    r_max = MLAT_POLE - MLAT_RIM

    Z = median[::-1, :]
    r_edges_plot = r_edges[::-1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)

    THETA, R = np.meshgrid(theta_edges, r_edges_plot)
    pcm = ax.pcolormesh(
        THETA,
        R,
        Z,
        cmap="viridis",
        shading="flat",
        vmin=CBAR_VMIN,
        vmax=CBAR_VMAX,
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label(r"median $\sigma_\phi$ (rad)")

    ax.set_ylim(0, r_max)
    mlat_ticks = np.arange(MLAT_RIM, MLAT_POLE + 1e-9, 10.0)
    r_ticks = MLAT_POLE - mlat_ticks
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{int(m)}" for m in mlat_ticks])
    ax.set_xticks(np.deg2rad(np.arange(0, 24, 3) * 15.0))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])

    condition_name = "quiet-time" if mode == "quiet" else "disturbed"
    mlt_bin_label = (
        "30 min" if abs(MLT_BIN_H - 0.5) < 1e-9 else f"{MLT_BIN_H:g} h"
    )
    ax.set_title(
        f"October 2024 {condition_name} median "
        r"$\sigma_\phi$"
        f"\n({selection_label}; "
        f"{mlt_bin_label} × {MLAT_BIN_DEG:g}° bins; "
        f"MLAT {MLAT_POLE:g}→{MLAT_RIM:g}; N={len(keep_days)} days)",
        pad=20,
    )
    ax.text(np.pi, r_max + 2.5, "12 MLT", ha="center", va="bottom", fontsize=11)

    if mode == "quiet":
        note = (
            f"Kept: {', '.join(d[-2:] for d in keep_days)}\n"
            f"Dropped (Kp>{KP_THRESHOLD:g}): "
            f"{', '.join(d[-2:] for d in other_days) or 'none'}"
        )
    else:
        note = (
            f"Kept (Kp≥{KP_THRESHOLD:g}): {', '.join(d[-2:] for d in keep_days)}\n"
            f"Not used (all Kp<{KP_THRESHOLD:g}): "
            f"{', '.join(d[-2:] for d in other_days) or 'none'}"
        )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def run_mode(mode: str) -> None:
    keep, other, selection_label = select_days(mode)
    print(f"\n=== mode={mode} ===")
    print(f"Selected days ({len(keep)}): {keep}")
    print(f"Other days ({len(other)}): {other}")
    print(f"MLAT edges: {mlat_bin_edges()}")

    df = load_samples(keep)
    print(f"Total finite samples: {len(df):,}")

    mlt_c, mlat_c, med, cnt = median_grid(df)
    filled = np.isfinite(med).sum()
    print(f"Filled bins: {filled} / {med.size}")
    print(f"Median σφ range: {np.nanmin(med):.4f}–{np.nanmax(med):.4f} rad")
    print(f"Total counts (finite only): {int(cnt.sum()):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"sigma_phi_median_{mode}_oct2024"
    # Keep prior quiet filenames for backward compatibility
    if mode == "quiet":
        grid_npz = OUT_DIR / "sigma_phi_median_grid_oct2024.npz"
        csv_npz = OUT_DIR / "sigma_phi_median_grid_oct2024.csv"
        plot_npz = OUT_DIR / "sigma_phi_median_polar_oct2024.png"
        artifact_name = "sigma_phi_median_polar_oct2024.png"
    else:
        grid_npz = OUT_DIR / f"{stem}_grid.npz"
        csv_npz = OUT_DIR / f"{stem}_grid.csv"
        plot_npz = OUT_DIR / f"{stem}_polar.png"
        artifact_name = f"{stem}_polar.png"

    np.savez_compressed(
        grid_npz,
        mlt_centers=mlt_c,
        mlat_centers=mlat_c,
        median_sigma_phi=med,
        counts=cnt,
        keep_days=np.array(keep),
        other_days=np.array(other),
        mode=mode,
        mlt_bin_h=MLT_BIN_H,
        mlat_bin_deg=MLAT_BIN_DEG,
        mlat_rim=MLAT_RIM,
        mlat_pole=MLAT_POLE,
        cbar_vmin=CBAR_VMIN,
        cbar_vmax=CBAR_VMAX,
    )
    print(f"wrote {grid_npz}")

    plot_polar(med, keep, other, mode, selection_label, plot_npz)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_polar(
        med, keep, other, mode, selection_label, ARTIFACT_DIR / artifact_name
    )

    rows = []
    for i, mlat in enumerate(mlat_c):
        for j, mlt in enumerate(mlt_c):
            if np.isfinite(med[i, j]):
                rows.append(
                    {
                        "mlat_center": mlat,
                        "mlt_center": mlt,
                        "median_sigma_phi_rad": med[i, j],
                        "n": int(cnt[i, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(csv_npz, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("quiet", "disturbed", "both"),
        default="both",
        help="Kp day selection (default: both)",
    )
    args = parser.parse_args()
    modes = ("quiet", "disturbed") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run_mode(mode)


if __name__ == "__main__":
    main()

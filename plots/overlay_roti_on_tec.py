#!/usr/bin/env python3
"""Overlay Madrigal ROTI pierce points on a GNSS TEC map image.

Matches the TEC map window:
  2025-11-12 (day 316) UT 04:20-04:40, lon -160..-40, lat 0..70
  ROTI filled circles for values in [0.2, 0.8] TECU/s
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

# Georeference of the TEC PNG (772x550), equirectangular
MAP_LEFT, MAP_RIGHT = 72, 708
MAP_TOP, MAP_BOTTOM = 113, 488
LON_MIN, LON_MAX = -160.0, -40.0
LAT_MIN, LAT_MAX = 0.0, 70.0
ROTI_MIN, ROTI_MAX = 0.2, 0.8


def load_roti(
    h5_path: Path,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    with h5py.File(h5_path, "r") as f:
        d = f["Data/Table Layout"]
        n = d.shape[0]
        chunk = 2_000_000
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            ut1 = d["ut1_unix"][sl]
            m = (ut1 >= t0) & (ut1 < t1)
            if not m.any():
                continue
            rows = d[sl][m]
            lon = np.asarray(rows["glon"], dtype=float)
            lat = np.asarray(rows["gdlat"], dtype=float)
            roti = np.asarray(rows["roti"], dtype=float)
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            sel = (
                (lon >= LON_MIN)
                & (lon <= LON_MAX)
                & (lat >= LAT_MIN)
                & (lat <= LAT_MAX)
                & (roti >= ROTI_MIN)
                & (roti <= ROTI_MAX)
            )
            if sel.any():
                chunks.append((lon[sel], lat[sel], roti[sel]))
    if not chunks:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    return (
        np.concatenate([c[0] for c in chunks]),
        np.concatenate([c[1] for c in chunks]),
        np.concatenate([c[2] for c in chunks]),
    )


def lonlat_to_px(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = MAP_LEFT + (lon - LON_MIN) / (LON_MAX - LON_MIN) * (MAP_RIGHT - MAP_LEFT)
    y = MAP_BOTTOM - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (MAP_BOTTOM - MAP_TOP)
    return x, y


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tec_path = Path(
        "/home/ubuntu/.cursor/projects/workspace/assets/"
        "3e99fdfb-b735-4418-824e-3e0c4a2189c2.png"
    )
    if not tec_path.exists():
        tec_path = root / "plots" / "tec_2025-11-12_0420-0440.png"
    h5_path = Path("/tmp/roti_316_2025.h5")
    out_path = root / "plots" / "roti_over_tec_2025-11-12_0420-0440.png"

    t0 = datetime(2025, 11, 12, 4, 20, 0, tzinfo=timezone.utc).timestamp()
    t1 = datetime(2025, 11, 12, 4, 40, 0, tzinfo=timezone.utc).timestamp()

    lon, lat, roti = load_roti(h5_path, t0, t1)
    print(f"ROTI points: {len(roti)}")

    tec = np.asarray(Image.open(tec_path))
    h, w = tec.shape[:2]
    x, y = lonlat_to_px(lon, lat)
    order = np.argsort(roti)
    x, y, roti = x[order], y[order], roti[order]

    fig, ax = plt.subplots(figsize=(11, 7.8), dpi=140)
    ax.imshow(tec, extent=[0, w, h, 0], origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    norm = Normalize(vmin=ROTI_MIN, vmax=ROTI_MAX)
    sc = ax.scatter(
        x,
        y,
        c=roti,
        s=18,
        cmap="plasma",
        norm=norm,
        alpha=0.75,
        edgecolors="k",
        linewidths=0.25,
        zorder=5,
    )

    cax = fig.add_axes([0.08, 0.22, 0.015, 0.45])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("ROTI (TECU/s)", fontsize=9)
    cb.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cb.ax.tick_params(labelsize=8)

    ax.text(
        0.02,
        0.02,
        f"ROTI {ROTI_MIN}–{ROTI_MAX} overlay  |  n={len(roti):,} pierce points  |  "
        "2025-11-12 04:20–04:40 UT",
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3),
        zorder=6,
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Global ROTI/TEC map in the style of the phase-scintillation/TEC reference plot.

Uses:
  - Madrigal gridded TEC HDF5 (native 1 deg x 1 deg x 5 min)
  - Rebinned to 3 deg x 3 deg with median over space and the 20-min window
  - Madrigal ROTI HDF5 table as magenta x markers (size ∝ 0.2-1.0 TECU/s)

Default window: 2026-03-22 13:40-14:00 UT
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np

ROTI_MIN, ROTI_MAX = 0.2, 1.0
SIZE_MIN, SIZE_MAX = 12.0, 140.0
MARKER_COLOR = "#E6007A"  # magenta on viridis TEC
BIN_DEG = 3.0


def load_tec_median_3deg(tec_path: Path, t0: float, t1: float):
    """Median TEC in 3x3 deg bins over all 1-deg cells and times in [t0, t1)."""
    with h5py.File(tec_path, "r") as f:
        ts = f["Data/Array Layout/timestamps"][:]
        lat1 = f["Data/Array Layout/gdlat"][:]
        lon1 = f["Data/Array Layout/glon"][:]
        idx = np.where((ts >= t0) & (ts < t1))[0]
        if idx.size == 0:
            raise ValueError("No TEC timestamps in requested window")
        tec_stack = f["Data/Array Layout/2D Parameters/tec"][:, :, idx]

    lat_edges = np.arange(-90.0, 90.0 + BIN_DEG, BIN_DEG)
    lon_edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
    nlat_b = len(lat_edges) - 1
    nlon_b = len(lon_edges) - 1
    tec3 = np.full((nlat_b, nlon_b), np.nan, dtype=float)

    lat_bin = np.clip(np.digitize(lat1, lat_edges) - 1, 0, nlat_b - 1)
    lon_bin = np.clip(np.digitize(lon1, lon_edges) - 1, 0, nlon_b - 1)

    for i in range(nlat_b):
        lat_m = lat_bin == i
        if not lat_m.any():
            continue
        for j in range(nlon_b):
            lon_m = lon_bin == j
            if not lon_m.any():
                continue
            block = tec_stack[np.ix_(lat_m, lon_m)]
            vals = block[np.isfinite(block)]
            if vals.size:
                tec3[i, j] = float(np.median(vals))

    return lat_edges, lon_edges, tec3


def load_roti(roti_path: Path, t0: float, t1: float):
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    with h5py.File(roti_path, "r") as f:
        d = f["Data/Table Layout"]
        n = d.shape[0]
        for i in range(0, n, 3_000_000):
            sl = slice(i, min(i + 3_000_000, n))
            ut = d["ut1_unix"][sl]
            m = (ut >= t0) & (ut < t1)
            if not m.any():
                continue
            rows = d[sl][m]
            lon = np.asarray(rows["glon"], dtype=float)
            lat = np.asarray(rows["gdlat"], dtype=float)
            roti = np.asarray(rows["roti"], dtype=float)
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            sel = (roti >= ROTI_MIN) & (roti <= ROTI_MAX)
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


def roti_to_size(roti: np.ndarray) -> np.ndarray:
    frac = (roti - ROTI_MIN) / (ROTI_MAX - ROTI_MIN)
    return SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tec_path = Path("/tmp/gps260322g.002.hdf5")
    roti_path = Path("/tmp/roti_081_2026.h5")
    out_path = root / "plots" / "roti_tec_global_2026-03-22_1340-1400.png"

    t0 = datetime(2026, 3, 22, 13, 40, 0, tzinfo=timezone.utc).timestamp()
    t1 = datetime(2026, 3, 22, 14, 0, 0, tzinfo=timezone.utc).timestamp()

    lat_edges, lon_edges, tec = load_tec_median_3deg(tec_path, t0, t1)
    rlon, rlat, roti = load_roti(roti_path, t0, t1)
    print(f"TEC grid: {tec.shape[0]} x {tec.shape[1]} ({BIN_DEG:.0f} deg bins)")
    print(f"ROTI points: {len(roti)}")

    sizes = roti_to_size(roti)
    order = np.argsort(roti)
    rlon, rlat, roti, sizes = rlon[order], rlat[order], roti[order], sizes[order]

    fig = plt.figure(figsize=(14, 7.2), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor("white")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    tec_masked = np.ma.masked_invalid(tec)

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        tec_masked,
        cmap=cmap,
        vmin=0,
        vmax=40,
        transform=ccrs.PlateCarree(),
        shading="flat",
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.5,
        edgecolor="k",
        zorder=2,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.3,
        edgecolor="0.3",
        zorder=2,
    )
    ax.gridlines(
        draw_labels=False,
        linewidth=0.4,
        color="0.5",
        alpha=0.5,
        linestyle="--",
        zorder=3,
    )
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    ax.scatter(
        rlon,
        rlat,
        s=sizes,
        c=MARKER_COLOR,
        marker="x",
        linewidths=1.15,
        alpha=0.95,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    cbar = fig.colorbar(
        mesh, ax=ax, orientation="vertical", fraction=0.025, pad=0.02, extend="max"
    )
    cbar.set_label("Total Electron Content [TECU]", fontsize=10)
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 30, 35, 40])

    handles = []
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        s = float(roti_to_size(np.array([v]))[0])
        handles.append(
            ax.scatter(
                [],
                [],
                s=s,
                c=MARKER_COLOR,
                marker="x",
                linewidths=1.2,
                label=f"ROTI {v:.1f}",
            )
        )
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        framealpha=0.92,
        fontsize=8,
        title="ROTI [TECU/s]",
        title_fontsize=9,
        borderpad=0.8,
        labelspacing=1.15,
        handletextpad=0.9,
        scatterpoints=1,
    )
    leg.set_zorder(10)

    ax.set_title(
        "ROTI/TEC map for 2026-03-22 13:40:00+00:00 - 14:00:00",
        fontsize=13,
        pad=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.15
    )
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

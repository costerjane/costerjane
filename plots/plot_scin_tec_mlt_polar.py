#!/usr/bin/env python3
"""Polar AACGM MLat / MLT map: Madrigal TEC + phase scintillation (sigma_phi).

Default: 2026-03-22 14:00-14:20 UT, mlat >= 40°, 12 MLT at top.
TEC from gps260322g.002.hdf5; scintillation from scin_20260322.001.hdf5.

Layout uses a GridSpec with a dedicated square polar axes so the full
MLT circle is visible (earlier tight/legend layout clipped to one sector).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aacgmv2
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

MLAT_MIN = 40.0
PIERCE_ALT = 350.0
TEC_ALT = 350.0


def mlt_mlat_to_polar(mlt, mlat):
    """12 MLT at top, 06 right, 00 bottom, 18 left; r = 90 - mlat."""
    theta = np.deg2rad(15.0 * np.asarray(mlt, dtype=float) - 90.0)
    r = 90.0 - np.asarray(mlat, dtype=float)
    return theta, r


def geo_to_aacgm_single_time(lat, lon, alt, d):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    d = d.replace(tzinfo=None) if d.tzinfo else d
    mlat, mlon, mlt = aacgmv2.get_aacgm_coord_arr(
        lat, lon, np.asarray(alt, dtype=float), d
    )
    return np.asarray(mlat, float), np.asarray(mlon, float), np.asarray(mlt, float)


def geo_to_aacgm_many_times(lat, lon, alt, times):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    alt = np.asarray(alt, float)
    mlat = np.full(lat.shape, np.nan)
    mlon = np.full(lat.shape, np.nan)
    mlt = np.full(lat.shape, np.nan)
    keys = np.array(
        [tt.replace(second=0, microsecond=0, tzinfo=None) for tt in times]
    )
    for u in np.unique(keys):
        sel = keys == u
        a, b, c = aacgmv2.get_aacgm_coord_arr(lat[sel], lon[sel], alt[sel], u)
        mlat[sel], mlon[sel], mlt[sel] = np.asarray(a), np.asarray(b), np.asarray(c)
    return mlat, mlon, mlt


def load_tec_mlt(tec_path: Path, t0: datetime, t1: datetime):
    """Load TEC frames with inclusive endpoint (t1 frame start counts)."""
    t0_ts = t0.timestamp()
    t1_ts = t1.timestamp()
    mlat_l, mlt_l, val_l = [], [], []
    with h5py.File(tec_path, "r") as f:
        ts = f["Data/Array Layout/timestamps"][:]
        lat1 = f["Data/Array Layout/gdlat"][:]
        lon1 = f["Data/Array Layout/glon"][:]
        idx = np.where((ts >= t0_ts) & (ts <= t1_ts))[0]
        lon_g, lat_g = np.meshgrid(lon1, lat1)
        for i in idx:
            tec = f["Data/Array Layout/2D Parameters/tec"][:, :, i]
            good = np.isfinite(tec)
            if not good.any():
                continue
            glat, glon, vals = lat_g[good], lon_g[good], tec[good]
            d = datetime.fromtimestamp(float(ts[i]), timezone.utc)
            mlat, _, mlt = geo_to_aacgm_single_time(
                glat, glon, np.full(glat.shape, TEC_ALT), d
            )
            sel = (
                np.isfinite(mlat)
                & np.isfinite(mlt)
                & (mlat >= MLAT_MIN)
                & (mlat <= 90.0)
            )
            mlat_l.append(mlat[sel])
            mlt_l.append(np.mod(mlt[sel], 24.0))
            val_l.append(vals[sel])
    return (
        np.concatenate(mlat_l),
        np.concatenate(mlt_l),
        np.concatenate(val_l),
        len(idx),
    )


def load_scin_mlt(scin_path: Path, t0: datetime, t1: datetime):
    t0_ts, t1_ts = t0.timestamp(), t1.timestamp()
    with h5py.File(scin_path, "r") as f:
        d = f["Data/Table Layout"]
        ut = d["ut1_unix"][:]
        rows = d[(ut >= t0_ts) & (ut < t1_ts)]
        use = (
            (rows["pierce_alt"] == PIERCE_ALT)
            & np.isfinite(rows["sigma_phi"])
            & np.isfinite(rows["gdlat"])
            & np.isfinite(rows["glon"])
        )
        glat = rows["gdlat"][use]
        glon = rows["glon"][use]
        sph = rows["sigma_phi"][use]
        alts = rows["pierce_alt"][use]
        times = [
            datetime.fromtimestamp(float(u), timezone.utc)
            for u in rows["ut1_unix"][use]
        ]
    mlat, _, mlt = geo_to_aacgm_many_times(glat, glon, alts, times)
    sel = np.isfinite(mlat) & np.isfinite(mlt) & (mlat >= MLAT_MIN) & (mlat <= 90.0)
    return mlat[sel], np.mod(mlt[sel], 24.0), sph[sel]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tec_path = Path("/tmp/gps260322g.002.hdf5")
    scin_path = Path("/tmp/scin_20260322.001.hdf5")
    out_path = root / "plots" / "scin_tec_mlt_polar_2026-03-22_1400-1420.png"

    t0 = datetime(2026, 3, 22, 14, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 3, 22, 14, 20, 0, tzinfo=timezone.utc)

    tec_mlat, tec_mlt, tec_val, n_tec_frames = load_tec_mlt(tec_path, t0, t1)
    sc_mlat, sc_mlt, sph = load_scin_mlt(scin_path, t0, t1)
    print(f"TEC dots={len(tec_val)} frames={n_tec_frames}")
    print(f"scin={len(sph)}  >=0.1={(sph >= 0.1).sum()}")

    fig = plt.figure(figsize=(11.5, 9.0), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 7.5, 0.35], wspace=0.15)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    ax_cb = fig.add_subplot(gs[0, 2])

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 90.0 - MLAT_MIN)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_yticks([90 - x for x in [80, 70, 60, 50, 40]])
    ax.set_yticklabels(["80°", "70°", "60°", "50°", "40°"], fontsize=8)
    mlt_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    ax.set_xticks([np.deg2rad((15 * h - 90) % 360) for h in mlt_hours])
    ax.set_xticklabels([f"{h:02d}" for h in mlt_hours], fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_aspect("equal", adjustable="box")

    th_t, r_t = mlt_mlat_to_polar(tec_mlt, tec_mlat)
    sc_tec = ax.scatter(
        th_t,
        r_t,
        c=tec_val,
        s=8,
        cmap="viridis",
        vmin=0,
        vmax=50,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    bins = [
        (None, 0.1, 10, "#2ca02c", "No significant SigmaPhi"),
        (0.1, 0.2, 22, "#d62728", r"SigmaPhi 0.1 – 0.2"),
        (0.2, 0.3, 36, "#d62728", r"SigmaPhi 0.2 – 0.3"),
        (0.3, 0.4, 52, "#d62728", r"SigmaPhi 0.3 – 0.4"),
        (0.4, 0.5, 72, "#d62728", r"SigmaPhi 0.4 – 0.5"),
        (0.5, 0.6, 95, "#d62728", r"SigmaPhi 0.5 – 0.6"),
        (0.6, None, 125, "#d62728", r"SigmaPhi > 0.6"),
    ]
    th_s, r_s = mlt_mlat_to_polar(sc_mlt, sc_mlat)
    for lo, hi, size, color, _ in bins:
        if lo is None:
            msk = sph < hi
        elif hi is None:
            msk = sph >= lo
        else:
            msk = (sph >= lo) & (sph < hi)
        if not msk.any():
            continue
        ax.scatter(
            th_s[msk],
            r_s[msk],
            s=size,
            c=color,
            marker="o",
            edgecolors="#1f77b4" if lo is not None else "none",
            linewidths=0.45 if lo is not None else 0.0,
            alpha=0.92,
            zorder=4 if lo else 3,
        )

    ax_leg.axis("off")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.text(
        0.05, 0.95, r"$\sigma_\phi$ legend", fontsize=11, fontweight="bold", va="top"
    )
    y = 0.85
    for lo, hi, size, color, label in bins:
        ax_leg.scatter(
            [0.12],
            [y],
            s=size,
            c=color,
            marker="o",
            edgecolors="#1f77b4" if lo is not None else "none",
            linewidths=0.5 if lo is not None else 0.0,
            clip_on=False,
        )
        ax_leg.text(0.28, y, label, fontsize=8, va="center")
        y -= 0.10

    cb = fig.colorbar(sc_tec, cax=ax_cb)
    cb.set_label("Vertical TEC (TECU)", fontsize=10)

    n_sig = int((sph >= 0.1).sum())
    fig.suptitle(
        f"Phase scintillation/TEC map for "
        f"{t0.strftime('%Y-%m-%d %H:%M:%S')}+00:00 - "
        f"{t1.strftime('%Y-%m-%d %H:%M:%S')}+00:00",
        fontsize=12,
        y=0.98,
    )
    ax.set_title(
        "AACGM magnetic latitude & MLT · 12 MLT at top · "
        "CEDAR Madrigal TEC (8000/3500) + scintillation (8010/20000)\n"
        fr"n_scin={len(sph)} ($\sigma_\phi \geq 0.1$: {n_sig}) · "
        f"TEC samples: {n_tec_frames} · mlat ≥ {MLAT_MIN:.0f}°",
        fontsize=8.5,
        pad=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25
    )
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

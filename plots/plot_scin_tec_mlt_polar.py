#!/usr/bin/env python3
"""North-pole AACGM polar map: MLT (angle) vs magnetic latitude (radial).

March 22, 2026 14:00-14:20 UT
  - TEC: /tmp/gps260322g.002.hdf5
  - Scintillation: /tmp/scin_20260322.001.hdf5

Coordinates (aacgmv2, height=350 km):
  - mlat  : AACGM magnetic latitude  (radial; 90 at center, 40 at rim)
  - mlon  : AACGM magnetic longitude (computed; used to form MLT)
  - mlt   : magnetic local time hours (angle; 12 at top, 06 right, 00 bottom, 18 left)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aacgmv2
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

T0 = datetime(2026, 3, 22, 14, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 3, 22, 14, 20, 0, tzinfo=timezone.utc)
MLAT_MIN = 40.0
ALT_KM = 350.0


def to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def geo_to_mag(glat, glon, alt_km, dtime):
    """Geographic lat/lon/alt -> AACGM mlat, mlon, mlt."""
    glat = np.asarray(glat, dtype=float)
    glon = np.asarray(glon, dtype=float)
    glon = np.where(glon > 180.0, glon - 360.0, glon)
    if np.ndim(alt_km) == 0:
        alt = np.full(glat.shape, float(alt_km))
    else:
        alt = np.asarray(alt_km, dtype=float)
    mlat, mlon, mlt = aacgmv2.get_aacgm_coord_arr(
        glat, glon, alt, to_naive_utc(dtime), method="ALLOWTRACE"
    )
    mlat = np.asarray(mlat, dtype=float)
    mlon = np.asarray(mlon, dtype=float)
    mlt = np.mod(np.asarray(mlt, dtype=float), 24.0)
    return mlat, mlon, mlt


def mlt_mlat_to_polar(mlt, mlat):
    """12 MLT at top, 06 right, 00 bottom, 18 left; r = 90 - mlat."""
    theta = np.deg2rad(15.0 * np.asarray(mlt, dtype=float) - 90.0)
    r = 90.0 - np.asarray(mlat, dtype=float)
    return theta, r


def load_tec(tec_path: Path):
    recs = []
    with h5py.File(tec_path, "r") as f:
        ts = f["Data/Array Layout/timestamps"][:]
        lat1 = f["Data/Array Layout/gdlat"][:]
        lon1 = f["Data/Array Layout/glon"][:]
        idx = np.where((ts >= T0.timestamp()) & (ts <= T1.timestamp()))[0]
        lon_g, lat_g = np.meshgrid(lon1, lat1)
        for i in idx:
            arr = f["Data/Array Layout/2D Parameters/tec"][:, :, i]
            good = np.isfinite(arr)
            if not good.any():
                continue
            glat, glon, vals = lat_g[good], lon_g[good], arr[good]
            dtime = datetime.fromtimestamp(float(ts[i]), timezone.utc)
            mlat, mlon, mlt = geo_to_mag(glat, glon, ALT_KM, dtime)
            sel = (
                np.isfinite(mlat)
                & np.isfinite(mlon)
                & np.isfinite(mlt)
                & (mlat >= MLAT_MIN)
                & (mlat <= 90.0)
            )
            recs.append((mlat[sel], mlon[sel], mlt[sel], vals[sel]))
    return (
        np.concatenate([r[0] for r in recs]),
        np.concatenate([r[1] for r in recs]),
        np.concatenate([r[2] for r in recs]),
        np.concatenate([r[3] for r in recs]),
        len(idx),
    )


def load_scin(scin_path: Path):
    with h5py.File(scin_path, "r") as f:
        tab = f["Data/Table Layout"]
        ut = tab["ut1_unix"][:]
        rows = tab[(ut >= T0.timestamp()) & (ut < T1.timestamp())]
        use = (
            (rows["pierce_alt"] == ALT_KM)
            & np.isfinite(rows["sigma_phi"])
            & np.isfinite(rows["gdlat"])
            & np.isfinite(rows["glon"])
        )
        glat = np.asarray(rows["gdlat"][use], float)
        glon = np.asarray(rows["glon"][use], float)
        sph = np.asarray(rows["sigma_phi"][use], float)
        uts = np.asarray(rows["ut1_unix"][use], float)

    mlat = np.full(glat.shape, np.nan)
    mlon = np.full(glat.shape, np.nan)
    mlt = np.full(glat.shape, np.nan)
    minutes = np.array(
        [
            datetime.fromtimestamp(u, timezone.utc).replace(second=0, microsecond=0)
            for u in uts
        ]
    )
    for u in np.unique(minutes):
        sel = minutes == u
        a, b, c = geo_to_mag(glat[sel], glon[sel], ALT_KM, u)
        mlat[sel], mlon[sel], mlt[sel] = a, b, c

    ok = (
        np.isfinite(mlat)
        & np.isfinite(mlon)
        & np.isfinite(mlt)
        & (mlat >= MLAT_MIN)
        & (mlat <= 90.0)
    )
    return mlat[ok], mlon[ok], mlt[ok], sph[ok]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tec_path = Path("/tmp/gps260322g.002.hdf5")
    scin_path = Path("/tmp/scin_20260322.001.hdf5")
    out_path = root / "plots" / "scin_tec_mlt_polar_2026-03-22_1400-1420.png"

    tec_mlat, tec_mlon, tec_mlt, tec_val, n_frames = load_tec(tec_path)
    sc_mlat, sc_mlon, sc_mlt, sph = load_scin(scin_path)
    print(
        f"TEC n={len(tec_val)} frames={n_frames} | "
        f"scin n={len(sph)} >=0.1={(sph >= 0.1).sum()}"
    )
    print(
        f"TEC mlon range [{tec_mlon.min():.1f}, {tec_mlon.max():.1f}]  "
        f"scin mlon range [{sc_mlon.min():.1f}, {sc_mlon.max():.1f}]"
    )

    fig = plt.figure(figsize=(12.0, 9.2), dpi=160)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 8.0, 0.38], wspace=0.12)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    cax = fig.add_subplot(gs[0, 2])

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 90.0 - MLAT_MIN)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([90 - la for la in [80, 70, 60, 50, 40]])
    ax.set_yticklabels(["80°", "70°", "60°", "50°", "40°"], fontsize=8, color="0.25")
    mlt_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    ax.set_xticks([np.deg2rad((15 * h - 90) % 360) for h in mlt_hours])
    ax.set_xticklabels([f"{h:02d}" for h in mlt_hours], fontsize=12)
    ax.grid(True, linestyle="--", color="0.55", alpha=0.7, linewidth=0.7)
    ax.set_facecolor("white")

    th_t, r_t = mlt_mlat_to_polar(tec_mlt, tec_mlat)
    tec_sc = ax.scatter(
        th_t,
        r_t,
        c=tec_val,
        s=7,
        cmap="viridis",
        vmin=0,
        vmax=35,
        linewidths=0,
        alpha=0.95,
        rasterized=True,
        zorder=2,
    )

    # Marker areas sized like the reference polar maps (largest very prominent)
    bins = [
        (None, 0.1, 22, "#2ca02c", "No significant SigmaPhi"),
        (0.1, 0.2, 80, "#d62728", "SigmaPhi 0.1 – 0.2"),
        (0.2, 0.3, 150, "#d62728", "SigmaPhi 0.2 – 0.3"),
        (0.3, 0.4, 250, "#d62728", "SigmaPhi 0.3 – 0.4"),
        (0.4, 0.5, 380, "#d62728", "SigmaPhi 0.4 – 0.5"),
        (0.5, 0.6, 550, "#d62728", "SigmaPhi 0.5 – 0.6"),
        (0.6, None, 750, "#d62728", "SigmaPhi > 0.6"),
    ]
    th_s, r_s = mlt_mlat_to_polar(sc_mlt, sc_mlat)
    for lo, hi, size, color, _lab in bins:
        if lo is None:
            msk = sph < hi
        elif hi is None:
            msk = sph >= lo
        else:
            msk = (sph >= lo) & (sph < hi)
        if not np.any(msk):
            continue
        ax.scatter(
            th_s[msk],
            r_s[msk],
            s=size,
            c=color,
            marker="o",
            edgecolors="#1f77b4" if lo is not None else "none",
            linewidths=0.4 if lo is not None else 0.0,
            alpha=0.9,
            zorder=4 if lo is not None else 3,
        )

    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.98, r"$\sigma_\phi$ legend", fontsize=12, fontweight="bold", va="top")
    yy = 0.88
    for lo, hi, size, color, lab in bins:
        ax_leg.scatter(
            [0.10],
            [yy],
            s=size,
            c=color,
            marker="o",
            edgecolors="#1f77b4" if lo is not None else "none",
            linewidths=0.5 if lo is not None else 0.0,
            clip_on=False,
        )
        ax_leg.text(0.26, yy, lab, fontsize=9, va="center")
        yy -= 0.10

    cb = fig.colorbar(tec_sc, cax=cax, extend="max")
    cb.set_label("Vertical TEC (TECU)", fontsize=11)
    cb.set_ticks([0, 5, 10, 15, 20, 25, 30, 35])

    n_sig = int(np.sum(sph >= 0.1))
    fig.suptitle(
        f"Phase scintillation/TEC map for "
        f"{T0.strftime('%Y-%m-%d %H:%M:%S')}+00:00 - "
        f"{T1.strftime('%Y-%m-%d %H:%M:%S')}+00:00",
        fontsize=13,
        y=0.985,
    )
    ax.set_title(
        "AACGM magnetic latitude & MLT · 12 MLT at top · "
        "CEDAR Madrigal TEC (8000/3500) + scintillation (8010/20000)\n"
        fr"n_scin={len(sph)} ($\sigma_\phi \geq 0.1$: {n_sig}) · "
        f"TEC samples: {n_frames} · mlat ≥ {MLAT_MIN:.0f}°",
        fontsize=9,
        pad=14,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.3
    )
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

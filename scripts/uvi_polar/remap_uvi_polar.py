#!/usr/bin/env python3
"""Remap SMILE UVI camera-frame quicklook PNGs onto a north-pole MLAT/MLT map.

Public EMDS UVI QL products are camera FOV images. This approximates the L2
LOS projection (Zhang et al. 2025):

  pixel LOS -> intersect R_E+110 km -> GEO -> AACGM MLAT/MLT -> polar plot

Geometry (public QL only; no private L1/L2 CDF):
  - UVI boresight toward geocenter
  - Circular FOV 9.97 deg over 512 px
  - Relative brightness from QL RGB

Note: public 2026-10-10 UVI QL exists only ~00:00-05:06 UT (not 12-24 UT).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aacgmv2
import cartopy.io.shapereader as shpreader
import cdflib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
from scipy.interpolate import griddata
from tqdm import tqdm

RE_KM = 6371.0
EMISSION_ALT_KM = 110.0
FOV_DEG = 9.97
IMG_SIZE = 512
MLAT_MIN = 50.0

logging.getLogger("aacgmv2").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class OrbitAtt:
    unix: np.ndarray
    pos_j2000_km: np.ndarray


def load_orbit(cdf_path: Path) -> OrbitAtt:
    cdf = cdflib.CDF(str(cdf_path))
    epoch = np.asarray(cdf.varget("EPOCH")).reshape(-1)
    return OrbitAtt(
        unix=np.asarray(cdflib.cdfepoch.unixtime(epoch), dtype=float),
        pos_j2000_km=np.asarray(cdf.varget("POSITION_J2000"), dtype=float),
    )


def parse_ql_time(name: str) -> datetime:
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if not m:
        raise ValueError(f"Cannot parse time from {name}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def interpolate_position(orbit: OrbitAtt, dt: datetime) -> np.ndarray:
    t = dt.timestamp()
    idx = int(np.searchsorted(orbit.unix, t))
    idx = min(max(idx, 1), len(orbit.unix) - 1)
    t0, t1 = orbit.unix[idx - 1], orbit.unix[idx]
    if t1 == t0:
        return orbit.pos_j2000_km[idx]
    w = (t - t0) / (t1 - t0)
    return (1.0 - w) * orbit.pos_j2000_km[idx - 1] + w * orbit.pos_j2000_km[idx]


def camera_basis(sc_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bore = -sc_pos / np.linalg.norm(sc_pos)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(bore, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(bore, ref)
    x /= np.linalg.norm(x)
    y = np.cross(bore, x)
    y = -y  # image y downward
    x = np.cross(y, bore)
    x /= np.linalg.norm(x)
    return bore, x, y


def j2000_to_geodetic(pos_km: np.ndarray, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Rotate J2000 -> approximate ECEF via GMST, return lat/lon degrees."""
    unix = dt.timestamp()
    jd = unix / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t
    gmst = np.deg2rad(gmst % 360.0)
    c, s = np.cos(gmst), np.sin(gmst)
    x = c * pos_km[..., 0] + s * pos_km[..., 1]
    y = -s * pos_km[..., 0] + c * pos_km[..., 1]
    z = pos_km[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def load_intensity(png: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=float)
    inten = rgb.max(axis=2)
    inten[450:, :300] = np.nan
    yy, xx = np.indices(inten.shape)
    r = np.sqrt((xx - 255.5) ** 2 + (yy - 255.5) ** 2)
    inten[r > 250] = np.nan
    return inten


def remap_frame(
    inten: np.ndarray,
    sc_pos: np.ndarray,
    dt: datetime,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bore, xhat, yhat = camera_basis(sc_pos)
    half = FOV_DEG / 2.0
    radius = RE_KM + EMISSION_ALT_KM

    ys = np.arange(0, IMG_SIZE, step)
    xs = np.arange(0, IMG_SIZE, step)
    XX, YY = np.meshgrid(xs, ys)
    vals = inten[YY, XX]
    valid = np.isfinite(vals)
    XX, YY, vals = XX[valid], YY[valid], vals[valid]
    if XX.size == 0:
        return np.array([]), np.array([]), np.array([])

    ax = (XX - 255.5) / (IMG_SIZE / 2.0) * half
    ay = (YY - 255.5) / (IMG_SIZE / 2.0) * half
    d = (
        bore[None, :]
        + np.tan(np.deg2rad(ax))[:, None] * xhat[None, :]
        + np.tan(np.deg2rad(ay))[:, None] * yhat[None, :]
    )
    d /= np.linalg.norm(d, axis=1)[:, None]

    # Ray-sphere intersection (vectorized)
    o = sc_pos
    b = 2.0 * (d @ o)
    c = np.dot(o, o) - radius * radius
    disc = b * b - 4 * c
    ok = disc >= 0
    d, vals, b, disc = d[ok], vals[ok], b[ok], disc[ok]
    if d.size == 0:
        return np.array([]), np.array([]), np.array([])
    s = np.sqrt(disc)
    t1 = (-b - s) / 2.0
    t2 = (-b + s) / 2.0
    t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    ok = t > 0
    d, vals, t = d[ok], vals[ok], t[ok]
    hit = o[None, :] + t[:, None] * d

    lat, lon = j2000_to_geodetic(hit, dt)
    keep = lat >= 40.0
    lat, lon, vals = lat[keep], lon[keep], vals[keep]
    if lat.size == 0:
        return np.array([]), np.array([]), np.array([])

    try:
        mlat, _mlon, mlt = aacgmv2.get_aacgm_coord_arr(lat, lon, EMISSION_ALT_KM, dt)
        mlat = np.asarray(mlat, dtype=float)
        mlt = np.asarray(mlt, dtype=float)
    except Exception:
        mlat = np.full(lat.shape, np.nan)
        mlt = np.full(lat.shape, np.nan)
        for j, (la, lo) in enumerate(zip(lat, lon)):
            try:
                a, _, m = aacgmv2.get_aacgm_coord(float(la), float(lo), EMISSION_ALT_KM, dt)
                mlat[j], mlt[j] = a, m
            except Exception:
                pass
    good = np.isfinite(mlat) & np.isfinite(mlt) & (mlat >= MLAT_MIN)
    return mlat[good], (mlt[good] % 24.0), vals[good]


def mlt_mlat_to_xy(mlt: np.ndarray, mlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad((mlt - 12.0) * 15.0)  # noon at top
    r = (90.0 - mlat) / (90.0 - MLAT_MIN)
    return r * np.sin(theta), r * np.cos(theta)


_COAST_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def coast_xy(dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    key = dt.strftime("%Y%m%d%H")
    if key in _COAST_CACHE:
        return _COAST_CACHE[key]
    xs, ys = [], []
    try:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="coastline")
        for rec in shpreader.Reader(shp).records():
            geom = rec.geometry
            lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                coords = np.asarray(line.coords)
                for lat, lon in zip(coords[::2, 1], coords[::2, 0]):
                    if lat < MLAT_MIN - 2:
                        continue
                    try:
                        mlat, _, mlt = aacgmv2.get_aacgm_coord(float(lat), float(lon), 0.0, dt)
                    except Exception:
                        continue
                    if mlat is None or not np.isfinite(mlat) or mlat < MLAT_MIN:
                        continue
                    x, y = mlt_mlat_to_xy(np.array([mlt % 24]), np.array([mlat]))
                    xs.append(x[0])
                    ys.append(y[0])
    except Exception:
        pass
    out = (np.asarray(xs), np.asarray(ys))
    _COAST_CACHE[key] = out
    return out


def render_polar(mlat, mlt, vals, dt: datetime, out_path: Path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.2, 7.8), dpi=110)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="#04070f", zorder=0))

    if len(vals) > 20:
        x, y = mlt_mlat_to_xy(mlt, mlat)
        # Interpolate onto regular Cartesian polar disk for filled look
        xi = np.linspace(-1, 1, 360)
        yi = np.linspace(-1, 1, 360)
        XX, YY = np.meshgrid(xi, yi)
        RR = np.sqrt(XX**2 + YY**2)
        Zi = griddata((x, y), vals, (XX, YY), method="linear")
        Zi = np.ma.array(Zi, mask=~np.isfinite(Zi) | (RR > 1.0))
        im = ax.pcolormesh(
            XX,
            YY,
            Zi,
            cmap="plasma",
            norm=Normalize(vmin=vmin, vmax=vmax),
            shading="auto",
            zorder=2,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        cbar.set_label("Ultraviolet brightness (relative)", color="white")
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")
        cbar.ax.set_xticks([vmin, vmax])
        cbar.ax.set_xticklabels(["Dim", "Bright"])

    for mlat_ring in [50, 60, 70, 80]:
        r = (90 - mlat_ring) / (90 - MLAT_MIN)
        ax.add_patch(
            plt.Circle((0, 0), r, fill=False, ls="--", lw=0.7, color="white", alpha=0.55, zorder=3)
        )
        ax.text(0.02, r + 0.015, f"{mlat_ring}°", color="white", fontsize=8, alpha=0.85)
    ax.text(0.72, 0.42, "70° MLAT", color="#7ec8ff", fontsize=8)
    ax.text(0.82, 0.72, "60° MLAT", color="#7ec8ff", fontsize=8)

    for mlt_h in range(0, 24, 3):
        x0, y0 = mlt_mlat_to_xy(np.array([float(mlt_h)]), np.array([MLAT_MIN]))
        ax.plot([0, x0[0]], [0, y0[0]], ls="--", lw=0.5, color="white", alpha=0.35, zorder=3)

    x_n, y_n = mlt_mlat_to_xy(np.array([12.0, 0.0]), np.array([MLAT_MIN, MLAT_MIN]))
    ax.plot([x_n[0], x_n[1]], [y_n[0], y_n[1]], ls="--", lw=1.5, color="white", alpha=0.85, zorder=4)

    cx, cy = coast_xy(dt)
    if len(cx):
        ax.scatter(cx, cy, s=0.35, c="#e6c35c", linewidths=0, zorder=5, alpha=0.9)

    ax.set_title(
        f"SMILE UVI quicklook\n{dt.strftime('%Y-%m-%d %H:%M:%S')} UT",
        color="white",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ql-dir", type=Path, required=True)
    ap.add_argument("--orbit-cdf", type=Path, required=True)
    ap.add_argument("--frame-dir", type=Path, required=True)
    ap.add_argument("--start", default="2026-10-10T00:00:00")
    ap.add_argument("--stop", default="2026-10-10T24:00:00")
    ap.add_argument("--step-px", type=int, default=2)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(args.stop).replace(tzinfo=timezone.utc)
    args.frame_dir.mkdir(parents=True, exist_ok=True)

    orbit = load_orbit(args.orbit_cdf)
    selected = []
    for fp in sorted(args.ql_dir.glob("SMILE_UVI_QL_AURORA_*.png")):
        dt = parse_ql_time(fp.name)
        if start <= dt < stop:
            selected.append((dt, fp))
    selected = selected[:: max(args.every, 1)]
    if args.limit:
        selected = selected[: args.limit]

    if not selected:
        print(f"No QL frames in {start} .. {stop}")
        print("Note: public Oct 10 UVI QL covers only ~00:00-05:06 UT.")
        return 1

    print(f"Rendering {len(selected)} frames from {selected[0][0]} to {selected[-1][0]}")

    sample = []
    for dt, fp in selected[:: max(1, len(selected) // 6)]:
        sc = interpolate_position(orbit, dt)
        _, _, vals = remap_frame(load_intensity(fp), sc, dt, step=max(args.step_px, 3))
        if len(vals):
            sample.append(vals)
    if sample:
        allv = np.concatenate(sample)
        vmin, vmax = np.nanpercentile(allv, [5, 99])
    else:
        vmin, vmax = 0.0, 255.0

    manifest = args.frame_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ut", "n_samples", "path"])
        for i, (dt, fp) in enumerate(tqdm(selected)):
            sc = interpolate_position(orbit, dt)
            mlat, mlt, vals = remap_frame(load_intensity(fp), sc, dt, step=args.step_px)
            out = args.frame_dir / f"frame_{i:04d}_{dt.strftime('%H%M%S')}.png"
            render_polar(mlat, mlt, vals, dt, out, vmin=vmin, vmax=vmax)
            w.writerow([i, dt.isoformat(), len(vals), out.name])
    print("Wrote", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

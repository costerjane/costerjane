#!/usr/bin/env python3
"""Remap SMILE UVI quicklook PNGs onto a north-pole MLAT/MLT map.

Follows the L2-style geometric chain:
  UVI image + SC position/attitude
  -> per-pixel camera LOS
  -> intersect emission shell at R_E + 110 km
  -> GEO (GMST)
  -> AACGM MLAT / MLT
  -> polar plot with AACGM coastlines

Camera model:
  - Body +Z is PLM normal; UVI optical axis is Ry(+24 deg)*+Z in the body frame
    (Zhang et al. 2025; SMILE spacecraft paper: SXI/UVI 24 deg offset in Z/X).
  - Quaternion_J2000 (qx,qy,qz,qw) rotates body -> J2000.
  - When the Earth disk is detectable in the QL PNG, boresight/roll are refined
    so the LOS through the Earth-disk centroid points at geocenter (fixes residual
    mounting/attitude error; critical for correct polar morphology).
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
from matplotlib.colors import Normalize, PowerNorm
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata
from tqdm import tqdm

RE_KM = 6371.0
EMISSION_ALT_KM = 110.0
FOV_DEG = 9.97
IMG_SIZE = 512
MLAT_MIN = 50.0
UVI_MOUNT_Y_DEG = 24.0  # UVI optical axis = Ry(+24°)·+Z in body frame

logging.getLogger("aacgmv2").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AuxData:
    unix: np.ndarray
    pos_j2000_km: np.ndarray
    quat_xyzw: np.ndarray  # body -> J2000


def load_aux(orbit_cdf: Path, att_cdf: Path) -> AuxData:
    orb = cdflib.CDF(str(orbit_cdf))
    att = cdflib.CDF(str(att_cdf))
    epoch = np.asarray(orb.varget("EPOCH")).reshape(-1)
    return AuxData(
        unix=np.asarray(cdflib.cdfepoch.unixtime(epoch), dtype=float),
        pos_j2000_km=np.asarray(orb.varget("POSITION_J2000"), dtype=float),
        quat_xyzw=np.asarray(att.varget("Quaternion_J2000"), dtype=float),
    )


def parse_ql_time(name: str) -> datetime:
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if not m:
        raise ValueError(f"Cannot parse time from {name}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def interp_row(unix: np.ndarray, arr: np.ndarray, t: float) -> np.ndarray:
    idx = int(np.searchsorted(unix, t))
    idx = min(max(idx, 1), len(unix) - 1)
    t0, t1 = unix[idx - 1], unix[idx]
    if t1 == t0:
        return arr[idx]
    w = (t - t0) / (t1 - t0)
    return (1.0 - w) * arr[idx - 1] + w * arr[idx]


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Body -> inertial rotation from quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def roty(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def uvi_boresight_body() -> np.ndarray:
    # PLM normal = +Z; UVI optical axis tipped +24° about +Y toward +X (Earth)
    v = roty(UVI_MOUNT_Y_DEG) @ np.array([0.0, 0.0, 1.0])
    return v / np.linalg.norm(v)


def load_intensity(png: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=float)
    # Prefer luminance; QL aurora/dayglow is bright cyan/white on dark blue
    inten = 0.2 * rgb[:, :, 0] + 0.3 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2]
    inten[450:, :300] = np.nan  # text overlay
    yy, xx = np.indices(inten.shape)
    r = np.sqrt((xx - 255.5) ** 2 + (yy - 255.5) ** 2)
    inten[r > 250] = np.nan
    return inten


def detect_earth_centroid(inten: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, radius_px) of Earth disk, or None if not found."""
    valid = np.isfinite(inten)
    if valid.sum() < 1000:
        return None
    vals = inten[valid]
    thr = np.nanpercentile(vals, 65)
    mask = valid & (inten > thr)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    earth_lab = 1 + int(np.argmax(sizes))
    earth = lab == earth_lab
    if earth.sum() < 2000:
        return None
    yy, xx = np.indices(inten.shape)
    w = np.where(np.isfinite(inten), inten, 0.0)
    cx = float((xx[earth] * w[earth]).sum() / w[earth].sum())
    cy = float((yy[earth] * w[earth]).sum() / w[earth].sum())
    rad = np.sqrt((xx[earth] - cx) ** 2 + (yy[earth] - cy) ** 2)
    r90 = float(np.percentile(rad, 90))
    # Sanity: Earth angular size should be plausible for SMILE distances
    if r90 < 40 or r90 > 240:
        return None
    return cx, cy, r90


def j2000_to_geodetic(pos_km: np.ndarray, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Convert J2000/GCRS cartesian km positions to geodetic lat/lon (ITRS)."""
    try:
        from astropy.coordinates import SkyCoord, ITRS
        from astropy.time import Time
        import astropy.units as u

        t = Time(dt)
        pos = np.asarray(pos_km, dtype=float)
        single = pos.ndim == 1
        if single:
            pos = pos[None, :]
        gcrs = SkyCoord(
            x=pos[:, 0] * u.km,
            y=pos[:, 1] * u.km,
            z=pos[:, 2] * u.km,
            representation_type="cartesian",
            frame="gcrs",
            obstime=t,
        )
        itrs = gcrs.transform_to(ITRS(obstime=t))
        lon = np.asarray(itrs.earth_location.geodetic.lon.deg, dtype=float)
        lat = np.asarray(itrs.earth_location.geodetic.lat.deg, dtype=float)
        if single:
            return float(lat[0]), float(lon[0])
        return lat, lon
    except Exception:
        # Fallback: GMST rotation about Z (approximate ECEF)
        unix = dt.timestamp()
        jd = unix / 86400.0 + 2440587.5
        tt = (jd - 2451545.0) / 36525.0
        gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * tt * tt
        gmst = np.deg2rad(gmst % 360.0)
        c, s = np.cos(gmst), np.sin(gmst)
        x = c * pos_km[..., 0] + s * pos_km[..., 1]
        y = -s * pos_km[..., 0] + c * pos_km[..., 1]
        z = pos_km[..., 2]
        r = np.sqrt(x * x + y * y + z * z)
        lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon


def build_camera(
    sc_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    earth_px: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (boresight, xhat, yhat) in J2000, or None if Earth not in FOV.

    xhat = +column (right), yhat = +row (down), RH with boresight.
    """
    R = quat_to_R(quat_xyzw)  # body -> J2000
    earth_dir = -sc_pos / np.linalg.norm(sc_pos)

    # Nominal UVI boresight from mounting + attitude
    bore = R @ uvi_boresight_body()
    bore /= np.linalg.norm(bore)
    ang_earth = float(np.degrees(np.arccos(np.clip(bore @ earth_dir, -1.0, 1.0))))

    # If attitude-based boresight points away from Earth, fall back to geocenter
    # pointing only when the Earth disk is visible in the image.
    if ang_earth > 45.0:
        if earth_px is None:
            return None
        bore = earth_dir.copy()

    # Nominal camera x from body +Y projected into focal plane
    by = R @ np.array([0.0, 1.0, 0.0])
    x0 = by - bore * (by @ bore)
    if np.linalg.norm(x0) < 1e-8:
        by = R @ np.array([1.0, 0.0, 0.0])
        x0 = by - bore * (by @ bore)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(bore, x0)  # RH: x × y = bore

    if earth_px is None:
        # Require Earth roughly in FOV (half-FOV ~5 deg + margin)
        if ang_earth > 8.0:
            return None
        return bore, x0, y0

    # Refine so LOS through Earth centroid equals geocenter direction.
    ex, ey, _r = earth_px
    plate = FOV_DEG / IMG_SIZE  # deg / pixel
    ax = (ex - 255.5) * plate
    ay = (ey - 255.5) * plate

    best = None
    for roll in np.linspace(0.0, 360.0, 721, endpoint=False):
        ca, sa = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        xhat = ca * x0 + sa * y0
        yhat = np.cross(bore, xhat)
        db, dx, dy = earth_dir @ bore, earth_dir @ xhat, earth_dir @ yhat
        if db <= 0:
            continue
        ax_p = np.degrees(np.arctan2(dx, db))
        ay_p = np.degrees(np.arctan2(dy, db))
        err = (ax_p - ax) ** 2 + (ay_p - ay) ** 2
        if best is None or err < best[0]:
            best = (err, xhat, yhat)

    if best is None:
        return bore, x0, y0
    _err, xhat, yhat = best

    bore2 = earth_dir - np.tan(np.deg2rad(ax)) * xhat - np.tan(np.deg2rad(ay)) * yhat
    if np.linalg.norm(bore2) < 1e-8:
        return bore, xhat, yhat
    bore2 /= np.linalg.norm(bore2)
    xhat2 = xhat - bore2 * (xhat @ bore2)
    if np.linalg.norm(xhat2) < 1e-8:
        return bore2, xhat, yhat
    xhat2 /= np.linalg.norm(xhat2)
    yhat2 = np.cross(bore2, xhat2)
    return bore2, xhat2, yhat2


def remap_frame(
    inten: np.ndarray,
    sc_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    dt: datetime,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    earth_px = detect_earth_centroid(inten)
    cam = build_camera(sc_pos, quat_xyzw, earth_px)
    if cam is None:
        return np.array([]), np.array([]), np.array([])
    bore, xhat, yhat = cam
    radius = RE_KM + EMISSION_ALT_KM

    ys = np.arange(0, IMG_SIZE, step)
    xs = np.arange(0, IMG_SIZE, step)
    XX, YY = np.meshgrid(xs, ys)
    vals = inten[YY, XX]
    valid = np.isfinite(vals)
    XX, YY, vals = XX[valid], YY[valid], vals[valid]
    if XX.size == 0:
        return np.array([]), np.array([]), np.array([])

    plate = FOV_DEG / IMG_SIZE
    ax = (XX - 255.5) * plate
    ay = (YY - 255.5) * plate
    d = (
        bore[None, :]
        + np.tan(np.deg2rad(ax))[:, None] * xhat[None, :]
        + np.tan(np.deg2rad(ay))[:, None] * yhat[None, :]
    )
    d /= np.linalg.norm(d, axis=1)[:, None]

    # Ray-sphere intersection with Earth-centered sphere
    o = sc_pos
    bcoef = 2.0 * (d @ o)
    ccoef = np.dot(o, o) - radius * radius
    disc = bcoef * bcoef - 4.0 * ccoef
    ok = disc >= 0
    d, vals, bcoef, disc = d[ok], vals[ok], bcoef[ok], disc[ok]
    if d.size == 0:
        return np.array([]), np.array([]), np.array([])
    s = np.sqrt(disc)
    t1 = (-bcoef - s) / 2.0
    t2 = (-bcoef + s) / 2.0
    t = np.where((t1 > 0) & ((t1 <= t2) | (t2 <= 0)), t1, t2)
    ok = t > 0
    d, vals, t = d[ok], vals[ok], t[ok]
    hit = o[None, :] + t[:, None] * d

    lat, lon = j2000_to_geodetic(hit, dt)
    keep = lat >= 35.0
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
    # Noon at top, midnight at bottom; MLT increases eastward (CCW from noon in this x=sin,y=cos)
    theta = np.deg2rad((mlt - 12.0) * 15.0)
    r = (90.0 - mlat) / (90.0 - MLAT_MIN)
    return r * np.sin(theta), r * np.cos(theta)


_COAST_SEGMENTS: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}


def coast_segments(dt: datetime) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coastline polylines in polar MLAT/MLT xy for this UT."""
    key = dt.strftime("%Y%m%d%H")
    if key in _COAST_SEGMENTS:
        return _COAST_SEGMENTS[key]
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="coastline")
        for rec in shpreader.Reader(shp).records():
            geom = rec.geometry
            lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                coords = np.asarray(line.coords)
                if len(coords) < 4:
                    continue
                lons = coords[:, 0]
                lats = coords[:, 1]
                if np.nanmax(lats) < MLAT_MIN - 5:
                    continue
                try:
                    mlat, _mlon, mlt = aacgmv2.get_aacgm_coord_arr(lats, lons, 0.0, dt)
                    mlat = np.asarray(mlat, float)
                    mlt = np.asarray(mlt, float) % 24.0
                except Exception:
                    continue
                good = np.isfinite(mlat) & (mlat >= MLAT_MIN)
                if good.sum() < 2:
                    continue
                x, y = mlt_mlat_to_xy(mlt, mlat)
                # Break across invalid gaps so we don't draw long chords
                x = np.where(good, x, np.nan)
                y = np.where(good, y, np.nan)
                segs.append((x, y))
    except Exception:
        pass
    _COAST_SEGMENTS[key] = segs
    return segs


def render_polar(mlat, mlt, vals, dt: datetime, out_path: Path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.2, 7.8), dpi=120)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="#02040a", zorder=0))

    if len(vals) > 50:
        x, y = mlt_mlat_to_xy(mlt, mlat)
        xi = np.linspace(-1, 1, 420)
        yi = np.linspace(-1, 1, 420)
        XX, YY = np.meshgrid(xi, yi)
        RR = np.sqrt(XX**2 + YY**2)
        Zi = griddata((x, y), vals, (XX, YY), method="linear")
        Zi = np.ma.array(Zi, mask=~np.isfinite(Zi) | (RR > 1.0))
        im = ax.pcolormesh(
            XX,
            YY,
            Zi,
            cmap="nipy_spectral",
            norm=PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax),
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
        ax.text(0.02, r + 0.015, f"{mlat_ring}°", color="0.85", fontsize=8)
    ax.text(0.72, 0.42, "70° MLAT", color="#7ec8ff", fontsize=8)
    ax.text(0.82, 0.72, "60° MLAT", color="#7ec8ff", fontsize=8)

    for mlt_h in range(0, 24, 3):
        x0, y0 = mlt_mlat_to_xy(np.array([float(mlt_h)]), np.array([MLAT_MIN]))
        ax.plot([0, x0[0]], [0, y0[0]], ls="--", lw=0.5, color="white", alpha=0.35, zorder=3)

    x_n, y_n = mlt_mlat_to_xy(np.array([12.0, 0.0]), np.array([MLAT_MIN, MLAT_MIN]))
    ax.plot([x_n[0], x_n[1]], [y_n[0], y_n[1]], ls="--", lw=1.6, color="white", alpha=0.9, zorder=4)

    for cx, cy in coast_segments(dt):
        ax.plot(cx, cy, color="#e6c35c", lw=0.7, solid_capstyle="round", zorder=5, alpha=0.95)

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
    ap.add_argument("--att-cdf", type=Path, required=True)
    ap.add_argument("--frame-dir", type=Path, required=True)
    ap.add_argument("--start", default="2026-10-10T00:00:00")
    ap.add_argument("--stop", default="2026-10-10T24:00:00")
    ap.add_argument("--step-px", type=int, default=1)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(args.stop).replace(tzinfo=timezone.utc)
    args.frame_dir.mkdir(parents=True, exist_ok=True)

    aux = load_aux(args.orbit_cdf, args.att_cdf)
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
        return 1

    print(f"Rendering {len(selected)} frames from {selected[0][0]} to {selected[-1][0]}")

    sample = []
    for dt, fp in selected[:: max(1, len(selected) // 6)]:
        sc = interp_row(aux.unix, aux.pos_j2000_km, dt.timestamp())
        q = interp_row(aux.unix, aux.quat_xyzw, dt.timestamp())
        _, _, vals = remap_frame(load_intensity(fp), sc, q, dt, step=max(args.step_px, 2))
        if len(vals):
            sample.append(vals)
    if sample:
        allv = np.concatenate(sample)
        vmin, vmax = np.nanpercentile(allv, [5, 99.5])
    else:
        vmin, vmax = 0.0, 255.0

    manifest = args.frame_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ut", "n_samples", "earth_px", "path"])
        for i, (dt, fp) in enumerate(tqdm(selected)):
            inten = load_intensity(fp)
            earth = detect_earth_centroid(inten)
            sc = interp_row(aux.unix, aux.pos_j2000_km, dt.timestamp())
            q = interp_row(aux.unix, aux.quat_xyzw, dt.timestamp())
            mlat, mlt, vals = remap_frame(inten, sc, q, dt, step=args.step_px)
            out = args.frame_dir / f"frame_{i:04d}_{dt.strftime('%H%M%S')}.png"
            render_polar(mlat, mlt, vals, dt, out, vmin=vmin, vmax=vmax)
            w.writerow(
                [
                    i,
                    dt.isoformat(),
                    len(vals),
                    "" if earth is None else f"{earth[0]:.1f},{earth[1]:.1f},{earth[2]:.1f}",
                    out.name,
                ]
            )
    print("Wrote", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Remap SMILE UVI quicklook PNGs onto a north-pole geographic map.

Follows the L2-style geometric chain:
  UVI image + SC position/attitude
  -> per-pixel camera LOS
  -> intersect emission shell at R_E + 110 km
  -> GEO (ITRS) lat/lon
  -> polar plot centered on the geographic North Pole

Official SMILE UVI polar quicklooks use *geographic* coordinates with a fixed
longitude orientation (North America / contiguous US at the bottom). The
auroral oval is therefore a ring around the magnetic pole and appears
offset from the geographic center — that is expected.

Camera model:
  - Body +Z is PLM normal; UVI optical axis is Ry(+24 deg)*+Z in the body frame
    (Zhang et al. 2025; SMILE spacecraft paper: SXI/UVI 24 deg offset in Z/X).
  - Quaternion_J2000 (qx,qy,qz,qw) rotates body -> J2000.
  - When the Earth disk is detectable, apply a *tip-only* correction that maps
    the LOS through the limb-fit Earth center onto the geocenter direction,
    preserving attitude roll (critical for FOV morphology).
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

import cartopy.io.shapereader as shpreader
import cdflib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata
from tqdm import tqdm

try:
    import aacgmv2
except ImportError:  # optional: only used for MLAT annotation contours
    aacgmv2 = None

RE_KM = 6371.0
EMISSION_ALT_KM = 110.0
FOV_DEG = 9.97
IMG_SIZE = 512
LAT_MIN = 50.0
# Fixed geographic orientation: 90°W (central Canada / contiguous US) at bottom.
LON_BOTTOM_DEG = -90.0
UVI_MOUNT_Y_DEG = 24.0  # UVI optical axis = Ry(+24°)·+Z in body frame

if aacgmv2 is not None:
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


def expected_earth_radius_px(sc_pos: np.ndarray) -> float:
    r = float(np.linalg.norm(sc_pos))
    ang = np.degrees(np.arcsin(min(0.999999, RE_KM / r)))
    return ang / (FOV_DEG / IMG_SIZE)


def detect_earth_center(
    inten: np.ndarray, sc_pos: np.ndarray
) -> tuple[float, float, float] | None:
    """Limb-based Earth center using expected angular radius from SC range.

    Brightness-weighted centroids bias toward the aurora and distort FOV
    morphology; instead fit the right-hand limb with the known Earth size.
    """
    valid = np.isfinite(inten)
    if valid.sum() < 1000:
        return None
    vals = inten[valid]
    thr = np.nanpercentile(vals, 55)
    mask = valid & (inten > thr)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    earth = lab == (1 + int(np.argmax(sizes)))
    if earth.sum() < 2000:
        return None

    exp_r = expected_earth_radius_px(sc_pos)
    yy, xx = np.indices(inten.shape)
    limb_x: list[float] = []
    limb_y: list[float] = []
    for y in range(IMG_SIZE):
        xs = xx[y][earth[y]]
        if xs.size > 10:
            limb_x.append(float(xs.max()))
            limb_y.append(float(y))
    if len(limb_x) < 30:
        return None
    limb_x_a = np.asarray(limb_x)
    limb_y_a = np.asarray(limb_y)
    cy = 0.5 * (limb_y_a.min() + limb_y_a.max())
    dy = limb_y_a - cy
    ok = np.abs(dy) < exp_r * 0.95
    if ok.sum() < 20:
        return None
    cx_est = limb_x_a[ok] - np.sqrt(np.maximum(exp_r**2 - dy[ok] ** 2, 0.0))
    cx = float(np.median(cx_est))
    cy = float(cy)
    if not (40.0 < cx < 470.0 and 40.0 < cy < 470.0):
        return None
    if exp_r < 40 or exp_r > 240:
        return None
    return cx, cy, float(exp_r)


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


def _attitude_camera(sc_pos: np.ndarray, quat_xyzw: np.ndarray):
    R = quat_to_R(quat_xyzw)  # body -> J2000
    earth_dir = -sc_pos / np.linalg.norm(sc_pos)
    bore = R @ uvi_boresight_body()
    bore /= np.linalg.norm(bore)
    ang_earth = float(np.degrees(np.arccos(np.clip(bore @ earth_dir, -1.0, 1.0))))
    # Camera +column from body -Y (not +Y). Matches QL raster convention:
    # dayglow on dawn/left when Earth is in FOV.
    by = R @ np.array([0.0, -1.0, 0.0])
    x0 = by - bore * (by @ bore)
    if np.linalg.norm(x0) < 1e-8:
        by = R @ np.array([1.0, 0.0, 0.0])
        x0 = by - bore * (by @ bore)
    x0 /= np.linalg.norm(x0)
    # Image +row is down. QL PNGs need the opposite of RH (x × y = bore).
    y0 = -np.cross(bore, x0)
    return bore, x0, y0, earth_dir, ang_earth


def _rodrigues(v: np.ndarray, k: np.ndarray, c: float, s: float) -> np.ndarray:
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1.0 - c)


def build_camera(
    sc_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    earth_px: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (boresight, xhat, yhat) in J2000, or None if Earth not in FOV.

    xhat = +column (right), yhat = +row (down), RH with boresight.

    Earth-disk refinement is *tip-only*: rotate the attitude camera by the
    minimal rotation that maps the LOS through the Earth-center pixel onto the
    geocenter direction. Roll about that axis stays locked to MEAS-ATT.
    """
    bore, x0, y0, earth_dir, ang_earth = _attitude_camera(sc_pos, quat_xyzw)

    if ang_earth > 45.0:
        if earth_px is None:
            return None
        # Far off; seed with geocenter pointing but keep body -Y roll sense
        bore = earth_dir.copy()
        by = quat_to_R(quat_xyzw) @ np.array([0.0, -1.0, 0.0])
        x0 = by - bore * (by @ bore)
        if np.linalg.norm(x0) < 1e-8:
            return None
        x0 /= np.linalg.norm(x0)
        y0 = -np.cross(bore, x0)

    if earth_px is None:
        if ang_earth > 8.0:
            return None
        return bore, x0, y0

    ex, ey, _r = earth_px
    plate = FOV_DEG / IMG_SIZE
    ax = (ex - 255.5) * plate
    ay = (ey - 255.5) * plate
    v0 = bore + np.tan(np.deg2rad(ax)) * x0 + np.tan(np.deg2rad(ay)) * y0
    v0 /= np.linalg.norm(v0)
    v1 = earth_dir
    c = float(np.clip(v0 @ v1, -1.0, 1.0))
    if c > 0.999999:
        return bore, x0, y0
    k = np.cross(v0, v1)
    kn = np.linalg.norm(k)
    if kn < 1e-12:
        return bore, x0, y0
    k /= kn
    s = float(np.sin(np.arccos(c)))
    bore2 = _rodrigues(bore, k, c, s)
    bore2 /= np.linalg.norm(bore2)
    x2 = _rodrigues(x0, k, c, s)
    x2 = x2 - bore2 * (x2 @ bore2)
    if np.linalg.norm(x2) < 1e-8:
        return bore2, x0, y0
    x2 /= np.linalg.norm(x2)
    y2 = -np.cross(bore2, x2)
    return bore2, x2, y2


def remap_frame(
    inten: np.ndarray,
    sc_pos: np.ndarray,
    quat_xyzw: np.ndarray,
    dt: datetime,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat_deg, lon_deg, intensity) samples on the emission shell."""
    earth_px = detect_earth_center(inten, sc_pos)
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
    keep = lat >= (LAT_MIN - 15.0)
    lat, lon, vals = lat[keep], lon[keep], vals[keep]
    if lat.size == 0:
        return np.array([]), np.array([]), np.array([])

    good = np.isfinite(lat) & np.isfinite(lon) & (lat >= LAT_MIN)
    return lat[good], lon[good], vals[good]


def lat_lon_to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geographic polar: N pole at origin, LON_BOTTOM_DEG at bottom (US)."""
    theta = np.deg2rad(lon - LON_BOTTOM_DEG)
    r = (90.0 - lat) / (90.0 - LAT_MIN)
    return r * np.sin(theta), -r * np.cos(theta)


_COAST_SEGMENTS: list[tuple[np.ndarray, np.ndarray]] | None = None


def coast_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Coastline polylines in geographic polar xy (static; no AACGM warp)."""
    global _COAST_SEGMENTS
    if _COAST_SEGMENTS is not None:
        return _COAST_SEGMENTS
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
                if np.nanmax(lats) < LAT_MIN - 5:
                    continue
                good = np.isfinite(lats) & (lats >= LAT_MIN)
                if good.sum() < 2:
                    continue
                x, y = lat_lon_to_xy(lats, lons)
                x = np.where(good, x, np.nan)
                y = np.where(good, y, np.nan)
                segs.append((x, y))
    except Exception:
        pass
    _COAST_SEGMENTS = segs
    return segs


def _subsolar_lon_lat(dt: datetime) -> tuple[float, float]:
    """Subsolar lon/lat (deg) for terminator drawing."""
    try:
        from astropy.coordinates import get_body, ITRS
        from astropy.time import Time
        import astropy.units as u

        t = Time(dt)
        # get_body('sun') returns GCRS; transform to ITRS for geographic lon/lat
        sun = get_body("sun", t).transform_to(ITRS(obstime=t))
        lon = float(sun.earth_location.geodetic.lon.deg)
        lat = float(sun.earth_location.geodetic.lat.deg)
        return lon, lat
    except Exception:
        pass
    try:
        from astropy.coordinates import get_sun, SkyCoord, ITRS
        from astropy.time import Time
        import astropy.units as u

        t = Time(dt)
        sun = get_sun(t)
        sc = SkyCoord(
            ra=sun.ra,
            dec=sun.dec,
            frame="gcrs",
            obstime=t,
            distance=1.0 * u.au,
        )
        itrs = sc.transform_to(ITRS(obstime=t))
        lon = float(itrs.earth_location.geodetic.lon.deg)
        lat = float(itrs.earth_location.geodetic.lat.deg)
        return lon, lat
    except Exception:
        pass
    # Fallback: mean solar longitude + approximate declination
    hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    lon = (180.0 - 15.0 * hours) % 360.0
    if lon > 180.0:
        lon -= 360.0
    doy = dt.timetuple().tm_yday
    lat = 23.44 * np.sin(2.0 * np.pi * (doy - 81) / 365.25)
    return lon, float(lat)


def terminator_xy(dt: datetime, n: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle terminator projected into geographic polar xy."""
    slon, slat = _subsolar_lon_lat(dt)
    # Pole of the terminator plane is the subsolar direction.
    # Sample longitudes and solve for lat where sun zenith = 90°.
    # cos(c) = sin(lat)*sin(slat) + cos(lat)*cos(slat)*cos(lon-slon) = 0
    lons = np.linspace(-180.0, 180.0, n, endpoint=False)
    # Rearrange: tan(lat) = -cos(slat)*cos(dlon)/sin(slat) when slat != 0
    # Better: lat = atan2(-cos(slat)*cos(dlon), sin(slat)) wait for cos c = 0:
    # sin(lat) sin(slat) = -cos(lat) cos(slat) cos(dlon)
    # tan(lat) = -cos(slat)/sin(slat) * cos(dlon) = -cot(slat)*cos(dlon)
    if abs(slat) < 1e-6:
        # Equinox: terminator is the 90°-from-subsolar meridian pair → circle through pole
        # Actually equator terminator: all lons at lat=0, plus the anti-meridians
        # For polar plot we want the curve of solar zenith 90 within lat>=LAT_MIN.
        # At equinox terminator is a diameter: lon = slon±90.
        lon_t = np.array([slon + 90.0, slon + 90.0, slon - 90.0, slon - 90.0])
        lat_t = np.array([90.0, LAT_MIN, LAT_MIN, 90.0])
        return lat_lon_to_xy(lat_t, lon_t)

    tan_lat = -np.cos(np.deg2rad(slat)) / np.sin(np.deg2rad(slat)) * np.cos(
        np.deg2rad(lons - slon)
    )
    lats = np.degrees(np.arctan(tan_lat))
    good = lats >= LAT_MIN
    if good.sum() < 2:
        return np.array([]), np.array([])
    # Keep contiguous high-lat arc; break wrap with nan
    x, y = lat_lon_to_xy(lats, lons)
    x = np.where(good, x, np.nan)
    y = np.where(good, y, np.nan)
    return x, y


def mlat_annotation_points(dt: datetime, mlat_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample geographic points near a given AACGM MLAT for text placement."""
    if aacgmv2 is None:
        return np.array([]), np.array([])
    lons = np.linspace(-180.0, 180.0, 72, endpoint=False)
    lats = np.linspace(LAT_MIN, 89.5, 40)
    LON, LAT = np.meshgrid(lons, lats)
    try:
        mlat, _mlon, _mlt = aacgmv2.get_aacgm_coord_arr(
            LAT.ravel(), LON.ravel(), EMISSION_ALT_KM, dt
        )
        mlat = np.asarray(mlat, dtype=float).reshape(LAT.shape)
    except Exception:
        return np.array([]), np.array([])
    # Pick the lon where |mlat - target| is minimal near the right side of the plot
    # (positive x ≈ lon_bottom+90 = 0°E with LON_BOTTOM=-90).
    band = np.abs(mlat - mlat_target) < 0.8
    if not np.any(band):
        return np.array([]), np.array([])
    x, y = lat_lon_to_xy(LAT[band], LON[band])
    # Prefer points on the right half of the disk for label placement.
    right = x > 0.15
    if np.any(right):
        x, y = x[right], y[right]
    if x.size == 0:
        return np.array([]), np.array([])
    # Median location as a single anchor
    return np.array([float(np.median(x))]), np.array([float(np.median(y))])


def render_polar(lat, lon, vals, dt: datetime, out_path: Path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.2, 7.8), dpi=120)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.add_patch(Circle((0, 0), 1.0, color="#000000", zorder=0))

    if len(vals) > 50:
        x, y = lat_lon_to_xy(lat, lon)
        xi = np.linspace(-1, 1, 480)
        yi = np.linspace(-1, 1, 480)
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
        im.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        cbar.set_label("Ultraviolet brightness (relative)", color="white")
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")
        cbar.ax.set_xticks([vmin, vmax])
        cbar.ax.set_xticklabels(["Dim", "Bright"])

    for lat_ring in [50, 60, 70, 80]:
        r = (90 - lat_ring) / (90 - LAT_MIN)
        ax.add_patch(
            Circle((0, 0), r, fill=False, ls="--", lw=0.7, color="white", alpha=0.55, zorder=3)
        )
        ax.text(
            -0.04,
            r + 0.012,
            f"{lat_ring}°",
            color="white",
            fontsize=8,
            ha="right",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.75),
            zorder=6,
        )

    # Longitude meridians every 45°
    for lon_m in np.arange(-180.0, 180.0, 45.0):
        x0, y0 = lat_lon_to_xy(np.array([LAT_MIN]), np.array([lon_m]))
        ax.plot([0, x0[0]], [0, y0[0]], ls="--", lw=0.5, color="white", alpha=0.35, zorder=3)

    # Solar terminator (dashed white, matches QL style)
    tx, ty = terminator_xy(dt)
    if len(tx) > 1:
        ax.plot(tx, ty, ls="--", lw=1.6, color="white", alpha=0.9, zorder=4)

    for cx, cy in coast_segments():
        ax.plot(cx, cy, color="#e6c35c", lw=0.7, solid_capstyle="round", zorder=5, alpha=0.95)

    # Optional MLAT text anchors (style match to official QL annotations)
    for mlat_t, fallback in ((70.0, (0.55, 0.05)), (60.0, (0.78, 0.18))):
        mx, my = mlat_annotation_points(dt, mlat_t)
        if len(mx):
            px, py = float(mx[0]), float(my[0])
        else:
            px, py = fallback
        ax.text(px, py, f"{int(mlat_t)}° MLAT", color="#7ec8ff", fontsize=8, zorder=6)

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
    ap.add_argument(
        "--lon-bottom",
        type=float,
        default=-90.0,
        help="Geographic longitude (deg) placed at the bottom of the polar plot "
        "(default -90 = US/Canada).",
    )
    args = ap.parse_args()

    global LON_BOTTOM_DEG, _COAST_SEGMENTS
    LON_BOTTOM_DEG = float(args.lon_bottom)
    _COAST_SEGMENTS = None  # rebuild coastlines for new lon orientation

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
    print(f"Geographic polar: N pole center, lon {LON_BOTTOM_DEG:g}° at bottom")

    sample = []
    for dt, fp in selected[:: max(1, len(selected) // 8)]:
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
            sc = interp_row(aux.unix, aux.pos_j2000_km, dt.timestamp())
            q = interp_row(aux.unix, aux.quat_xyzw, dt.timestamp())
            earth = detect_earth_center(inten, sc)
            lat, lon, vals = remap_frame(inten, sc, q, dt, step=args.step_px)
            out = args.frame_dir / f"frame_{i:04d}_{dt.strftime('%H%M%S')}.png"
            render_polar(lat, lon, vals, dt, out, vmin=vmin, vmax=vmax)
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

#!/usr/bin/env python3
"""Merge CEDAR Madrigal gridded VTEC with Google phone VTEC maps and make a movie.

Madrigal product: World-wide GNSS Receiver Network, instrument 8000,
kindat 3500 (1° × 1° × 5 min binned vertical TEC) from
https://cedar.openmadrigal.org

Phone product: Android dual-frequency GNSS VTEC maps (S2 level-7 cells,
10 min) from Smith et al., Nature 2024, Code Ocean
https://doi.org/10.24433/CO.9149928.v1

The published phone archive covers 11 September 2023 – 24 May 2024.
12 May 2026 is not in that archive; the default day is 12 May 2024
(the first full day after the G5 storm, and the May 12 that exists in
both products).

Merged VTEC on the Madrigal grid:

* Madrigal-only cells keep the GNSS-station value
* phone-only cells (typical gap-fill over Africa, India, South America)
  take the inverse-variance mean of phones in that 1° bin
* overlapping cells use inverse-variance weighting of the two estimates
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

MADRIGAL_URL = "https://cedar.openmadrigal.org"
INST_CODE = 8000
KINDAT_VTEC = 3500
PHONE_FILES_BASE = (
    "https://files.codeocean.com/files/verified/"
    "ec6fe27c-d523-4b43-bf5b-a5a046b49f03_v1.1"
)
PHONE_DOI = "https://doi.org/10.24433/CO.9149928.v1"
PHONE_ARCHIVE_START = dt.date(2023, 9, 11)
PHONE_ARCHIVE_END = dt.date(2024, 5, 24)
DEFAULT_DATE = dt.date(2024, 5, 12)

TEC_VMIN = 0.0
TEC_VMAX = 100.0
PHONE_MAX_STDDEV = 8.0
PHONE_VTEC_MAX = 200.0
USER_AGENT = (
    "costerjane-merged-tec/1.0 "
    "(MIT Haystack; costera@mit.edu; research visualization)"
)

DEFAULT_USER = os.environ.get("MADRIGAL_USER", "Anthea Coster")
DEFAULT_EMAIL = os.environ.get("MADRIGAL_EMAIL", "costera@mit.edu")
DEFAULT_AFFIL = os.environ.get("MADRIGAL_AFFILIATION", "MIT Haystack Observatory")


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def phone_vtec_url(date: dt.date) -> str:
    return (
        f"{PHONE_FILES_BASE}/data/vtec_maps/"
        f"vtec_{date:%Y_%m_%d}.csv.gz"
    )


def download_file(url: str, dest: Path, timeout: float = 180.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(f"Empty download for {url}")
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_phone_vtec(date: dt.date, data_dir: Path) -> Path:
    if date < PHONE_ARCHIVE_START or date > PHONE_ARCHIVE_END:
        raise FileNotFoundError(
            f"Google phone VTEC maps are published only for "
            f"{PHONE_ARCHIVE_START.isoformat()} through {PHONE_ARCHIVE_END.isoformat()} "
            f"({PHONE_DOI}). {date.isoformat()} is outside that archive. "
            f"Use --date {DEFAULT_DATE.isoformat()} for a day present in both products."
        )
    dest = data_dir / "phone_tec" / f"vtec_{date:%Y_%m_%d}.csv.gz"
    try:
        return download_file(phone_vtec_url(date), dest)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                f"No phone VTEC map at {phone_vtec_url(date)}"
            ) from exc
        raise


def download_madrigal_vtec(
    date: dt.date,
    data_dir: Path,
    user_fullname: str,
    user_email: str,
    user_affiliation: str,
) -> Path:
    import madrigalWeb.madrigalWeb as madrigal_web

    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob(f"gps{date:%y%m%d}g*.hdf5"))
    if existing and existing[0].stat().st_size > 0:
        print(f"Using existing {existing[0]}")
        return existing[0]
    local_path = data_dir / f"gps{date:%y%m%d}g.hdf5"
    mad = madrigal_web.MadrigalData(MADRIGAL_URL)
    experiments = mad.getExperiments(
        INST_CODE,
        date.year, date.month, date.day, 0, 0, 0,
        date.year, date.month, date.day, 23, 59, 59,
    )
    if not experiments:
        raise FileNotFoundError(f"No GNSS TEC experiments in Madrigal for {date.isoformat()}")
    experiment = None
    for exp in experiments:
        if exp.startyear == date.year and exp.startmonth == date.month and exp.startday == date.day:
            experiment = exp
            break
    if experiment is None:
        experiment = experiments[-1]
    files = [f for f in mad.getExperimentFiles(experiment.id) if int(f.kindat) == KINDAT_VTEC]
    if not files:
        raise FileNotFoundError(f"No kindat {KINDAT_VTEC} VTEC file for {date.isoformat()}")
    remote = files[0]
    print(f"Downloading Madrigal {remote.name}")
    mad.downloadFile(
        remote.name,
        str(local_path),
        user_fullname,
        user_email,
        user_affiliation,
        "hdf5",
    )
    print(f"Saved {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
    return local_path


def load_madrigal_grid(path: Path) -> dict:
    with h5py.File(path, "r") as hdf:
        layout = hdf["Data/Array Layout"]
        gdlat = layout["gdlat"][:].astype(np.float64)
        glon = layout["glon"][:].astype(np.float64)
        timestamps = layout["timestamps"][:].astype(np.float64)
        tec = layout["2D Parameters"]["tec"][:].astype(np.float64)
        dtec = layout["2D Parameters"]["dtec"][:].astype(np.float64)
    times = np.array([dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in timestamps])
    tec[~np.isfinite(tec)] = np.nan
    dtec[~np.isfinite(dtec)] = np.nan
    dtec = np.where(np.isfinite(dtec) & (dtec > 0), dtec, np.nan)
    return {"gdlat": gdlat, "glon": glon, "times": times, "tec": tec, "dtec": dtec}


def s2_tokens_to_latlon(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude of S2 cell centers (degrees)."""
    import s2sphere as s2

    unique, inverse = np.unique(tokens, return_inverse=True)
    lats = np.empty(unique.size, dtype=np.float64)
    lons = np.empty(unique.size, dtype=np.float64)
    for i, token in enumerate(unique):
        latlng = s2.CellId.from_token(str(token)).to_lat_lng()
        lats[i] = latlng.lat().degrees
        lons[i] = latlng.lng().degrees
    return lats[inverse], lons[inverse]


def load_phone_vtec(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    needed = {"utc_sec", "pierce_s2_token", "vtec", "vtec_stddev"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    ok = (
        np.isfinite(df["vtec"])
        & np.isfinite(df["vtec_stddev"])
        & (df["vtec"] >= 0.0)
        & (df["vtec"] <= PHONE_VTEC_MAX)
        & (df["vtec_stddev"] > 0.0)
        & (df["vtec_stddev"] <= PHONE_MAX_STDDEV)
    )
    df = df.loc[ok].copy()
    lats, lons = s2_tokens_to_latlon(df["pierce_s2_token"].to_numpy())
    df["pierce_lat"] = lats
    df["pierce_lng"] = lons
    df["timestamp"] = pd.to_datetime(df["utc_sec"], unit="s", utc=True)
    return df


def latlon_to_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    gdlat: np.ndarray,
    glon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest Madrigal 1° cell for each pierce point."""
    dlat = float(np.median(np.diff(gdlat)))
    dlon = float(np.median(np.diff(glon)))
    i = np.rint((lat - gdlat[0]) / dlat).astype(np.int64)
    j = np.rint((lon - glon[0]) / dlon).astype(np.int64)
    valid = (i >= 0) & (i < gdlat.size) & (j >= 0) & (j < glon.size)
    return i, j, valid


def bin_phone_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    vtec: np.ndarray,
    stddev: np.ndarray,
    gdlat: np.ndarray,
    glon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance mean of phone VTEC on the Madrigal lat/lon grid."""
    nlat, nlon = gdlat.size, glon.size
    i, j, valid = latlon_to_indices(lat, lon, gdlat, glon)
    weight = np.zeros((nlat, nlon), dtype=np.float64)
    accum = np.zeros((nlat, nlon), dtype=np.float64)
    w = 1.0 / np.maximum(stddev[valid], 1e-3) ** 2
    np.add.at(weight, (i[valid], j[valid]), w)
    np.add.at(accum, (i[valid], j[valid]), w * vtec[valid])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weight > 0, accum / weight, np.nan)
        var = np.where(weight > 0, 1.0 / weight, np.nan)
    return mean, np.sqrt(var)


def merge_tec(
    madrigal: np.ndarray,
    madrigal_sigma: np.ndarray,
    phone: np.ndarray,
    phone_sigma: np.ndarray,
) -> np.ndarray:
    """Inverse-variance merge; use whichever source is present."""
    m_ok = np.isfinite(madrigal)
    p_ok = np.isfinite(phone)
    merged = np.full(madrigal.shape, np.nan, dtype=np.float64)
    merged[m_ok & ~p_ok] = madrigal[m_ok & ~p_ok]
    merged[p_ok & ~m_ok] = phone[p_ok & ~m_ok]
    both = m_ok & p_ok
    sm = np.where(np.isfinite(madrigal_sigma) & (madrigal_sigma > 0), madrigal_sigma, 2.0)
    sp = np.where(np.isfinite(phone_sigma) & (phone_sigma > 0), phone_sigma, 3.0)
    wm = 1.0 / sm[both] ** 2
    wp = 1.0 / sp[both] ** 2
    merged[both] = (wm * madrigal[both] + wp * phone[both]) / (wm + wp)
    return merged


def nearest_time_index(times: np.ndarray, when: dt.datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    epoch = np.array([t.timestamp() for t in times], dtype=np.float64)
    return int(np.argmin(np.abs(epoch - when.timestamp())))


def solar_zenith_deg(lat_deg: np.ndarray, lon_deg: np.ndarray, when: dt.datetime) -> np.ndarray:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    when = when.astimezone(dt.timezone.utc)
    doy = when.timetuple().tm_yday + (when.hour + when.minute / 60.0) / 24.0
    decl = np.radians(23.44) * np.sin(2.0 * np.pi * (doy - 81.0) / 365.25)
    utc_hours = when.hour + when.minute / 60.0 + when.second / 3600.0
    subsolar_lon = 15.0 * (12.0 - utc_hours)
    hour_angle = np.radians(lon_deg - subsolar_lon)
    lat = np.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arccos(np.clip(cosz, -1.0, 1.0)))


def _lon_lat_edges(gdlat: np.ndarray, glon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dlat = float(np.median(np.diff(gdlat)))
    dlon = float(np.median(np.diff(glon)))
    lat_edges = np.concatenate([gdlat - 0.5 * dlat, gdlat[-1:] + 0.5 * dlat])
    lon_edges = np.concatenate([glon - 0.5 * dlon, glon[-1:] + 0.5 * dlon])
    return lon_edges, lat_edges


def _decorate_map(ax) -> None:
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d9e6f2", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4f1ea", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.45, edgecolor="#222222", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="#555555", zorder=4)
    ax.gridlines(linewidth=0.25, color="0.5", alpha=0.35, linestyle="--")


def plot_triple_frame(
    gdlat: np.ndarray,
    glon: np.ndarray,
    madrigal: np.ndarray,
    phone: np.ndarray,
    merged: np.ndarray,
    when: dt.datetime,
    outfile: Path,
    cmap,
) -> Path:
    lon_edges, lat_edges = _lon_lat_edges(gdlat, glon)
    lon2d, lat2d = np.meshgrid(glon, gdlat)
    sza = solar_zenith_deg(lat2d, lon2d, when)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 3, figsize=(16.8, 5.15), dpi=110,
        subplot_kw={"projection": proj},
        facecolor="white",
    )
    titles = (
        "Madrigal GNSS stations\n1° × 1° × 5 min",
        "Google Android phones\nS2 ~75 km, 10 min",
        "Merged (gap-filled)\ninverse-variance",
    )
    fields = (madrigal, phone, merged)
    mesh = None
    for ax, field, title in zip(axes, fields, titles):
        _decorate_map(ax)
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(field),
            transform=proj,
            cmap=cmap,
            norm=Normalize(vmin=TEC_VMIN, vmax=TEC_VMAX),
            shading="flat",
            zorder=2,
        )
        ax.contour(
            lon2d, lat2d, sza, levels=[90.0], colors="k",
            linestyles="--", linewidths=0.7, transform=proj, zorder=5,
        )
        n = int(np.isfinite(field).sum())
        ax.set_title(f"{title}\n{n:,} cells", fontsize=10)
    fig.suptitle(
        f"Global vertical TEC  ·  {when:%Y-%m-%d  %H:%M} UT\n"
        "CEDAR Madrigal GNSS network  +  Smith et al. phone VTEC maps",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.86, top=0.78, bottom=0.06, wspace=0.08)
    cax = fig.add_axes([0.88, 0.14, 0.015, 0.58])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("VTEC (TECU)")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def _run_ffmpeg(args: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to assemble the movie")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + (result.stderr or result.stdout or "(no output)"))


def assemble_movie(frames: list[Path], outfile: Path, fps: int = 8) -> Path:
    if not frames:
        raise ValueError("No frames to assemble")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(frames):
            dest = Path(tmp) / f"frame_{i:05d}{src.suffix.lower()}"
            dest.symlink_to(src.resolve())
        pattern = Path(tmp) / f"frame_%05d{frames[0].suffix.lower()}"
        _run_ffmpeg(
            [
                "-framerate", str(fps),
                "-i", str(pattern),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-movflags", "+faststart",
                str(outfile),
            ]
        )
    print(f"Wrote {outfile} ({outfile.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    return outfile


def phone_times(df: pd.DataFrame) -> np.ndarray:
    uniq = np.sort(df["utc_sec"].unique())
    return np.array([dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in uniq])


def coverage_stats(madrigal: np.ndarray, phone: np.ndarray, merged: np.ndarray) -> dict:
    m = np.isfinite(madrigal)
    p = np.isfinite(phone)
    u = np.isfinite(merged)
    n = madrigal.size
    return {
        "madrigal_frac": float(m.mean()),
        "phone_frac": float(p.mean()),
        "merged_frac": float(u.mean()),
        "phone_only": int((p & ~m).sum()),
        "madrigal_only": int((m & ~p).sum()),
        "both": int((m & p).sum()),
        "n": n,
    }


def build_day_movie(
    date: dt.date,
    madrigal: dict,
    phone_df: pd.DataFrame,
    fig_dir: Path,
    frame_dir: Path,
    fps: int,
) -> tuple[Path, list[Path], dict]:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f4f1ea")
    times = phone_times(phone_df)
    frames: list[Path] = []
    stats_noon: dict | None = None
    grouped = {int(t): g for t, g in phone_df.groupby("utc_sec", sort=True)}

    for k, when in enumerate(times):
        g = grouped[int(when.timestamp())]
        phone_tec, phone_sig = bin_phone_to_grid(
            g["pierce_lat"].to_numpy(),
            g["pierce_lng"].to_numpy(),
            g["vtec"].to_numpy(),
            g["vtec_stddev"].to_numpy(),
            madrigal["gdlat"],
            madrigal["glon"],
        )
        idx = nearest_time_index(madrigal["times"], when)
        mad = madrigal["tec"][:, :, idx]
        mad_s = madrigal["dtec"][:, :, idx]
        merged = merge_tec(mad, mad_s, phone_tec, phone_sig)
        frame = frame_dir / f"merged_tec_{when:%Y%m%dT%H%M}.png"
        plot_triple_frame(
            madrigal["gdlat"], madrigal["glon"],
            mad, phone_tec, merged, when, frame, cmap,
        )
        frames.append(frame)
        if when.hour == 12 and when.minute == 0:
            stats_noon = coverage_stats(mad, phone_tec, merged)
        if (k + 1) % 12 == 0 or k == 0:
            print(f"  frame {k + 1}/{len(times)} {when:%H:%M} UT")

    movie = assemble_movie(
        frames,
        fig_dir / f"merged_tec_{date.isoformat()}_global.mp4",
        fps=fps,
    )
    # Preview stills at 00 / 12 / 20 UT
    stills = []
    for hour in (0, 12, 20):
        when = dt.datetime(date.year, date.month, date.day, hour, 0, tzinfo=dt.timezone.utc)
        idx = nearest_time_index(times, when)
        src = frames[idx]
        dest = fig_dir / f"merged_tec_{date.isoformat()}_{hour:02d}ut.png"
        shutil.copyfile(src, dest)
        stills.append(dest)
    return movie, stills, stats_noon or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_DATE.isoformat(), type=_parse_date)
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--fig-dir", default=Path("figures"), type=Path)
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--affiliation", default=DEFAULT_AFFIL)
    args = parser.parse_args(argv)

    print(f"Phone VTEC archive: {PHONE_ARCHIVE_START} – {PHONE_ARCHIVE_END}  ({PHONE_DOI})")
    if args.date != DEFAULT_DATE and (
        args.date < PHONE_ARCHIVE_START or args.date > PHONE_ARCHIVE_END
    ):
        print(
            f"Note: {args.date.isoformat()} has no published phone maps; "
            f"the overlapping May 12 in the archive is {DEFAULT_DATE.isoformat()}."
        )

    phone_path = download_phone_vtec(args.date, args.data_dir)
    mad_path = download_madrigal_vtec(
        args.date, args.data_dir, args.user, args.email, args.affiliation,
    )
    print("Loading Madrigal grid…")
    madrigal = load_madrigal_grid(mad_path)
    print("Loading phone VTEC maps…")
    phone_df = load_phone_vtec(phone_path)
    print(
        f"Phone samples after quality filter: {len(phone_df):,}  "
        f"in {phone_df['utc_sec'].nunique()} maps"
    )
    frame_dir = args.data_dir / "tec_frames" / args.date.isoformat()
    frame_dir.mkdir(parents=True, exist_ok=True)
    movie, stills, noon = build_day_movie(
        args.date, madrigal, phone_df, args.fig_dir, frame_dir, args.fps,
    )
    if noon:
        print(
            "12 UT coverage: "
            f"Madrigal {100 * noon['madrigal_frac']:.1f}%  "
            f"phones {100 * noon['phone_frac']:.1f}%  "
            f"merged {100 * noon['merged_frac']:.1f}%  "
            f"(phones add {noon['phone_only']:,} cells)"
        )
    print("Stills:", ", ".join(str(p) for p in stills))
    print("Movie:", movie)
    return 0


if __name__ == "__main__":
    sys.exit(main())

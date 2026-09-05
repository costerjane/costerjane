#!/usr/bin/env python3
"""Download CEDAR Madrigal GNSS vertical TEC and plot it over the USA.

Uses the MIT Haystack World-wide GNSS Receiver Network product
(instrument 8000, kindat 3500: 1 deg x 1 deg x 5 min binned VTEC)
from https://cedar.openmadrigal.org
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import Normalize

MADRIGAL_URL = "https://cedar.openmadrigal.org"
INST_CODE = 8000  # World-wide GNSS Receiver Network
KINDAT_VTEC = 3500  # TEC binned 1 degree by 1 degree by 5 min

# CONUS map window (includes a little of Canada / Mexico for ionosphere context)
CONUS_EXTENT = [-125.0, -66.0, 24.0, 50.0]  # lon_min, lon_max, lat_min, lat_max
TEC_VMIN = 0.0
TEC_VMAX = 50.0

DEFAULT_USER = os.environ.get("MADRIGAL_USER", "Anthea Coster")
DEFAULT_EMAIL = os.environ.get("MADRIGAL_EMAIL", "costera@mit.edu")
DEFAULT_AFFIL = os.environ.get(
    "MADRIGAL_AFFILIATION", "MIT Haystack Observatory"
)


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def download_vtec_file(
    date: dt.date,
    data_dir: Path,
    user_fullname: str,
    user_email: str,
    user_affiliation: str,
) -> Path:
    """Download the default 1x1 deg VTEC HDF5 for ``date`` if it is not local."""
    import madrigalWeb.madrigalWeb as madrigal_web

    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob(f"gps{date:%y%m%d}g*.hdf5"))
    if existing and existing[0].stat().st_size > 0:
        print(f"Using existing file {existing[0]}")
        return existing[0]
    local_path = data_dir / f"gps{date:%y%m%d}g.hdf5"

    mad = madrigal_web.MadrigalData(MADRIGAL_URL)
    experiments = mad.getExperiments(
        INST_CODE,
        date.year, date.month, date.day, 0, 0, 0,
        date.year, date.month, date.day, 23, 59, 59,
    )
    if not experiments:
        raise FileNotFoundError(
            f"No GNSS TEC experiments found in Madrigal for {date.isoformat()}"
        )

    # Prefer the experiment whose start day matches the requested date.
    experiment = None
    for exp in experiments:
        if exp.startyear == date.year and exp.startmonth == date.month and exp.startday == date.day:
            experiment = exp
            break
    if experiment is None:
        experiment = experiments[-1]

    files = mad.getExperimentFiles(experiment.id)
    vtec_files = [f for f in files if int(f.kindat) == KINDAT_VTEC]
    if not vtec_files:
        raise FileNotFoundError(
            f"No kindat {KINDAT_VTEC} VTEC file in experiment {experiment.id}"
        )

    remote = vtec_files[0]
    print(f"Downloading {remote.name}")
    print(f"  kindat: {remote.kindat} ({remote.kindatdesc})")
    print(f"  status: {remote.status}")
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


def load_vtec_grid(path: Path) -> dict:
    """Load the Array Layout VTEC cube from a CEDAR Madrigal HDF5 file."""
    with h5py.File(path, "r") as hdf:
        layout = hdf["Data/Array Layout"]
        gdlat = layout["gdlat"][:].astype(np.float64)
        glon = layout["glon"][:].astype(np.float64)
        timestamps = layout["timestamps"][:].astype(np.float64)
        tec = layout["2D Parameters"]["tec"][:].astype(np.float64)
        dtec = layout["2D Parameters"]["dtec"][:].astype(np.float64)
        notes = [
            row[0].decode("utf-8", errors="replace").strip()
            if isinstance(row[0], (bytes, np.bytes_))
            else str(row[0]).strip()
            for row in hdf["Metadata/Experiment Notes"][:]
        ]

    times = np.array(
        [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in timestamps]
    )
    return {
        "gdlat": gdlat,
        "glon": glon,
        "times": times,
        "tec": tec,
        "dtec": dtec,
        "notes": notes,
    }


def nearest_time_index(times: np.ndarray, when: dt.datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    epoch = np.array([t.timestamp() for t in times])
    return int(np.argmin(np.abs(epoch - when.timestamp())))


def solar_zenith_deg(lat_deg: np.ndarray, lon_deg: np.ndarray, when: dt.datetime) -> np.ndarray:
    """Approximate solar zenith angle in degrees on a lat/lon grid."""
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


def add_terminator(ax, when: dt.datetime) -> None:
    lat = np.linspace(CONUS_EXTENT[2] - 5, CONUS_EXTENT[3] + 5, 200)
    lon = np.linspace(CONUS_EXTENT[0] - 5, CONUS_EXTENT[1] + 5, 200)
    lon2d, lat2d = np.meshgrid(lon, lat)
    sza = solar_zenith_deg(lat2d, lon2d, when)
    ax.contour(
        lon2d,
        lat2d,
        sza,
        levels=[90.0],
        colors="k",
        linestyles="--",
        linewidths=1.1,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )


def _conus_axes(fig, *spec):
    proj = ccrs.LambertConformal(
        central_longitude=-96.0,
        central_latitude=39.0,
        standard_parallels=(33.0, 45.0),
    )
    ax = fig.add_subplot(*spec, projection=proj)
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d9e6f2", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f1ea", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, edgecolor="#444444", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#d9e6f2", edgecolor="#6a8aa8", linewidth=0.3, zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=range(-130, -60, 10),
        ylocs=range(25, 55, 5),
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
        linestyle="--",
        zorder=4,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.x_inline = False
    gl.y_inline = False
    gl.rotate_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xformatter = LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = LatitudeFormatter()
    return ax


def _pcolor_tec(ax, gdlat, glon, tec_slice, cmap):
    """Plot one VTEC time slice. tec_slice is (nlat, nlon)."""
    lon_edges = np.concatenate([glon - 0.5, glon[-1:] + 0.5])
    lat_edges = np.concatenate([gdlat - 0.5, gdlat[-1:] + 0.5])
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_invalid(tec_slice),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=TEC_VMAX),
        shading="flat",
        zorder=2,
    )
    return mesh


def plot_conus_panels(grid: dict, hours: list[int], outfile: Path, date: dt.date) -> Path:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f4f1ea")

    n = len(hours)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(12.5, 4.6 * nrows + 0.8), facecolor="white")
    fig.suptitle(
        f"GNSS Vertical TEC over the United States\n{date:%d %B %Y}  |  "
        "CEDAR Madrigal  ·  MIT Haystack GNSS network  ·  1° × 1° × 5 min",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    mesh = None
    for i, hour in enumerate(hours):
        when = dt.datetime(date.year, date.month, date.day, hour, 0, tzinfo=dt.timezone.utc)
        idx = nearest_time_index(grid["times"], when)
        stamp = grid["times"][idx]
        ax = _conus_axes(fig, nrows, ncols, i + 1)
        mesh = _pcolor_tec(ax, grid["gdlat"], grid["glon"], grid["tec"][:, :, idx], cmap)
        add_terminator(ax, stamp)
        usa = _usa_stats(grid, idx)
        ax.set_title("")
        ax.text(
            0.02,
            0.97,
            f"{stamp:%H:%M} UT\nmedian {usa['median']:.1f} TECU",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
            zorder=6,
        )

    fig.subplots_adjust(left=0.04, right=0.88, top=0.88, bottom=0.07, wspace=0.10, hspace=0.08)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.58])
    cbar = fig.colorbar(mesh, cax=cax, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.text(
        0.04,
        0.015,
        "Data: CEDAR Madrigal (cedar.openmadrigal.org), instrument 8000 / kindat 3500.  "
        "PI: Anthea Coster.  Rideout & Coster 2006; Vierinen et al. 2016.",
        fontsize=7.5,
        color="0.35",
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_conus_snapshot(grid: dict, hour: int, outfile: Path, date: dt.date) -> Path:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f4f1ea")

    when = dt.datetime(date.year, date.month, date.day, hour, 0, tzinfo=dt.timezone.utc)
    idx = nearest_time_index(grid["times"], when)
    stamp = grid["times"][idx]
    usa = _usa_stats(grid, idx)

    fig = plt.figure(figsize=(11.5, 7.6), facecolor="white")
    ax = _conus_axes(fig, 111)
    mesh = _pcolor_tec(ax, grid["gdlat"], grid["glon"], grid["tec"][:, :, idx], cmap)
    add_terminator(ax, stamp)
    ax.set_title(
        f"GNSS Vertical TEC over the United States\n"
        f"{stamp:%d %B %Y  %H:%M} UT   |   CEDAR Madrigal / MIT Haystack  "
        f"(1° × 1° × 5 min)\n"
        f"CONUS median {usa['median']:.1f} TECU   ·   "
        f"5–95% {usa['p5']:.1f}–{usa['p95']:.1f} TECU   ·   dashed line: terminator",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.03, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    fig.text(
        0.12,
        0.03,
        "Data: CEDAR Madrigal World-wide GNSS Receiver Network (instrument 8000, kindat 3500).  "
        "PI: Anthea Coster.  Rideout & Coster (2006); Vierinen et al. (2016).",
        fontsize=8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def _usa_stats(grid: dict, idx: int) -> dict:
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    lat_mask = (grid["gdlat"] >= lat_min) & (grid["gdlat"] <= lat_max)
    lon_mask = (grid["glon"] >= lon_min) & (grid["glon"] <= lon_max)
    slc = grid["tec"][np.ix_(lat_mask, lon_mask)][:, :, idx]
    finite = slc[np.isfinite(slc)]
    if finite.size == 0:
        return {"median": np.nan, "p5": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "median": float(np.median(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-01-19", type=_parse_date)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--fig-dir", default="figures", type=Path)
    parser.add_argument("--hours", default="0,6,12,20",
                        help="Comma-separated UT hours for the multi-panel figure")
    parser.add_argument("--snapshot-hour", default=20, type=int,
                        help="UT hour for the single CONUS snapshot")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--affiliation", default=DEFAULT_AFFIL)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.skip_download:
        candidates = sorted(args.data_dir.glob(f"gps{args.date:%y%m%d}g*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"No local VTEC file in {args.data_dir}")
        hdf_path = candidates[0]
    else:
        hdf_path = download_vtec_file(
            args.date, args.data_dir, args.user, args.email, args.affiliation
        )

    grid = load_vtec_grid(hdf_path)
    hours = [int(h.strip()) for h in args.hours.split(",") if h.strip()]
    plot_conus_panels(
        grid,
        hours,
        args.fig_dir / f"usa_tec_{args.date.isoformat()}_panels.png",
        args.date,
    )
    plot_conus_snapshot(
        grid,
        args.snapshot_hour,
        args.fig_dir / f"usa_tec_{args.date.isoformat()}_{args.snapshot_hour:02d}ut.png",
        args.date,
    )


if __name__ == "__main__":
    main()

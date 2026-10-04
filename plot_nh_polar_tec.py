#!/usr/bin/env python3
"""Northern-hemisphere polar GNSS VTEC in magnetic latitude / MLT.

Downloads CEDAR Madrigal World-wide GNSS Receiver Network vertical TEC
(instrument 8000, kindat 3500: 1° × 1° × 5 min) and plots a north-pole
view (looking down on the magnetic north pole) using AACGM magnetic
latitude and magnetic local time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import aacgmv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

MADRIGAL_URL = "https://cedar.openmadrigal.org"
INST_CODE = 8000  # World-wide GNSS Receiver Network
KINDAT_VTEC = 3500  # TEC binned 1 degree by 1 degree by 5 min

# Outer ring of the polar plot (magnetic latitude)
MLAT_OUTER = 40.0
IONO_HEIGHT_KM = 350.0
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

    experiment = None
    for exp in experiments:
        if (
            exp.startyear == date.year
            and exp.startmonth == date.month
            and exp.startday == date.day
        ):
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

    times = np.array(
        [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in timestamps]
    )
    return {
        "gdlat": gdlat,
        "glon": glon,
        "times": times,
        "tec": tec,
        "dtec": dtec,
    }


def nearest_time_index(times: np.ndarray, when: dt.datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    epoch = np.array([t.timestamp() for t in times])
    return int(np.argmin(np.abs(epoch - when.timestamp())))


def geo_to_mlat_mlt(
    gdlat: np.ndarray,
    glon: np.ndarray,
    when: dt.datetime,
    height_km: float = IONO_HEIGHT_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert geographic lat/lon grid to AACGM magnetic latitude and MLT."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)

    lon2d, lat2d = np.meshgrid(glon, gdlat)
    flat_lat = lat2d.ravel()
    flat_lon = ((lon2d.ravel() + 180.0) % 360.0) - 180.0
    height = np.full(flat_lat.shape, height_km, dtype=np.float64)

    mlat, mlon, _ = aacgmv2.convert_latlon_arr(
        flat_lat, flat_lon, height, when, method_code="G2A"
    )
    mlt = aacgmv2.convert_mlt(mlon, when, m2a=False)
    return (
        np.asarray(mlat, dtype=np.float64).reshape(lat2d.shape),
        np.asarray(mlt, dtype=np.float64).reshape(lat2d.shape),
    )


def polar_coords(mlat: np.ndarray, mlt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map magnetic lat/MLT to matplotlib polar (theta, r).

    Looking down on the north magnetic pole:
    00 MLT at bottom, 06 dawn right, 12 noon top, 18 dusk left.
    Radius is magnetic colatitude (0 at the pole).
    """
    theta = np.deg2rad(mlt * 15.0)
    radius = 90.0 - mlat
    return theta, radius


def _configure_polar_ax(ax, mlat_outer: float) -> None:
    ax.set_theta_zero_location("S")  # 00 MLT at bottom
    ax.set_theta_direction(1)  # CCW: dawn (06) to the right
    ax.set_ylim(0.0, 90.0 - mlat_outer)
    ax.set_yticks([90.0 - m for m in (80.0, 70.0, 60.0, 50.0, mlat_outer)])
    ax.set_yticklabels(
        [f"{int(m)}°" for m in (80, 70, 60, 50, int(mlat_outer))],
        fontsize=8,
        color="0.25",
    )
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(
        ["00", "03", "06", "09", "12", "15", "18", "21"],
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", linewidth=0.5, color="0.45", alpha=0.8)
    ax.set_facecolor("#f7f7f7")


def plot_nh_polar_snapshot(
    grid: dict,
    when: dt.datetime,
    outfile: Path,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
) -> Path:
    idx = nearest_time_index(grid["times"], when)
    stamp = grid["times"][idx]
    tec = grid["tec"][:, :, idx]
    mlat, mlt = geo_to_mlat_mlt(grid["gdlat"], grid["glon"], stamp)
    theta, radius = polar_coords(mlat, mlt)

    nh = (
        np.isfinite(mlat)
        & np.isfinite(mlt)
        & np.isfinite(tec)
        & (mlat >= mlat_outer)
    )
    if not np.any(nh):
        raise RuntimeError(f"No NH magnetic-latitude samples at {stamp:%Y-%m-%d %H:%M} UT")

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f7f7f7")

    fig = plt.figure(figsize=(8.6, 8.8), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    _configure_polar_ax(ax, mlat_outer)

    sc = ax.scatter(
        theta[nh],
        radius[nh],
        c=tec[nh],
        s=14,
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
        linewidths=0,
        alpha=0.95,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.78, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    finite = tec[nh]
    ax.set_title(
        "GNSS Vertical TEC — Northern Hemisphere (magnetic)\n"
        f"{stamp:%d %B %Y  %H:%M} UT   |   CEDAR Madrigal / MIT Haystack\n"
        f"AACGM magnetic latitude & MLT   ·   looking down on magnetic north pole\n"
        f"median {np.median(finite):.1f} TECU   ·   "
        f"5–95% {np.percentile(finite, 5):.1f}–{np.percentile(finite, 95):.1f} TECU   ·   "
        f"mlat ≥ {mlat_outer:.0f}°",
        fontsize=11,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.015,
        "Data: CEDAR Madrigal World-wide GNSS Receiver Network "
        "(instrument 8000, kindat 3500).  PI: Anthea Coster.  "
        "Rideout & Coster (2006); Vierinen et al. (2016).",
        ha="center",
        fontsize=7.5,
        color="0.35",
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_nh_polar_panels(
    grid: dict,
    hours: list[int],
    date: dt.date,
    outfile: Path,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
) -> Path:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f7f7f7")

    n = len(hours)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(11.5, 5.4 * nrows + 0.6), facecolor="white")
    fig.suptitle(
        f"GNSS Vertical TEC — Northern Hemisphere (magnetic)\n"
        f"{date:%d %B %Y}  |  CEDAR Madrigal · MIT Haystack · AACGM mlat / MLT",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    sc = None
    for i, hour in enumerate(hours):
        when = dt.datetime(
            date.year, date.month, date.day, hour, 0, tzinfo=dt.timezone.utc
        )
        idx = nearest_time_index(grid["times"], when)
        stamp = grid["times"][idx]
        tec = grid["tec"][:, :, idx]
        mlat, mlt = geo_to_mlat_mlt(grid["gdlat"], grid["glon"], stamp)
        theta, radius = polar_coords(mlat, mlt)
        nh = (
            np.isfinite(mlat)
            & np.isfinite(mlt)
            & np.isfinite(tec)
            & (mlat >= mlat_outer)
        )

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        _configure_polar_ax(ax, mlat_outer)
        sc = ax.scatter(
            theta[nh],
            radius[nh],
            c=tec[nh],
            s=9,
            cmap=cmap,
            norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
            linewidths=0,
            alpha=0.95,
            zorder=2,
        )
        finite = tec[nh]
        ax.set_title(
            f"{stamp:%H:%M} UT   ·   median {np.median(finite):.1f} TECU",
            fontsize=10,
            fontweight="bold",
            pad=12,
        )

    fig.subplots_adjust(
        left=0.04, right=0.88, top=0.88, bottom=0.06, wspace=0.25, hspace=0.30
    )
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.55])
    cbar = fig.colorbar(sc, cax=cax, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    fig.text(
        0.04,
        0.015,
        "Looking down on magnetic north pole: 12 MLT at top, 00 at bottom, "
        "06 dawn right, 18 dusk left.  "
        "Data: CEDAR Madrigal instrument 8000 / kindat 3500.  PI: Anthea Coster.",
        fontsize=7.5,
        color="0.35",
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def nh_stats(grid: dict, idx: int, mlat_outer: float = MLAT_OUTER) -> dict:
    stamp = grid["times"][idx]
    tec = grid["tec"][:, :, idx]
    mlat, mlt = geo_to_mlat_mlt(grid["gdlat"], grid["glon"], stamp)
    nh = (
        np.isfinite(mlat)
        & np.isfinite(mlt)
        & np.isfinite(tec)
        & (mlat >= mlat_outer)
    )
    finite = tec[nh]
    if finite.size == 0:
        return {"median": np.nan, "p5": np.nan, "p95": np.nan, "n": 0}
    return {
        "median": float(np.median(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "n": int(finite.size),
        "mlat_min": float(np.nanmin(mlat[nh])),
        "mlat_max": float(np.nanmax(mlat[nh])),
        "mlt_min": float(np.nanmin(mlt[nh])),
        "mlt_max": float(np.nanmax(mlt[nh])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-01-20", type=_parse_date)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--fig-dir", default="figures", type=Path)
    parser.add_argument(
        "--hours",
        default="0,6,12,18",
        help="Comma-separated UT hours for the multi-panel figure",
    )
    parser.add_argument(
        "--snapshot-hour",
        default=20,
        type=int,
        help="UT hour for the single polar snapshot",
    )
    parser.add_argument("--mlat-outer", default=MLAT_OUTER, type=float)
    parser.add_argument("--tec-vmax", default=TEC_VMAX, type=float)
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
    plot_nh_polar_panels(
        grid,
        hours,
        args.date,
        args.fig_dir / f"nh_polar_tec_{args.date.isoformat()}_panels.png",
        mlat_outer=args.mlat_outer,
        tec_vmax=args.tec_vmax,
    )
    when = dt.datetime(
        args.date.year,
        args.date.month,
        args.date.day,
        args.snapshot_hour,
        0,
        tzinfo=dt.timezone.utc,
    )
    plot_nh_polar_snapshot(
        grid,
        when,
        args.fig_dir
        / f"nh_polar_tec_{args.date.isoformat()}_{args.snapshot_hour:02d}ut.png",
        mlat_outer=args.mlat_outer,
        tec_vmax=args.tec_vmax,
    )


if __name__ == "__main__":
    main()

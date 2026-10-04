#!/usr/bin/env python3
"""Northern-hemisphere polar GNSS VTEC with scintillation overlay.

Downloads CEDAR Madrigal:
  * World-wide GNSS Receiver Network VTEC (instrument 8000, kindat 3500)
  * GNSS Scintillation Network (instrument 8010, kindat 20000)

Plots a north-pole view in AACGM magnetic latitude / MLT and overplots
scintillation pierce-point σφ (phase scintillation) using the same
magnetic conversion approach as
``New Code for Magnetic Conversion of Scintillation.ipynb``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path

import aacgmv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

MADRIGAL_URL = "https://cedar.openmadrigal.org"
INST_CODE_TEC = 8000  # World-wide GNSS Receiver Network
KINDAT_VTEC = 3500  # TEC binned 1 degree by 1 degree by 5 min
INST_CODE_SCIN = 8010  # GNSS Scintillation Network
KINDAT_SCIN = 20000  # Ionospheric scintillation

# Outer ring of the polar plot (magnetic latitude)
MLAT_OUTER = 40.0
TEC_HEIGHT_KM = 350.0
SCIN_HEIGHT_KM = 250.0  # pierce_alt index 1 in Madrigal scintillation files
PIERCE_ALT_INDEX = 1
TEC_VMIN = 0.0
TEC_VMAX = 50.0
SIGMA_PHI_VMIN = 0.2
SIGMA_PHI_VMAX = 0.9
SCIN_TIME_WINDOW_MIN = 2.5  # match nearest 5-min sample
SCIN_ELEV_MIN = 20.0
SCIN_MARKER_SIZE = 55.0

DEFAULT_USER = os.environ.get("MADRIGAL_USER", "Anthea Coster")
DEFAULT_EMAIL = os.environ.get("MADRIGAL_EMAIL", "costera@mit.edu")
DEFAULT_AFFIL = os.environ.get(
    "MADRIGAL_AFFILIATION", "MIT Haystack Observatory"
)


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _madrigal() -> "madrigalWeb.madrigalWeb.MadrigalData":
    import madrigalWeb.madrigalWeb as madrigal_web

    return madrigal_web.MadrigalData(MADRIGAL_URL)


def _pick_experiment(experiments, date: dt.date):
    for exp in experiments:
        if (
            exp.startyear == date.year
            and exp.startmonth == date.month
            and exp.startday == date.day
        ):
            return exp
    return experiments[-1]


def download_vtec_file(
    date: dt.date,
    data_dir: Path,
    user_fullname: str,
    user_email: str,
    user_affiliation: str,
) -> Path:
    """Download the default 1x1 deg VTEC HDF5 for ``date`` if it is not local."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob(f"gps{date:%y%m%d}g*.hdf5"))
    if existing and existing[0].stat().st_size > 0:
        print(f"Using existing file {existing[0]}")
        return existing[0]
    local_path = data_dir / f"gps{date:%y%m%d}g.hdf5"

    mad = _madrigal()
    experiments = mad.getExperiments(
        INST_CODE_TEC,
        date.year, date.month, date.day, 0, 0, 0,
        date.year, date.month, date.day, 23, 59, 59,
    )
    if not experiments:
        raise FileNotFoundError(
            f"No GNSS TEC experiments found in Madrigal for {date.isoformat()}"
        )
    experiment = _pick_experiment(experiments, date)
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


def download_scintillation_file(
    date: dt.date,
    data_dir: Path,
    user_fullname: str,
    user_email: str,
    user_affiliation: str,
) -> Path:
    """Download GNSS Scintillation Network HDF5 for ``date`` if not local."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob(f"scin_{date:%Y%m%d}*.hdf5"))
    if existing and existing[0].stat().st_size > 0:
        print(f"Using existing file {existing[0]}")
        return existing[0]
    local_path = data_dir / f"scin_{date:%Y%m%d}.001.hdf5"

    mad = _madrigal()
    experiments = mad.getExperiments(
        INST_CODE_SCIN,
        date.year, date.month, date.day, 0, 0, 0,
        date.year, date.month, date.day, 23, 59, 59,
    )
    if not experiments:
        raise FileNotFoundError(
            f"No GNSS scintillation experiments in Madrigal for {date.isoformat()}"
        )
    experiment = _pick_experiment(experiments, date)
    files = mad.getExperimentFiles(experiment.id)
    scin_files = [f for f in files if int(f.kindat) == KINDAT_SCIN]
    if not scin_files:
        # Fall back to any file whose basename looks like scin_YYYYMMDD
        scin_files = [
            f for f in files if Path(f.name).name.startswith(f"scin_{date:%Y%m%d}")
        ]
    if not scin_files:
        raise FileNotFoundError(
            f"No scintillation file in experiment {experiment.id}"
        )

    remote = scin_files[0]
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


def load_scintillation_records(path: Path, pierce_alt_index: int = PIERCE_ALT_INDEX) -> pd.DataFrame:
    """Flatten Madrigal scintillation Array Layout into pierce-point records.

    Follows the user's magnetic-conversion notebook: use pierce-point
    geographic lat/lon at the chosen pierce altitude (default 250 km).
    """
    rows: list[dict] = []
    with h5py.File(path, "r") as hdf:
        layout = hdf["Data/Array Layout"]
        sites = [
            key for key in layout.keys() if key.startswith("Array with gps_site")
        ]
        for site_key in sites:
            match = re.search(r"b'(\w+)'", str(site_key))
            site_name = match.group(1) if match else str(site_key)
            site = layout[site_key]
            gnss_types = [
                g.decode().strip() if isinstance(g, (bytes, np.bytes_)) else str(g).strip()
                for g in site["gnss_type"][:]
            ]
            timestamps = site["timestamps"][:].astype(np.float64)
            pierce_alt = site["pierce_alt"][:].astype(np.float64)
            sat_id = site["sat_id"][:].astype(np.int64)
            azm = site["2D Parameters"]["azm"][:].astype(np.float64)
            elm = site["2D Parameters"]["elm"][:].astype(np.float64)
            gdlat = site["2D Parameters"]["gdlat"][:].astype(np.float64)
            glon = site["2D Parameters"]["glon"][:].astype(np.float64)
            s4 = site["2D Parameters"]["s4_scin"][:].astype(np.float64)
            sigma_phi = site["2D Parameters"]["sigma_phi"][:].astype(np.float64)

            if pierce_alt_index >= pierce_alt.size:
                raise IndexError(
                    f"pierce_alt_index {pierce_alt_index} out of range for "
                    f"{site_name}: {pierce_alt}"
                )
            alt_km = float(pierce_alt[pierce_alt_index])

            n_chan, n_gnss, _n_alt, n_time = gdlat.shape
            times = [
                dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc) for ts in timestamps
            ]
            for i in range(n_chan):
                sat = int(sat_id[i])
                for j in range(n_gnss):
                    constellation = gnss_types[j] if j < len(gnss_types) else str(j)
                    lat = gdlat[i, j, pierce_alt_index, :]
                    lon = glon[i, j, pierce_alt_index, :]
                    az = azm[i, j, pierce_alt_index, :]
                    el = elm[i, j, pierce_alt_index, :]
                    s4v = s4[i, j, pierce_alt_index, :]
                    sig = sigma_phi[i, j, pierce_alt_index, :]
                    good = (
                        np.isfinite(lat)
                        & np.isfinite(lon)
                        & (lat != 0.0)
                        & (lon != 0.0)
                        & (lat >= -90.0)
                        & (lat <= 90.0)
                        & (lon >= -180.0)
                        & (lon <= 180.0)
                    )
                    for k in np.flatnonzero(good):
                        rows.append(
                            {
                                "site": site_name,
                                "channel": i,
                                "constellation": constellation,
                                "satellite_id": sat,
                                "time": times[k],
                                "azimuth": float(az[k]),
                                "elevation": float(el[k]),
                                "s4": float(s4v[k]),
                                "sigma_phi": float(sig[k]),
                                "gdlat": float(lat[k]),
                                "glon": float(lon[k]),
                                "pierce_alt": alt_km,
                            }
                        )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} scintillation pierce points from {path.name}")
    if len(df):
        print(f"  sites: {sorted(df['site'].unique())}")
    return df


def add_magnetic_coordinates(df: pd.DataFrame, height_km: float = SCIN_HEIGHT_KM) -> pd.DataFrame:
    """Add AACGM mlat/mlon/mlt columns (notebook conversion, vectorized per time)."""
    if df.empty:
        out = df.copy()
        out["mlat"] = []
        out["mlon"] = []
        out["mlt"] = []
        return out

    out = df.copy()
    out["mlat"] = np.nan
    out["mlon"] = np.nan
    out["mlt"] = np.nan

    unique_times = out["time"].unique()
    print(f"Converting {len(out)} points at {len(unique_times)} times to AACGM...")
    for time_idx, timestamp in enumerate(unique_times):
        if time_idx % 50 == 0:
            print(f"  time {time_idx}/{len(unique_times)}: {timestamp}")
        mask = out["time"] == timestamp
        lats = out.loc[mask, "gdlat"].to_numpy(dtype=np.float64)
        lons = out.loc[mask, "glon"].to_numpy(dtype=np.float64)
        timestamp_dt = pd.Timestamp(timestamp).to_pydatetime()
        if timestamp_dt.tzinfo is not None:
            timestamp_dt = timestamp_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)

        mlat, mlon, _ = aacgmv2.convert_latlon_arr(
            lats, lons, height_km, timestamp_dt, method_code="G2A"
        )
        mlt = np.asarray(
            aacgmv2.convert_mlt(mlon, timestamp_dt, m2a=False), dtype=np.float64
        )
        mlt = np.mod(mlt, 24.0)
        out.loc[mask, "mlat"] = np.asarray(mlat, dtype=np.float64)
        out.loc[mask, "mlon"] = np.asarray(mlon, dtype=np.float64)
        out.loc[mask, "mlt"] = mlt

    before = len(out)
    out = out[np.isfinite(out["mlat"]) & np.isfinite(out["mlt"])].reset_index(drop=True)
    print(
        f"Magnetic coordinate conversion complete "
        f"(kept {len(out)} / {before} finite AACGM points)"
    )
    return out


def filter_scintillation_hours(
    df: pd.DataFrame,
    date: dt.date,
    hours: list[int],
    window_min: float = SCIN_TIME_WINDOW_MIN,
) -> pd.DataFrame:
    """Keep scintillation samples near the requested UT hours."""
    if df.empty or not hours:
        return df
    times = df["time"]
    if getattr(times.dt, "tz", None) is None:
        times = times.dt.tz_localize("UTC")
    keep = np.zeros(len(df), dtype=bool)
    for hour in hours:
        center = dt.datetime(
            date.year, date.month, date.day, hour, 0, tzinfo=dt.timezone.utc
        )
        keep |= (
            (times >= center - dt.timedelta(minutes=window_min))
            & (times <= center + dt.timedelta(minutes=window_min))
        ).to_numpy()
    out = df.loc[keep].reset_index(drop=True)
    print(
        f"Kept {len(out)} / {len(df)} scintillation rows near UT hours {hours} "
        f"(±{window_min} min)"
    )
    return out


def load_or_build_scintillation_mag(
    hdf_path: Path,
    csv_path: Path | None = None,
    elev_min: float = SCIN_ELEV_MIN,
    date: dt.date | None = None,
    hours: list[int] | None = None,
) -> pd.DataFrame:
    """Load magnetic-coordinate CSV if present, otherwise build from HDF5."""
    if csv_path is not None and csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"Using existing scintillation CSV {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["time"])
        # Ensure timezone-aware UTC
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")
        return df

    df = load_scintillation_records(hdf_path)
    if elev_min is not None:
        before = len(df)
        df = df[df["elevation"] >= elev_min].reset_index(drop=True)
        print(f"Elevation ≥ {elev_min:.0f}°: kept {len(df)} / {before}")
    if date is not None and hours:
        df = filter_scintillation_hours(df, date, hours)
    df = add_magnetic_coordinates(df, height_km=SCIN_HEIGHT_KM)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)")
    return df


def nearest_time_index(times: np.ndarray, when: dt.datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    epoch = np.array([t.timestamp() for t in times])
    return int(np.argmin(np.abs(epoch - when.timestamp())))


def geo_to_mlat_mlt(
    gdlat: np.ndarray,
    glon: np.ndarray,
    when: dt.datetime,
    height_km: float = TEC_HEIGHT_KM,
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
    theta = np.deg2rad(np.asarray(mlt, dtype=np.float64) * 15.0)
    radius = 90.0 - np.asarray(mlat, dtype=np.float64)
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


def select_scintillation_slice(
    scin: pd.DataFrame,
    when: dt.datetime,
    mlat_outer: float = MLAT_OUTER,
    window_min: float = SCIN_TIME_WINDOW_MIN,
    sigma_vmin: float = SIGMA_PHI_VMIN,
) -> pd.DataFrame:
    """Return NH scintillation rows near ``when`` with σφ in the plot range."""
    if scin is None or scin.empty:
        return pd.DataFrame()
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    t0 = when - dt.timedelta(minutes=window_min)
    t1 = when + dt.timedelta(minutes=window_min)
    times = scin["time"]
    if getattr(times.dt, "tz", None) is None:
        times = times.dt.tz_localize("UTC")
    slc = scin[
        (times >= t0)
        & (times <= t1)
        & np.isfinite(scin["mlat"])
        & np.isfinite(scin["mlt"])
        & np.isfinite(scin["sigma_phi"])
        & (scin["mlat"] >= mlat_outer)
        & (scin["sigma_phi"] >= sigma_vmin)
    ].copy()
    return slc


def overplot_scintillation(
    ax,
    scin_slice: pd.DataFrame,
    sigma_vmin: float = SIGMA_PHI_VMIN,
    sigma_vmax: float = SIGMA_PHI_VMAX,
):
    """Scatter elevated σφ pierce points as filled circles on a polar TEC axis."""
    if scin_slice is None or scin_slice.empty:
        return None
    theta, radius = polar_coords(
        scin_slice["mlat"].to_numpy(), scin_slice["mlt"].to_numpy()
    )
    sc = ax.scatter(
        theta,
        radius,
        c=scin_slice["sigma_phi"].to_numpy(),
        s=SCIN_MARKER_SIZE,
        marker="o",
        cmap="plasma",
        norm=Normalize(vmin=sigma_vmin, vmax=sigma_vmax),
        edgecolors="black",
        linewidths=0.9,
        alpha=1.0,
        zorder=5,
        clip_on=False,
    )
    return sc


def plot_nh_polar_snapshot(
    grid: dict,
    when: dt.datetime,
    outfile: Path,
    scin: pd.DataFrame | None = None,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
    sigma_vmin: float = SIGMA_PHI_VMIN,
    sigma_vmax: float = SIGMA_PHI_VMAX,
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

    fig = plt.figure(figsize=(9.2, 8.8), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    _configure_polar_ax(ax, mlat_outer)

    sc_tec = ax.scatter(
        theta[nh],
        radius[nh],
        c=tec[nh],
        s=14,
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
        linewidths=0,
        alpha=0.90,
        zorder=2,
    )
    scin_slice = select_scintillation_slice(
        scin, stamp, mlat_outer=mlat_outer, sigma_vmin=sigma_vmin
    )
    sc_scin = overplot_scintillation(
        ax, scin_slice, sigma_vmin=sigma_vmin, sigma_vmax=sigma_vmax
    )

    cbar = fig.colorbar(sc_tec, ax=ax, pad=0.08, shrink=0.72, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    if sc_scin is not None:
        cbar2 = fig.colorbar(sc_scin, ax=ax, pad=0.02, shrink=0.72, extend="both")
        cbar2.set_label(r"Phase scintillation $\sigma_\phi$ (rad)", fontsize=11)
        cbar2.ax.tick_params(labelsize=9)

    finite = tec[nh]
    n_scin = 0 if scin_slice is None else len(scin_slice)
    med_sig = (
        float(np.nanmedian(scin_slice["sigma_phi"])) if n_scin else float("nan")
    )
    ax.set_title(
        "GNSS VTEC + scintillation — Northern Hemisphere (magnetic)\n"
        f"{stamp:%d %B %Y  %H:%M} UT   |   CEDAR Madrigal / MIT Haystack\n"
        f"AACGM magnetic latitude & MLT   ·   looking down on magnetic north pole\n"
        f"TEC median {np.median(finite):.1f} TECU   ·   "
        f"σφ circles ≥ {sigma_vmin:.1f} (n={n_scin}, scale {sigma_vmin:.1f}–{sigma_vmax:.1f})"
        + (f"   ·   median σφ {med_sig:.2f}" if n_scin else "")
        + f"   ·   mlat ≥ {mlat_outer:.0f}°",
        fontsize=11,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.012,
        "TEC: instrument 8000 / kindat 3500.  "
        "Scintillation: instrument 8010 / kindat 20000 "
        f"(pierce alt {SCIN_HEIGHT_KM:.0f} km).  "
        "PI: Anthea Coster.  Mag. conversion: aacgmv2 (notebook method).",
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
    scin: pd.DataFrame | None = None,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
    sigma_vmin: float = SIGMA_PHI_VMIN,
    sigma_vmax: float = SIGMA_PHI_VMAX,
) -> Path:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f7f7f7")

    n = len(hours)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(12.0, 5.4 * nrows + 0.6), facecolor="white")
    fig.suptitle(
        f"GNSS VTEC + scintillation — Northern Hemisphere (magnetic)\n"
        f"{date:%d %B %Y}  |  CEDAR Madrigal · AACGM mlat / MLT · "
        f"σφ circles {sigma_vmin:.1f}–{sigma_vmax:.1f} "
        f"(pierce {SCIN_HEIGHT_KM:.0f} km)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    sc_tec = None
    sc_scin = None
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
        sc_tec = ax.scatter(
            theta[nh],
            radius[nh],
            c=tec[nh],
            s=9,
            cmap=cmap,
            norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
            linewidths=0,
            alpha=0.90,
            zorder=2,
        )
        scin_slice = select_scintillation_slice(
            scin, stamp, mlat_outer=mlat_outer, sigma_vmin=sigma_vmin
        )
        sc = overplot_scintillation(
            ax, scin_slice, sigma_vmin=sigma_vmin, sigma_vmax=sigma_vmax
        )
        if sc is not None:
            sc_scin = sc
        finite = tec[nh]
        n_scin = 0 if scin_slice is None else len(scin_slice)
        ax.set_title(
            f"{stamp:%H:%M} UT   ·   TEC median {np.median(finite):.1f}   ·   "
            f"n_σφ≥{sigma_vmin:.1f}={n_scin}",
            fontsize=10,
            fontweight="bold",
            pad=12,
        )

    fig.subplots_adjust(
        left=0.04, right=0.84, top=0.88, bottom=0.06, wspace=0.25, hspace=0.30
    )
    cax = fig.add_axes([0.86, 0.20, 0.016, 0.55])
    cbar = fig.colorbar(sc_tec, cax=cax, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    if sc_scin is not None:
        cax2 = fig.add_axes([0.93, 0.20, 0.016, 0.55])
        cbar2 = fig.colorbar(sc_scin, cax=cax2, extend="both")
        cbar2.set_label(r"$\sigma_\phi$ (rad)", fontsize=11)
    fig.text(
        0.04,
        0.015,
        "Looking down on magnetic north pole: 12 MLT at top, 00 at bottom, "
        "06 dawn right, 18 dusk left.  "
        "TEC: 8000/3500.  Scintillation: 8010/20000.  PI: Anthea Coster.",
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
    parser.add_argument("--sigma-vmin", default=SIGMA_PHI_VMIN, type=float)
    parser.add_argument("--sigma-vmax", default=SIGMA_PHI_VMAX, type=float)
    parser.add_argument("--elev-min", default=SCIN_ELEV_MIN, type=float)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--affiliation", default=DEFAULT_AFFIL)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-scintillation",
        action="store_true",
        help="Plot TEC only (no scintillation download/overlay)",
    )
    args = parser.parse_args()

    if args.skip_download:
        tec_candidates = sorted(args.data_dir.glob(f"gps{args.date:%y%m%d}g*.hdf5"))
        if not tec_candidates:
            raise FileNotFoundError(f"No local VTEC file in {args.data_dir}")
        tec_path = tec_candidates[0]
        scin_path = None
        if not args.skip_scintillation:
            scin_candidates = sorted(
                args.data_dir.glob(f"scin_{args.date:%Y%m%d}*.hdf5")
            )
            if not scin_candidates:
                raise FileNotFoundError(
                    f"No local scintillation file in {args.data_dir}"
                )
            scin_path = scin_candidates[0]
    else:
        tec_path = download_vtec_file(
            args.date, args.data_dir, args.user, args.email, args.affiliation
        )
        scin_path = None
        if not args.skip_scintillation:
            scin_path = download_scintillation_file(
                args.date, args.data_dir, args.user, args.email, args.affiliation
            )

    grid = load_vtec_grid(tec_path)
    hours = [int(h.strip()) for h in args.hours.split(",") if h.strip()]
    plot_hours = sorted(set(hours + [args.snapshot_hour]))

    scin = None
    if scin_path is not None:
        csv_path = (
            args.data_dir
            / f"scint_mag_coords_{args.date.isoformat()}_hours-{'-'.join(map(str, plot_hours))}.csv"
        )
        scin = load_or_build_scintillation_mag(
            scin_path,
            csv_path=csv_path,
            elev_min=args.elev_min,
            date=args.date,
            hours=plot_hours,
        )

    plot_nh_polar_panels(
        grid,
        hours,
        args.date,
        args.fig_dir / f"nh_polar_tec_{args.date.isoformat()}_panels.png",
        scin=scin,
        mlat_outer=args.mlat_outer,
        tec_vmax=args.tec_vmax,
        sigma_vmin=args.sigma_vmin,
        sigma_vmax=args.sigma_vmax,
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
        scin=scin,
        sigma_vmin=args.sigma_vmin,
        mlat_outer=args.mlat_outer,
        tec_vmax=args.tec_vmax,
        sigma_vmax=args.sigma_vmax,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Northern-hemisphere polar GNSS VTEC with scintillation overlay.

Downloads CEDAR Madrigal:
  * World-wide GNSS Receiver Network VTEC (instrument 8000, kindat 3500)
  * GNSS Scintillation Network (instrument 8010, kindat 20000)

Default product is an AACGM magnetic-latitude / MLT north-polar map
(12 MLT at top) of TEC with reference-style σφ circles (green = not
significant; red sized bins 0.1–0.6+ with blue edges). Optional
``--geo-also`` also writes the geographic north-polar map.
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
GEO_LAT_MIN = 40.0  # outer latitude for geographic north-polar map
SCIN_TIME_WINDOW_MIN = 2.5  # match nearest 5-min sample
SCIN_ELEV_MIN = 20.0

# Reference-style σφ legend bins (green = not significant; red sized by bin).
SIGMA_PHI_EDGE = "#1f4e79"  # blue border as in the reference figure
SIGMA_PHI_QUIET_FACE = "#2ca02c"
SIGMA_PHI_ACTIVE_FACE = "#d62728"
SIGMA_PHI_QUIET_THRESHOLD = 0.1
SIGMA_PHI_BINS = [
    # (label, low, high_exclusive_or_None, marker_size)
    ("No significant SigmaPhi", None, SIGMA_PHI_QUIET_THRESHOLD, 22.0),
    ("SigmaPhi 0.1 - 0.2", 0.1, 0.2, 55.0),
    ("SigmaPhi 0.2 - 0.3", 0.2, 0.3, 110.0),
    ("SigmaPhi 0.3 - 0.4", 0.3, 0.4, 190.0),
    ("SigmaPhi 0.4 - 0.5", 0.4, 0.5, 290.0),
    ("SigmaPhi 0.5 - 0.6", 0.5, 0.6, 420.0),
    ("SigmaPhi > 0.6", 0.6, None, 600.0),
]
# Keep old names for CLI/tests that still mention a continuous scale.
SIGMA_PHI_VMIN = 0.2
SIGMA_PHI_VMAX = 0.9
SCIN_MARKER_SIZE_MIN = SIGMA_PHI_BINS[2][3]
SCIN_MARKER_SIZE_MAX = SIGMA_PHI_BINS[-1][3]
SCIN_MARKER_FACE = SIGMA_PHI_ACTIVE_FACE

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
    times = _aware_times(df["time"])
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


def filter_scintillation_time_range(
    df: pd.DataFrame,
    t0: dt.datetime,
    t1: dt.datetime,
) -> pd.DataFrame:
    """Keep scintillation samples inside an absolute UTC window."""
    if df.empty:
        return df
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=dt.timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=dt.timezone.utc)
    times = _aware_times(df["time"])
    out = df.loc[(times >= t0) & (times <= t1)].reset_index(drop=True)
    print(f"Kept {len(out)} / {len(df)} scintillation rows in {t0} – {t1}")
    return out


def _read_scin_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")
    return df


def load_or_build_scintillation_mag(
    hdf_path: Path,
    csv_path: Path | None = None,
    elev_min: float = SCIN_ELEV_MIN,
    date: dt.date | None = None,
    hours: list[int] | None = None,
    t0: dt.datetime | None = None,
    t1: dt.datetime | None = None,
    convert_magnetic: bool = True,
) -> pd.DataFrame:
    """Load scintillation CSV if present, otherwise build from HDF5."""
    if csv_path is not None and csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"Using existing scintillation CSV {csv_path}")
        return _read_scin_csv(csv_path)

    df = load_scintillation_records(hdf_path)
    if elev_min is not None:
        before = len(df)
        df = df[df["elevation"] >= elev_min].reset_index(drop=True)
        print(f"Elevation ≥ {elev_min:.0f}°: kept {len(df)} / {before}")
    if t0 is not None and t1 is not None:
        df = filter_scintillation_time_range(df, t0, t1)
    elif date is not None and hours:
        df = filter_scintillation_hours(df, date, hours)
    if convert_magnetic:
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


def _aware_times(series: pd.Series) -> pd.Series:
    if getattr(series.dt, "tz", None) is None:
        return series.dt.tz_localize("UTC")
    return series.dt.tz_convert("UTC")


def select_scintillation_window(
    scin: pd.DataFrame,
    t0: dt.datetime,
    t1: dt.datetime,
    *,
    gdlat_min: float | None = None,
    mlat_min: float | None = None,
) -> pd.DataFrame:
    """Return scintillation rows inside ``[t0, t1]`` with finite σφ."""
    if scin is None or scin.empty:
        return pd.DataFrame()
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=dt.timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=dt.timezone.utc)
    times = _aware_times(scin["time"])
    mask = (
        (times >= t0)
        & (times <= t1)
        & np.isfinite(scin["sigma_phi"])
        & np.isfinite(scin["gdlat"])
        & np.isfinite(scin["glon"])
    )
    if gdlat_min is not None:
        mask &= scin["gdlat"] >= gdlat_min
    if mlat_min is not None:
        mask &= np.isfinite(scin["mlat"]) & np.isfinite(scin["mlt"])
        mask &= scin["mlat"] >= mlat_min
    return scin.loc[mask].copy()


def select_scintillation_slice(
    scin: pd.DataFrame,
    when: dt.datetime,
    mlat_outer: float = MLAT_OUTER,
    window_min: float = SCIN_TIME_WINDOW_MIN,
    sigma_vmin: float = 0.0,
) -> pd.DataFrame:
    """Return NH magnetic scintillation rows near ``when``."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    t0 = when - dt.timedelta(minutes=window_min)
    t1 = when + dt.timedelta(minutes=window_min)
    slc = select_scintillation_window(scin, t0, t1, mlat_min=mlat_outer)
    if sigma_vmin > 0 and not slc.empty:
        slc = slc[slc["sigma_phi"] >= sigma_vmin].copy()
    return slc


def sigma_phi_bin_style(sigma_phi: float) -> tuple[str, float, str]:
    """Return (face_color, size, label) for one σφ value using reference bins."""
    for label, low, high, size in SIGMA_PHI_BINS:
        if low is None:
            if sigma_phi < high:
                return SIGMA_PHI_QUIET_FACE, size, label
        elif high is None:
            if sigma_phi >= low:
                return SIGMA_PHI_ACTIVE_FACE, size, label
        elif low <= sigma_phi < high:
            return SIGMA_PHI_ACTIVE_FACE, size, label
    return SIGMA_PHI_ACTIVE_FACE, SIGMA_PHI_BINS[-1][3], SIGMA_PHI_BINS[-1][0]


def sigma_phi_marker_sizes(
    sigma_phi: np.ndarray,
    sigma_vmin: float = SIGMA_PHI_VMIN,
    sigma_vmax: float = SIGMA_PHI_VMAX,
) -> np.ndarray:
    """Map σφ values to reference-style discrete marker sizes."""
    del sigma_vmin, sigma_vmax  # discrete bins; kept for call-site compatibility
    values = np.asarray(sigma_phi, dtype=np.float64)
    sizes = np.empty(values.shape, dtype=np.float64)
    for i, val in enumerate(values.ravel()):
        sizes.ravel()[i] = sigma_phi_bin_style(float(val))[1]
    return sizes


def _sigma_phi_legend_handles(ax):
    """Build empty scatter handles for the reference-style σφ legend."""
    handles = []
    for label, _low, _high, size in SIGMA_PHI_BINS:
        face = (
            SIGMA_PHI_QUIET_FACE
            if label.startswith("No significant")
            else SIGMA_PHI_ACTIVE_FACE
        )
        handles.append(
            ax.scatter(
                [],
                [],
                s=size,
                marker="o",
                facecolors=face,
                edgecolors=SIGMA_PHI_EDGE,
                linewidths=1.1,
                label=label,
            )
        )
    return handles


def add_sigma_phi_legend(
    ax,
    loc: str = "upper right",
    bbox_to_anchor: tuple[float, float] | None = None,
    fontsize: float = 8,
) -> None:
    """Add the reference-style discrete σφ legend on ``ax``."""
    handles = _sigma_phi_legend_handles(ax)
    kwargs = dict(
        handles=handles,
        loc=loc,
        frameon=True,
        fontsize=fontsize,
        labelspacing=1.15,
        borderpad=0.8,
        scatterpoints=1,
        framealpha=0.95,
        fancybox=True,
    )
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
        kwargs["bbox_transform"] = ax.transAxes
    ax.legend(**kwargs)


def add_sigma_phi_legend_left(fig, fontsize: float = 35) -> None:
    """Draw σφ legend in a dedicated white panel left of the polar globe."""
    # Wide left panel so 35 pt bold labels fit without overlapping the globe.
    leg_ax = fig.add_axes([0.01, 0.08, 0.36, 0.70])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    for spine in leg_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.35")
        spine.set_linewidth(1.2)
    leg_ax.set_facecolor("white")
    leg_ax.set_title(
        r"$\sigma_\phi$ legend", fontsize=fontsize, fontweight="bold", pad=12
    )
    ys = np.linspace(0.88, 0.08, len(SIGMA_PHI_BINS))
    for y, (label, _low, _high, size) in zip(ys, SIGMA_PHI_BINS):
        face = (
            SIGMA_PHI_QUIET_FACE
            if label.startswith("No significant")
            else SIGMA_PHI_ACTIVE_FACE
        )
        # Scale marker size with the large legend font so symbols stay readable.
        marker_size = max(size * 1.8, 80.0)
        leg_ax.scatter(
            [0.08],
            [y],
            s=marker_size,
            marker="o",
            facecolors=face,
            edgecolors=SIGMA_PHI_EDGE,
            linewidths=1.4,
            clip_on=False,
        )
        leg_ax.text(
            0.18,
            y,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            fontweight="bold",
        )


def overplot_sigma_phi_binned(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    sigma_phi: np.ndarray,
    *,
    transform=None,
    add_legend: bool = False,
    legend_loc: str = "upper right",
    zorder: int = 20,
):
    """Plot reference-style filled σφ circles (green quiet / red sized bins)."""
    if len(sigma_phi) == 0:
        return None
    faces = []
    sizes = []
    for val in np.asarray(sigma_phi, dtype=np.float64):
        face, size, _ = sigma_phi_bin_style(float(val))
        faces.append(face)
        sizes.append(size)
    kwargs = dict(
        s=np.asarray(sizes),
        marker="o",
        c=faces,
        edgecolors=SIGMA_PHI_EDGE,
        linewidths=1.2,
        alpha=0.95,
        zorder=zorder,
        clip_on=True,
    )
    if transform is not None:
        kwargs["transform"] = transform
    sc = ax.scatter(x, y, **kwargs)
    if add_legend:
        add_sigma_phi_legend(ax, loc=legend_loc)
    return sc


def overplot_scintillation(
    ax,
    scin_slice: pd.DataFrame,
    sigma_vmin: float = 0.0,
    sigma_vmax: float = SIGMA_PHI_VMAX,
    add_size_legend: bool = False,
):
    """Overplot σφ on a magnetic polar axis using the reference circle style."""
    del sigma_vmax
    if scin_slice is None or scin_slice.empty:
        return None
    slc = scin_slice
    if sigma_vmin > 0:
        slc = slc[slc["sigma_phi"] >= sigma_vmin]
    if slc.empty:
        return None
    theta, radius = polar_coords(slc["mlat"].to_numpy(), slc["mlt"].to_numpy())
    return overplot_sigma_phi_binned(
        ax,
        theta,
        radius,
        slc["sigma_phi"].to_numpy(),
        add_legend=add_size_legend,
        legend_loc="upper left",
    )


def mean_tec_slice(
    grid: dict, t0: dt.datetime, t1: dt.datetime
) -> tuple[np.ndarray, list[dt.datetime]]:
    """Mean VTEC over all 5-min samples inside ``[t0, t1]``."""
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=dt.timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=dt.timezone.utc)
    idxs = [
        i
        for i, stamp in enumerate(grid["times"])
        if t0 <= stamp <= t1
    ]
    if not idxs:
        idx = nearest_time_index(grid["times"], t0)
        idxs = [idx]
    stack = np.stack([grid["tec"][:, :, i] for i in idxs], axis=-1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=-1)
    used = [grid["times"][i] for i in idxs]
    return mean, used


def plot_mag_north_polar_tec_scint(
    grid: dict,
    scin: pd.DataFrame | None,
    t0: dt.datetime,
    t1: dt.datetime,
    outfile: Path,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
) -> Path:
    """Magnetic north-polar TEC + σφ in AACGM mlat/MLT (12 MLT at top)."""
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=dt.timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=dt.timezone.utc)

    tec_mean, used_times = mean_tec_slice(grid, t0, t1)
    # Convert geographic TEC grid at window midpoint (standard for a mean map).
    mid = t0 + (t1 - t0) / 2
    mlat, mlt = geo_to_mlat_mlt(grid["gdlat"], grid["glon"], mid)
    theta, radius = polar_coords(mlat, mlt)
    nh = (
        np.isfinite(mlat)
        & np.isfinite(mlt)
        & np.isfinite(tec_mean)
        & (mlat >= mlat_outer)
    )
    if not np.any(nh):
        raise RuntimeError(f"No NH magnetic TEC samples in window {t0} – {t1}")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f7f7f7")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f7f7f7")

    # Explicit figure layout:
    #   - title block at top of the figure
    #   - boxed σφ legend in left white panel
    #   - polar globe shifted right
    #   - TEC colorbar on the right
    fig = plt.figure(figsize=(18.0, 13.0), facecolor="white")
    fig.text(
        0.5,
        0.985,
        "Phase scintillation/TEC map",
        ha="center",
        va="top",
        fontsize=35,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        f"{t0} - {t1}",
        ha="center",
        va="top",
        fontsize=35,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "AACGM magnetic latitude & MLT  ·  12 MLT at top  ·  "
        "CEDAR Madrigal TEC (8000/3500) + scintillation (8010/20000)",
        ha="center",
        va="top",
        fontsize=14,
    )

    ax = fig.add_axes([0.42, 0.04, 0.46, 0.72], projection="polar")
    _configure_polar_ax(ax, mlat_outer)
    # Explicit noon-at-top orientation (00 at bottom, 06 dawn right, 18 dusk left).
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)

    sc_tec = ax.scatter(
        theta[nh],
        radius[nh],
        c=tec_mean[nh],
        s=16,
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
        linewidths=0,
        alpha=0.90,
        zorder=2,
    )

    scin_slice = select_scintillation_window(scin, t0, t1, mlat_min=mlat_outer)
    if scin_slice is not None and not scin_slice.empty:
        th_s, r_s = polar_coords(
            scin_slice["mlat"].to_numpy(), scin_slice["mlt"].to_numpy()
        )
        overplot_sigma_phi_binned(
            ax,
            th_s,
            r_s,
            scin_slice["sigma_phi"].to_numpy(),
            add_legend=False,
            zorder=10,
        )
        n_scin = len(scin_slice)
        n_sig = int((scin_slice["sigma_phi"] >= SIGMA_PHI_QUIET_THRESHOLD).sum())
    else:
        n_scin = 0
        n_sig = 0

    fig.text(
        0.5,
        0.845,
        f"n_scin={n_scin} (σφ≥0.1: {n_sig})  ·  TEC samples: {len(used_times)}  ·  "
        f"mlat ≥ {mlat_outer:.0f}°",
        ha="center",
        va="top",
        fontsize=13,
    )
    add_sigma_phi_legend_left(fig, fontsize=35)

    cax = fig.add_axes([0.91, 0.16, 0.02, 0.55])
    cbar = fig.colorbar(sc_tec, cax=cax, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "Magnetic north-pole view: 12 MLT top, 00 bottom, 06 dawn right, 18 dusk left.  "
        "σφ: green = not significant; red size bins 0.1–0.6+ (blue edges).  "
        "PI: Anthea Coster.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    # Fixed figure coordinates — do not use bbox_inches='tight' (it can
    # collapse the left legend back onto the globe).
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_geo_north_polar_tec_scint(
    grid: dict,
    scin: pd.DataFrame | None,
    t0: dt.datetime,
    t1: dt.datetime,
    outfile: Path,
    lat_min: float = GEO_LAT_MIN,
    tec_vmax: float = TEC_VMAX,
) -> Path:
    """Geographic north-polar TEC heatmap + reference-style σφ circles."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=dt.timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=dt.timezone.utc)

    tec_mean, used_times = mean_tec_slice(grid, t0, t1)
    lon_edges = np.concatenate([grid["glon"] - 0.5, grid["glon"][-1:] + 0.5])
    lat_edges = np.concatenate([grid["gdlat"] - 0.5, grid["gdlat"][-1:] + 0.5])

    proj = ccrs.NorthPolarStereo(central_longitude=0.0)
    fig = plt.figure(figsize=(9.5, 9.2), facecolor="white")
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent([-180, 180, lat_min, 90], crs=ccrs.PlateCarree())
    try:
        # Circular map boundary like the reference figure.
        import matplotlib.path as mpath

        theta = np.linspace(0, 2 * np.pi, 361)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
    except Exception:
        pass

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_invalid(tec_mean),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
        shading="flat",
        zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.7, edgecolor="k", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, edgecolor="0.3", zorder=3)
    gl = ax.gridlines(
        draw_labels=False,
        xlocs=range(-180, 181, 30),
        ylocs=range(int(lat_min), 91, 10),
        linewidth=0.6,
        color="0.55",
        alpha=0.85,
        linestyle="--",
        zorder=4,
    )
    del gl

    scin_slice = select_scintillation_window(scin, t0, t1, gdlat_min=lat_min)
    if scin_slice is not None and not scin_slice.empty:
        overplot_sigma_phi_binned(
            ax,
            scin_slice["glon"].to_numpy(),
            scin_slice["gdlat"].to_numpy(),
            scin_slice["sigma_phi"].to_numpy(),
            transform=ccrs.PlateCarree(),
            add_legend=True,
            legend_loc="upper right",
            zorder=10,
        )
        n_scin = len(scin_slice)
        n_sig = int((scin_slice["sigma_phi"] >= SIGMA_PHI_QUIET_THRESHOLD).sum())
    else:
        n_scin = 0
        n_sig = 0
        add_sigma_phi_legend(ax, loc="upper right")

    ax.set_title(
        f"Phase scintillation/TEC map for {t0} - {t1}\n"
        f"CEDAR Madrigal TEC (8000/3500) + scintillation (8010/20000)  ·  "
        f"n_scin={n_scin} (σφ≥0.1: {n_sig})  ·  "
        f"TEC samples: {len(used_times)}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.06, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "North-polar geographic map.  σφ circles: green = not significant; "
        "red size bins follow 0.1–0.6+ (blue edges).  PI: Anthea Coster.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_nh_polar_snapshot(
    grid: dict,
    when: dt.datetime,
    outfile: Path,
    scin: pd.DataFrame | None = None,
    mlat_outer: float = MLAT_OUTER,
    tec_vmax: float = TEC_VMAX,
    sigma_vmin: float = 0.0,
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
        s=10,
        cmap=cmap,
        norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
        linewidths=0,
        alpha=0.75,
        zorder=2,
    )
    scin_slice = select_scintillation_slice(
        scin, stamp, mlat_outer=mlat_outer, sigma_vmin=sigma_vmin
    )
    sc_scin = overplot_scintillation(
        ax,
        scin_slice,
        sigma_vmin=sigma_vmin,
        sigma_vmax=sigma_vmax,
        add_size_legend=True,
    )

    cbar = fig.colorbar(sc_tec, ax=ax, pad=0.08, shrink=0.72, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

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
        f"σφ circles (reference bins) n={n_scin}"
        + (f"   ·   median σφ {med_sig:.2f}" if n_scin else "")
        + f"   ·   mlat ≥ {mlat_outer:.0f}°",
        fontsize=11,
        fontweight="bold",
        pad=18,
    )
    del sigma_vmin, sigma_vmax, sc_scin
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
    sigma_vmin: float = 0.0,
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
        f"σφ reference-style bins (pierce {SCIN_HEIGHT_KM:.0f} km)",
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
            s=7,
            cmap=cmap,
            norm=Normalize(vmin=TEC_VMIN, vmax=tec_vmax),
            linewidths=0,
            alpha=0.75,
            zorder=2,
        )
        scin_slice = select_scintillation_slice(
            scin, stamp, mlat_outer=mlat_outer, sigma_vmin=sigma_vmin
        )
        sc = overplot_scintillation(
            ax,
            scin_slice,
            sigma_vmin=sigma_vmin,
            sigma_vmax=sigma_vmax,
            add_size_legend=False,
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
        left=0.04, right=0.90, top=0.88, bottom=0.06, wspace=0.25, hspace=0.30
    )
    cax = fig.add_axes([0.92, 0.20, 0.016, 0.55])
    cbar = fig.colorbar(sc_tec, cax=cax, extend="max")
    cbar.set_label("Vertical TEC (TECU)", fontsize=11)
    del sc_scin, sigma_vmin, sigma_vmax
    fig.text(
        0.04,
        0.015,
        "Looking down on magnetic north pole: 12 MLT at top, 00 at bottom, "
        "06 dawn right, 18 dusk left.  "
        "σφ: green = not significant; red sized bins 0.1–0.6+.  "
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
        "--window-start",
        default="17:40",
        help="UT start HH:MM for the geographic reference-style map",
    )
    parser.add_argument(
        "--window-end",
        default="18:00",
        help="UT end HH:MM for the geographic reference-style map",
    )
    parser.add_argument(
        "--hours",
        default="0,6,12,18",
        help="Comma-separated UT hours for magnetic multi-panel figure",
    )
    parser.add_argument(
        "--snapshot-hour",
        default=18,
        type=int,
        help="UT hour for the magnetic polar snapshot",
    )
    parser.add_argument("--mlat-outer", default=MLAT_OUTER, type=float)
    parser.add_argument("--lat-min", default=GEO_LAT_MIN, type=float)
    parser.add_argument("--tec-vmax", default=TEC_VMAX, type=float)
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
    parser.add_argument(
        "--geo-also",
        action="store_true",
        help="Also write the geographic north-polar map",
    )
    parser.add_argument(
        "--panels-also",
        action="store_true",
        help="Also write multi-hour AACGM mlat/MLT panels",
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
    start_h, start_m = [int(x) for x in args.window_start.split(":")]
    end_h, end_m = [int(x) for x in args.window_end.split(":")]
    t0 = dt.datetime(
        args.date.year, args.date.month, args.date.day,
        start_h, start_m, tzinfo=dt.timezone.utc,
    )
    t1 = dt.datetime(
        args.date.year, args.date.month, args.date.day,
        end_h, end_m, tzinfo=dt.timezone.utc,
    )

    # Primary product: both TEC and σφ in AACGM mlat / MLT, noon at top.
    scin_mag = None
    if scin_path is not None:
        mag_csv = (
            args.data_dir
            / (
                f"scint_mag_coords_{args.date.isoformat()}_"
                f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}.csv"
            )
        )
        scin_mag = load_or_build_scintillation_mag(
            scin_path,
            csv_path=mag_csv,
            elev_min=args.elev_min,
            t0=t0,
            t1=t1,
            convert_magnetic=True,
        )

    plot_mag_north_polar_tec_scint(
        grid,
        scin_mag,
        t0,
        t1,
        args.fig_dir
        / (
            f"phase_scintillation_tec_mlt_{args.date.isoformat()}_"
            f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}_leftlegend.png"
        ),
        mlat_outer=args.mlat_outer,
        tec_vmax=args.tec_vmax,
    )
    # Keep a stable alias for the primary magnetic product.
    alias = (
        args.fig_dir
        / (
            f"phase_scintillation_tec_mlt_{args.date.isoformat()}_"
            f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}.png"
        )
    )
    src = (
        args.fig_dir
        / (
            f"phase_scintillation_tec_mlt_{args.date.isoformat()}_"
            f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}_leftlegend.png"
        )
    )
    if src.exists():
        alias.write_bytes(src.read_bytes())
        print(f"Wrote {alias}")

    if args.geo_also:
        geo_csv = (
            args.data_dir
            / (
                f"scint_geo_{args.date.isoformat()}_"
                f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}.csv"
            )
        )
        scin_geo = load_or_build_scintillation_mag(
            scin_path,
            csv_path=geo_csv,
            elev_min=args.elev_min,
            t0=t0,
            t1=t1,
            convert_magnetic=False,
        )
        plot_geo_north_polar_tec_scint(
            grid,
            scin_geo,
            t0,
            t1,
            args.fig_dir
            / (
                f"phase_scintillation_tec_{args.date.isoformat()}_"
                f"{start_h:02d}{start_m:02d}-{end_h:02d}{end_m:02d}.png"
            ),
            lat_min=args.lat_min,
            tec_vmax=args.tec_vmax,
        )

    if args.panels_also:
        hours = [int(h.strip()) for h in args.hours.split(",") if h.strip()]
        plot_hours = sorted(set(hours + [args.snapshot_hour]))
        panel_csv = (
            args.data_dir
            / f"scint_mag_coords_{args.date.isoformat()}_hours-{'-'.join(map(str, plot_hours))}.csv"
        )
        scin_panels = load_or_build_scintillation_mag(
            scin_path,
            csv_path=panel_csv,
            elev_min=args.elev_min,
            date=args.date,
            hours=plot_hours,
            convert_magnetic=True,
        )
        plot_nh_polar_panels(
            grid,
            hours,
            args.date,
            args.fig_dir / f"nh_polar_tec_{args.date.isoformat()}_panels.png",
            scin=scin_panels,
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
            scin=scin_panels,
            mlat_outer=args.mlat_outer,
            tec_vmax=args.tec_vmax,
        )


if __name__ == "__main__":
    main()

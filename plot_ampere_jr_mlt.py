#!/usr/bin/env python3
r"""Line plot of AMPERE fitted radial current at a fixed MLT.

Recovers \(j_R\) from the public Type 1/2 three-panel summary plots
(the same third panel shown in the JHUAPL survey movies) by inverting
the plot colorbar, then takes the maximum along a chosen magnetic-local-
time meridian.

The official GRD netCDF files are behind the AMPERE login portal; the
summary PNGs are the public product that matches the movies.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import plot_ampere_movies as ampere

DEFAULT_MLT = 16.0
JR_PANEL = 2  # third panel: fitted radial current
COLORBAR_VMAX = 1.0  # μA/m² as labeled on the Type 1/2 plots


@dataclass(frozen=True)
class PolarLayout:
    r"""Geometry and color scale of one \(j_R\) polar panel."""

    cx: float
    cy: float
    radius: float
    mlat_outer: float
    lut_rgb: np.ndarray  # (N, 3) float
    lut_val: np.ndarray  # (N,) Jr in μA/m²


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB."""
    from matplotlib import image as mpimg

    arr = mpimg.imread(path)
    if arr.dtype.kind == "f":
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    else:
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, :3]


def jr_panel(image: np.ndarray, panel: int = JR_PANEL) -> np.ndarray:
    """Return one of the three equal-width AMPERE summary-plot panels."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")
    height, width, _ = image.shape
    if width % 3 != 0:
        raise ValueError(f"Expected width divisible by 3, got {width}")
    pw = width // 3
    return image[:, panel * pw : (panel + 1) * pw, :3]


def _edge_points(panel: np.ndarray, threshold: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    gray = panel.mean(axis=2)
    gx = np.diff(gray.astype(np.float32), axis=1, prepend=gray[:, :1])
    gy = np.diff(gray.astype(np.float32), axis=0, prepend=gray[:1, :])
    mag = np.hypot(gx, gy)
    ys, xs = np.where(mag > threshold)
    return xs.astype(np.float64), ys.astype(np.float64)


def _vote_circle(
    xs: np.ndarray,
    ys: np.ndarray,
    cx_range: range,
    cy_range: range,
    r_bins: np.ndarray,
) -> tuple[int, int, int, float]:
    best_count = -1
    best = (0, 0, 0.0)
    for cy in cy_range:
        for cx in cx_range:
            rr = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            hist, edges = np.histogram(rr, bins=r_bins)
            k = int(np.argmax(hist))
            count = int(hist[k])
            if count > best_count:
                best_count = count
                best = (cx, cy, 0.5 * (edges[k] + edges[k + 1]))
    return best_count, best[0], best[1], best[2]


def locate_polar_circle(panel: np.ndarray, mlat_outer: float) -> tuple[float, float, float]:
    """Hough-fit the outermost latitude circle of an AMPERE polar panel.

    Inner latitude rings (50°/60°) can out-vote the 40° circle on busy
    frames, so after locating the center we take the *outermost* strong
    radius peak rather than the single highest bin.
    """
    xs, ys = _edge_points(panel)
    if xs.size < 100:
        raise RuntimeError("Not enough edges to locate the AMPERE polar plot")
    h, w = panel.shape[:2]
    rmin, rmax = 0.30 * min(h, w), 0.52 * min(h, w)
    coarse = _vote_circle(
        xs,
        ys,
        range(int(0.35 * w), int(0.65 * w), 4),
        range(int(0.40 * h), int(0.65 * h), 4),
        np.arange(rmin, rmax + 4, 4),
    )
    _, cx0, cy0, _r0 = coarse
    fine = _vote_circle(
        xs,
        ys,
        range(cx0 - 6, cx0 + 7, 1),
        range(cy0 - 6, cy0 + 7, 1),
        np.arange(rmin, rmax + 1, 1),
    )
    _, cx, cy, _r_any = fine
    rr = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    hist, edges = np.histogram(rr, bins=np.arange(rmin, rmax + 1, 1))
    if hist.max() < 20:
        raise RuntimeError("Could not find a latitude circle in the j_R panel")
    strong = np.where(hist >= 0.45 * hist.max())[0]
    radius = float(0.5 * (edges[strong[-1]] + edges[strong[-1] + 1]))
    if radius < 50:
        raise RuntimeError(f"Implausible polar radius {radius:.1f}px")
    return float(cx), float(cy), radius


def locate_colorbar(panel: np.ndarray) -> tuple[int, int, int]:
    r"""Return (x, y0, y1) of the vertical \(j_R\) colorbar (red at top)."""
    h, w, _ = panel.shape
    best: tuple[float, int, int, int] | None = None
    x_start = int(0.80 * w)
    for x in range(x_start, w - 3):
        col = panel[: int(0.40 * h), x].astype(np.float32)
        for y0 in range(20, 90):
            length = 120
            y1 = y0 + length
            if y1 >= col.shape[0]:
                break
            strip = col[y0:y1]
            top = strip[6:14].mean(axis=0)
            bot = strip[-14:-6].mean(axis=0)
            mid = strip[length // 2 - 4 : length // 2 + 4].mean(axis=0)
            if (
                top[0] > 140
                and top[0] > top[2] + 40
                and bot[2] > 140
                and bot[2] > bot[0] + 40
                and mid.mean() > 140
            ):
                score = float((top[0] - top[2]) + (bot[2] - bot[0]))
                if best is None or score > best[0]:
                    best = (score, x, y0, y1)
    if best is None:
        raise RuntimeError("Could not locate the j_R colorbar in the third panel")
    return best[1], best[2], best[3]


def colorbar_lut(panel: np.ndarray, x: int, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Map colorbar pixels from +vmax (top, red) to -vmax (bottom, blue)."""
    lut = panel[y0:y1, x].astype(np.float32)

    def _white(c: np.ndarray) -> bool:
        return float(c.min()) > 230

    def _black(c: np.ndarray) -> bool:
        return float(c.max()) < 30

    lo, hi = 0, len(lut) - 1
    while lo < hi and (_white(lut[lo]) or _black(lut[lo])):
        lo += 1
    while hi > lo and (_white(lut[hi]) or _black(lut[hi])):
        hi -= 1
    lut = lut[lo : hi + 1]
    if len(lut) < 20:
        raise RuntimeError("Colorbar LUT is too short after trimming ticks/padding")
    values = np.linspace(COLORBAR_VMAX, -COLORBAR_VMAX, len(lut))
    return lut, values


def detect_layout(panel: np.ndarray, mlat_outer: float) -> PolarLayout:
    cx, cy, radius = locate_polar_circle(panel, mlat_outer)
    x, y0, y1 = locate_colorbar(panel)
    lut_rgb, lut_val = colorbar_lut(panel, x, y0, y1)
    return PolarLayout(cx, cy, radius, mlat_outer, lut_rgb, lut_val)


def mlt_xy(layout: PolarLayout, mlt: float, radius: float) -> tuple[float, float]:
    """Image coordinates of a point at the given MLT and pixel radius.

    AMPERE polar plots put 12 MLT at the top and 18 MLT at the left, so
    MLT increases counterclockwise from noon.
    """
    angle = (mlt - 12.0) * (np.pi / 12.0)
    x = layout.cx - radius * np.sin(angle)
    y = layout.cy - radius * np.cos(angle)
    return float(x), float(y)


def rgb_to_jr(rgb: np.ndarray, layout: PolarLayout, max_delta: float = 40.0) -> float:
    """Invert one RGB triple through the panel colorbar. NaN if not a data color."""
    rgb = np.asarray(rgb, dtype=np.float32).reshape(3)
    if float(rgb.max()) < 40:
        return float("nan")  # grid / text
    delta = np.linalg.norm(layout.lut_rgb - rgb[None, :], axis=1)
    k = int(np.argmin(delta))
    if float(delta[k]) > max_delta:
        return float("nan")
    return float(layout.lut_val[k])


def sample_jr_along_mlt(
    panel: np.ndarray,
    layout: PolarLayout,
    mlt: float,
    n: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mlat, j_R) samples from the pole to the outer latitude circle."""
    height, width, _ = panel.shape
    radii = np.linspace(4.0, 0.98 * layout.radius, n)
    mlats = 90.0 - (radii / layout.radius) * (90.0 - layout.mlat_outer)
    jrs = np.empty(n, dtype=np.float64)
    for i, radius in enumerate(radii):
        x, y = mlt_xy(layout, mlt, float(radius))
        xi, yi = int(round(x)), int(round(y))
        if not (1 <= xi < width - 1 and 1 <= yi < height - 1):
            jrs[i] = np.nan
            continue
        patch = panel[yi - 1 : yi + 2, xi - 1 : xi + 2].reshape(-1, 3).mean(axis=0)
        jrs[i] = rgb_to_jr(patch, layout)
    return mlats, jrs


def max_jr_at_mlt(
    panel: np.ndarray,
    layout: PolarLayout,
    mlt: float,
) -> tuple[float, float]:
    r"""Maximum (most positive) \(j_R\) along an MLT meridian and its MLAT."""
    mlats, jrs = sample_jr_along_mlt(panel, layout, mlt)
    if not np.isfinite(jrs).any():
        return float("nan"), float("nan")
    k = int(np.nanargmax(jrs))
    return float(jrs[k]), float(mlats[k])


def min_jr_at_mlt(
    panel: np.ndarray,
    layout: PolarLayout,
    mlt: float,
) -> tuple[float, float]:
    mlats, jrs = sample_jr_along_mlt(panel, layout, mlt)
    if not np.isfinite(jrs).any():
        return float("nan"), float("nan")
    k = int(np.nanargmin(jrs))
    return float(jrs[k]), float(mlats[k])


def overlay_mlt_ray(
    image: np.ndarray,
    layout: PolarLayout,
    mlt: float,
    panel: int = JR_PANEL,
) -> np.ndarray:
    """Copy of the full three-panel image with the sampled meridian drawn."""
    out = image.copy()
    height, width, _ = image.shape
    pw = width // 3
    xoff = panel * pw
    yy, xx = np.ogrid[:height, :pw]
    ring = np.abs(np.sqrt((xx - layout.cx) ** 2 + (yy - layout.cy) ** 2) - layout.radius) < 1.5
    out[:, xoff : xoff + pw, :][ring] = (0, 180, 0)
    for radius in np.linspace(4.0, 0.98 * layout.radius, 280):
        x, y = mlt_xy(layout, mlt, float(radius))
        xi, yi = int(round(x)) + xoff, int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            out[yi, xi] = (220, 0, 0)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_ut", "jr_max", "mlat_at_max", "jr_min", "mlat_at_min"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(
    times: list[dt.datetime],
    jr_max: np.ndarray,
    outfile: Path,
    *,
    mlt: float,
    start: dt.date,
    end: dt.date,
    pole: str,
    boundary: int,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=140)
    ax.plot(times, jr_max, color="#b2182b", lw=1.15, label=rf"max $j_R$ at {mlt:.0f} MLT")
    ax.axhline(COLORBAR_VMAX, color="0.55", ls="--", lw=0.8, label=fr"plot colorbar limit ($\pm{COLORBAR_VMAX:.0f}$)")
    ax.set_ylabel(r"max $j_R$ ($\mu$A m$^{-2}$)")
    ax.set_xlabel("Universal Time")
    kind = "Type 1" if boundary == 40 else "Type 2"
    ax.set_title(
        f"AMPERE {pole} {kind} $j_R$ (fit)  |  {start.isoformat()}–{end.isoformat()}  |  "
        f"{mlt:.0f}:00 MLT, {boundary}°–90° MLAT"
    )
    ax.set_ylim(-0.05, COLORBAR_VMAX + 0.12)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", frameon=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H UT"))
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def collect_timeseries(
    dates: list[dt.date],
    *,
    pole: str,
    boundary: int,
    fit: str,
    mlt: float,
    data_dir: Path,
    overlay_dir: Path | None,
) -> tuple[list[dt.datetime], np.ndarray, np.ndarray, list[dict[str, object]]]:
    layout: PolarLayout | None = None
    times: list[dt.datetime] = []
    jr_max_vals: list[float] = []
    mlat_vals: list[float] = []
    rows: list[dict[str, object]] = []
    overlay_written = False

    for date in dates:
        for when in ampere.day_frame_times(date):
            dest = (
                data_dir
                / "plots"
                / date.isoformat()
                / f"ampere_{when:%Y%m%dT%H%M}_{pole}_{boundary:02d}.png"
            )
            url = ampere.plot_url(when, pole=pole, boundary=boundary, fit=fit)
            try:
                ampere.download_file(url, dest)
            except ampere.AmpereDownloadError as exc:
                print(f"Skipping {when:%Y-%m-%d %H:%M} UT ({exc})")
                continue
            image = load_rgb(dest)
            panel = jr_panel(image)
            if layout is None:
                layout = detect_layout(panel, mlat_outer=float(boundary))
                print(
                    f"Polar layout: center=({layout.cx:.1f},{layout.cy:.1f}) "
                    f"R={layout.radius:.1f}px  LUT={len(layout.lut_val)} colors"
                )
            jr_max, mlat_max = max_jr_at_mlt(panel, layout, mlt)
            jr_min, mlat_min = min_jr_at_mlt(panel, layout, mlt)
            times.append(when)
            jr_max_vals.append(jr_max)
            mlat_vals.append(mlat_max)
            rows.append(
                {
                    "time_ut": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "jr_max": f"{jr_max:.4f}",
                    "mlat_at_max": f"{mlat_max:.2f}",
                    "jr_min": f"{jr_min:.4f}",
                    "mlat_at_min": f"{mlat_min:.2f}",
                }
            )
            if overlay_dir is not None and not overlay_written and np.isfinite(jr_max):
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = overlay_mlt_ray(image, layout, mlt)
                from matplotlib import image as mpimg

                overlay_path = overlay_dir / (
                    f"ampere_{when:%Y-%m-%dT%H%M}_{pole}_jr_{mlt:.0f}mlt_ray.png"
                )
                mpimg.imsave(overlay_path, overlay)
                print(f"Wrote meridian overlay {overlay_path}")
                overlay_written = True

    if not times:
        raise RuntimeError("No AMPERE summary plots were available for the requested interval")
    return times, np.asarray(jr_max_vals), np.asarray(mlat_vals), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-05-10", type=ampere._parse_date)
    parser.add_argument("--end", default="2025-05-12", type=ampere._parse_date)
    parser.add_argument("--pole", default=ampere.DEFAULT_POLE, choices=("north", "south"))
    parser.add_argument("--boundary", default=ampere.DEFAULT_BOUNDARY, type=int, choices=(40, 60))
    parser.add_argument("--fit", default=ampere.DEFAULT_FIT)
    parser.add_argument("--mlt", default=DEFAULT_MLT, type=float, help="Magnetic local time (hours)")
    parser.add_argument("--data-dir", default=Path("data/ampere"), type=Path)
    parser.add_argument("--fig-dir", default=Path("figures"), type=Path)
    args = parser.parse_args(argv)

    dates = ampere.daterange(args.start, args.end)
    times, jr_max, _mlat, rows = collect_timeseries(
        dates,
        pole=args.pole,
        boundary=args.boundary,
        fit=args.fit,
        mlt=args.mlt,
        data_dir=args.data_dir,
        overlay_dir=args.fig_dir,
    )
    stem = (
        f"ampere_{args.start.isoformat()}_{args.end.isoformat()}"
        f"_{args.pole}_jrmax_{args.mlt:02.0f}mlt"
    )
    write_csv(args.fig_dir / f"{stem}.csv", rows)
    plot_timeseries(
        times,
        jr_max,
        args.fig_dir / f"{stem}.png",
        mlt=args.mlt,
        start=args.start,
        end=args.end,
        pole=args.pole,
        boundary=args.boundary,
    )
    finite = jr_max[np.isfinite(jr_max)]
    print(
        f"Finished {len(times)} samples; "
        f"max j_R range {finite.min():.2f} to {finite.max():.2f} uA/m^2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

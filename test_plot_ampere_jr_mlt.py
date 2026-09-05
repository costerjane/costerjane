#!/usr/bin/env python3
"""Tests for AMPERE j_R-at-MLT extraction from summary-plot panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import plot_ampere_jr_mlt as jr
import plot_ampere_movies as ampere


def _rd_bu_r(n: int = 256) -> np.ndarray:
    r"""Red-white-blue LUT similar to the AMPERE j_R colorbar (red = +)."""
    x = np.linspace(-1.0, 1.0, n)
    rgb = np.zeros((n, 3), dtype=np.float32)
    # negative: blue -> white
    neg = x <= 0
    t = (x[neg] + 1.0)
    rgb[neg, 0] = t
    rgb[neg, 1] = t
    rgb[neg, 2] = 1.0
    # positive: white -> red
    pos = x > 0
    t = 1.0 - x[pos]
    rgb[pos, 0] = 1.0
    rgb[pos, 1] = t
    rgb[pos, 2] = t
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def _paint_colorbar(panel: np.ndarray, x: int, y0: int, y1: int, lut: np.ndarray) -> None:
    ys = np.linspace(0, len(lut) - 1, y1 - y0)
    for i, y in enumerate(range(y0, y1)):
        panel[y, x : x + 3] = lut[len(lut) - 1 - int(round(ys[i]))]


def synthetic_jr_panel(
    *,
    jr_peak: float = 0.72,
    mlt_peak: float = 16.0,
    mlat_peak: float = 72.0,
    mlat_outer: float = 40.0,
    size: int = 640,
) -> tuple[np.ndarray, jr.PolarLayout]:
    """Make a 640x640 polar panel with a known j_R blob at one MLT/MLAT."""
    panel = np.full((size, size, 3), 255, dtype=np.uint8)
    cx = cy = 320.0
    radius = 260.0
    lut = _rd_bu_r()
    _paint_colorbar(panel, 580, 40, 180, lut)

    yy, xx = np.ogrid[:size, :size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # 12 MLT at top, 18 MLT at left: angle CCW from noon
    ang = np.arctan2(cx - xx, cy - yy)  # 0 at 12 MLT, positive toward 18 MLT
    mlt = (12.0 + 12.0 * ang / np.pi) % 24.0
    mlat = 90.0 - (r / radius) * (90.0 - mlat_outer)

    # Gaussian blob in (MLT, MLAT)
    dmlt = (mlt - mlt_peak + 12.0) % 24.0 - 12.0
    field = jr_peak * np.exp(-0.5 * ((dmlt / 0.35) ** 2 + ((mlat - mlat_peak) / 2.5) ** 2))
    field = np.where(r <= radius, field, 0.0)
    field = np.clip(field, -1.0, 1.0)

    idx = np.clip(np.round((field + 1.0) * 0.5 * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    inside = r <= radius
    panel[inside] = lut[idx[inside]]

    # latitude circle (edge for Hough)
    ring = np.abs(r - radius) < 1.2
    panel[ring] = (30, 30, 30)

    lut_rgb = lut[::-1].astype(np.float32)  # top = +1 (red) like the real colorbar
    lut_val = np.linspace(1.0, -1.0, len(lut_rgb))
    layout = jr.PolarLayout(cx, cy, radius, mlat_outer, lut_rgb, lut_val)
    return panel, layout


def test_mlt_xy_16_is_upper_left() -> None:
    layout = jr.PolarLayout(320, 320, 200, 40, np.zeros((10, 3)), np.linspace(1, -1, 10))
    x, y = jr.mlt_xy(layout, 16.0, 200.0)
    assert x < layout.cx  # toward 18 MLT (left)
    assert y < layout.cy  # toward 12 MLT (up)
    x18, y18 = jr.mlt_xy(layout, 18.0, 200.0)
    assert abs(x18 - (layout.cx - 200)) < 1e-6
    assert abs(y18 - layout.cy) < 1e-6


def test_rgb_to_jr_colorbar_ends() -> None:
    _panel, layout = synthetic_jr_panel()
    assert jr.rgb_to_jr(layout.lut_rgb[0], layout) == 1.0
    assert jr.rgb_to_jr(layout.lut_rgb[-1], layout) == -1.0
    mid = layout.lut_rgb[len(layout.lut_rgb) // 2]
    assert abs(jr.rgb_to_jr(mid, layout)) < 0.05


def test_max_jr_recovers_synthetic_blob() -> None:
    panel, layout = synthetic_jr_panel(jr_peak=0.72, mlt_peak=16.0, mlat_peak=72.0)
    jr_max, mlat = jr.max_jr_at_mlt(panel, layout, mlt=16.0)
    assert 0.60 < jr_max < 0.80, jr_max
    assert 68.0 < mlat < 76.0, mlat


def test_jr_panel_is_right_third() -> None:
    image = np.zeros((10, 30, 3), dtype=np.uint8)
    image[:, 20:30, 0] = 255
    panel = jr.jr_panel(image)
    assert panel.shape == (10, 10, 3)
    assert int(panel[:, :, 0].mean()) == 255


def test_detect_layout_on_synthetic() -> None:
    panel, expected = synthetic_jr_panel()
    layout = jr.detect_layout(panel, mlat_outer=40.0)
    assert abs(layout.cx - expected.cx) < 8
    assert abs(layout.cy - expected.cy) < 8
    assert abs(layout.radius - expected.radius) < 8
    jr_max, mlat = jr.max_jr_at_mlt(panel, layout, mlt=16.0)
    assert jr_max > 0.5
    assert 65.0 < mlat < 80.0


def test_committed_preview_has_afternoon_current() -> None:
    preview = Path("figures/ampere_2025-05-10_north_40deg_preview.png")
    assert preview.exists(), preview
    image = jr.load_rgb(preview)
    panel = jr.jr_panel(image)
    layout = jr.detect_layout(panel, mlat_outer=40.0)
    jr_max, mlat = jr.max_jr_at_mlt(panel, layout, mlt=16.0)
    assert 0.5 < jr_max <= 1.0, jr_max
    assert 60.0 < mlat < 85.0, mlat


def test_live_official_plot_matches_preview_scale() -> None:
    when = __import__("datetime").datetime(2025, 5, 10, 12, 0, tzinfo=__import__("datetime").timezone.utc)
    url = ampere.plot_url(when)
    dest = Path("data/ampere/plots/2025-05-10/ampere_20250510T1200_north_40.png")
    ampere.download_file(url, dest)
    image = jr.load_rgb(dest)
    panel = jr.jr_panel(image)
    layout = jr.detect_layout(panel, mlat_outer=40.0)
    jr_max, mlat = jr.max_jr_at_mlt(panel, layout, mlt=16.0)
    assert 0.7 < jr_max <= 1.0, (jr_max, mlat)
    assert 68.0 < mlat < 80.0, (jr_max, mlat)
    # Must use the 40° circle (~287 px), not the 50° ring (~230 px).
    assert layout.radius > 260, layout.radius


if __name__ == "__main__":
    test_mlt_xy_16_is_upper_left()
    test_rgb_to_jr_colorbar_ends()
    test_max_jr_recovers_synthetic_blob()
    test_jr_panel_is_right_third()
    test_detect_layout_on_synthetic()
    test_committed_preview_has_afternoon_current()
    test_live_official_plot_matches_preview_scale()
    print("all tests passed")

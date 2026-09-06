#!/usr/bin/env python3
"""Tests for Madrigal + Google phone VTEC merge."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import plot_merged_tec_movie as tec


def test_phone_url() -> None:
    url = tec.phone_vtec_url(dt.date(2024, 5, 12))
    assert url.endswith("vtec_2024_05_12.csv.gz")
    assert "ec6fe27c-d523-4b43-bf5b-a5a046b49f03_v1.1" in url


def test_phone_date_outside_archive_raises() -> None:
    try:
        tec.download_phone_vtec(dt.date(2026, 5, 12), Path("data"))
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "2023-09-11" in msg
        assert "2024-05-24" in msg
        assert "2024-05-12" in msg
    else:
        raise AssertionError("expected FileNotFoundError for 2026-05-12")


def test_merge_fills_phone_only_gaps() -> None:
    mad = np.array([[1.0, np.nan], [np.nan, 4.0]])
    ms = np.array([[1.0, np.nan], [np.nan, 1.0]])
    ph = np.array([[np.nan, 2.0], [3.0, 8.0]])
    ps = np.array([[np.nan, 1.0], [1.0, 1.0]])
    merged = tec.merge_tec(mad, ms, ph, ps)
    assert merged[0, 0] == 1.0
    assert merged[0, 1] == 2.0
    assert merged[1, 0] == 3.0
    # both present, equal sigma -> average
    assert abs(merged[1, 1] - 6.0) < 1e-9


def test_bin_phone_to_grid_inverse_variance() -> None:
    gdlat = np.array([-1.0, 0.0, 1.0])
    glon = np.array([-1.0, 0.0, 1.0])
    lat = np.array([0.0, 0.0])
    lon = np.array([0.0, 0.0])
    vtec = np.array([10.0, 20.0])
    std = np.array([1.0, 2.0])
    mean, sigma = tec.bin_phone_to_grid(lat, lon, vtec, std, gdlat, glon)
    # weights 1 and 0.25 -> mean = (10 + 5) / 1.25 = 12
    assert abs(mean[1, 1] - 12.0) < 1e-9
    assert np.isnan(mean[0, 0])
    assert sigma[1, 1] > 0


def test_s2_token_known_cell() -> None:
    lat, lon = tec.s2_tokens_to_latlon(np.array(["0007c"]))
    assert -40 < lat[0] < -25
    assert -50 < lon[0] < -40


def test_nearest_time_index() -> None:
    times = np.array(
        [
            dt.datetime(2024, 5, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 5, 12, 0, 10, tzinfo=dt.timezone.utc),
        ]
    )
    when = dt.datetime(2024, 5, 12, 0, 7, tzinfo=dt.timezone.utc)
    assert tec.nearest_time_index(times, when) == 1


def test_load_madrigal_if_present() -> None:
    matches = sorted(Path("data").glob("gps240512g*.hdf5"))
    if not matches:
        print("skip madrigal load (no hdf5)")
        return
    grid = tec.load_madrigal_grid(matches[0])
    assert grid["tec"].shape[0] == grid["gdlat"].size == 180
    assert grid["tec"].shape[1] == grid["glon"].size == 360
    assert grid["times"].size == 288
    noon = tec.nearest_time_index(
        grid["times"], dt.datetime(2024, 5, 12, 12, 0, tzinfo=dt.timezone.utc)
    )
    sl = grid["tec"][:, :, noon]
    assert np.isfinite(sl).sum() > 1000
    print(f"Madrigal 12 UT finite cells={np.isfinite(sl).sum()} max={np.nanmax(sl):.1f} TECU")


if __name__ == "__main__":
    test_phone_url()
    test_phone_date_outside_archive_raises()
    test_merge_fills_phone_only_gaps()
    test_bin_phone_to_grid_inverse_variance()
    test_s2_token_known_cell()
    test_nearest_time_index()
    test_load_madrigal_if_present()
    print("all tests passed")

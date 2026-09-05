#!/usr/bin/env python3
"""Tests for AMPERE survey-movie URL construction and date handling."""

from __future__ import annotations

import datetime as dt
import io
from contextlib import redirect_stdout
from pathlib import Path

import plot_ampere_movies as ampere


def test_movie_url() -> None:
    url = ampere.movie_url(dt.date(2025, 5, 10), pole="north", boundary=40)
    assert url == (
        "https://ampere.jhuapl.edu/products/smr.movies/k060_m08/2025/20250510/"
        "ampere.20250510.k060_m08.north.40.smr.mp4"
    )


def test_plot_url_wraps_10_minute_window() -> None:
    when = dt.datetime(2025, 5, 10, 12, 0, tzinfo=dt.timezone.utc)
    url = ampere.plot_url(when, pole="north", boundary=40)
    assert url.endswith(
        "ampere.20250510.120000-121000.k060_m08.north.40.smr.png"
    )
    assert "/smr.plots/k060_m08/2025/20250510/" in url


def test_daterange_inclusive() -> None:
    days = ampere.daterange(dt.date(2025, 5, 10), dt.date(2025, 5, 12))
    assert [d.isoformat() for d in days] == [
        "2025-05-10",
        "2025-05-11",
        "2025-05-12",
    ]


def test_day_frame_times_are_10_min_cadence() -> None:
    times = ampere.day_frame_times(dt.date(2025, 5, 10))
    assert len(times) == 144
    assert times[0] == dt.datetime(2025, 5, 10, 0, 0, tzinfo=dt.timezone.utc)
    assert times[1] == dt.datetime(2025, 5, 10, 0, 10, tzinfo=dt.timezone.utc)
    assert times[-1] == dt.datetime(2025, 5, 10, 23, 50, tzinfo=dt.timezone.utc)


def test_download_listing_includes_curl_and_urls() -> None:
    dates = ampere.daterange(dt.date(2025, 5, 10), dt.date(2025, 5, 12))
    text = ampere.format_download_listing(dates, pole="north", boundary=40)
    assert "ampere.20250510.k060_m08.north.40.smr.mp4" in text
    assert "ampere.20250512.k060_m08.north.40.smr.mp4" in text
    assert "curl -L --fail --retry 4" in text
    assert "https://ampere.jhuapl.edu/download-sandbox/" in text
    urls_block, curl_block = text.split("# curl", 1)
    assert urls_block.count("https://ampere.jhuapl.edu/products/smr.movies/") == 3
    assert curl_block.count("https://ampere.jhuapl.edu/products/smr.movies/") == 3


def test_print_urls_exits_without_download() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ampere.main(
            ["--start", "2025-05-10", "--end", "2025-05-12", "--print-urls"]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "curl -L --fail --retry 4" in out
    urls_block, curl_block = out.split("# curl", 1)
    assert urls_block.count("https://ampere.jhuapl.edu/products/smr.movies/") == 3
    assert curl_block.count("https://ampere.jhuapl.edu/products/smr.movies/") == 3


def test_live_may_2025_products_exist() -> None:
    """Live check against the public AMPERE product tree used for the movies."""
    movie = ampere.movie_url(dt.date(2025, 5, 10), pole="north", boundary=40)
    plot = ampere.plot_url(
        dt.datetime(2025, 5, 10, 12, 0, tzinfo=dt.timezone.utc),
        pole="north",
        boundary=40,
    )
    assert ampere.http_exists(movie), movie
    assert ampere.http_exists(plot), plot


def test_assemble_tiny_movie(tmp_path: Path | None = None) -> None:
    """Encode two copied PNGs into a short MP4 to verify the ffmpeg path."""
    root = Path(tmp_path) if tmp_path is not None else Path("data/ampere/test_frames")
    root.mkdir(parents=True, exist_ok=True)
    src = Path("figures/ampere_2025-05-10_north_40deg_preview.png")
    if not src.exists():
        when = dt.datetime(2025, 5, 10, 12, 0, tzinfo=dt.timezone.utc)
        src = root / "live.png"
        ampere.download_file(ampere.plot_url(when), src)
    frames = [root / "a.png", root / "b.png"]
    for dest in frames:
        dest.write_bytes(src.read_bytes())
    outfile = root / "tiny.mp4"
    ampere.assemble_movie_from_frames(frames, outfile, fps=2)
    assert outfile.exists() and outfile.stat().st_size > 0


if __name__ == "__main__":
    test_movie_url()
    test_plot_url_wraps_10_minute_window()
    test_daterange_inclusive()
    test_day_frame_times_are_10_min_cadence()
    test_download_listing_includes_curl_and_urls()
    test_print_urls_exits_without_download()
    test_live_may_2025_products_exist()
    test_assemble_tiny_movie()
    print("all tests passed")

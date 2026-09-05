#!/usr/bin/env python3
"""Download JHUAPL AMPERE survey movies and assemble northern-hemisphere films.

Uses the public AMPERE Science Data Center products at
https://ampere.jhuapl.edu :

* Daily Type 1 (40° MLAT) / Type 2 (60° MLAT) MP4 survey movies
* 10-minute three-panel summary-plot PNGs as a fallback when a movie
  is not published

Default fit product is ``k060_m08`` (degree 60 / order 8 spherical-cap
harmonic fit), which is what the AMPERE quick-look page plays.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

AMPERE_BASE = "https://ampere.jhuapl.edu/products"
DEFAULT_FIT = "k060_m08"
DEFAULT_POLE = "north"
DEFAULT_BOUNDARY = 40
USER_AGENT = (
    "costerjane-ampere-movies/1.0 "
    "(MIT Haystack; costera@mit.edu; research visualization)"
)
FRAME_CADENCE = dt.timedelta(minutes=10)
FRAME_WINDOW = dt.timedelta(minutes=10)


class AmpereDownloadError(RuntimeError):
    """HTTP or local I/O failure while fetching an AMPERE product."""


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    days = (end - start).days + 1
    return [start + dt.timedelta(days=i) for i in range(days)]


def movie_url(
    date: dt.date,
    *,
    pole: str = DEFAULT_POLE,
    boundary: int = DEFAULT_BOUNDARY,
    fit: str = DEFAULT_FIT,
) -> str:
    """Official AMPERE daily survey-movie URL (MP4)."""
    ymd = date.strftime("%Y%m%d")
    return (
        f"{AMPERE_BASE}/smr.movies/{fit}/{date.year}/{ymd}/"
        f"ampere.{ymd}.{fit}.{pole}.{boundary}.smr.mp4"
    )


def movie_filename(
    date: dt.date,
    *,
    pole: str = DEFAULT_POLE,
    boundary: int = DEFAULT_BOUNDARY,
    fit: str = DEFAULT_FIT,
) -> str:
    """Basename of the official daily survey-movie MP4."""
    ymd = date.strftime("%Y%m%d")
    return f"ampere.{ymd}.{fit}.{pole}.{boundary}.smr.mp4"


def curl_command(url: str, dest_name: str) -> str:
    """Copy-pasteable curl line for a public AMPERE product URL."""
    return f"curl -L --fail --retry 4 -A '{USER_AGENT}' -o '{dest_name}' '{url}'"


def format_download_listing(
    dates: list[dt.date],
    *,
    pole: str = DEFAULT_POLE,
    boundary: int = DEFAULT_BOUNDARY,
    fit: str = DEFAULT_FIT,
) -> str:
    """Human-readable list of official movie URLs and curl commands."""
    lines = [
        f"# Official JHUAPL AMPERE survey movies ({pole}, {boundary}° MLAT, {fit})",
        "# Direct HTTP from the public product tree; login is not required.",
        "# Portal browse/download: https://ampere.jhuapl.edu/download-sandbox/",
        "",
        "# URLs",
    ]
    urls = [movie_url(date, pole=pole, boundary=boundary, fit=fit) for date in dates]
    lines.extend(urls)
    lines.extend(["", "# curl"])
    for date, url in zip(dates, urls):
        lines.append(curl_command(url, movie_filename(date, pole=pole, boundary=boundary, fit=fit)))
    lines.append("")
    return "\n".join(lines)


def plot_url(
    when: dt.datetime,
    *,
    pole: str = DEFAULT_POLE,
    boundary: int = DEFAULT_BOUNDARY,
    fit: str = DEFAULT_FIT,
) -> str:
    """Official AMPERE 10-minute Type 1/2 summary-plot PNG URL."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    when = when.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
    end = when + FRAME_WINDOW
    ymd = when.strftime("%Y%m%d")
    return (
        f"{AMPERE_BASE}/smr.plots/{fit}/{when.year}/{ymd}/"
        f"ampere.{ymd}.{when:%H%M%S}-{end:%H%M%S}.{fit}.{pole}.{boundary}.smr.png"
    )


def day_frame_times(date: dt.date) -> list[dt.datetime]:
    start = dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc)
    n = int(dt.timedelta(days=1) / FRAME_CADENCE)
    return [start + i * FRAME_CADENCE for i in range(n)]


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener()


def http_exists(url: str, timeout: float = 30.0) -> bool:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with _opener().open(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        if exc.code in {405, 501}:
            return http_get_ok(url, timeout=timeout)
        raise AmpereDownloadError(f"HEAD {url} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AmpereDownloadError(f"HEAD {url} failed: {exc.reason}") from exc


def http_get_ok(url: str, timeout: float = 30.0) -> bool:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
    try:
        with _opener().open(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise AmpereDownloadError(f"GET {url} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AmpereDownloadError(f"GET {url} failed: {exc.reason}") from exc


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    retries: int = 4,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with _opener().open(request, timeout=timeout) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if tmp.stat().st_size == 0:
                raise AmpereDownloadError(f"Empty download for {url}")
            tmp.replace(dest)
            print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if tmp.exists():
                    tmp.unlink()
                raise AmpereDownloadError(f"Not found: {url}") from exc
            last_error = AmpereDownloadError(f"GET {url} failed: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = AmpereDownloadError(f"GET {url} failed: {exc}")
        if tmp.exists():
            tmp.unlink()
        sleep_s = 2 ** attempt
        print(f"Retry {attempt + 1}/{retries} in {sleep_s}s for {url}")
        time.sleep(sleep_s)
    raise last_error or AmpereDownloadError(f"Failed to download {url}")


def _run_ffmpeg(args: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to assemble or concatenate movies")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n"
            + (result.stderr.strip() or result.stdout.strip() or "(no output)")
        )


def concatenate_movies(paths: list[Path], outfile: Path) -> Path:
    if not paths:
        raise ValueError("No movies to concatenate")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "concat.txt"
        listing.write_text(
            "".join(f"file '{path.resolve()}'\n" for path in paths),
            encoding="utf-8",
        )
        try:
            _run_ffmpeg(
                [
                    "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart",
                    str(outfile),
                ]
            )
        except RuntimeError:
            _run_ffmpeg(
                [
                    "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart",
                    str(outfile),
                ]
            )
    print(f"Wrote {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")
    return outfile


def assemble_movie_from_frames(
    frames: list[Path],
    outfile: Path,
    *,
    fps: int = 12,
) -> Path:
    if not frames:
        raise ValueError("No frames to assemble")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged: list[Path] = []
        for i, src in enumerate(frames):
            dest = Path(tmp) / f"frame_{i:05d}{src.suffix.lower()}"
            dest.symlink_to(src.resolve())
            staged.append(dest)
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
    print(f"Wrote {outfile} from {len(frames)} frames ({outfile.stat().st_size / 1e6:.1f} MB)")
    return outfile


def download_day_movie(
    date: dt.date,
    movie_dir: Path,
    *,
    pole: str,
    boundary: int,
    fit: str,
    from_frames: bool,
    frame_dir: Path,
    fps: int,
) -> Path:
    label = f"ampere_{date.isoformat()}_{pole}_{boundary:02d}deg"
    outfile = movie_dir / f"{label}.mp4"
    official = movie_url(date, pole=pole, boundary=boundary, fit=fit)

    if not from_frames:
        try:
            return download_file(official, outfile)
        except AmpereDownloadError as exc:
            print(f"Official movie missing ({exc}); assembling from 10-min PNGs")

    frames: list[Path] = []
    missing = 0
    for when in day_frame_times(date):
        dest = (
            frame_dir
            / f"{date.isoformat()}"
            / f"ampere_{when:%Y%m%dT%H%M}_{pole}_{boundary:02d}.png"
        )
        try:
            frames.append(download_file(plot_url(when, pole=pole, boundary=boundary, fit=fit), dest))
        except AmpereDownloadError:
            missing += 1
            print(f"Skipping missing frame {when:%Y-%m-%d %H:%M} UT")
    if not frames:
        raise AmpereDownloadError(
            f"No AMPERE frames available for {date.isoformat()} {pole} {boundary}°"
        )
    if missing:
        print(f"Assembled {date.isoformat()} with {len(frames)} frames ({missing} missing)")
    return assemble_movie_from_frames(frames, outfile, fps=fps)


def extract_preview_frame(movie: Path, outfile: Path, timestamp: str = "12:00:00") -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-ss", timestamp, "-i", str(movie),
            "-frames:v", "1", "-q:v", "2",
            str(outfile),
        ]
    )
    print(f"Wrote preview {outfile}")
    return outfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-05-10", type=_parse_date)
    parser.add_argument("--end", default="2025-05-12", type=_parse_date)
    parser.add_argument("--pole", default=DEFAULT_POLE, choices=("north", "south"))
    parser.add_argument(
        "--boundary",
        default=DEFAULT_BOUNDARY,
        type=int,
        choices=(40, 60),
        help="Magnetic-latitude cutoff used by AMPERE Type 1 (40) / Type 2 (60) plots",
    )
    parser.add_argument("--fit", default=DEFAULT_FIT)
    parser.add_argument("--data-dir", default=Path("data/ampere"), type=Path)
    parser.add_argument("--fig-dir", default=Path("figures"), type=Path)
    parser.add_argument("--from-frames", action="store_true",
                        help="Ignore official MP4s and assemble each day from PNG frames")
    parser.add_argument("--fps", default=12, type=int,
                        help="Frame rate used when assembling from PNGs")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument(
        "--print-urls",
        action="store_true",
        help="Print official JHUAPL MP4 URLs and curl commands, then exit",
    )
    args = parser.parse_args(argv)

    dates = daterange(args.start, args.end)
    listing = format_download_listing(
        dates, pole=args.pole, boundary=args.boundary, fit=args.fit
    )
    if args.print_urls:
        print(listing, end="")
        return 0

    movie_dir = args.fig_dir
    frame_dir = args.data_dir / "frames"
    movie_dir.mkdir(parents=True, exist_ok=True)
    url_list = movie_dir / (
        f"ampere_{args.start.isoformat()}_{args.end.isoformat()}"
        f"_{args.pole}_{args.boundary:02d}deg_urls.txt"
    )
    url_list.write_text(listing, encoding="utf-8")
    print(f"Wrote download URLs to {url_list}")
    daily: list[Path] = []
    for date in dates:
        daily.append(
            download_day_movie(
                date,
                movie_dir,
                pole=args.pole,
                boundary=args.boundary,
                fit=args.fit,
                from_frames=args.from_frames,
                frame_dir=frame_dir,
                fps=args.fps,
            )
        )

    preview_stamp = "00:01:00"
    for movie, date in zip(daily, dates):
        extract_preview_frame(
            movie,
            args.fig_dir / f"ampere_{date.isoformat()}_{args.pole}_{args.boundary:02d}deg_preview.png",
            timestamp=preview_stamp,
        )

    if not args.skip_combined and len(daily) > 1:
        start = args.start.isoformat()
        end = args.end.isoformat()
        combined = args.fig_dir / (
            f"ampere_{start}_{end}_{args.pole}_{args.boundary:02d}deg.mp4"
        )
        concatenate_movies(daily, combined)
        extract_preview_frame(
            combined,
            args.fig_dir / f"ampere_{start}_{end}_{args.pole}_{args.boundary:02d}deg_preview.png",
            timestamp=preview_stamp,
        )

    print(f"Finished {len(daily)} daily movie(s) for {args.pole} {args.boundary}° MLAT")
    return 0


if __name__ == "__main__":
    sys.exit(main())

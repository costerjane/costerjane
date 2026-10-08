#!/usr/bin/env python3
"""Download October 2024 scint_mag_coords CSVs from the shared Google Drive folder."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

FOLDER_ID = "1BN3HBuqxXpBJAElz7DedXt3IwQOA6ljk"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"


def list_files(folder_id: str) -> list[tuple[str, str]]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", errors="ignore")
    return re.findall(
        r'id="entry-([a-zA-Z0-9_-]+)"[\s\S]*?flip-entry-title[^>]*>([^<]+)',
        html,
    )


def download(file_id: str, dest: Path) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = list_files(FOLDER_ID)
    if not entries:
        raise SystemExit("No files found — is the Drive folder shared as 'Anyone with the link'?")
    for file_id, name in entries:
        dest = OUT_DIR / name.strip()
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip existing {dest.name}")
            continue
        download(file_id, dest)
    print(f"done — {len(list(OUT_DIR.glob('*.csv')))} CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()

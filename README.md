# costerjane

Scripts for ionosphere / magnetosphere-ionosphere coupling plots.

## AMPERE northern-hemisphere movies

`plot_ampere_movies.py` downloads JHUAPL AMPERE survey movies from the
[AMPERE Science Data Center](https://ampere.jhuapl.edu) quick-look archive
and optionally concatenates a date range into one film.

Each daily movie is the official Type 1 (40° magnetic latitude) or Type 2
(60°) three-panel northern/southern hemisphere product: observed δB,
fitted δB, and fitted radial current \(j_R\). If an official MP4 is not
published, the script assembles the day from 10-minute summary-plot PNGs.

```bash
python3 plot_ampere_movies.py --start 2025-05-10 --end 2025-05-12 --pole north --boundary 40
```

Outputs land in `figures/` (MP4s are gitignored because each daily
file is ~20 MB):

- `ampere_YYYY-MM-DD_north_40deg.mp4` — official daily movie (2 min at 25 fps)
- `ampere_YYYY-MM-DD_YYYY-MM-DD_north_40deg.mp4` — concatenated interval
- matching `*_preview.png` stills at 12 UT
- `ampere_YYYY-MM-DD_YYYY-MM-DD_north_40deg_urls.txt` — official download URLs

Requires `ffmpeg` on `PATH`.

## Downloading the movies

The daily films are public HTTP objects on the AMPERE product tree
(no portal login required for these MP4s). Print the URLs and curl
commands:

```bash
python3 plot_ampere_movies.py --start 2025-05-10 --end 2025-05-12 --print-urls
```

Northern Type 1 movies for 10–12 May 2025:

```
https://ampere.jhuapl.edu/products/smr.movies/k060_m08/2025/20250510/ampere.20250510.k060_m08.north.40.smr.mp4
https://ampere.jhuapl.edu/products/smr.movies/k060_m08/2025/20250511/ampere.20250511.k060_m08.north.40.smr.mp4
https://ampere.jhuapl.edu/products/smr.movies/k060_m08/2025/20250512/ampere.20250512.k060_m08.north.40.smr.mp4
```

Example:

```bash
curl -L --fail --retry 4 \
  -o ampere.20250510.k060_m08.north.40.smr.mp4 \
  'https://ampere.jhuapl.edu/products/smr.movies/k060_m08/2025/20250510/ampere.20250510.k060_m08.north.40.smr.mp4'
```

To browse or bulk-download other products (GRD, IRD, plots, movies),
register at the [AMPERE download portal](https://ampere.jhuapl.edu/download-sandbox/).
The concatenated three-day MP4 is produced only by this script, not by
JHUAPL.

Data: AMPERE / Johns Hopkins University Applied Physics Laboratory
(Anderson et al.; Waters et al.). PI: Brian Anderson.

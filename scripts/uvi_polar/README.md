# SMILE UVI polar (MLAT/MLT) remapping

Remaps public EMDS UVI camera-frame quicklook PNGs onto a north-pole
**MLAT / MLT** map (approximate L2-style LOS projection) and builds a movie.

## Important: Oct 10 coverage

Public UVI quicklook for **2026-10-10** exists only from about
**00:00–05:06 UT**. There are **no** public UVI QL (or public science CDF)
files for **12:00–24:00 UT** that day. The movie produced here uses the
available morning window.

## Dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install cdflib numpy matplotlib pillow scipy imageio imageio-ffmpeg \
  tqdm cartopy aacgmv2
# also needs ffmpeg on PATH for make_movie.sh
```

## Data inputs

1. UVI QL PNGs from EMDS `smile.v_ql_data_file` (descriptor `AURORA`)
2. Public GSS measured orbit CDF, e.g. `SMILE_GSS_AUX_MEAS-ORBIT_YYYYMMDD_V*.cdf`

## Run

```bash
python scripts/uvi_polar/remap_uvi_polar.py \
  --ql-dir /path/to/ql_pngs \
  --orbit-cdf /path/to/SMILE_GSS_AUX_MEAS-ORBIT_20261010_V02.cdf \
  --frame-dir /path/to/frames \
  --start 2026-10-10T00:00:00 \
  --stop 2026-10-10T06:00:00 \
  --step-px 2

bash scripts/uvi_polar/make_movie.sh /path/to/frames /path/to/out.mp4
```

## Geometry notes

This is a **best-effort public-data remapping**, not the official L1/L2
pipeline. Boresight is assumed toward geocenter; roll uses a J2000-Z
reference. Official L2 maps LOS to a 110 km shell in GEO and AACGM
(Zhang et al., Space Sci Rev 2025). Private L0/L1 CDFs would improve fidelity.

# SMILE UVI polar (geographic, sun-synchronous) remapping

Remaps public EMDS UVI **camera-frame** quicklook PNGs onto a north-pole
**geographic** map and builds a movie.

The polar view is centered on the **geographic North Pole**, extending down to
**40°N**. Default orientation is **sun-synchronous**: the subsolar longitude is
kept on the **left** so dayglow stays fixed while continents rotate — matching
SMILE’s sun-locked science attitude (body +Y ≈ GSE +Y). Pass `--lon-bottom -90`
for a fixed US/Canada-at-bottom view instead.

The auroral oval is a ring around the *magnetic* pole, so it appears offset
from the geographic center — that is expected. Optional AACGM MLAT labels are
annotated for style match to official QL products.

## Geometric chain (required)

1. Load UVI image + spacecraft **position** and **attitude** at the frame UT  
2. Build camera LOS (sun-sync roll from `−Sun×Earth`; UVI tip auto from
   MEAS-ATT Z–Earth angle, or `--mount-deg`)  
3. Limb-fit tip-only correction to geocenter (preserves sun-sync roll)  
4. Intersect each LOS with the emission shell at \(R_E + 110\,\mathrm{km}\)  
5. Convert hit points J2000 → GEO (ITRS) lat/lon  
6. Plot on a geographic polar grid (lat rings 40/50/60/70/80°, lon spokes,
   coastlines, solar terminator)

## Public coverage notes

- **2026-10-05**: UVI QL available through the day; Earth is in FOV for the
  early hours used here (**00:00–06:00 UT**).
- **2026-10-10 00–05 UT**: public QL is mostly star fields (attitude ~145° from
  Earth) — not usable for an aurora polar movie.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/uvi_polar/requirements.txt

python scripts/uvi_polar/remap_uvi_polar.py \
  --ql-dir /path/to/ql_pngs \
  --orbit-cdf /path/to/SMILE_GSS_AUX_MEAS-ORBIT_YYYYMMDD_V*.cdf \
  --att-cdf   /path/to/SMILE_GSS_AUX_MEAS-ATT_YYYYMMDD_V*.cdf \
  --frame-dir /path/to/frames \
  --start 2026-10-05T00:00:00 \
  --stop  2026-10-05T06:00:00

# Optional: fixed US-at-bottom instead of sun-sync
#   --lon-bottom -90
# Optional: force 25° mount tip instead of auto
#   --mount-deg 25

bash scripts/uvi_polar/make_movie.sh /path/to/frames /path/to/out.mp4
```

## Caveats

- Public inputs are **QL PNGs**, not private L1/L2 CDFs — intensity is relative
  display brightness, not calibrated Rayleighs.
- Camera \(+\)column from sun-synchronous \(-\mathrm{Sun}\times\mathrm{Earth}\)
  (≈ body \(-Y\)), with QL row-axis flip. Limb-fit tip-only refinement then
  re-applies that roll.
- Not an official SOC/SAS product; do not publish as calibrated L2.

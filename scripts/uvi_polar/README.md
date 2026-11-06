# SMILE UVI polar (geographic) remapping

Remaps public EMDS UVI **camera-frame** quicklook PNGs onto a north-pole
**geographic** map and builds a movie.

Official SMILE UVI polar quicklooks are geographic (not AACGM/MLT): the plot is
centered on the **geographic North Pole**, with a fixed longitude orientation
so **North America / the contiguous US sits at the bottom**. The auroral oval
is a ring around the magnetic pole and therefore appears offset from the plot
center — that is expected.

## Geometric chain (required)

1. Load UVI image + spacecraft **position** and **attitude** at the frame UT  
2. Build camera LOS for each pixel (UVI boresight from body attitude; limb-fit
   Earth center applies a **tip-only** correction so the geocenter LOS matches
   the Earth disk while preserving MEAS-ATT roll)  
3. Intersect each LOS with the emission shell at \(R_E + 110\,\mathrm{km}\)  
4. Convert hit points J2000 → GEO (ITRS) lat/lon  
5. Plot on a geographic polar grid (latitude rings, longitude meridians,
   geographic coastlines, solar terminator). Optional AACGM MLAT text anchors
   match the official QL annotations.

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

bash scripts/uvi_polar/make_movie.sh /path/to/frames /path/to/out.mp4
```

`--lon-bottom` (default `-90`) sets which geographic longitude sits at the
bottom of the polar disk.

## Caveats

- Public inputs are **QL PNGs**, not private L1/L2 CDFs — intensity is relative
  display brightness, not calibrated Rayleighs.
- UVI mounting uses PLM \(+Z\) tipped by \(R_y(+24^\circ)\) (mission papers); camera
  \(+\)column from body \(-Y\), with QL row-axis flip. Limb-fit tip-only refinement
  absorbs residual pointing error without scrambling roll.
- Not an official SOC/SAS product; do not publish as calibrated L2.

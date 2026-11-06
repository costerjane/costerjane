# SMILE UVI polar (MLAT/MLT) remapping

Remaps public EMDS UVI **camera-frame** quicklook PNGs onto a north-pole
**MLAT / MLT** map and builds a movie.

## Geometric chain (required)

1. Load UVI image + spacecraft **position** and **attitude** at the frame UT  
2. Build camera LOS for each pixel (UVI boresight from body attitude; Earth-disk
   centroid refines pointing so the geocenter LOS matches the Earth disk)  
3. Intersect each LOS with the emission shell at \(R_E + 110\,\mathrm{km}\)  
4. Convert hit points J2000 → GEO (ITRS) → AACGM **MLAT / MLT**  
5. Plot on a polar grid (MLAT rings, MLT spokes) with AACGM-warped coastlines  

## Oct 10 public coverage

Public UVI QL for **2026-10-10** exists only **~00:00–05:06 UT**. There are
**no** public UVI files for **12:00–24:00 UT**.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/uvi_polar/requirements.txt

python scripts/uvi_polar/remap_uvi_polar.py \
  --ql-dir /path/to/ql_pngs \
  --orbit-cdf /path/to/SMILE_GSS_AUX_MEAS-ORBIT_YYYYMMDD_V*.cdf \
  --att-cdf   /path/to/SMILE_GSS_AUX_MEAS-ATT_YYYYMMDD_V*.cdf \
  --frame-dir /path/to/frames \
  --start 2026-10-10T00:00:00 \
  --stop  2026-10-10T06:00:00

bash scripts/uvi_polar/make_movie.sh /path/to/frames /path/to/out.mp4
```

## Caveats

- Public inputs are **QL PNGs**, not private L1/L2 CDFs — intensity is relative
  display brightness, not calibrated Rayleighs.
- UVI mounting uses PLM \(+Z\) tipped by \(R_y(+24^\circ)\) (mission papers); Earth-disk
  refinement absorbs residual alignment error.
- Not an official SOC/SAS product; do not publish as calibrated L2.

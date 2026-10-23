# costerjane

Tools for EIA / VTEC analysis, including conversion of quiet monthly median
TEC maps into quasi-dipole (QD) latitude and magnetic local time (MLT).

## 2017 quiet monthly medians → QD / MLT

Source pickles (geographic latitude/longitude × local time; not in git):

`data/quiet_monthly_median/2017/vtec_avg_*_kp_4_median.pkl`

Convert:

```bash
python scripts/convert_vtec_qd_mlt.py
```

Outputs in `data/quiet_monthly_median/2017_qd_mlt/`:

| File pattern | Grid | Use |
|---|---|---|
| `vtec_qd_mlt_YYYYMM.npz` | QD lat (1°) × MLT (0.25 h) | MLT morphology |
| `vtec_qdlat_qdlon_mlt14_YYYYMM.npz` | QD lat (1°) × QD lon (5°), MLT≈14±0.5 h | **Longitude structure** |

Conversion uses `apexpy` at 350 km. MLT from `UT = LT - glon/15`, then `mlon2mlt`.

## EIA coefficients vs QD longitude (MLT ≈ 14)

Fit the 9-parameter EIA profile at each QD-longitude bin of the
`vtec_qdlat_qdlon_mlt14_*.npz` maps:

```bash
python scripts/fit_eia_coeffs_qdlon.py
```

Outputs in `data/quiet_monthly_median/2017_eia_coeffs_qdlon/`:

- `eia_coeffs_qdlon_mlt14_YYYYMM.npz` with `qdlon_centers`, `coefficients` `(n_lon, 9)`,
  `rmse`, resolved flags, and reported crest positions/widths.

## Fourier smoothing in QD longitude

```bash
python scripts/fit_eia_fourier_qdlon.py --order 3
```

Fits \(c(\phi)=a_0+\sum_{k=1}^{K}[a_k\cos(k\phi)+b_k\sin(k\phi)]\) to each
coefficient, reconstructs VTEC, and compares to the original maps. Outputs in
`data/quiet_monthly_median/2017_eia_fourier_qdlon/`.

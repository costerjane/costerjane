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

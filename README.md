# costerjane

Scripts for ionosphere / magnetosphere-ionosphere coupling plots.

## Merged global TEC movie (Madrigal + Android phones)

`plot_merged_tec_movie.py` downloads the CEDAR Madrigal **gridded VTEC**
product (instrument 8000, kindat 3500: 1° × 1° × 5 min) and the Google
**phone VTEC maps** from Smith et al., *Nature* (2024), then builds a
full-day three-panel global movie: Madrigal, phones, and an
inverse-variance merge that fills station gaps with phone cells.

The published phone archive
([doi:10.24433/CO.9149928.v1](https://doi.org/10.24433/CO.9149928.v1))
covers **11 September 2023 – 24 May 2024**. There is no phone map for
12 May 2026. The overlapping May 12 in both products is **12 May 2024**.

```bash
python3 plot_merged_tec_movie.py --date 2024-05-12
```

Outputs in `figures/`:

- `merged_tec_2024-05-12_global.mp4` — 144 frames, 10 min cadence, 8 fps
- `merged_tec_2024-05-12_00ut.png` / `_12ut.png` / `_20ut.png` stills

Phone VTEC is quality-filtered (`0 ≤ VTEC ≤ 200` TECU,
`σ ≤ 8` TECU) and binned onto the Madrigal grid. Merged cells use
inverse-variance weighting when both sources observe the same 1° bin.

Data: CEDAR Madrigal GNSS TEC (MIT Haystack) and Smith et al. phone
VTEC maps (Google / Code Ocean).

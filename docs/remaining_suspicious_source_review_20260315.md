# Remaining Suspicious Spectra Review (2026-03-15)

This review covers the final full-raw SIAC package:

- `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`

Detector summary:

- total exported spectra: `77125`
- suspicious spectra: `187`
- suspicious fraction: `0.24%`

Source-by-source suspicious plots were generated under:

- `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/full_review/source_suspicious_plots`

Important interpretation:

- "suspicious" here means flagged by the final detector, not automatically broken.
- Many spectra are simply unusual materials or edge-case vegetation types that disagree with the smoother.
- This review separates likely-valid residual outliers from spectra that still look like repair candidates.

## Priority Summary

Review-only / likely valid:

- `ossl`
- `bssl`
- `antarctic_vegetation_speclib`
- `cabo_leaf_v2`
- `drylands_emit`
- `ecostress_v1`
- `emit_adjusted_surface`
- `montesinho_plants`
- `neon_field_spectra`
- `ngee_arctic_leaf_reflectance_barrow_2013`
- `ngee_arctic_leaf_reflectance_kougarok_2016`
- `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016`
- `santa_barbara_urban_reflectance`
- `branch_tree_spectra_boreal_temperate`
- `hyspiri_ground_targets`

Inspect-first / possible targeted repair if needed:

- `usgs_v7`
- `emit_l2a_surface`
- `emit_adjusted_vegetation`
- `ngee_arctic_2018`
- `sispec`
- `hsdos`

## Source-by-Source Findings

| Source | Count | Assessment | Recommended action |
|---|---:|---|---|
| `ossl` | 51 | The flagged set is dominated by plausible soil-shape variation, not parser failure. The suspicious full-spectrum sheet shows mostly normal soil curves with varying depths near the water bands and longwave tail. | Keep. No automatic repair recommended. |
| `usgs_v7` | 29 | Mixed set. Many curves are plausible rare materials, but a subset still shows visible spikes, abrupt local steps, or out-of-range behavior. The visible-panel review shows the clearest residual artifacts in some mineral and synthetic-material cases. | Inspect by material family before any further repair. If another pass is done, make it spectrum-level and local-window only. |
| `bssl` | 28 | These look like valid soil spectra with some rough longwave behavior. After removing the old longwave-only subset, the remaining suspicious curves are mostly strong but plausible soil shapes. | Keep. No automatic repair recommended. |
| `emit_l2a_surface` | 10 | Most flagged spectra still look physically plausible, but a few retain small residual structure in the `1800-1950 nm` region and minor shoulder roughness. | Optional small targeted repair if these matter for downstream use; otherwise keep. |
| `emit_adjusted_surface` | 8 | The remaining suspicious curves are unlabeled surface spectra with unusual but coherent shapes. They do not look like parser corruption. | Keep. Review-only. |
| `antarctic_vegetation_speclib` | 8 | The flagged spectra are dark vegetation / lichen / moss-like signatures with strong but plausible structure. The detector is likely penalizing them for being atypical vegetation rather than broken. | Keep. Review-only. |
| `ecostress_v1` | 8 | Mixed group. Some mineral/chemical spectra are high and abrupt, but still plausible for the source. A few vegetation-like cases remain flagged by residual shape rather than obvious corruption. | Keep. Review-only. |
| `neon_field_spectra` | 7 | The flagged vegetation curves are plausible leaf/canopy shapes with stronger-than-average visible or water-band structure. | Keep. Review-only. |
| `emit_adjusted_vegetation` | 6 | Mostly plausible vegetation curves with minor remaining structure near the second water-absorption region. Not obviously broken. | Keep. Optional targeted repair only if vegetation means still show artifacts. |
| `montesinho_plants` | 6 | The flagged curves still look like plausible plant spectra. The detector is reacting more to shape deviation than artifact patterns. | Keep. Review-only. |
| `cabo_leaf_v2` | 5 | These look like valid leaf spectra. The detector is likely oversensitive to the source's high smoothness and strong red edge. | Keep. Review-only. |
| `santa_barbara_urban_reflectance` | 4 | The broad Santa problems were already addressed and the 37 worst spectra were excluded. The 4 remaining flagged cases look much more plausible than the excluded set. | Keep. No new exclusion by default. |
| `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016` | 3 | These spectra look coherent and vegetation-like. The detector is likely reacting to atypical arctic leaf shapes rather than corruption. | Keep. Review-only. |
| `ngee_arctic_leaf_reflectance_kougarok_2016` | 3 | Similar to the previous source: unusual but plausible arctic vegetation curves. | Keep. Review-only. |
| `ngee_arctic_2018` | 2 | Small residual longwave / water-band oddities remain. These are mild and localized. | Keep. Optional tiny targeted repair if needed. |
| `drylands_emit` | 2 | The flagged spectra look like plausible dry vegetation cases, not parser failures. | Keep. Review-only. |
| `branch_tree_spectra_boreal_temperate` | 2 | After tail repair, the remaining flagged spectra look acceptable. | Keep. Review-only. |
| `ngee_arctic_leaf_reflectance_barrow_2013` | 2 | Plausible vegetation curves with stronger detector residuals than the average source member. | Keep. Review-only. |
| `hsdos` | 1 | Single flagged soil spectrum. This is a good candidate for a one-off manual check because the count is so small. | Inspect one spectrum manually before deciding on any action. |
| `sispec` | 1 | Single suspicious water/snow spectrum with a localized absorption-region irregularity. | Optional one-off local-window repair if it matters; otherwise keep. |
| `hyspiri_ground_targets` | 1 | Single flagged soil/playa spectrum. The residual issue looks minor relative to earlier source problems. | Keep. Review-only. |

## Overall Conclusion

The detector still flags `187` spectra, but the majority of the remaining set no
longer looks like systematic parser or scaling failure.

Practical interpretation:

- clear source-wide problems have already been addressed
- the largest remaining flagged sets (`ossl`, `bssl`) are mostly acceptable
- the main residual repair candidate is still a small subset of `usgs_v7`
- `emit_l2a_surface`, `emit_adjusted_vegetation`, `ngee_arctic_2018`, `sispec`,
  and `hsdos` are optional fine-tuning targets, not urgent cleanup blockers

If another cleanup round is done, it should be:

1. `usgs_v7` spectrum-level and local-window only
2. then optional single-spectrum fixes for `hsdos` and `sispec`
3. leave `ossl`, `bssl`, `cabo_leaf_v2`, `ecostress_v1`, and the arctic leaf sources untouched

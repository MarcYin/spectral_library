# Remaining Suspicious Spectra Repair Plan

This plan is based on the final integrated build:

- Normalized root: `build/real_siac_pipeline_e2e_integrated/11_source_artifacts_fixed`
- SIAC package: `build/siac_spectral_library_e2e_integrated_no_ghisacasia`
- Full review summary: `build/siac_spectral_library_e2e_integrated_no_ghisacasia/full_review/review_summary.json`
- Per-source issue table: `build/siac_spectral_library_e2e_integrated_no_ghisacasia/full_review/remaining_repair_plan_by_source.csv`

Current state:

- Total exported spectra: `77,627`
- Remaining suspicious spectra: `350`
- Suspicious fraction: `0.45%`

The remaining set is small enough that repairs should now be source-specific. Broad global smoothing is no longer justified.

## Repair Priority

### P1: Repair now

These sources still show clear artifact-like behavior and should be repaired in code.

| Source | Flagged | Main issue pattern | Proposed fix |
| --- | ---: | --- | --- |
| `understory_estonia_czech` | 92 | residual-heavy, some out-of-range values, remaining longwave instability | Add a final source-specific cleanup pass for negative/over-1 bands and local tail instability after the current water-band repair. Clamp only repaired local segments, not whole spectra. |
| `bssl` | 47 | mostly `>2300 nm` tail jump/drift, some residual-only cases | Add a BSSL tail repair for the flagged subset only, using a left-shoulder fit and blended transition in the longwave tail. Keep the previously removed `600-3000 nm` subset excluded. |
| `branch_tree_spectra_boreal_temperate` | 29 | almost entirely tail drift/jump after `2300 nm` | Add a source-specific tail repair for flagged spectra only. This is the cleanest next repair because the visible region looks stable. |
| `usgs_v7` | 24 | strong visible jumps in a small number of spectra, some tail drift | Add local segment interpolation for flagged visible jumps. Limit the fix to the flagged windows instead of any library-wide smoothing. |
| `hyspiri_ground_targets` | 18 | tail jump/drift and a few out-of-range values, especially Spectralon-like targets | Split HyspIRI flagged spectra into calibration/bright reference targets vs ground materials. Repair or drop only the calibration/reference-like spectra from exported SIAC spectra if they are not physically representative. |

### P2: Probably repairable, but inspect first

These sources still have a small number of suspicious spectra, but the evidence is weaker or mixed.

| Source | Flagged | Main issue pattern | Proposed fix |
| --- | ---: | --- | --- |
| `emit_l2a_surface` | 13 | residual-heavy with a few visible jumps | Review flagged spectra by surface subgroup. Only repair if the issue is repeatable within one EMIT subgroup; otherwise leave as unusual-but-valid. |
| `emit_adjusted_vegetation` | 8 | residual-heavy, a few visible jumps | Same approach as `emit_l2a_surface`. Do not apply a new global EMIT correction without subgroup confirmation. |
| `emit_adjusted_surface` | 8 | residual-heavy with some tail drift | Inspect whether the tail behavior comes from a specific adjusted surface subset. Repair only if it repeats within that subset. |
| `antarctic_vegetation_speclib` | 8 | mixed residual and a few visible/tail issues | Likely heterogeneous biological/lichen signals. Check the flagged members individually before modifying. |
| `ecostress_v1` | 8 | mixed residual, one out-of-range/tail case | Fix individual bad spectra only if they are true parse/repair failures. Avoid source-wide changes. |

### P3: Review-only unless a repeatable artifact is confirmed

These are mainly residual-only or threshold-edge cases and should not be repaired automatically yet.

| Source | Flagged | Why not auto-fix yet |
| --- | ---: | --- |
| `ossl` | 51 | Mostly residual-heavy and flagged-band-heavy, but not dominated by jumps, spikes, or out-of-range values. These are likely unusual but valid soils mixed with threshold-edge cases. |
| `neon_field_spectra` | 7 | Residual-only cases. No clear localized artifact family. |
| `natural_snow_twigs` | 6 | Very small source, likely intrinsically unusual spectra. |
| `montesinho_plants` | 6 | Mixed residual-only cases, not one consistent artifact. |
| `cabo_leaf_v2` | 5 | Low count and mostly residual-only after thresholds were made source-relative. |

## Concrete Next Pass

1. `understory_estonia_czech`
   Add a final post-repair validator:
   - detect any remaining values `< -0.01` or `> 1.02`
   - detect any local tail jump `> 0.03` after `2300 nm`
   - repair only the flagged local segments with shoulder-based interpolation and blend windows

2. `bssl`
   Add a flagged-subset tail repair:
   - only for spectra with `tail_end_drift_2400 > 0.05` or `max_abs_jump_tail_2300 > 0.03`
   - interpolate from the stable left shoulder and blend across a `+-50 nm` window

3. `branch_tree_spectra_boreal_temperate`
   Add the same tail repair as `bssl`, but source-specific:
   - this is the cleanest high-yield remaining fix because `28/29` flagged spectra are tail-jump cases

4. `usgs_v7`
   Add a visible-jump detector:
   - repair only spectra with `max_abs_jump_visible > 0.03`
   - apply local linear interpolation across the jump neighborhood, then blend the repaired segment
   - do not touch normal USGS spectra

5. `hyspiri_ground_targets`
   Separate target types before repair:
   - bright calibration/reference-like samples should not be treated the same as soil-like ground targets
   - if the flagged subset is mostly calibration targets, exclude those from SIAC spectral export while keeping metadata

## What Should Not Be Done

- Do not reintroduce a global smoother over all flagged spectra.
- Do not use residual-only flags alone as a repair trigger.
- Do not treat all remaining suspicious spectra as broken; a large part of the remaining set is likely genuine spectral diversity.

## Recommended Success Criteria For The Next Repair Pass

- Reduce suspicious spectra from `350` to below `200`
- Keep suspicious fraction below `0.30%`
- Reduce `understory_estonia_czech`, `bssl`, `branch_tree_spectra_boreal_temperate`, `usgs_v7`, and `hyspiri_ground_targets` by at least `50%`
- Avoid increasing suspicious counts in `ossl`, `cabo_leaf_v2`, `neon_field_spectra`, or `antarctic_vegetation_speclib`

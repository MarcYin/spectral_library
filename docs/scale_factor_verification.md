# Scale Factor Verification

This log records only scale-factor decisions that were checked against original source files or official source metadata. It avoids inferred unit changes without evidence.

## Verified Fixes
| Source | Original source or metadata checked | Verified unit basis | Mixed units allowed by source metadata | Parser action |
|---|---|---|---|---|
| `ossl` | Local raw file [ossl_visnir_v1.rds](/Users/fengyin/Documents/spectral_library/build/local_sources/ossl/data/ossl_visnir_v1.rds) | Raw spectral columns are named `scan_visnir.*_pcnt`, so the stored values are percent-like. | No mixed unit evidence found in the raw file structure. | Force `value_scale_hint = 100` for `ossl`. |
| `ghisacasia_v001` | Official LP DAAC [User Guide](https://lpdaac.usgs.gov/documents/616/GHISACASIA_User_Guide_V1.pdf) and [ATBD](https://lpdaac.usgs.gov/documents/615/GHISACASIA_ATBD.pdf) | The official documentation describes ASD/Hyperion outputs as surface reflectance in percent, including “surface reflectance (%) after atmospheric correction”. | No mixed unit declaration was found in the official documentation checked here. | Force `value_scale_hint = 100` for `ghisacasia_v001`. The direct workbook download was still Earthdata-protected during this verification pass. |
| `hyspiri_ground_targets` | Official EcoSIS package metadata from [ecosis.org/package/uw-bnl-nasa-hyspiri-california-airborne-campaign-ground-target-spectra](https://ecosis.org/package/uw-bnl-nasa-hyspiri-california-airborne-campaign-ground-target-spectra), local raw [ivanpah_dry_lake_spectra_20130603.zip](/Users/fengyin/Documents/spectral_library/build/local_sources/hyspiri_ground_targets/ivanpah_dry_lake_spectra_20130603.zip), and local raw [uw-bnl_nasa_hyspiri_airborne_campaign_ground_cal_target_spectra_spectral_measurements.csv](/Users/fengyin/Documents/spectral_library/build/local_sources/hyspiri_ground_targets/uw-bnl_nasa_hyspiri_airborne_campaign_ground_cal_target_spectra_spectral_measurements.csv) | Package metadata explicitly lists measurement units `ratio`, `percent`, `%`. The Ivanpah member files contain `Norm. DN` calibration columns plus a `Reflect. %` column. The package CSV contains both ratio-like rows and percent-like rows. | Yes. Mixed `1` and `100` scales are supported by the source metadata. | For Ivanpah ZIP files, ingest only the `Reflect. %` column and drop `Norm. DN` columns. For the package CSV, use source-specific row-wise detection that only resolves to `1` or `100`. |
| `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016` | Local raw [NGEE-Arctic_Barrow_2015_HR1024i_Leaf_Spectral_Transmittance.xlsx](/Users/fengyin/Documents/spectral_library/build/local_sources_vegetation_all/ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016/NGEE-Arctic_Barrow_2015_HR1024i_Leaf_Spectral_Transmittance.xlsx), local raw [NGEE-Arctic_Barrow_2015_HR1024i_Leaf_Spectral_Transmittance.csv](/Users/fengyin/Documents/spectral_library/build/local_sources_vegetation_all/ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016/NGEE-Arctic_Barrow_2015_HR1024i_Leaf_Spectral_Transmittance.csv), and local raw ZIP members | The local files explicitly include `Spectra_Units = Percent_Reflectance_0_100`. | No mixed unit declaration was found in the checked files for this source. | Force `value_scale_hint = 100` for `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016`. |
| `understory_estonia_czech` | Local raw workbook [DatasetOfTreeCanopyStructureUnderstoryReflectanceSpectraAndFractionalCoverInHemiborealAndTemperateForestAreasInEstoniaAndCzechRepublic_V2.xlsx](/Users/fengyin/Documents/spectral_library/build/local_sources_vegetation_all/understory_estonia_czech/DatasetOfTreeCanopyStructureUnderstoryReflectanceSpectraAndFractionalCoverInHemiborealAndTemperateForestAreasInEstoniaAndCzechRepublic_V2.xlsx) | The workbook values are already stored in `0-1` reflectance space. | No mixed unit declaration was found in the workbook. | Force `value_scale_hint = 1` for `understory_estonia_czech`. |

## Final Export Audit
These values are from the rebuilt SIAC export [siac_spectral_library_real_e2e_scalefixed_no_ghisacasia](/Users/fengyin/Documents/spectral_library/build/siac_spectral_library_real_e2e_scalefixed_no_ghisacasia), using [siac_spectra_metadata.csv](/Users/fengyin/Documents/spectral_library/build/siac_spectral_library_real_e2e_scalefixed_no_ghisacasia/tabular/siac_spectra_metadata.csv). Final normalized range is `400-2500 nm` at `1 nm` for all exported spectra.

| Source | Exported spectra | Scale factor(s) applied after fix | Original native wavelength range in export | Final wavelength range | Mixed scale in export | Varying native wavelength in export | Status |
|---|---:|---|---|---|---|---|---|
| `ossl` | 37874 | `100` | `350-408` to `2500` | `400-2500` | No | Yes | Fixed for scale. Native start still varies across spectra in the export and should be treated as a source-range characteristic, not a unit decision. |
| `hyspiri_ground_targets` | 84 | `1`, `100` | `350-2500` | `400-2500` | Yes | No | Mixed scale is source-backed and expected for this source. Unsupported `1000/10000` cases were removed by the parser fix. |
| `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016` | 1367 | `100` | `350-2500` | `400-2500` | No | No | Fixed. The previous `1000` cases were not supported by the source metadata. |
| `understory_estonia_czech` | 465 | `1` | `350-2500` | `400-2500` | No | No | Fixed. This source had a real parser scale bug before the source-specific override. |
| `ghisacasia_v001` | 0 | not exported | cached source metadata showed `351-2491` before exclusion | metadata only | not applicable | not applicable | Excluded from SIAC spectra export by source ID, but parser logic is now pinned to `100` based on official LP DAAC documentation. |

## Remaining Variability Flags In The Current Export
These are not scale-factor fixes. They are listed because the processed metadata still show native-range variation.

| Source | Native range seen in export | Evidence-backed interpretation |
|---|---|---|
| `ossl` | `350-408` to `2500` | Raw file confirms percent units. The native start variation remains in processed metadata; this document does not assert a cause beyond what is directly observed. |
| `antarctic_vegetation_speclib` | `350-351` to `2299-2500` | Native range varies in processed metadata. No scale change was applied. |
| `ecostress_v1` | `300-405` to `2499-15000` | Native range varies in processed metadata. No scale change was applied in this audit. |
| `emit_l2a_surface` | `350-400` to `2500` | Native range varies in processed metadata. No scale change was applied in this audit. |
| `sispec` | `350-352` to `2500` | Native range varies in processed metadata. No scale change was applied in this audit. |
| `usgs_v7` | `350-392` to `2498-2500` | Native range varies in processed metadata. No scale change was applied in this audit. |

Notes:

- This log is about unit scaling first. Other source-specific spectral repairs are tracked separately in the processing diagnostics.
- Only `hyspiri_ground_targets` remains mixed-scale in the export, and that mixed `1`/`100` state is explicitly supported by the source metadata and raw files.

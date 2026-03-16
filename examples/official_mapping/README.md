# Official Sensor Mapping Example

This directory contains a fully reproducible mapping example built from:

- official Terra MODIS relative spectral response tables from NASA MCST,
- official Sentinel-2A MSI spectral response tables from ESA Copernicus,
- official Landsat 8 and Landsat 9 band JSON files from the USGS Spectral Characteristics Viewer.
- the previously composed full SIAC library at `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`

Contents:

- `official_source_manifest.json`: upstream provenance for the selected band subsets
- `srfs/`: package-ready sensor JSON files for the selected bands
- `queries/`: held-out single-sample and batch query CSVs used in the docs
- `results/`: generated mapping outputs, truth tables, and pairwise error summaries

The public examples use four held-out targets from the full SIAC library:

- `blue_spruce_needles`
- `pale_brown_silty_loam`
- `tap_water`
- `asphalt_road`

Those targets are simulated to each source sensor. Single-sample examples
exclude the exact prepared `row_id`, and the batch input carries a per-sample
`exclude_row_id` column so each query removes only its own source row from the
77,125-row retrieval library.
All generated outputs use the public default `k = 10`.

Regenerate everything from the official upstream sources with:

```bash
python3 -m pip install ".[internal-build]"
python3 scripts/build_official_mapping_examples.py --siac-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37
```

The public write-up that explains the example commands and figures is in
[`docs/official_sensor_examples.md`](../../docs/official_sensor_examples.md).

# Official Sensor Mapping Example

This directory contains a fully reproducible mapping example built from:

- official Terra MODIS relative spectral response tables from NASA MCST,
- official Sentinel-2A MSI spectral response tables from ESA Copernicus,
- official Landsat 8 and Landsat 9 band JSON files from the USGS Spectral Characteristics Viewer.

Contents:

- `official_source_manifest.json`: upstream provenance for the selected band subsets
- `srfs/`: package-ready sensor JSON files for the selected bands
- `siac/`: a compact synthetic SIAC-style candidate library on the public 400-2500 nm grid
- `queries/`: held-out single-sample and batch query CSVs used in the docs
- `results/`: generated mapping outputs, truth tables, and pairwise error summaries

The public examples use four held-out targets from the synthetic catalogue:

- `dense_vegetation`
- `bright_soil`
- `turbid_water`
- `asphalt`

Those target spectra are simulated to each source sensor. Each example query
then excludes only the matching `sample_name` from candidate retrieval, so the
mapper cannot self-match while the rest of the catalogue remains available.

Regenerate everything from the official upstream sources with:

```bash
python3 -m pip install ".[internal-build]"
python3 scripts/build_official_mapping_examples.py
```

The public write-up that explains the example commands and figures is in
[`docs/official_sensor_examples.md`](../../docs/official_sensor_examples.md).

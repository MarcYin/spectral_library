# Official Example Bundle

The repository includes a reproducible official-source example bundle under
`examples/official_mapping`.

This bundle is the companion dataset for
[Official Sensor Examples](official_sensor_examples.md). It contains the exact
inputs, outputs, and provenance records used by the public examples.

## Contents

- `examples/official_mapping/official_source_manifest.json`
  upstream URLs, retrieval timestamps, and SHA-256 digests for the official
  MODIS, Sentinel-2A, Landsat 8, and Landsat 9 source assets
- `examples/official_mapping/srfs/`
  package-ready sensor JSON files for the selected semantic bands
- `examples/official_mapping/siac/`
  a compact SIAC-style hyperspectral library on the canonical `400-2500 nm`
  grid
- `examples/official_mapping/queries/`
  single-sample and batch query CSVs used in the docs
- `examples/official_mapping/results/`
  mapped outputs, truth tables, and pairwise metrics

## Key Files

- [Source manifest](../examples/official_mapping/official_source_manifest.json)
- [Pairwise metrics CSV](../examples/official_mapping/results/metrics/pairwise_band_metrics.csv)
- [MODIS to Sentinel-2A example output](../examples/official_mapping/results/selected/veg_soil_mix_modis_to_sentinel2a.csv)
- [Batch diagnostics example](../examples/official_mapping/results/selected/landsat8_to_sentinel2a_batch_diagnostics.json)

## Regeneration

Rebuild the entire example bundle from the official upstream sources with:

```bash
python3 -m pip install ".[internal-build]"
python3 scripts/build_official_mapping_examples.py
```

That script refreshes the reduced SRF JSON files, the synthetic SIAC-style
runtime fixture, the query/result artifacts, and the figures used in the docs.

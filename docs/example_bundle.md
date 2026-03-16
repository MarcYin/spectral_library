# Official Example Bundle

The repository ships a reproducible example bundle under
`examples/official_mapping`.

This bundle is the companion dataset for
[Official Sensor Examples](official_sensor_examples.md). It contains the exact
inputs, outputs, and provenance records used by the public cross-sensor
examples.

## What Is Included

| Path | Contents |
| --- | --- |
| `examples/official_mapping/official_source_manifest.json` | official-source provenance, retrieval timestamps, and SHA-256 digests |
| `examples/official_mapping/srfs/` | selected-band SRF JSON files |
| `examples/official_mapping/siac/` | compact SIAC-style candidate fixture used by the prepared runtime |
| `examples/official_mapping/queries/` | held-out single-sample and batch query CSVs |
| `examples/official_mapping/results/` | mapped outputs, truth tables, and pairwise metrics |

## Key Artifacts

- [Source manifest](../examples/official_mapping/official_source_manifest.json)
- [Pairwise metrics CSV](../examples/official_mapping/results/metrics/pairwise_band_metrics.csv)
- [Vegetation holdout MODIS to Sentinel-2A output](../examples/official_mapping/results/selected/dense_vegetation_modis_to_sentinel2a.csv)
- [Batch diagnostics example](../examples/official_mapping/results/selected/landsat8_to_sentinel2a_holdout_batch_diagnostics.json)

## Why This Bundle Exists

The bundle keeps the public examples reproducible:

- the SRF sources are official and recorded with hashes
- the input queries are committed
- the four public targets are explicit held-out library spectra recorded in the manifest
- the output tables and figures are generated from the same committed fixture

## Regenerate The Bundle

```bash
python3 -m pip install ".[internal-build]"
python3 scripts/build_official_mapping_examples.py
```

That script refreshes:

- the reduced SRF JSON files
- the synthetic SIAC-style runtime fixture
- the query/result artifacts
- the figures used in the docs

## Related Docs

- [Official Sensor Examples](official_sensor_examples.md)
- [Mathematical Foundations](theory.md)

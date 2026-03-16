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
| `examples/official_mapping/queries/` | held-out single-sample and batch query CSVs |
| `examples/official_mapping/results/` | mapped outputs, truth tables, pairwise metrics, estimator benchmarks, and neighbor-review exports |

The example does not vendor the retrieval library itself. It expects the
previously composed full SIAC root recorded in
`official_source_manifest.json`, currently
`build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`.

## Key Artifacts

- [Source manifest](../examples/official_mapping/official_source_manifest.json)
- [Pairwise metrics CSV](../examples/official_mapping/results/metrics/pairwise_band_metrics.csv)
- [Estimator comparison CSV](../examples/official_mapping/results/metrics/neighbor_estimator_holdout_comparison.csv)
- [Vegetation holdout MODIS to Sentinel-2A output](../examples/official_mapping/results/selected/blue_spruce_needles_modis_to_sentinel2a.csv)
- [Batch diagnostics example](../examples/official_mapping/results/selected/landsat8_to_sentinel2a_holdout_batch_diagnostics.json)
- [Batch neighbor review CSV](../examples/official_mapping/results/selected/landsat8_to_sentinel2a_holdout_neighbor_review.csv)

## Why This Bundle Exists

The bundle keeps the public examples reproducible:

- the SRF sources are official and recorded with hashes
- the input queries are committed
- the four public targets are explicit held-out library spectra recorded in the manifest
- the runtime uses the previously composed full SIAC library and each published example excludes only the matching query row
- the output tables and figures are generated from that same full-library design

## Regenerate The Bundle

```bash
python3 -m pip install ".[internal-build]"
python3 scripts/build_official_mapping_examples.py \
  --siac-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37
```

That script refreshes:

- the reduced SRF JSON files
- the query/result artifacts
- the figures used in the docs

## Related Docs

- [Official Sensor Examples](official_sensor_examples.md)
- [Mathematical Foundations](theory.md)

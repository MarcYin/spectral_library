# Documentation

`spectral-library` has two layers of documentation:

- the public mapping package and CLI you can install and use directly
- the retained internal build pipeline that creates SIAC-style source data

Start with these public pages:

- [Mapping Quickstart](mapping_quickstart.md)
  for install, prepare, validate, single-sample mapping, batch mapping, and a
  minimal Python example
- [Official Sensor Examples](official_sensor_examples.md)
  for reproducible MODIS, Sentinel-2A, Landsat 8, and Landsat 9 examples built
  from official RSR sources
- [CLI Reference](cli_reference.md)
  for the public commands, required flags, CSV layouts, and outputs
- [Python API Reference](python_api_reference.md)
  for the stable import surface, result objects, and public errors
- [Prepared Runtime Contract](prepared_runtime_contract.md)
  for the on-disk layout, manifest fields, checksums, and compatibility policy

Reference and release pages:

- [Release Notes 0.1.0](releases/0.1.0.md)
- [Release Process](../RELEASE.md)

Internal and design pages:

- [Internal Build Pipeline](internal_build_pipeline.md)
- [Spectral Mapping Usage Plan](spectral_mapping_usage_plan.md)
- [Production Release Standard Plan](production_release_standard_plan.md)

Example data bundled in the repository:

- [Official mapping example bundle](../examples/official_mapping/README.md)
  with official-source SRF JSON, a compact SIAC-style example library, example
  queries, generated outputs, and plot assets

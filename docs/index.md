# spectral-library

<div class="hero" markdown>
<p class="hero__eyebrow">Prepared Spectral Mapping</p>
<h1 class="hero__title">Map between sensors. Reconstruct spectra. Keep the runtime stable.</h1>
<p class="hero__lead">
`spectral-library` is a Python package and CLI for preparing row-aligned mapping
runtimes from SIAC-style hyperspectral exports, retrieving nearest spectral
neighbors, simulating target-sensor reflectance, and reconstructing VNIR, SWIR,
or full `400-2500 nm` spectra.
</p>
[Get Started](mapping_quickstart.md){ .md-button .md-button--primary }
[See Official Examples](official_sensor_examples.md){ .md-button }
[Read The Theory](theory.md){ .md-button }
</div>

## What This Site Covers

<p class="section-intro">
The public documentation is organized around one practical flow:
prepare a runtime, validate it, map reflectance, and understand the retrieval
model well enough to trust the outputs.
</p>

<div class="grid cards" markdown>

- __1. Prepare__

  Build a compact prepared runtime from a SIAC-style spectral library and one
  or more sensor SRF definitions.

  [Prepared runtime workflow](mapping_quickstart.md#step-2-get-a-prepared-runtime)

- __2. Map__

  Run single-sample or batch mapping to a target sensor, or reconstruct VNIR,
  SWIR, or the full hyperspectral spectrum.

  [CLI mapping examples](mapping_quickstart.md#step-4-map-a-single-sample)

- __3. Understand__

  Follow the exact forward model, retrieval distance, overlap blending, and
  benchmark definitions used by the implementation.

  [Mathematical foundations](theory.md)

</div>

## Start Here

<div class="page-links">
  <a href="mapping_quickstart.html"><strong>Getting Started</strong>Install the package, prepare a runtime, validate it, and run your first mappings.</a>
  <a href="official_sensor_examples.html"><strong>Official Sensor Examples</strong>MODIS, Sentinel-2A, Landsat 8, and Landsat 9 examples built from official response functions.</a>
  <a href="theory.html"><strong>Mathematical Foundations</strong>The equations and assumptions behind retrieval, target simulation, and overlap blending.</a>
  <a href="prepared_runtime_contract.html"><strong>Prepared Runtime Contract</strong>The stable on-disk standard used by the mapper.</a>
</div>

## Public Documentation Map

### Guides

- [Getting Started](mapping_quickstart.md)
  for installation, runtime preparation, validation, single-sample mapping,
  batch mapping, and minimal Python usage
- [Official Sensor Examples](official_sensor_examples.md)
  for reproducible examples and figures built from official MODIS,
  Sentinel-2A, Landsat 8, and Landsat 9 spectral response functions
- [Troubleshooting](troubleshooting.md)
  for common issues, confidence score interpretation, and KNN backend
  selection

### Concepts

- [Mathematical Foundations](theory.md)
  for the sensor forward model, segment-wise nearest-neighbor retrieval,
  target-sensor simulation, and benchmark metrics
- [Prepared Runtime Contract](prepared_runtime_contract.md)
  for the stable files, schema rules, checksum behavior, and compatibility
  policy
- [Official Example Bundle](example_bundle.md)
  for the bundled example data and provenance artifacts

### Reference

- [CLI Reference](cli_reference.md)
  for the public commands, required flags, CSV layouts, and outputs
- [Python API Reference](python_api_reference.md)
  for the stable imports, result objects, and public errors
- [FAQ](faq.md)
  for answers to common questions

### Project

- [Security and Provenance](security_provenance.md)
- [Release Process](release_process.md)
- [Migration Guide](migration.md)
- [Release Notes 0.2.0](releases/0.2.0.md)
- [Release Notes 0.1.0](releases/0.1.0.md)

## Package Facts

<div class="fact-grid">
  <div><strong>Distribution</strong><span>`spectral-library`</span></div>
  <div><strong>Import</strong><span>`spectral_library`</span></div>
  <div><strong>CLI</strong><span>`spectral-library`</span></div>
  <div><strong>Python</strong><span>`3.9` to `3.14`</span></div>
</div>

!!! info "Public vs internal material"
    This site includes a small internal section for maintainers. If you are
    using the package, you can ignore the internal build/design pages and stay
    within the Start Here, Concepts, and Reference sections.

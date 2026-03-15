# Internal Documentation

These pages document the retained build pipeline, design notes, and release
planning material that still lives in this repository.

They are useful for maintainers, but they are not part of the stable public
package contract.

!!! warning "Not part of the public API contract"
    The public contract is limited to the installable package, public CLI, and
    prepared-runtime standard documented in the main docs. The internal pages
    below can change more freely.

## Internal Pages

- [Internal Build Pipeline](internal_build_pipeline.md)
  describes the retained SIAC-oriented source assembly workflow
- [Mapping Design](spectral_mapping_usage_plan.md)
  records the original retrieval-based mapping design and benchmark goals
- [Release Design](production_release_standard_plan.md)
  records the release-standardization plan that shaped the package surface

## Public Docs

If you are using the package rather than maintaining the repository, start with:

- [Getting Started](mapping_quickstart.md)
- [Official Sensor Examples](official_sensor_examples.md)
- [Mathematical Foundations](theory.md)
- [CLI Reference](cli_reference.md)
- [Python API Reference](python_api_reference.md)

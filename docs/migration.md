# Migration Guide

## Current prepared-runtime version

Prepared runtimes written by the current package use schema version `3.0.0`.
This package only loads prepared runtimes with schema major version `3`, so
older `1.2.0` and `2.0.0` runtimes must be rebuilt before use with the current
package.

## Upgrading from 0.2.x to 0.3.x

### Prepared runtimes must be rebuilt

The prepared-runtime schema version changed from `1.2.0` to `2.0.0`. Runtimes
built with `0.2.x` are **not** loadable by `0.3.x` and newer.

**Action:** Re-run `build-mapping-library` with the current package to rebuild
your runtimes.

```bash
spectral-library build-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

### Prepared sensor schemas now use full `rsrf` sensor definitions

Prepared `sensor_schema.json` files now store each sensor as an
`rsrf_sensor_definition` document. Band segment metadata lives in
`bands[].extensions.spectral_library.segment`, and band responses stay in
`response_definition`.

**Action:** If you distribute prepared runtimes, regenerate and republish them
with the current package so downstream users receive `schema_version: 3.0.0`
and `rsrf_sensor_definition`-based sensor schemas.

### Summary of changes

| Area | 0.2.x | 0.3.x and newer |
| --- | --- | --- |
| Schema version | `1.2.0` | `3.0.0` |
| Prepared sensor schema bands | top-level sampled arrays | `rsrf_sensor_definition` with `extensions.spectral_library.segment` |
| Runtime compatibility | `0.2.x` runtimes only | `3.x` prepared runtimes only |

### Checklist

- [ ] Install `spectral-library` 0.3.x or newer
- [ ] Rebuild all prepared runtimes with `build-mapping-library`
- [ ] Verify runtimes with `validate-prepared-library`
- [ ] Republish any distributed runtimes that were built with `0.2.x`

## Related Docs

- [Release Notes 0.2.0](releases/0.2.0.md)
- [Release Notes 0.1.0](releases/0.1.0.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)

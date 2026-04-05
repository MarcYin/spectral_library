# Migration Guide

## Current prepared-runtime version

Prepared runtimes written by the current package use schema version `2.0.0`.
This package only loads prepared runtimes with schema major version `2`, so
older `1.2.0` runtimes must be rebuilt before use with `0.3.x` and newer.

## Upgrading from 0.2.x to 0.3.x

### Prepared runtimes must be rebuilt

The prepared-runtime schema version changed from `1.2.0` to `2.0.0`. Runtimes
built with `0.2.x` are **not** loadable by `0.3.x` and newer.

**Action:** Re-run `prepare-mapping-library` with the current package to rebuild
your runtimes.

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

### Prepared sensor schemas now use `rsrf` response definitions

Prepared `sensor_schema.json` files now store each band as an `rsrf`-compatible
`response_definition`. The runtime loader hands those definitions back to
`rsrf`, rather than accepting legacy top-level sampled arrays such as
`wavelength_nm` and `rsr`.

**Action:** If you distribute prepared runtimes, regenerate and republish them
with the current package so downstream users receive `schema_version:
2.0.0` and `response_definition`-based sensor schemas.

### Summary of changes

| Area | 0.2.x | 0.3.x and newer |
| --- | --- | --- |
| Schema version | `1.2.0` | `2.0.0` |
| Prepared sensor schema bands | top-level sampled arrays | `rsrf` `response_definition` |
| Runtime compatibility | `0.2.x` runtimes only | `2.x` prepared runtimes only |

### Checklist

- [ ] Install `spectral-library` 0.3.x or newer
- [ ] Rebuild all prepared runtimes with `prepare-mapping-library`
- [ ] Verify runtimes with `validate-prepared-library`
- [ ] Republish any distributed runtimes that were built with `0.2.x`

## Related Docs

- [Release Notes 0.2.0](releases/0.2.0.md)
- [Release Notes 0.1.0](releases/0.1.0.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)

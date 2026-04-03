# Migration Guide

## Upgrading from 0.1.x to 0.2.x

### Prepared runtimes must be rebuilt

The prepared-runtime schema version changed from `1.0.0` to `1.2.0`. Runtimes
built with 0.1.x are **not** loadable by 0.2.x.

**Action:** Re-run `prepare-mapping-library` with the 0.2.x package to rebuild
your runtimes.

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

### New features (no breaking changes)

These are additive features. Existing code continues to work without
modification.

#### KNN backend selection

Mapping commands and the Python API now accept an optional `knn_backend`
parameter. The default remains `numpy`, so existing calls are unchanged.

```python
# Before (still works)
result = mapper.map_reflectance(source_sensor="s", reflectance={...}, output_mode="target_sensor")

# New option
result = mapper.map_reflectance(source_sensor="s", reflectance={...}, output_mode="target_sensor",
                                knn_backend="scipy_ckdtree", knn_eps=0.05)
```

#### Persisted ANN indexes

You can now persist ANN indexes at prepare time for faster first-query
performance:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --knn-index-backend faiss \
  --output-root build/mapping_runtime
```

#### Confidence diagnostics

Mapping results now include `confidence_score` and `confidence_components` in
the diagnostics payload. These are new additive fields.

#### Benchmark `max_test_rows`

`benchmark_mapping(...)` now accepts `max_test_rows` to cap the held-out test
split size.

### Summary of changes

| Area | 0.1.x | 0.2.x |
| --- | --- | --- |
| Schema version | `1.0.0` | `1.2.0` |
| KNN backends | `numpy` only | `numpy`, `scipy_ckdtree`, `faiss`, `pynndescent`, `scann` |
| Persisted indexes | Not available | `faiss`, `pynndescent`, `scann` |
| Confidence score | Not available | Heuristic score in diagnostics |
| Neighbor review | Not available | CSV export of shortlisted neighbors |

### Checklist

- [ ] Install `spectral-library` 0.2.x
- [ ] Rebuild all prepared runtimes with `prepare-mapping-library`
- [ ] Optionally install KNN backend extras (`.[knn]`, `.[knn-faiss]`, etc.)
- [ ] Optionally add `--knn-index-backend` to your prepare step
- [ ] Verify runtimes with `validate-prepared-library`

## Related Docs

- [Release Notes 0.2.0](releases/0.2.0.md)
- [Release Notes 0.1.0](releases/0.1.0.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)

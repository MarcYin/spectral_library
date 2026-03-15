# Spectral Mapping Usage Design

## Purpose

This document defines how the current SIAC spectral library should be used for
sensor-to-sensor reflectance mapping. It covers:

- the traditional regression-based spectral mapping baseline,
- a new retrieval-based mapping workflow built on nearest-neighbor search,
- direct partial-to-hyperspectral reconstruction from multispectral or other
  partial spectral inputs,
- the data products and runtime preparation needed to make mapping fast and
  memory efficient,
- proposed CLI and Python interfaces,
- validation and benchmark requirements.

This document records the mapping design and usage model that the current
implementation is following. It does not change the current build pipeline or
the existing build plan in `plan.md`.

## Current Library Constraints

The design is anchored to the current canonical SIAC package:

- package root:
  `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`
- spectral grid:
  `400-2500 nm` at `1 nm`
- total spectra:
  `77,125`
- spectra covering `400-1000 nm`:
  `77,056`
- spectra covering `1000-2500 nm`:
  `72,900`
- spectra covering full `400-2500 nm`:
  `72,868`
- normalized spectra Parquet:
  about `895 MB`
- SIAC DuckDB:
  about `1.6 GB`

These constraints imply the following:

- the archive is already large enough that repeated wide-table scans should be
  avoided in runtime mapping,
- the mapping workflow should prepare smaller row-aligned arrays for retrieval
  rather than repeatedly querying the full CSV or DuckDB export,
- the design should remain sensor-agnostic because no relative spectral
  response (RSR/SRF) assets are currently committed in this repository.

## Problem Definition

The library should support two related use cases:

1. sensor-to-sensor mapping
   Convert reflectance observed by one sensor into a physically informed
   estimate of reflectance for another sensor using the spectral library as an
   intermediate hyperspectral prior.

2. partial-to-hyperspectral reconstruction
   Use partial spectral input, such as multispectral VNIR, multispectral SWIR,
   or mixed band subsets, to reconstruct a hyperspectral estimate over:
   - VNIR only,
   - SWIR only,
   - full VNIR-SWIR range.

The same retrieval framework should support both use cases. The only difference
is what output representation is requested after the nearest-neighbor step.

### Notation

Let:

- `N` be the number of library spectra,
- `L` be the number of hyperspectral wavelengths on the canonical grid,
- `L = 2101` for wavelengths `400, 401, ..., 2500 nm`,
- `H ∈ R^{N × L}` be the library hyperspectral reflectance matrix,
- `H_i ∈ R^L` be the `i`th library spectrum,
- `r_{s,b} ∈ R^L` be the relative spectral response vector for sensor `s`,
  band `b`, sampled on the same `400-2500 nm` grid,
- `X^{(src)} ∈ R^{N × B_src}` be the library convolved to the source sensor,
- `Y^{(tgt)} ∈ R^{N × B_tgt}` be the library convolved to the target sensor,
- `q ∈ R^{B_src}` be an observed source-sensor reflectance vector for one
  input sample.

Each SRF vector is assumed to be zero outside the supported wavelength range of
that band after resampling to the canonical grid.

### Sensor Convolution

For any library spectrum `H_i` and sensor `s`, the simulated reflectance for
band `b` is:

$$
x_{i,b}^{(s)} =
\frac{\sum_{\ell=1}^{L} H_{i,\ell} r_{s,b,\ell}}
     {\sum_{\ell=1}^{L} r_{s,b,\ell}}
$$

This equation is the common forward model used by both the traditional baseline
and the new retrieval-based workflow.

## Traditional Baseline

The reference baseline is the traditional spectral mapping workflow:

1. convolve every hyperspectral library spectrum to the source sensor,
2. convolve the same library spectra to the target sensor,
3. fit a regression from simulated source reflectance to simulated target
   reflectance,
4. apply the fitted regression to real source-sensor reflectance.

In matrix form:

$$
Y^{(tgt)} = \mathbf{1}\alpha^\top + X^{(src)} B + E
$$

where:

- `α` is the intercept vector for target bands,
- `B` is the regression coefficient matrix,
- `E` is the residual matrix.

This baseline remains important because it provides:

- a familiar reference implementation,
- a benchmark against the retrieval-based method,
- a fallback path for users who want a fixed conversion model rather than
  per-sample library lookup.

## Retrieval-Based Mapping Design

### Core Idea

Instead of fitting one global regression from many simulated source-target
pairs, the new method uses the observed source reflectance to retrieve the most
similar library spectra, then uses the retrieved hyperspectral neighborhood to
either:

- simulate target sensor reflectance, or
- reconstruct a hyperspectral spectrum directly.

The default v1 estimator is:

- retrieve top `k` nearest library spectra using source-sensor reflectance,
- use `k = 10` by default,
- average the matched hyperspectral spectra without distance weighting,
- convolve the averaged hyperspectral spectrum with the target sensor SRFs.

### Segment-Wise Processing

The workflow is split into two overlapping spectral segments:

- `VNIR = 400-1000 nm`
- `NIR-SWIR = 900-2500 nm`

The overlap over `900-1000 nm` exists to preserve information near the
silicon/SWIR transition and to avoid unstable boundary behavior during
retrieval and target simulation.

The overlap is not a duplicate-band merge step. Final output is formed as the
ordered union of:

- target bands assigned to `vnir`,
- target bands assigned to `swir`.

Each sensor band must therefore be assigned exactly one segment in sensor
metadata. Bands are not emitted twice.

### Segment Logic

For each segment `p ∈ {vnir, swir}`:

1. build the source-sensor query vector using only source bands assigned to
   that segment,
2. search for nearest library spectra using only the same segment-specific
   source-sensor simulation matrix,
3. average the matched hyperspectral library spectra,
4. convolve that mean hyperspectral spectrum to target bands assigned to the
   same segment.

This gives segment-specific retrieval neighborhoods:

- VNIR input only queries VNIR-like library behavior,
- SWIR input only queries SWIR-like library behavior.

That separation is deliberate. It prevents weak or missing source support in
one spectral region from influencing retrieval in the other region.

### Segment Outputs

Each segment retrieval produces a segment-specific hyperspectral estimate:

- `\bar{h}^{(vnir)}` over `400-1000 nm`,
- `\bar{h}^{(swir)}` over `900-2500 nm`.

Those segment outputs can then be used in more than one way:

- convolve to a target sensor,
- return the reconstructed VNIR spectrum directly,
- return the reconstructed SWIR spectrum directly,
- blend the overlap and return a full reconstructed `400-2500 nm` spectrum.

### Distance Metric

The default retrieval metric is weighted Euclidean distance over valid
source-sensor bands in the current segment:

$$
d_i^{(p)} =
\sqrt{
\frac{
\sum_{b \in B_p^{valid}} w_b \left(q_b - X_{i,b}^{(src,p)}\right)^2
}{
\sum_{b \in B_p^{valid}} w_b
}
}
$$

where:

- `B_p^{valid}` is the set of valid query bands in segment `p`,
- `w_b` is the per-band weight,
- `X^{(src,p)}` is the segment-specific source simulation matrix.

Default weighting:

- `w_b = 1` for all valid bands unless a later benchmark justifies another
  scheme,
- invalid, missing, or masked source bands are excluded from both the numerator
  and denominator.

### Neighbor Estimator

Let `N_k^{(p)}` be the top-`k` nearest neighbors in segment `p`. The default
retrieved hyperspectral estimate is the unweighted mean:

$$
\bar{h}^{(p)} = \frac{1}{k}\sum_{i \in N_k^{(p)}} H_i
$$

This is intentionally simple for v1:

- it matches the method requested here,
- it avoids another tuning parameter,
- it gives a stable default for benchmarking.

Distance-weighted means can be evaluated later, but they are not the v1 default.

### Target-Sensor Simulation From Retrieved Spectra

For each target band `b` assigned to segment `p`, the mapped target reflectance
is:

$$
\hat{y}_b^{(tgt,p)} =
\frac{\sum_{\ell=1}^{L} \bar{h}_{\ell}^{(p)} r_{tgt,b,\ell}}
     {\sum_{\ell=1}^{L} r_{tgt,b,\ell}}
$$

The final target output vector is the ordered concatenation of:

- mapped target bands from the `vnir` segment,
- mapped target bands from the `swir` segment.

## Direct Hyperspectral Reconstruction

### Output Modes

The retrieval workflow should support the following output modes:

- `target_sensor`
  Return target-sensor reflectance after convolving the retrieved
  hyperspectral estimate.
- `vnir_spectrum`
  Return reconstructed hyperspectral VNIR reflectance over `400-1000 nm`.
- `swir_spectrum`
  Return reconstructed hyperspectral SWIR reflectance over `900-2500 nm`.
- `full_spectrum`
  Return reconstructed hyperspectral reflectance over `400-2500 nm`.

This makes the same prepared library useful for both conventional spectral
mapping and direct spectral reconstruction from partial input.

### Full-Spectrum Reconstruction

When full-range output is requested, the VNIR and SWIR segment estimates should
be merged into one hyperspectral spectrum:

- use the VNIR estimate directly for `400-899 nm`,
- use the SWIR estimate directly for `1001-2500 nm`,
- use a weighted average in the overlap `900-1000 nm`.

Let `λ` be wavelength in nanometers. Define a blending weight
`ω(λ)` over the overlap:

$$
\omega(\lambda) = \frac{1000 - \lambda}{100}
$$

for `900 \le \lambda \le 1000`.

Then the reconstructed full spectrum `\hat{h}_\lambda` is:

$$
\hat{h}_\lambda =
\begin{cases}
\bar{h}^{(vnir)}_\lambda, & 400 \le \lambda < 900 \\
\omega(\lambda)\bar{h}^{(vnir)}_\lambda +
\left(1 - \omega(\lambda)\right)\bar{h}^{(swir)}_\lambda, &
900 \le \lambda \le 1000 \\
\bar{h}^{(swir)}_\lambda, & 1000 < \lambda \le 2500
\end{cases}
$$

This weighted overlap merge gives a continuous full-range output while still
allowing each segment retrieval to be driven only by its own source support.

### Partial Input To Full Output

The design should allow partial spectral input to reconstruct more complete
spectral output. Examples:

- multispectral VNIR input reconstructing VNIR hyperspectral output,
- multispectral SWIR input reconstructing SWIR hyperspectral output,
- source-sensor input reconstructing full `400-2500 nm` hyperspectral output,
- source-sensor input reconstructing target-sensor reflectance after an
  intermediate hyperspectral estimate.

The critical rule is that output mode must be explicit. The same retrieval step
can feed different output products depending on user intent.

## Runtime Data Preparation

### Why A Prepared Layer Is Needed

The current archive is optimized for completeness and inspection. It is not yet
optimized for low-latency mapping. Runtime mapping should avoid repeatedly:

- reading a `895 MB` wide Parquet file,
- scanning a `1.6 GB` DuckDB table for every query,
- re-convolving the full library for the same source sensor over and over.

### Prepared Runtime Layer

Create a prepared mapping layer on top of the canonical SIAC export with the
following row-aligned assets:

1. a compact metadata table
2. overlapping hyperspectral arrays in `float32`
3. precomputed source-sensor convolution matrices
4. target-sensor convolution metadata
5. run metadata and provenance
6. output-mode metadata

Recommended prepared assets:

| Asset | Purpose |
| --- | --- |
| `mapping_metadata.parquet` | Row-aligned spectrum identifiers, source ids, landcover labels, native range, and coverage metrics. |
| `hyperspectral_vnir.f32` or `.npy` | `N × 601` array for `400-1000 nm`. |
| `hyperspectral_swir.f32` or `.npy` | `N × 1601` array for `900-2500 nm`. |
| `source_<sensor_id>_vnir.f32` or `.npy` | `N × B_src_vnir` convolved source matrix for VNIR retrieval. |
| `source_<sensor_id>_swir.f32` or `.npy` | `N × B_src_swir` convolved source matrix for SWIR retrieval. |
| `sensor_schema.json` | Sensor and band definitions, segment assignments, wavelength support, and file references. |
| `build_info.json` | SIAC package path, sensor ids, versioning, preparation parameters, and supported output modes. |

The key property is row alignment: row `i` must refer to the same spectrum in
every prepared asset.

### Storage Recommendations

Preferred v1 storage choices:

- metadata in Parquet,
- dense numeric arrays in `float32`,
- array files memory-mappable with NumPy,
- preparation scripts in Python first.

Rationale:

- `float32` is sufficient for reflectance mapping and cuts memory roughly in
  half relative to `float64`,
- memory-mapped arrays avoid loading the full library into RAM when not needed,
- row-major dense arrays are better for KNN retrieval than the current wide CSV
  format.

DuckDB and Parquet remain the canonical distribution formats. The prepared layer
is a runtime acceleration layer derived from them.

## Sensor SRF Input Schema

The mapping workflow needs a sensor SRF representation that is explicit about
segment assignment and easy to resample onto the canonical wavelength grid.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | string | Stable identifier for the sensor. |
| `band_id` | string | Stable identifier for the band within the sensor. |
| `segment` | string | Either `vnir` or `swir`. |
| `wavelength_nm` | array of float | Wavelength axis for the band SRF before resampling. |
| `rsr` | array of float | Relative spectral response values aligned to `wavelength_nm`. |

Optional summary fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `center_nm` | float | Band center wavelength if known. |
| `fwhm_nm` | float | Full width at half maximum if known. |
| `support_min_nm` | float | Minimum wavelength with non-zero or effective support. |
| `support_max_nm` | float | Maximum wavelength with non-zero or effective support. |

Rules:

- every band must have exactly one segment assignment,
- the SRF must be resampled to the canonical `400-2500 nm` grid before
  convolution,
- RSR normalization is handled by the convolution denominator, so raw RSR input
  does not need to sum to one.

## Implementation Approach

### Preparation Phase

The preparation phase should:

1. load the SIAC package metadata and normalized spectra,
2. build row-aligned arrays for the VNIR and SWIR hyperspectral windows,
3. ingest one or more source-sensor SRF definitions,
4. precompute the convolved source-sensor retrieval matrices by segment,
5. write prepared assets and provenance metadata.

This phase is where most of the expensive work happens. It should be done once
per SIAC package and source sensor configuration, not once per mapping query.

### Mapping Phase

For one input sample:

1. validate and mask source-sensor reflectance inputs,
2. split the input into segment-specific query vectors,
3. compute distances against the prepared source retrieval matrices,
4. retrieve top `k` library rows per segment,
5. average the matched hyperspectral rows per segment,
6. either convolve the retrieved means to target bands or assemble a
   hyperspectral output product,
7. if full-range hyperspectral output is requested, merge the overlap with the
   defined weighted average,
8. combine the segment outputs into the requested final representation,
9. return mapped reflectance plus neighbor and diagnostic information.

### Search Backend

The default implementation should be Python-first and use exact KNN methods
first because the search space is:

- moderate in row count for v1 (`77,125` spectra),
- low in dimensionality after source-sensor convolution,
- easier to validate numerically with exact search before introducing
  approximate methods.

Recommended backend progression:

1. exact vectorized distance computation in NumPy for the first correct
   implementation,
2. optional tree-based exact KNN if profiling shows benefit,
3. optional approximate nearest-neighbor backends only if profiling shows exact
   methods are insufficient.

Approximate search is therefore allowed, but not required by design. It should
only be introduced after benchmark evidence, not pre-emptively.

### Rust Positioning

Rust is not a v1 requirement. It is reserved as a later optimization path for:

- convolution kernels,
- masked weighted distance kernels,
- memory-efficient batch retrieval routines.

Any Rust acceleration should preserve the same prepared file layout and
high-level API so that the Python-first implementation remains the reference
behavior.

## Proposed Interfaces

### CLI

#### `spectral-library prepare-mapping-library`

Purpose:

- build the prepared runtime layer for one SIAC package and one or more source
  sensors.

Expected responsibilities:

- validate SIAC inputs,
- ingest SRF definitions,
- write row-aligned hyperspectral arrays,
- write source-sensor retrieval matrices,
- write provenance metadata.

Representative arguments:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37 \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

#### `spectral-library map-reflectance`

Purpose:

- map source-sensor reflectance to either target-sensor reflectance or
  reconstructed spectral output using the prepared library and retrieval
  workflow.

Expected responsibilities:

- load prepared assets,
- validate query reflectance,
- run segment-wise retrieval,
- emit mapped target reflectance, reconstructed spectral output, and
  diagnostics.

Representative arguments:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/source_reflectance.csv \
  --output-mode target_sensor \
  --k 10 \
  --output path/to/mapped_reflectance.csv
```

Representative full-spectrum reconstruction call:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --input path/to/source_reflectance.csv \
  --output-mode full_spectrum \
  --k 10 \
  --output path/to/reconstructed_spectrum.csv
```

#### `spectral-library benchmark-mapping`

Purpose:

- benchmark regression and retrieval-based mapping on simulated source-target
  pairs generated from the same spectral library.

Expected responsibilities:

- build train/test splits from library spectra,
- compare regression baseline against retrieval mapping,
- report per-band RMSE, MAE, and bias.

Representative arguments:

```bash
spectral-library benchmark-mapping \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --report path/to/benchmark.json
```

### Python API

The Python API should mirror the CLI responsibilities.

#### Preparation Entry Point

```python
prepare_mapping_library(
    siac_root: Path,
    srf_root: Path,
    output_root: Path,
    source_sensors: list[str],
    *,
    dtype: str = "float32",
)
```

Responsibilities:

- materialize the prepared runtime layer,
- validate row alignment and sensor schema,
- persist provenance and preparation settings.

#### Mapper Object

```python
mapper = SpectralMapper(prepared_root=Path("build/mapping_runtime"))
result = mapper.map_reflectance(
    source_sensor="SENSOR_A",
    reflectance=query_vector,
    valid_mask=query_mask,
    output_mode="target_sensor",
    target_sensor="SENSOR_B",
    k=10,
)
```

Responsibilities:

- lazy-load prepared arrays,
- perform segment-specific retrieval and target simulation,
- support direct reconstructed spectral outputs in addition to target-sensor
  outputs,
- expose deterministic behavior for the same inputs and settings.

#### Result Type

The result object should contain:

- `target_reflectance`
- `target_band_ids`
- `reconstructed_vnir`
- `reconstructed_swir`
- `reconstructed_full_spectrum`
- `reconstructed_wavelength_nm`
- `neighbor_ids_by_segment`
- `neighbor_distances_by_segment`
- `segment_outputs`
- `segment_valid_band_counts`
- `diagnostics`

Diagnostics should include enough information to explain how the result was
constructed, especially:

- how many valid source bands were used in each segment,
- whether a segment fell below the minimum-valid-band threshold,
- which library rows were retrieved.

Only the fields relevant to the selected output mode need to be populated.

## Missing Data And Validity Rules

The retrieval design must treat missing or invalid source reflectance
explicitly.

Rules:

- invalid or absent source bands are masked out of the distance calculation,
- a segment must satisfy a minimum-valid-band rule before retrieval is allowed,
- if a segment does not satisfy the rule, that segment should return a clear
  diagnostic rather than a silent estimate,
- target bands are only produced for segments that were successfully mapped.

This keeps the behavior transparent and prevents spurious nearest-neighbor
matches driven by a nearly empty query vector.

## Validation And Benchmark Plan

### Functional Tests

1. Convolution sanity
   A constant hyperspectral spectrum must produce the same constant band
   reflectance after convolution for every band.

2. Identity retrieval
   A query generated from an existing library spectrum must return that spectrum
   as the top-1 neighbor and reproduce the same reflectance when
   `source == target`.

3. Segment isolation
   Changing only VNIR inputs must not change SWIR outputs, and changing only
   SWIR inputs must not change VNIR outputs.

4. Overlap support
   Queries and target bands around `900-1000 nm` must remain supported without
   duplication or boundary dropout.

5. Full-spectrum blend behavior
   Full reconstructed output must use VNIR values below `900 nm`, SWIR values
   above `1000 nm`, and weighted overlap blending inside `900-1000 nm`.

6. Missing-band behavior
   Masked source bands must be excluded from the weighted Euclidean distance,
   and segments below the minimum-valid-band threshold must fail explicitly.

7. Storage consistency
   Convolution results computed from prepared arrays, DuckDB, and Parquet must
   agree within a defined numerical tolerance.

### Benchmark Scenarios

Benchmark the traditional regression baseline and the retrieval-based method on
simulated source-target pairs derived from the same library.

Required metrics:

- per-band RMSE
- per-band MAE
- per-band bias

Recommended benchmark slices:

- all spectra,
- by landcover group when labels exist,
- by VNIR bands and SWIR bands separately,
- by output mode: target sensor, VNIR spectrum, SWIR spectrum, and full
  spectrum,
- by different `k` values with `k = 10` as the default reference.

## Defaults And Assumptions

- this document is documentation only and does not change code or schema in the
  current build pipeline,
- the design is sensor-agnostic because no SRF datasets are currently committed
  in the repository,
- default segment windows are `400-1000 nm` and `900-2500 nm`,
- default retrieval metric is weighted Euclidean distance with uniform per-band
  weights over valid bands,
- default estimator is unweighted top-`k` mean with `k = 10`,
- default full-spectrum merge uses a linear weighted average over `900-1000 nm`,
- landcover labels are optional evaluation strata and not a mandatory retrieval
  filter,
- Python-first implementation is the default path,
- Rust is a later optimization option, not a v1 dependency.

## Recommended Next Step

After this design is accepted, the first implementation milestone should be a
minimal Python prototype that:

1. ingests one source sensor SRF definition and one target sensor SRF
   definition,
2. prepares row-aligned VNIR and SWIR runtime arrays,
3. runs exact top-`k` retrieval on source-sensor reflectance,
4. returns either target-sensor reflectance or reconstructed spectral output
   plus neighbor diagnostics,
5. supports full-spectrum reconstruction by weighted overlap blending,
6. benchmarks the retrieval result against the regression baseline on a held-out
   simulated test set.

# Mathematical Foundations

This page explains the model that `spectral-library` implements.

It answers four questions:

1. how sensor bands are simulated from hyperspectral library rows
2. how nearest neighbors are retrieved segment by segment
3. how target-sensor reflectance is produced from the retrieved spectra
4. how the regression benchmark is defined

Use this page together with:

- [Getting Started](mapping_quickstart.md) for the workflow
- [Python API Reference](python_api_reference.md) for the public objects
- [Prepared Runtime Contract](prepared_runtime_contract.md) for the runtime
  standard

## Model Summary

<div class="grid cards" markdown>

- __Forward model__

  Every sensor band is simulated by integrating hyperspectral reflectance
  against a normalized resampled SRF.

- __Retrieval model__

  Each segment uses nearest-neighbor search in source-sensor feature space,
  then averages the matched hyperspectral rows.

- __Output model__

  Retrieved hyperspectral segments are either convolved to a target sensor or
  blended into a full `400-2500 nm` spectrum.

</div>

## Problem Setup

The package starts from a row-aligned hyperspectral library sampled on the
canonical integer grid:

$$
\lambda \in \{400, 401, \ldots, 2500\}\ \text{nm}
$$

Let:

- $N$ be the number of library rows
- $L = 2101$ be the number of wavelengths on the canonical grid
- $H \in \mathbb{R}^{N \times L}$ be the hyperspectral reflectance matrix
- $H_i \in \mathbb{R}^{L}$ be the $i$th library spectrum
- $q$ be one observed source-sensor reflectance vector

The current runtime splits the spectrum into two overlapping segments:

- `VNIR = 400-1000 nm`
- `SWIR = 800-2500 nm`

The overlap is intentional. It preserves continuity near the detector
transition and lets the system reconstruct a stable full `400-2500 nm`
spectrum.

For retrieval, the SWIR query can also reuse the semantic `nir` source band
when that band exists in the sensor schema. This gives the SWIR neighbor search
one bridge feature near the VNIR-SWIR transition instead of relying only on
`swir1` and `swir2`.

## Sensor Forward Model

For each sensor band, the package uses the resampled relative spectral response
(SRF) as a discrete weighting function on the canonical wavelength grid.

For sensor $s$, band $b$, and library spectrum $H_i$, the simulated band
reflectance is:

$$
x_{i,b}^{(s)} =
\frac{\sum_{\ell=1}^{L} H_{i,\ell} r_{s,b,\ell}}
     {\sum_{\ell=1}^{L} r_{s,b,\ell}}
$$

where:

- $r_{s,b,\ell}$ is the resampled SRF weight for wavelength index $\ell$
- the denominator normalizes the band response so the output stays in
  reflectance units

This is the same forward model used in:

- `prepare_mapping_library(...)` when precomputing source-sensor matrices
- target-sensor simulation during mapping
- the regression baseline inside `benchmark_mapping(...)`

## Prepared Runtime Matrices

The prepared runtime stores three aligned data products:

1. the segment hyperspectral arrays
2. the source-sensor simulation matrices
3. the row metadata table that keeps every array on the same row index

For each source sensor $s$ and segment $p \in \{\text{vnir}, \text{swir}\}$,
the runtime contains a precomputed matrix:

$$
X^{(s,p)} \in \mathbb{R}^{N \times B_{s,p}}
$$

where $B_{s,p}$ is the number of source bands assigned to segment $p$.

This is the reason mapping stays fast at runtime: the package does not
re-convolve the entire hyperspectral library for every query.

## Segment-Wise Retrieval

Given one source query $q$, the package separates the input into segment-level
queries:

- $q^{(\text{vnir})}$
- $q^{(\text{swir})}$

Each segment is retrieved independently. Missing VNIR support does not distort
the SWIR neighborhood, and vice versa.

This remains true even when the source query provides a complete visible-to-SWIR
band vector. The runtime still performs two independent nearest-neighbor
searches, one for the VNIR query and one for the NIR-SWIR query, and only
combines the reconstructed segments after retrieval.

The SWIR retrieval query is slightly augmented: if the source sensor defines a
semantic `nir` band, that band is appended to the SWIR query feature set so the
retrieval sees one overlap feature across the detector boundary.

If $B_p^{\text{valid}}$ is the valid source-band set for segment $p$, the
implementation uses a root-mean-square Euclidean distance:

$$
d_i^{(p)} =
\sqrt{
\frac{1}{|B_p^{\text{valid}}|}
\sum_{b \in B_p^{\text{valid}}}
\left(q_b - X_{i,b}^{(src,p)}\right)^2
}
$$

Important implementation details:

- only valid source bands are scored
- the SWIR feature set includes `nir` when available, then the SWIR bands
- all valid bands currently use uniform weight
- top-$k$ neighbors are ordered deterministically by distance and row index

The package also exposes multiple search backends for this same metric:

- `numpy`: direct dense distance evaluation
- `scipy_ckdtree`: SciPy `cKDTree` search, with `knn_eps > 0` allowing an
  approximate shortlist
- `faiss`: FAISS HNSW search
- `pynndescent`: PyNNDescent approximate search
- `scann`: ScaNN approximate search

The backend changes how the shortlist is found, not the metric itself. The
returned neighbors are still re-ordered by exact RMS distance before the
estimator runs.

For `faiss`, `pynndescent`, and `scann`, the prepare step can also persist
full-feature ANN indexes for each prepared source sensor and segment. Those
saved indexes are only used when a query uses the full segment feature set and
the full candidate pool. Queries with missing bands or per-query exclusions
fall back to on-the-fly search in the reduced feature space.

Pre-sorting the full library rows by global spectral similarity usually does
not improve this retrieval. The runtime searches two different segment spaces,
and each query may activate a different valid-band subset, so there is no
single fixed row order that makes the real KNN computation cheap. Indexing or
tree-based search is the useful acceleration path instead.

If a segment has fewer than `min_valid_bands` valid inputs, that segment is
reported as unavailable instead of forcing a low-information retrieval.

## Neighbor Estimator

Let $\mathcal{N}_k^{(p)}$ be the top-$k$ nearest neighbors for segment $p$.
The default retrieved hyperspectral segment is the unweighted neighbor mean:

$$
\bar{h}^{(p)} = \frac{1}{k} \sum_{i \in \mathcal{N}_k^{(p)}} H_i^{(p)}
$$

This is intentionally simple in v1:

- it is stable
- it has no extra tuning parameter
- it exposes the retrieval behavior directly in diagnostics
- it is query-centric, so neighbors are not required to be mutually similar or
  from the same landcover class

An optional estimator mode is also available:

`distance_weighted_mean`

For non-zero distances, the implementation uses inverse-distance weights
$w_i = 1 / d_i$. If one or more neighbors have distance `0`, only those exact
matches are averaged.

`simplex_mixture`

This estimator keeps the same top-$k$ shortlist, but then solves a constrained
local mixture fit in source-band space:

$$
\min_{w} \left\| \sum_{i \in \mathcal{N}_k^{(p)}} w_i x_i^{(src,p)} - q^{(p)} \right\|_2^2
$$

subject to:

$$
w_i \ge 0,\qquad \sum_i w_i = 1
$$

So the reconstructed segment remains a convex combination of the retrieved
library spectra. In practice, this makes the shortlist more interpretable:
the diagnostics now expose both the KNN ranking and the fitted estimator
weights.

The returned diagnostics carry the chosen `neighbor_estimator`, plus:

- segment query band ids
- query band values and validity masks
- neighbor identities and neighbor distances
- neighbor weights and source-fit RMSE
- heuristic confidence scores and their components
- the selected neighbors' simulated source-band values for the same segment

The current confidence score is a bounded heuristic, not a calibrated
probability. It combines:

- valid-band coverage
- mean neighbor distance
- source-space fit RMSE after estimator weighting
- estimator weight concentration

Current production interpretation policy:

- `high` / `accept` for `confidence_score >= 0.85`
- `medium` / `manual_review` for `0.60 <= confidence_score < 0.85`
- `low` / `reject` for `confidence_score < 0.60`

These thresholds are intentionally conservative. They are useful as QA routing
rules, not as a substitute for benchmark validation on the target
sensor/library regime.

## Target-Sensor Mapping

Once a segment-level hyperspectral reconstruction is available, target bands in
the same segment are simulated with the same normalized SRF integral:

$$
\hat{y}_b^{(tgt,p)} =
\frac{\sum_{\ell} \bar{h}_{\ell}^{(p)} r_{tgt,b,\ell}}
     {\sum_{\ell} r_{tgt,b,\ell}}
$$

This produces the `target_sensor` output mode.

If only one segment is retrievable, target bands assigned to that segment are
still emitted. The public batch CLI leaves the unmapped target-band cells blank
for the missing segment.

## Full-Spectrum Assembly

When both segment reconstructions are available, the package assembles the full
`400-2500 nm` spectrum piecewise:

$$
\hat{h}(\lambda) =
\begin{cases}
\bar{h}^{(\text{vnir})}(\lambda), & 400 \le \lambda < 800 \\
w(\lambda)\bar{h}^{(\text{vnir})}(\lambda) + \left(1 - w(\lambda)\right)\bar{h}^{(\text{swir})}(\lambda), & 800 \le \lambda \le 1000 \\
\bar{h}^{(\text{swir})}(\lambda), & 1000 < \lambda \le 2500
\end{cases}
$$

with a linear overlap weight:

$$
w(\lambda) = \frac{1000 - \lambda}{200}
$$

So the blend is fully VNIR at `800 nm`, fully SWIR at `1000 nm`, and linear in
between.

## Regression Baseline

The benchmark report compares retrieval against a traditional regression
baseline fitted on the prepared-library rows.

For the source design matrix $X^{(src)}$ and target matrix $Y^{(tgt)}$, the
baseline model is:

$$
Y^{(tgt)} = \mathbf{1}\alpha^\top + X^{(src)}B + E
$$

where:

- $\alpha$ is the intercept vector
- $B$ is the regression coefficient matrix
- $E$ is the residual matrix

`benchmark_mapping(...)` fits this linear model on the train split and compares
it against retrieval on a held-out test split from the same prepared runtime.

## Error Metrics

The benchmark reports three families of metrics.

For predictions $\hat{y}_{j,b}$ and truth $y_{j,b}$ over test rows
$j = 1, \ldots, M$:

Root-mean-square error:

$$
\operatorname{RMSE}_b =
\sqrt{
\frac{1}{M} \sum_{j=1}^{M} \left(\hat{y}_{j,b} - y_{j,b}\right)^2
}
$$

Mean absolute error:

$$
\operatorname{MAE}_b =
\frac{1}{M} \sum_{j=1}^{M} \left|\hat{y}_{j,b} - y_{j,b}\right|
$$

Signed bias:

$$
\operatorname{Bias}_b =
\frac{1}{M} \sum_{j=1}^{M} \left(\hat{y}_{j,b} - y_{j,b}\right)
$$

The report also returns the mean of each metric across all target bands.

## Why The Method Is Physically Constrained

The package is not doing an unconstrained band-to-band interpolation. It uses
the hyperspectral library as a prior at every stage:

1. source observations retrieve spectrally similar library rows
2. the mapped target output is derived from retrieved hyperspectral structure
3. target reflectance is generated through the target sensor's own SRFs

That gives the method two useful properties:

- it stays tied to spectra that already exist in the library manifold
- it can reconstruct hyperspectral outputs and target-sensor outputs from the
  same retrieval step

## Practical Interpretation

- Larger `k` smooths the reconstruction and reduces neighbor sensitivity.
- Smaller `k` keeps sharper library detail but can increase local variance.
- Missing source bands only reduce information within their own segment.
- Full-spectrum output is available only when both VNIR and SWIR retrievals
  succeed.

For worked examples using official MODIS, Sentinel-2A, Landsat 8, and Landsat 9
response functions, see [Official Sensor Examples](official_sensor_examples.md).

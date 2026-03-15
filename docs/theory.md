# Mathematical Foundations

This page explains the theory behind the public mapping package and the equations
implemented by `spectral-library`.

It complements:

- [Mapping Quickstart](mapping_quickstart.md) for usage
- [Python API Reference](python_api_reference.md) for the public objects
- [Prepared Runtime Contract](prepared_runtime_contract.md) for the on-disk
  runtime standard

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
- `SWIR = 900-2500 nm`

The overlap is intentional. It preserves continuity near the detector
transition and lets the system reconstruct a stable full `400-2500 nm`
spectrum.

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
- all valid bands currently use uniform weight
- top-$k$ neighbors are ordered deterministically by distance and row index

If a segment has fewer than `min_valid_bands` valid inputs, that segment is
reported as unavailable instead of forcing a low-information retrieval.

## Neighbor Estimator

Let $\mathcal{N}_k^{(p)}$ be the top-$k$ nearest neighbors for segment $p$.
The retrieved hyperspectral segment is the unweighted neighbor mean:

$$
\bar{h}^{(p)} = \frac{1}{k} \sum_{i \in \mathcal{N}_k^{(p)}} H_i^{(p)}
$$

This is intentionally simple in v1:

- it is stable
- it has no extra tuning parameter
- it exposes the retrieval behavior directly in diagnostics

The returned diagnostics carry both neighbor identities and neighbor distances
for each segment.

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
\bar{h}^{(\text{vnir})}(\lambda), & 400 \le \lambda < 900 \\
w(\lambda)\bar{h}^{(\text{vnir})}(\lambda) + \left(1 - w(\lambda)\right)\bar{h}^{(\text{swir})}(\lambda), & 900 \le \lambda \le 1000 \\
\bar{h}^{(\text{swir})}(\lambda), & 1000 < \lambda \le 2500
\end{cases}
$$

with a linear overlap weight:

$$
w(\lambda) = \frac{1000 - \lambda}{100}
$$

So the blend is fully VNIR at `900 nm`, fully SWIR at `1000 nm`, and linear in
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

use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyReadwriteArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;

const CANONICAL_START_NM: usize = 400;
const CANONICAL_END_NM: usize = 2500;
const VNIR_START_NM: usize = 400;
const VNIR_END_NM: usize = 1000;
const SWIR_START_NM: usize = 800;
const SWIR_END_NM: usize = 2500;
const FULL_BLEND_START_NM: usize = 800;
const FULL_BLEND_END_NM: usize = 1000;
const FULL_WAVELENGTH_COUNT: usize = CANONICAL_END_NM - CANONICAL_START_NM + 1;
const VNIR_WAVELENGTH_COUNT: usize = VNIR_END_NM - VNIR_START_NM + 1;
const SWIR_WAVELENGTH_COUNT: usize = SWIR_END_NM - SWIR_START_NM + 1;
const FULL_BLEND_START_INDEX: usize = FULL_BLEND_START_NM - CANONICAL_START_NM;
const FULL_BLEND_STOP_INDEX: usize = FULL_BLEND_END_NM - CANONICAL_START_NM + 1;
const VNIR_OVERLAP_START_INDEX: usize = FULL_BLEND_START_NM - VNIR_START_NM;
const SWIR_OVERLAP_START_INDEX: usize = FULL_BLEND_START_NM - SWIR_START_NM;
const SWIR_OVERLAP_STOP_INDEX: usize = FULL_BLEND_END_NM - SWIR_START_NM + 1;
const INLINE_SMALL_CAPACITY: usize = 16;
const NARROW_EXACT_MEAN_OUTPUT_MAX_WIDTH: usize = 32;

type InlineUsizeVec = SmallVec<[usize; INLINE_SMALL_CAPACITY]>;
type InlineF64Vec = SmallVec<[f64; INLINE_SMALL_CAPACITY]>;

trait FloatLike: Copy + Send + Sync + Element {
    fn to_f64(self) -> f64;
}

impl FloatLike for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Clone, Copy)]
enum TargetFinalizeStatus {
    Ok = 0,
    NoTargetBands = 1,
}

#[derive(Clone, Copy)]
enum NeighborEstimator {
    Mean,
    DistanceWeightedMean,
    SimplexMixture,
}

impl NeighborEstimator {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "mean" => Ok(Self::Mean),
            "distance_weighted_mean" => Ok(Self::DistanceWeightedMean),
            "simplex_mixture" => Ok(Self::SimplexMixture),
            _ => Err(format!("unsupported neighbor_estimator `{name}`")),
        }
    }
}

fn can_use_exact_mean_narrow_fast_path(
    estimator: NeighborEstimator,
    output_width: usize,
    has_shortlists: bool,
    has_shortlist_distances: bool,
) -> bool {
    matches!(estimator, NeighborEstimator::Mean)
        && output_width <= NARROW_EXACT_MEAN_OUTPUT_MAX_WIDTH
        && has_shortlists
        && has_shortlist_distances
}

fn assemble_full_spectrum_row_into(
    vnir: &[f64],
    swir: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    if vnir.len() != VNIR_WAVELENGTH_COUNT {
        return Err(format!(
            "vnir rows must have width {VNIR_WAVELENGTH_COUNT}, got {}",
            vnir.len()
        ));
    }
    if swir.len() != SWIR_WAVELENGTH_COUNT {
        return Err(format!(
            "swir rows must have width {SWIR_WAVELENGTH_COUNT}, got {}",
            swir.len()
        ));
    }
    if output.len() != FULL_WAVELENGTH_COUNT {
        return Err(format!(
            "full-spectrum output rows must have width {FULL_WAVELENGTH_COUNT}, got {}",
            output.len()
        ));
    }

    output[..FULL_BLEND_START_INDEX].copy_from_slice(&vnir[..FULL_BLEND_START_INDEX]);
    for overlap_offset in 0..(FULL_BLEND_STOP_INDEX - FULL_BLEND_START_INDEX) {
        let wavelength_nm = (FULL_BLEND_START_NM + overlap_offset) as f64;
        let weight = (FULL_BLEND_END_NM as f64 - wavelength_nm)
            / (FULL_BLEND_END_NM as f64 - FULL_BLEND_START_NM as f64);
        output[FULL_BLEND_START_INDEX + overlap_offset] = weight
            * vnir[VNIR_OVERLAP_START_INDEX + overlap_offset]
            + (1.0 - weight) * swir[SWIR_OVERLAP_START_INDEX + overlap_offset];
    }
    output[FULL_BLEND_STOP_INDEX..].copy_from_slice(&swir[SWIR_OVERLAP_STOP_INDEX..]);
    Ok(())
}

fn validate_output_indices(indices: &[i64], output_width: usize) -> Result<InlineUsizeVec, String> {
    let mut resolved = InlineUsizeVec::with_capacity(indices.len());
    for &value in indices {
        if value < 0 {
            return Err("output indices must be non-negative".to_string());
        }
        let index = value as usize;
        if index >= output_width {
            return Err("output index exceeded target output width".to_string());
        }
        resolved.push(index);
    }
    Ok(resolved)
}

fn ordering_for_distances(left: (usize, f64), right: (usize, f64)) -> Ordering {
    match left.1.partial_cmp(&right.1).unwrap_or(Ordering::Greater) {
        Ordering::Equal => left.0.cmp(&right.0),
        other => other,
    }
}

fn top_neighbor_pairs(
    mut pairs: SmallVec<[(usize, f64); INLINE_SMALL_CAPACITY]>,
    k: usize,
) -> Result<(InlineUsizeVec, InlineF64Vec), String> {
    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if pairs.is_empty() {
        return Err("neighbor search backend returned no candidate rows".to_string());
    }
    let neighbor_count = k.min(pairs.len());
    if neighbor_count == pairs.len() {
        pairs.sort_unstable_by(|left, right| ordering_for_distances(*left, *right));
    } else {
        pairs.select_nth_unstable_by(neighbor_count - 1, |left, right| {
            ordering_for_distances(*left, *right)
        });
        pairs[..neighbor_count].sort_unstable_by(|left, right| ordering_for_distances(*left, *right));
    }
    let mut row_ids = InlineUsizeVec::with_capacity(neighbor_count);
    let mut distances = InlineF64Vec::with_capacity(neighbor_count);
    for (row_id, distance) in pairs.into_iter().take(neighbor_count) {
        row_ids.push(row_id);
        distances.push(distance);
    }
    Ok((row_ids, distances))
}

fn project_simplex(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0;
    let mut rho = 0usize;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += *value;
        let candidate = (cumulative - 1.0) / ((index + 1) as f64);
        if *value - candidate > 0.0 {
            rho = index;
            theta = candidate;
        }
    }
    let _ = rho;
    values
        .iter()
        .map(|value| (value - theta).max(0.0))
        .collect()
}

fn max_eigenvalue_symmetric(matrix: &[f64], width: usize) -> f64 {
    if width == 0 {
        return 0.0;
    }
    let mut vector = vec![1.0 / (width as f64).sqrt(); width];
    let mut estimate = 0.0;
    for _ in 0..64 {
        let mut next = vec![0.0; width];
        for row in 0..width {
            let row_offset = row * width;
            let mut value = 0.0;
            for column in 0..width {
                value += matrix[row_offset + column] * vector[column];
            }
            next[row] = value;
        }
        let norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 1e-12 {
            return 0.0;
        }
        for value in &mut next {
            *value /= norm;
        }
        let mut numerator = 0.0;
        for row in 0..width {
            let row_offset = row * width;
            let mut value = 0.0;
            for column in 0..width {
                value += matrix[row_offset + column] * next[column];
            }
            numerator += next[row] * value;
        }
        estimate = numerator;
        vector = next;
    }
    estimate
}

fn resolve_global_row_index(row_index: i64, row_count: usize) -> Result<usize, String> {
    if row_index < 0 {
        return Err("neighbor indices must be non-negative".to_string());
    }
    let row = row_index as usize;
    if row >= row_count {
        return Err("neighbor index exceeded the prepared row count".to_string());
    }
    Ok(row)
}

fn validate_neighbor_rows<T: FloatLike>(
    source: &ArrayView2<'_, T>,
    hyperspectral: &ArrayView2<'_, T>,
    neighbor_indices: &[i64],
) -> Result<Vec<usize>, String> {
    let row_count = source.nrows();
    if row_count != hyperspectral.nrows() {
        return Err("source and hyperspectral arrays must have the same row count".to_string());
    }
    let mut rows = Vec::with_capacity(neighbor_indices.len());
    for &neighbor_index in neighbor_indices {
        rows.push(resolve_global_row_index(neighbor_index, row_count)?);
    }
    Ok(rows)
}

fn compute_weights_into<'a, T: FloatLike>(
    source: &'a ArrayView2<'a, T>,
    neighbor_rows: &[usize],
    neighbor_distances: &[f64],
    valid_columns: &[usize],
    query: &[f64],
    estimator: NeighborEstimator,
    weights_out: &mut [f64],
) -> Result<(), String> {
    let neighbor_count = neighbor_rows.len();
    if neighbor_count == 0 {
        return Err("at least one neighbor is required".to_string());
    }
    if neighbor_count != neighbor_distances.len() {
        return Err("neighbor distance count must match neighbor index count".to_string());
    }
    if valid_columns.len() != query.len() {
        return Err("query width must match the selected valid-band count".to_string());
    }
    if weights_out.len() != neighbor_count {
        return Err("weights output width must match neighbor count".to_string());
    }
    match estimator {
        NeighborEstimator::Mean => {
            let weight = 1.0 / (neighbor_count as f64);
            weights_out.fill(weight);
            Ok(())
        }
        NeighborEstimator::DistanceWeightedMean => {
            let exact_count = neighbor_distances
                .iter()
                .filter(|distance| **distance <= 1e-12)
                .count();
            if exact_count > 0 {
                let exact_weight = 1.0 / (exact_count as f64);
                for (weight, distance) in weights_out.iter_mut().zip(neighbor_distances.iter()) {
                    *weight = if *distance <= 1e-12 {
                        exact_weight
                    } else {
                        0.0
                    };
                }
                return Ok(());
            }
            let mut weight_sum = 0.0;
            for (weight, distance) in weights_out.iter_mut().zip(neighbor_distances.iter()) {
                *weight = 1.0 / *distance;
                weight_sum += *weight;
            }
            for value in weights_out.iter_mut() {
                *value /= weight_sum;
            }
            Ok(())
        }
        NeighborEstimator::SimplexMixture => {
            if neighbor_count == 1 {
                weights_out[0] = 1.0;
                return Ok(());
            }
            let exact_count = neighbor_distances
                .iter()
                .filter(|distance| **distance <= 1e-12)
                .count();
            if exact_count > 0 {
                let exact_weight = 1.0 / (exact_count as f64);
                for (weight, distance) in weights_out.iter_mut().zip(neighbor_distances.iter()) {
                    *weight = if *distance <= 1e-12 {
                        exact_weight
                    } else {
                        0.0
                    };
                }
                return Ok(());
            }

            let valid_count = valid_columns.len();
            let source_flat = source
                .as_slice_memory_order()
                .ok_or_else(|| "source must be contiguous".to_string())?;
            let source_width = source.ncols();
            let mut candidate_matrix = vec![0.0; neighbor_count * valid_count];
            for (neighbor_offset, &row) in neighbor_rows.iter().enumerate() {
                let row_offset = neighbor_offset * valid_count;
                let source_start = row * source_width;
                let source_row = &source_flat[source_start..source_start + source_width];
                for (column_offset, &column) in valid_columns.iter().enumerate() {
                    candidate_matrix[row_offset + column_offset] = source_row[column].to_f64();
                }
            }

            let mut gram = vec![0.0; neighbor_count * neighbor_count];
            let mut linear = vec![0.0; neighbor_count];
            for left in 0..neighbor_count {
                let left_offset = left * valid_count;
                let left_row = &candidate_matrix[left_offset..left_offset + valid_count];
                linear[left] = left_row
                    .iter()
                    .zip(query.iter())
                    .map(|(candidate, target)| candidate * target)
                    .sum::<f64>();
                for right in left..neighbor_count {
                    let right_offset = right * valid_count;
                    let right_row = &candidate_matrix[right_offset..right_offset + valid_count];
                    let value = left_row
                        .iter()
                        .zip(right_row.iter())
                        .map(|(left_value, right_value)| left_value * right_value)
                        .sum::<f64>();
                    gram[left * neighbor_count + right] = value;
                    gram[right * neighbor_count + left] = value;
                }
            }

            let max_eigenvalue = max_eigenvalue_symmetric(&gram, neighbor_count);
            if !max_eigenvalue.is_finite() || max_eigenvalue <= 1e-12 {
                let weight = 1.0 / (neighbor_count as f64);
                weights_out.fill(weight);
                return Ok(());
            }

            let step = 1.0 / max_eigenvalue;
            let mut weights = vec![1.0 / (neighbor_count as f64); neighbor_count];
            let mut momentum = weights.clone();
            let mut t_prev = 1.0;
            for _ in 0..250 {
                let mut gradient = vec![0.0; neighbor_count];
                for row in 0..neighbor_count {
                    let row_offset = row * neighbor_count;
                    let mut value = 0.0;
                    for column in 0..neighbor_count {
                        value += gram[row_offset + column] * momentum[column];
                    }
                    gradient[row] = value - linear[row];
                }
                let proposed_input: Vec<f64> = momentum
                    .iter()
                    .zip(gradient.iter())
                    .map(|(weight, grad)| weight - step * grad)
                    .collect();
                let updated = project_simplex(&proposed_input);
                let l1_delta = updated
                    .iter()
                    .zip(weights.iter())
                    .map(|(left, right)| (left - right).abs())
                    .sum::<f64>();
                if l1_delta <= 1e-10 {
                    weights = updated;
                    break;
                }
                let t_next = 0.5_f64 * (1.0_f64 + (1.0_f64 + 4.0_f64 * t_prev * t_prev).sqrt());
                momentum = updated
                    .iter()
                    .zip(weights.iter())
                    .map(|(new_weight, old_weight)| {
                        new_weight + ((t_prev - 1.0) / t_next) * (new_weight - old_weight)
                    })
                    .collect();
                weights = updated;
                t_prev = t_next;
            }
            let weight_sum = weights.iter().sum::<f64>();
            if !weight_sum.is_finite() || weight_sum <= 0.0 {
                let weight = 1.0 / (neighbor_count as f64);
                weights_out.fill(weight);
                return Ok(());
            }
            for (output, value) in weights_out.iter_mut().zip(weights.iter()) {
                *output = *value / weight_sum;
            }
            Ok(())
        }
    }
}

fn compute_sample<'a, T: FloatLike>(
    source: &'a ArrayView2<'a, T>,
    hyperspectral: &'a ArrayView2<'a, T>,
    neighbor_rows: &[usize],
    neighbor_distances: &[f64],
    query: &[f64],
    valid_columns: &[usize],
    estimator: NeighborEstimator,
    reconstructed_out: &mut [f64],
    weights_out: &mut [f64],
) -> Result<f64, String> {
    if source.nrows() != hyperspectral.nrows() {
        return Err("source and hyperspectral arrays must have the same row count".to_string());
    }
    if reconstructed_out.len() != hyperspectral.ncols() {
        return Err("reconstructed output width must match hyperspectral width".to_string());
    }
    if weights_out.len() != neighbor_rows.len() {
        return Err("weights output width must match neighbor count".to_string());
    }
    let source_flat = source
        .as_slice_memory_order()
        .ok_or_else(|| "source must be contiguous".to_string())?;
    let hyperspectral_flat = hyperspectral
        .as_slice_memory_order()
        .ok_or_else(|| "hyperspectral must be contiguous".to_string())?;
    let source_width = source.ncols();
    let hyperspectral_width = hyperspectral.ncols();

    reconstructed_out.fill(0.0);
    let mut source_rows = SmallVec::<[&[T]; INLINE_SMALL_CAPACITY]>::with_capacity(neighbor_rows.len());
    let mut hyperspectral_rows = SmallVec::<[&[T]; INLINE_SMALL_CAPACITY]>::with_capacity(neighbor_rows.len());
    for &row in neighbor_rows.iter() {
        let source_offset = row
            .checked_mul(source_width)
            .ok_or_else(|| "source row offset overflowed".to_string())?;
        let source_end = source_offset
            .checked_add(source_width)
            .ok_or_else(|| "source row end overflowed".to_string())?;
        let hyperspectral_offset = row
            .checked_mul(hyperspectral_width)
            .ok_or_else(|| "hyperspectral row offset overflowed".to_string())?;
        let hyperspectral_end = hyperspectral_offset
            .checked_add(hyperspectral_width)
            .ok_or_else(|| "hyperspectral row end overflowed".to_string())?;
        source_rows.push(
            source_flat
                .get(source_offset..source_end)
                .ok_or_else(|| "source row exceeded matrix bounds".to_string())?,
        );
        hyperspectral_rows.push(
            hyperspectral_flat
                .get(hyperspectral_offset..hyperspectral_end)
                .ok_or_else(|| "hyperspectral row exceeded matrix bounds".to_string())?,
        );
    }
    let mut predicted_query = SmallVec::<[f64; INLINE_SMALL_CAPACITY]>::with_capacity(valid_columns.len());
    predicted_query.resize(valid_columns.len(), 0.0);

    let use_full_width_columns = valid_columns.len() == source.ncols()
        && valid_columns
            .iter()
            .enumerate()
            .all(|(query_offset, &column)| column == query_offset);
    let mut accumulate_row = |row_offset: usize, weight: f64| {
        for (output, value) in reconstructed_out.iter_mut().zip(hyperspectral_rows[row_offset].iter()) {
            *output += weight * value.to_f64();
        }
        if use_full_width_columns {
            for (predicted, value) in predicted_query.iter_mut().zip(source_rows[row_offset].iter()) {
                *predicted += weight * value.to_f64();
            }
        } else {
            for (predicted, &source_column) in predicted_query.iter_mut().zip(valid_columns.iter()) {
                *predicted += weight * source_rows[row_offset][source_column].to_f64();
            }
        }
    };

    match estimator {
        NeighborEstimator::Mean => {
            let inverse_neighbor_count = 1.0 / (neighbor_rows.len() as f64);
            for row_offset in 0..neighbor_rows.len() {
                weights_out[row_offset] = inverse_neighbor_count;
                accumulate_row(row_offset, inverse_neighbor_count);
            }
        }
        NeighborEstimator::DistanceWeightedMean => {
            let exact_count = neighbor_distances
                .iter()
                .filter(|distance| **distance <= 1e-12)
                .count();
            if exact_count > 0 {
                let weight = 1.0 / (exact_count as f64);
                for (row_offset, &distance) in neighbor_distances.iter().enumerate() {
                    if distance <= 1e-12 {
                        weights_out[row_offset] = weight;
                        accumulate_row(row_offset, weight);
                    }
                }
            } else {
                let inv_distance_sum = neighbor_distances.iter().map(|distance| 1.0 / *distance).sum::<f64>();
                for (row_offset, &distance) in neighbor_distances.iter().enumerate() {
                    let weight = (1.0 / distance) / inv_distance_sum;
                    weights_out[row_offset] = weight;
                    accumulate_row(row_offset, weight);
                }
            }
        }
        NeighborEstimator::SimplexMixture => {
            compute_weights_into(
                source,
                neighbor_rows,
                neighbor_distances,
                valid_columns,
                query,
                estimator,
                weights_out,
            )?;
            for (row_offset, weight) in weights_out.iter().enumerate() {
                accumulate_row(row_offset, *weight);
            }
        }
    }

    let mut sum_squared_error = 0.0;
    for (predicted, target) in predicted_query.iter().zip(query.iter()) {
        let difference = *predicted - *target;
        sum_squared_error += difference * difference;
    }
    let source_fit_rmse = (sum_squared_error / (valid_columns.len() as f64)).sqrt();
    Ok(source_fit_rmse)
}

fn resolve_exact_shortlist(
    row_indices: &numpy::ndarray::ArrayView1<'_, i64>,
    shortlist: &[i64],
    shortlist_distances: &[f64],
    requested_neighbor_count: usize,
    row_count: usize,
) -> Result<Option<(InlineUsizeVec, InlineF64Vec)>, String> {
    if shortlist.len() < requested_neighbor_count || shortlist_distances.len() != shortlist.len() {
        return Ok(None);
    }
    let mut resolved_rows = InlineUsizeVec::with_capacity(requested_neighbor_count);
    let mut resolved_distances = InlineF64Vec::with_capacity(requested_neighbor_count);
    let mut previous_distance = f64::NEG_INFINITY;
    for offset in 0..requested_neighbor_count {
        let distance = shortlist_distances[offset];
        if !distance.is_finite() || distance < previous_distance {
            return Ok(None);
        }
        previous_distance = distance;
        let local_index = shortlist[offset];
        if local_index < 0 {
            return Ok(None);
        }
        let local_index = local_index as usize;
        if local_index >= row_indices.len() {
            return Ok(None);
        }
        let global_row = resolve_global_row_index(row_indices[local_index], row_count)?;
        if resolved_rows.iter().any(|&existing_row| existing_row == global_row) {
            return Ok(None);
        }
        resolved_rows.push(global_row);
        resolved_distances.push(distance);
    }
    Ok(Some((resolved_rows, resolved_distances)))
}

fn compute_sample_exact_shortlist_mean_narrow<'a, 'b, T: FloatLike>(
    candidates: &'a ArrayView2<'a, T>,
    row_indices: &numpy::ndarray::ArrayView1<'_, i64>,
    hyperspectral: &'b ArrayView2<'b, T>,
    shortlist_row: Option<&[i64]>,
    shortlist_distance_row: Option<&[f64]>,
    requested_neighbor_count: usize,
    row_count: usize,
    query: &[f64],
    reconstructed_out: &mut [f64],
) -> Result<Option<f64>, String> {
    let shortlist = match shortlist_row {
        Some(value) => value,
        None => return Ok(None),
    };
    let shortlist_distances = match shortlist_distance_row {
        Some(value) => value,
        None => return Ok(None),
    };
    if requested_neighbor_count == 0
        || requested_neighbor_count > shortlist.len()
        || shortlist_distances.len() != shortlist.len()
    {
        return Ok(None);
    }
    if candidates.ncols() != query.len() {
        return Ok(None);
    }
    if reconstructed_out.len() != hyperspectral.ncols() {
        return Ok(None);
    }

    let candidate_flat = candidates
        .as_slice_memory_order()
        .ok_or_else(|| "candidate_matrix must be contiguous".to_string())?;
    let candidate_width = candidates.ncols();
    let hyperspectral_flat = hyperspectral
        .as_slice_memory_order()
        .ok_or_else(|| "hyperspectral_rows must be contiguous".to_string())?;
    let hyperspectral_width = hyperspectral.ncols();

    reconstructed_out.fill(0.0);
    let mut predicted_query = SmallVec::<[f64; INLINE_SMALL_CAPACITY]>::with_capacity(query.len());
    predicted_query.resize(query.len(), 0.0);
    let mut previous_distance = f64::NEG_INFINITY;
    let mut seen_global_rows = InlineUsizeVec::with_capacity(requested_neighbor_count);

    for offset in 0..requested_neighbor_count {
        let distance = shortlist_distances[offset];
        if !distance.is_finite() || distance < previous_distance {
            return Ok(None);
        }
        previous_distance = distance;

        let local_index = shortlist[offset];
        if local_index < 0 {
            return Ok(None);
        }
        let local_index = local_index as usize;
        if local_index >= row_indices.len() {
            return Ok(None);
        }
        let global_row = resolve_global_row_index(row_indices[local_index], row_count)?;
        if seen_global_rows.iter().any(|&row| row == global_row) {
            return Ok(None);
        }
        seen_global_rows.push(global_row);
        let candidate_offset = local_index
            .checked_mul(candidate_width)
            .ok_or_else(|| "candidate row offset overflowed".to_string())?;
        let candidate_end = candidate_offset
            .checked_add(candidate_width)
            .ok_or_else(|| "candidate row end overflowed".to_string())?;
        let hyperspectral_offset = global_row
            .checked_mul(hyperspectral_width)
            .ok_or_else(|| "hyperspectral row offset overflowed".to_string())?;
        let hyperspectral_end = hyperspectral_offset
            .checked_add(hyperspectral_width)
            .ok_or_else(|| "hyperspectral row end overflowed".to_string())?;
        let candidate_row = candidate_flat
            .get(candidate_offset..candidate_end)
            .ok_or_else(|| "candidate row exceeded matrix bounds".to_string())?;
        let hyperspectral_row = hyperspectral_flat
            .get(hyperspectral_offset..hyperspectral_end)
            .ok_or_else(|| "hyperspectral row exceeded matrix bounds".to_string())?;
        for (predicted, value) in predicted_query.iter_mut().zip(candidate_row.iter()) {
            *predicted += value.to_f64();
        }
        for (output, value) in reconstructed_out.iter_mut().zip(hyperspectral_row.iter()) {
            *output += value.to_f64();
        }
    }

    let inverse_neighbor_count = 1.0 / (requested_neighbor_count as f64);
    for predicted in predicted_query.iter_mut() {
        *predicted *= inverse_neighbor_count;
    }
    for output in reconstructed_out.iter_mut() {
        *output *= inverse_neighbor_count;
    }

    let mut sum_squared_error = 0.0;
    for (predicted, target) in predicted_query.iter().zip(query.iter()) {
        let difference = *predicted - *target;
        sum_squared_error += difference * difference;
    }
    Ok(Some((sum_squared_error / (query.len() as f64)).sqrt()))
}

fn resolve_top_neighbors<T: FloatLike>(
    candidates: &ArrayView2<'_, T>,
    row_indices: &numpy::ndarray::ArrayView1<'_, i64>,
    query_row: &[f64],
    requested_neighbor_count: usize,
    shortlist_row: Option<&[i64]>,
    shortlist_distance_row: Option<&[f64]>,
    row_count: usize,
) -> Result<(InlineUsizeVec, InlineF64Vec), String> {
    if requested_neighbor_count == 0 {
        return Err("k must be at least 1".to_string());
    }
    let candidate_flat = candidates
        .as_slice_memory_order()
        .ok_or_else(|| "candidate_matrix must be contiguous".to_string())?;
    let candidate_width = candidates.ncols();
    let pairs = if let Some(shortlist) = shortlist_row {
        if let Some(distances) = shortlist_distance_row {
            if distances.len() != shortlist.len() {
                return Err("local_candidate_distances width must match local_candidate_indices width".to_string());
            }
            if let Some(exact_neighbors) =
                resolve_exact_shortlist(row_indices, shortlist, distances, requested_neighbor_count, row_count)?
            {
                return Ok(exact_neighbors);
            }
        }
        let mut resolved_local = SmallVec::<[(usize, f64); INLINE_SMALL_CAPACITY]>::with_capacity(shortlist.len());
        for (offset, &value) in shortlist.iter().enumerate() {
            if value < 0 {
                continue;
            }
            let index = value as usize;
            if index >= row_indices.len() {
                continue;
            }
            let distance = if let Some(distances) = shortlist_distance_row {
                distances[offset]
            } else {
                let candidate_offset = index
                    .checked_mul(candidate_width)
                    .ok_or_else(|| "candidate row offset overflowed".to_string())?;
                let candidate_end = candidate_offset
                    .checked_add(candidate_width)
                    .ok_or_else(|| "candidate row end overflowed".to_string())?;
                let candidate_row = candidate_flat
                    .get(candidate_offset..candidate_end)
                    .ok_or_else(|| "candidate row exceeded matrix bounds".to_string())?;
                let mut sum_squared_error = 0.0;
                for (candidate_value, query_value) in candidate_row.iter().zip(query_row.iter()) {
                    let difference = candidate_value.to_f64() - *query_value;
                    sum_squared_error += difference * difference;
                }
                (sum_squared_error / (query_row.len() as f64)).sqrt()
            };
            if distance.is_finite() {
                resolved_local.push((index, distance));
            }
        }
        resolved_local.sort_unstable_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Greater)
            })
        });
        resolved_local.dedup_by(|left, right| left.0 == right.0);
        if resolved_local.len() < requested_neighbor_count {
            return Err(
                "neighbor shortlist did not contain enough unique candidates for exact reranking"
                    .to_string(),
            );
        }
        let mut resolved = SmallVec::<[(usize, f64); INLINE_SMALL_CAPACITY]>::with_capacity(resolved_local.len());
        for (local_index, distance) in resolved_local {
            resolved.push((resolve_global_row_index(row_indices[local_index], row_count)?, distance));
        }
        resolved
    } else {
        let mut resolved = SmallVec::<[(usize, f64); INLINE_SMALL_CAPACITY]>::with_capacity(candidates.nrows());
        for local_index in 0..candidates.nrows() {
            let candidate_offset = local_index
                .checked_mul(candidate_width)
                .ok_or_else(|| "candidate row offset overflowed".to_string())?;
            let candidate_end = candidate_offset
                .checked_add(candidate_width)
                .ok_or_else(|| "candidate row end overflowed".to_string())?;
            let candidate_row = candidate_flat
                .get(candidate_offset..candidate_end)
                .ok_or_else(|| "candidate row exceeded matrix bounds".to_string())?;
            let mut sum_squared_error = 0.0;
            for (candidate_value, query_value) in candidate_row.iter().zip(query_row.iter()) {
                let difference = candidate_value.to_f64() - *query_value;
                sum_squared_error += difference * difference;
            }
            let distance = (sum_squared_error / (query_row.len() as f64)).sqrt();
            resolved.push((resolve_global_row_index(row_indices[local_index], row_count)?, distance));
        }
        resolved
    };
    top_neighbor_pairs(pairs, requested_neighbor_count)
}

fn refine_neighbor_rows_batch_impl<T: FloatLike + Element>(
    py: Python<'_>,
    candidate_matrix: PyReadonlyArray2<'_, T>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f64>>)> {
    let candidates = candidate_matrix.as_array();
    let query_rows = query_values.as_array();
    let row_indices = candidate_row_indices.as_array();
    if candidates.nrows() != row_indices.len() {
        return Err(PyValueError::new_err(
            "candidate_matrix row count must match candidate_row_indices length",
        ));
    }
    if candidates.ncols() != query_rows.ncols() {
        return Err(PyValueError::new_err(
            "query_values column count must match candidate_matrix width",
        ));
    }
    let local_shortlists = local_candidate_indices.as_ref().map(|indices| indices.as_array());
    if let Some(shortlists) = &local_shortlists {
        if shortlists.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_indices row count must match query_values row count",
            ));
        }
    }
    let local_shortlist_distances = local_candidate_distances.as_ref().map(|distances| distances.as_array());
    if let Some(distances) = &local_shortlist_distances {
        if local_shortlists.is_none() {
            return Err(PyValueError::new_err(
                "local_candidate_distances requires local_candidate_indices",
            ));
        }
        if distances.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_distances row count must match query_values row count",
            ));
        }
        if distances.ncols()
            != local_shortlists
                .as_ref()
                .expect("validated local shortlist presence")
                .ncols()
        {
            return Err(PyValueError::new_err(
                "local_candidate_distances width must match local_candidate_indices width",
            ));
        }
    }

    let batch_size = query_rows.nrows();
    let requested_neighbor_count = if let Some(shortlists) = &local_shortlists {
        k.min(shortlists.ncols())
    } else {
        k.min(candidates.nrows())
    };
    if requested_neighbor_count == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }

    let global_row_count = row_indices
        .iter()
        .map(|&index| if index >= 0 { index as usize + 1 } else { 0 })
        .max()
        .unwrap_or(0);

    let results = py.allow_threads(|| {
        (0..batch_size)
            .into_par_iter()
            .map(|batch_index| {
                let query_row_binding = query_rows.row(batch_index);
                let query_row = query_row_binding
                    .as_slice()
                    .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                let shortlist_binding = local_shortlists
                    .as_ref()
                    .map(|shortlists| shortlists.row(batch_index));
                let shortlist_row = match shortlist_binding.as_ref() {
                    Some(row) => Some(
                        row.as_slice()
                            .ok_or_else(|| "local_candidate_indices rows must be contiguous".to_string())?,
                    ),
                    None => None,
                };
                let shortlist_distance_binding = local_shortlist_distances
                    .as_ref()
                    .map(|distances| distances.row(batch_index));
                let shortlist_distance_row = match shortlist_distance_binding.as_ref() {
                    Some(row) => Some(
                        row.as_slice()
                            .ok_or_else(|| "local_candidate_distances rows must be contiguous".to_string())?,
                    ),
                    None => None,
                };
                resolve_top_neighbors(
                    &candidates,
                    &row_indices,
                    query_row,
                    requested_neighbor_count,
                    shortlist_row,
                    shortlist_distance_row,
                    global_row_count,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut neighbor_indices_flat = vec![0_i64; batch_size * requested_neighbor_count];
    let mut neighbor_distances_flat = vec![0.0_f64; batch_size * requested_neighbor_count];
    for (batch_index, result) in results.into_iter().enumerate() {
        let (row_ids, distances) = result
            .map_err(|message| PyValueError::new_err(format!("batch row {batch_index}: {message}")))?;
        let offset = batch_index * requested_neighbor_count;
        for (value_offset, row_id) in row_ids.into_iter().enumerate() {
            neighbor_indices_flat[offset + value_offset] = row_id as i64;
        }
        neighbor_distances_flat[offset..offset + requested_neighbor_count].copy_from_slice(&distances);
    }

    let neighbor_indices_array =
        Array2::from_shape_vec((batch_size, requested_neighbor_count), neighbor_indices_flat)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let neighbor_distances_array =
        Array2::from_shape_vec((batch_size, requested_neighbor_count), neighbor_distances_flat)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        neighbor_indices_array.into_pyarray_bound(py).unbind(),
        neighbor_distances_array.into_pyarray_bound(py).unbind(),
    ))
}

fn reconstruct_neighbor_spectra_batch_impl<T: FloatLike + Element>(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, T>,
    hyperspectral_rows: PyReadonlyArray2<'_, T>,
    candidate_matrix: PyReadonlyArray2<'_, T>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
    neighbor_estimator: &str,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let estimator = NeighborEstimator::parse(neighbor_estimator).map_err(PyValueError::new_err)?;
    let source = source_matrix.as_array();
    let hyperspectral = hyperspectral_rows.as_array();
    let candidates = candidate_matrix.as_array();
    let query_rows = query_values.as_array();
    let row_indices = candidate_row_indices.as_array();
    if candidates.nrows() != row_indices.len() {
        return Err(PyValueError::new_err(
            "candidate_matrix row count must match candidate_row_indices length",
        ));
    }
    if candidates.ncols() != query_rows.ncols() {
        return Err(PyValueError::new_err(
            "query_values column count must match candidate_matrix width",
        ));
    }
    if source.nrows() != hyperspectral.nrows() {
        return Err(PyValueError::new_err(
            "source_matrix and hyperspectral_rows must have the same row count",
        ));
    }
    let valid_columns: Vec<usize> = match valid_indices {
        Some(indices) => {
            let values = indices.as_array();
            let mut columns = Vec::with_capacity(values.len());
            for &value in values.iter() {
                if value < 0 {
                    return Err(PyValueError::new_err("valid_indices must be non-negative"));
                }
                let column = value as usize;
                if column >= source.ncols() {
                    return Err(PyValueError::new_err("valid_indices exceeded source band count"));
                }
                columns.push(column);
            }
            columns
        }
        None => (0..source.ncols()).collect(),
    };
    if query_rows.ncols() != valid_columns.len() {
        return Err(PyValueError::new_err(
            "query_values column count must match valid_indices or the full source band count",
        ));
    }
    let local_shortlists = local_candidate_indices.as_ref().map(|indices| indices.as_array());
    if let Some(shortlists) = &local_shortlists {
        if shortlists.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_indices row count must match query_values row count",
            ));
        }
    }
    let local_shortlist_distances = local_candidate_distances.as_ref().map(|distances| distances.as_array());
    if let Some(distances) = &local_shortlist_distances {
        if local_shortlists.is_none() {
            return Err(PyValueError::new_err(
                "local_candidate_distances requires local_candidate_indices",
            ));
        }
        if distances.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_distances row count must match query_values row count",
            ));
        }
        if distances.ncols()
            != local_shortlists
                .as_ref()
                .expect("validated local shortlist presence")
                .ncols()
        {
            return Err(PyValueError::new_err(
                "local_candidate_distances width must match local_candidate_indices width",
            ));
        }
    }

    let batch_size = query_rows.nrows();
    let requested_neighbor_count = if let Some(shortlists) = &local_shortlists {
        k.min(shortlists.ncols())
    } else {
        k.min(candidates.nrows())
    };
    if requested_neighbor_count == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }

    let output_width = hyperspectral.ncols();
    let mut reconstructed_flat = vec![0.0; batch_size * output_width];
    let mut rmse_flat = vec![0.0; batch_size];
    let use_exact_mean_narrow_fast_path = can_use_exact_mean_narrow_fast_path(
        estimator,
        output_width,
        local_shortlists.is_some(),
        local_shortlist_distances.is_some(),
    );
    if use_exact_mean_narrow_fast_path {
        py.allow_threads(|| {
            reconstructed_flat
                .par_chunks_mut(output_width)
                .zip(rmse_flat.par_iter_mut())
                .enumerate()
                .try_for_each(|(batch_index, (reconstructed_out, rmse_out))| -> Result<(), String> {
                    let query_row_binding = query_rows.row(batch_index);
                    let query_row = query_row_binding
                        .as_slice()
                        .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                    let shortlist_binding = local_shortlists
                        .as_ref()
                        .map(|shortlists| shortlists.row(batch_index));
                    let shortlist_row = shortlist_binding
                        .as_ref()
                        .map(|row| {
                            row.as_slice().ok_or_else(|| {
                                "local_candidate_indices rows must be contiguous".to_string()
                            })
                        })
                        .transpose()?;
                    let shortlist_distance_binding = local_shortlist_distances
                        .as_ref()
                        .map(|distances| distances.row(batch_index));
                    let shortlist_distance_row = shortlist_distance_binding
                        .as_ref()
                        .map(|row| {
                            row.as_slice().ok_or_else(|| {
                                "local_candidate_distances rows must be contiguous".to_string()
                            })
                        })
                        .transpose()?;
                    let fast_rmse = compute_sample_exact_shortlist_mean_narrow(
                        &candidates,
                        &row_indices,
                        &hyperspectral,
                        shortlist_row,
                        shortlist_distance_row,
                        requested_neighbor_count,
                        source.nrows(),
                        query_row,
                        reconstructed_out,
                    )?;
                    if let Some(rmse) = fast_rmse {
                        *rmse_out = rmse;
                        return Ok(());
                    }
                    let (neighbor_rows, neighbor_distances) = resolve_top_neighbors(
                        &candidates,
                        &row_indices,
                        query_row,
                        requested_neighbor_count,
                        shortlist_row,
                        shortlist_distance_row,
                        source.nrows(),
                    )?;
                    let mut weights_out = SmallVec::<[f64; INLINE_SMALL_CAPACITY]>::with_capacity(
                        requested_neighbor_count,
                    );
                    weights_out.resize(requested_neighbor_count, 0.0);
                    let rmse = compute_sample(
                        &source,
                        &hyperspectral,
                        &neighbor_rows,
                        &neighbor_distances,
                        query_row,
                        &valid_columns,
                        estimator,
                        reconstructed_out,
                        weights_out.as_mut_slice(),
                    )?;
                    *rmse_out = rmse;
                    Ok(())
                })
        })
        .map_err(PyValueError::new_err)?;
    } else {
        let mut weight_scratch_flat = vec![0.0; batch_size * requested_neighbor_count];
        py.allow_threads(|| {
            reconstructed_flat
                .par_chunks_mut(output_width)
                .zip(weight_scratch_flat.par_chunks_mut(requested_neighbor_count))
                .zip(rmse_flat.par_iter_mut())
                .enumerate()
                .try_for_each(|(batch_index, ((reconstructed_out, weights_out), rmse_out))| -> Result<(), String> {
                    let query_row_binding = query_rows.row(batch_index);
                    let query_row = query_row_binding
                        .as_slice()
                        .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                    let shortlist_binding = local_shortlists
                        .as_ref()
                        .map(|shortlists| shortlists.row(batch_index));
                    let shortlist_row = match shortlist_binding.as_ref() {
                        Some(row) => Some(
                            row.as_slice()
                                .ok_or_else(|| "local_candidate_indices rows must be contiguous".to_string())?,
                        ),
                        None => None,
                    };
                    let shortlist_distance_binding = local_shortlist_distances
                        .as_ref()
                        .map(|distances| distances.row(batch_index));
                    let shortlist_distance_row = match shortlist_distance_binding.as_ref() {
                        Some(row) => Some(
                            row.as_slice()
                                .ok_or_else(|| "local_candidate_distances rows must be contiguous".to_string())?,
                        ),
                        None => None,
                    };
                    let (neighbor_rows, neighbor_distances) = resolve_top_neighbors(
                        &candidates,
                        &row_indices,
                        query_row,
                        requested_neighbor_count,
                        shortlist_row,
                        shortlist_distance_row,
                        source.nrows(),
                    )?;
                    let rmse = compute_sample(
                        &source,
                        &hyperspectral,
                        &neighbor_rows,
                        &neighbor_distances,
                        query_row,
                        &valid_columns,
                        estimator,
                        reconstructed_out,
                        weights_out,
                    )?;
                    *rmse_out = rmse;
                    Ok(())
                })
        })
        .map_err(PyValueError::new_err)?;
    }

    let reconstructed_array = Array2::from_shape_vec((batch_size, output_width), reconstructed_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let rmse_array = Array1::from_vec(rmse_flat);
    Ok((
        reconstructed_array.into_pyarray_bound(py).unbind(),
        rmse_array.into_pyarray_bound(py).unbind(),
    ))
}

fn reconstruct_neighbor_spectra_batch_into_impl<T: FloatLike + Element>(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, T>,
    hyperspectral_rows: PyReadonlyArray2<'_, T>,
    candidate_matrix: PyReadonlyArray2<'_, T>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
    neighbor_estimator: &str,
    mut output_reconstructed: PyReadwriteArray2<'_, f64>,
    mut output_rmse: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let estimator = NeighborEstimator::parse(neighbor_estimator).map_err(PyValueError::new_err)?;
    let source = source_matrix.as_array();
    let hyperspectral = hyperspectral_rows.as_array();
    let candidates = candidate_matrix.as_array();
    let query_rows = query_values.as_array();
    let row_indices = candidate_row_indices.as_array();
    if candidates.nrows() != row_indices.len() {
        return Err(PyValueError::new_err(
            "candidate_matrix row count must match candidate_row_indices length",
        ));
    }
    if candidates.ncols() != query_rows.ncols() {
        return Err(PyValueError::new_err(
            "query_values column count must match candidate_matrix width",
        ));
    }
    if source.nrows() != hyperspectral.nrows() {
        return Err(PyValueError::new_err(
            "source_matrix and hyperspectral_rows must have the same row count",
        ));
    }
    let valid_columns: Vec<usize> = match valid_indices {
        Some(indices) => {
            let values = indices.as_array();
            let mut columns = Vec::with_capacity(values.len());
            for &value in values.iter() {
                if value < 0 {
                    return Err(PyValueError::new_err("valid_indices must be non-negative"));
                }
                let column = value as usize;
                if column >= source.ncols() {
                    return Err(PyValueError::new_err("valid_indices exceeded source band count"));
                }
                columns.push(column);
            }
            columns
        }
        None => (0..source.ncols()).collect(),
    };
    if query_rows.ncols() != valid_columns.len() {
        return Err(PyValueError::new_err(
            "query_values column count must match valid_indices or the full source band count",
        ));
    }
    let local_shortlists = local_candidate_indices.as_ref().map(|indices| indices.as_array());
    if let Some(shortlists) = &local_shortlists {
        if shortlists.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_indices row count must match query_values row count",
            ));
        }
    }
    let local_shortlist_distances = local_candidate_distances.as_ref().map(|distances| distances.as_array());
    if let Some(distances) = &local_shortlist_distances {
        if local_shortlists.is_none() {
            return Err(PyValueError::new_err(
                "local_candidate_distances requires local_candidate_indices",
            ));
        }
        if distances.nrows() != query_rows.nrows() {
            return Err(PyValueError::new_err(
                "local_candidate_distances row count must match query_values row count",
            ));
        }
        if distances.ncols()
            != local_shortlists
                .as_ref()
                .expect("validated local shortlist presence")
                .ncols()
        {
            return Err(PyValueError::new_err(
                "local_candidate_distances width must match local_candidate_indices width",
            ));
        }
    }

    let batch_size = query_rows.nrows();
    let requested_neighbor_count = if let Some(shortlists) = &local_shortlists {
        k.min(shortlists.ncols())
    } else {
        k.min(candidates.nrows())
    };
    if requested_neighbor_count == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }

    let output_width = hyperspectral.ncols();
    let mut reconstructed_output = output_reconstructed.as_array_mut();
    if reconstructed_output.nrows() != batch_size || reconstructed_output.ncols() != output_width {
        return Err(PyValueError::new_err(
            "out_reconstructed must match the batch size and hyperspectral output width",
        ));
    }
    let reconstructed_flat = reconstructed_output
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_reconstructed must be contiguous"))?;
    let mut rmse_output = output_rmse.as_array_mut();
    if rmse_output.len() != batch_size {
        return Err(PyValueError::new_err(
            "out_source_fit_rmse must match the batch size",
        ));
    }
    let rmse_flat = rmse_output
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_source_fit_rmse must be contiguous"))?;
    let use_exact_mean_narrow_fast_path = can_use_exact_mean_narrow_fast_path(
        estimator,
        output_width,
        local_shortlists.is_some(),
        local_shortlist_distances.is_some(),
    );
    if use_exact_mean_narrow_fast_path {
        py.allow_threads(|| {
            reconstructed_flat
                .par_chunks_mut(output_width)
                .zip(rmse_flat.par_iter_mut())
                .enumerate()
                .try_for_each(|(batch_index, (reconstructed_out, rmse_out))| -> Result<(), String> {
                    let query_row_binding = query_rows.row(batch_index);
                    let query_row = query_row_binding
                        .as_slice()
                        .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                    let shortlist_binding = local_shortlists
                        .as_ref()
                        .map(|shortlists| shortlists.row(batch_index));
                    let shortlist_row = shortlist_binding
                        .as_ref()
                        .map(|row| {
                            row.as_slice().ok_or_else(|| {
                                "local_candidate_indices rows must be contiguous".to_string()
                            })
                        })
                        .transpose()?;
                    let shortlist_distance_binding = local_shortlist_distances
                        .as_ref()
                        .map(|distances| distances.row(batch_index));
                    let shortlist_distance_row = shortlist_distance_binding
                        .as_ref()
                        .map(|row| {
                            row.as_slice().ok_or_else(|| {
                                "local_candidate_distances rows must be contiguous".to_string()
                            })
                        })
                        .transpose()?;
                    let fast_rmse = compute_sample_exact_shortlist_mean_narrow(
                        &candidates,
                        &row_indices,
                        &hyperspectral,
                        shortlist_row,
                        shortlist_distance_row,
                        requested_neighbor_count,
                        source.nrows(),
                        query_row,
                        reconstructed_out,
                    )?;
                    if let Some(rmse) = fast_rmse {
                        *rmse_out = rmse;
                        return Ok(());
                    }
                    let (neighbor_rows, neighbor_distances) = resolve_top_neighbors(
                        &candidates,
                        &row_indices,
                        query_row,
                        requested_neighbor_count,
                        shortlist_row,
                        shortlist_distance_row,
                        source.nrows(),
                    )?;
                    let mut weights_out = SmallVec::<[f64; INLINE_SMALL_CAPACITY]>::with_capacity(
                        requested_neighbor_count,
                    );
                    weights_out.resize(requested_neighbor_count, 0.0);
                    let rmse = compute_sample(
                        &source,
                        &hyperspectral,
                        &neighbor_rows,
                        &neighbor_distances,
                        query_row,
                        &valid_columns,
                        estimator,
                        reconstructed_out,
                        weights_out.as_mut_slice(),
                    )?;
                    *rmse_out = rmse;
                    Ok(())
                })
        })
        .map_err(PyValueError::new_err)?;
    } else {
        let mut weight_scratch_flat = vec![0.0; batch_size * requested_neighbor_count];
        py.allow_threads(|| {
            reconstructed_flat
                .par_chunks_mut(output_width)
                .zip(weight_scratch_flat.par_chunks_mut(requested_neighbor_count))
                .zip(rmse_flat.par_iter_mut())
                .enumerate()
                .try_for_each(|(batch_index, ((reconstructed_out, weights_out), rmse_out))| -> Result<(), String> {
                    let query_row_binding = query_rows.row(batch_index);
                    let query_row = query_row_binding
                        .as_slice()
                        .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                    let shortlist_binding = local_shortlists
                        .as_ref()
                        .map(|shortlists| shortlists.row(batch_index));
                    let shortlist_row = match shortlist_binding.as_ref() {
                        Some(row) => Some(
                            row.as_slice()
                                .ok_or_else(|| "local_candidate_indices rows must be contiguous".to_string())?,
                        ),
                        None => None,
                    };
                    let shortlist_distance_binding = local_shortlist_distances
                        .as_ref()
                        .map(|distances| distances.row(batch_index));
                    let shortlist_distance_row = match shortlist_distance_binding.as_ref() {
                        Some(row) => Some(
                            row.as_slice()
                                .ok_or_else(|| "local_candidate_distances rows must be contiguous".to_string())?,
                        ),
                        None => None,
                    };
                    let (neighbor_rows, neighbor_distances) = resolve_top_neighbors(
                        &candidates,
                        &row_indices,
                        query_row,
                        requested_neighbor_count,
                        shortlist_row,
                        shortlist_distance_row,
                        source.nrows(),
                    )?;
                    let rmse = compute_sample(
                        &source,
                        &hyperspectral,
                        &neighbor_rows,
                        &neighbor_distances,
                        query_row,
                        &valid_columns,
                        estimator,
                        reconstructed_out,
                        weights_out,
                    )?;
                    *rmse_out = rmse;
                    Ok(())
                })
        })
        .map_err(PyValueError::new_err)?;
    }
    Ok(())
}

fn combine_neighbor_spectra_batch_impl<T: FloatLike + Element>(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, T>,
    hyperspectral_rows: PyReadonlyArray2<'_, T>,
    neighbor_indices: PyReadonlyArray2<'_, i64>,
    neighbor_distances: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
    neighbor_estimator: &str,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let estimator = NeighborEstimator::parse(neighbor_estimator)
        .map_err(PyValueError::new_err)?;
    let source = source_matrix.as_array();
    let hyperspectral = hyperspectral_rows.as_array();
    let neighbor_index_rows = neighbor_indices.as_array();
    let neighbor_distance_rows = neighbor_distances.as_array();
    let query_rows = query_values.as_array();

    if neighbor_index_rows.shape() != neighbor_distance_rows.shape() {
        return Err(PyValueError::new_err(
            "neighbor_indices and neighbor_distances must have the same shape",
        ));
    }
    if source.nrows() != hyperspectral.nrows() {
        return Err(PyValueError::new_err(
            "source_matrix and hyperspectral_rows must have the same row count",
        ));
    }
    if neighbor_index_rows.nrows() != query_rows.nrows() {
        return Err(PyValueError::new_err(
            "query_values row count must match the batch size",
        ));
    }
    if neighbor_index_rows.ncols() == 0 {
        return Err(PyValueError::new_err("neighbor batches must include at least one neighbor"));
    }

    let valid_columns: Vec<usize> = match valid_indices {
        Some(indices) => {
            let values = indices.as_array();
            let mut columns = Vec::with_capacity(values.len());
            for &value in values.iter() {
                if value < 0 {
                    return Err(PyValueError::new_err("valid_indices must be non-negative"));
                }
                let column = value as usize;
                if column >= source.ncols() {
                    return Err(PyValueError::new_err("valid_indices exceeded source band count"));
                }
                columns.push(column);
            }
            columns
        }
        None => (0..source.ncols()).collect(),
    };
    if query_rows.ncols() != valid_columns.len() {
        return Err(PyValueError::new_err(
            "query_values column count must match valid_indices or the full source band count",
        ));
    }

    let batch_size = neighbor_index_rows.nrows();
    let neighbor_count = neighbor_index_rows.ncols();
    let output_width = hyperspectral.ncols();
    let mut reconstructed_flat = vec![0.0; batch_size * output_width];
    let mut weights_flat = vec![0.0; batch_size * neighbor_count];
    let mut rmse_flat = vec![0.0; batch_size];
    py.allow_threads(|| {
        reconstructed_flat
            .par_chunks_mut(output_width)
            .zip(weights_flat.par_chunks_mut(neighbor_count))
            .zip(rmse_flat.par_iter_mut())
            .enumerate()
            .try_for_each(|(batch_index, ((reconstructed_out, weights_out), rmse_out))| -> Result<(), String> {
                let neighbor_indices_binding = neighbor_index_rows.row(batch_index);
                let neighbor_indices_row = neighbor_indices_binding
                    .as_slice()
                    .ok_or_else(|| "neighbor_indices rows must be contiguous".to_string())?;
                let neighbor_distances_binding = neighbor_distance_rows.row(batch_index);
                let neighbor_distances_row = neighbor_distances_binding
                    .as_slice()
                    .ok_or_else(|| "neighbor_distances rows must be contiguous".to_string())?;
                let query_binding = query_rows.row(batch_index);
                let query_row = query_binding
                    .as_slice()
                    .ok_or_else(|| "query_values rows must be contiguous".to_string())?;
                let neighbor_rows = validate_neighbor_rows(&source, &hyperspectral, neighbor_indices_row)?;
                let rmse = compute_sample(
                    &source,
                    &hyperspectral,
                    &neighbor_rows,
                    neighbor_distances_row,
                    query_row,
                    &valid_columns,
                    estimator,
                    reconstructed_out,
                    weights_out,
                )?;
                *rmse_out = rmse;
                Ok(())
            })
    })
    .map_err(PyValueError::new_err)?;

    let reconstructed_array = Array2::from_shape_vec((batch_size, output_width), reconstructed_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let weights_array = Array2::from_shape_vec((batch_size, neighbor_count), weights_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let rmse_array = Array1::from_vec(rmse_flat);
    Ok((
        reconstructed_array.into_pyarray_bound(py).unbind(),
        weights_array.into_pyarray_bound(py).unbind(),
        rmse_array.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
fn assemble_full_spectrum_batch(
    py: Python<'_>,
    vnir: PyReadonlyArray2<'_, f64>,
    swir: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let vnir_rows = vnir.as_array();
    let swir_rows = swir.as_array();
    if vnir_rows.nrows() != swir_rows.nrows() {
        return Err(PyValueError::new_err(
            "vnir and swir batches must have the same row count",
        ));
    }
    if vnir_rows.ncols() != VNIR_WAVELENGTH_COUNT {
        return Err(PyValueError::new_err(format!(
            "vnir rows must have width {VNIR_WAVELENGTH_COUNT}",
        )));
    }
    if swir_rows.ncols() != SWIR_WAVELENGTH_COUNT {
        return Err(PyValueError::new_err(format!(
            "swir rows must have width {SWIR_WAVELENGTH_COUNT}",
        )));
    }

    let batch_size = vnir_rows.nrows();
    let mut full_flat = vec![0.0; batch_size * FULL_WAVELENGTH_COUNT];
    py.allow_threads(|| {
        full_flat
            .par_chunks_mut(FULL_WAVELENGTH_COUNT)
            .enumerate()
            .try_for_each(|(batch_index, output_row)| -> Result<(), String> {
                let vnir_binding = vnir_rows.row(batch_index);
                let vnir_row = vnir_binding
                    .as_slice()
                    .ok_or_else(|| "vnir rows must be contiguous".to_string())?;
                let swir_binding = swir_rows.row(batch_index);
                let swir_row = swir_binding
                    .as_slice()
                    .ok_or_else(|| "swir rows must be contiguous".to_string())?;
                assemble_full_spectrum_row_into(vnir_row, swir_row, output_row)
            })
    })
    .map_err(PyValueError::new_err)?;
    let full_array = Array2::from_shape_vec((batch_size, FULL_WAVELENGTH_COUNT), full_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(full_array.into_pyarray_bound(py).unbind())
}

fn finalize_target_sensor_batch_core(
    py: Python<'_>,
    vnir_rows: &ArrayView2<'_, f64>,
    swir_rows: &ArrayView2<'_, f64>,
    vnir_success_values: &ArrayView1<'_, bool>,
    swir_success_values: &ArrayView1<'_, bool>,
    vnir_projection: &ArrayView2<'_, f64>,
    swir_projection: &ArrayView2<'_, f64>,
    vnir_indices: &[usize],
    swir_indices: &[usize],
    output_width: usize,
    output_flat: &mut [f64],
    status_values: &mut [i32],
) -> Result<(), String> {
    let batch_size = vnir_rows.nrows();
    if output_flat.len() != batch_size * output_width {
        return Err("output_rows must match the batch size and target output width".to_string());
    }
    if status_values.len() != batch_size {
        return Err("status_codes must match the batch size".to_string());
    }
    py.allow_threads(|| {
        output_flat
            .par_chunks_mut(output_width)
            .zip(status_values.par_iter_mut())
            .enumerate()
            .for_each(|(batch_index, (output_row, status_value))| {
                output_row.fill(f64::NAN);
                let mut wrote_any = false;
                if vnir_success_values[batch_index] && !vnir_indices.is_empty() {
                    let vnir_row = vnir_rows.row(batch_index);
                    for (projection_column, &output_index) in vnir_indices.iter().enumerate() {
                        let mut value = 0.0;
                        for projection_row in 0..vnir_projection.nrows() {
                            value += vnir_row[projection_row] * vnir_projection[[projection_row, projection_column]];
                        }
                        output_row[output_index] = value;
                    }
                    wrote_any = true;
                }
                if swir_success_values[batch_index] && !swir_indices.is_empty() {
                    let swir_row = swir_rows.row(batch_index);
                    for (projection_column, &output_index) in swir_indices.iter().enumerate() {
                        let mut value = 0.0;
                        for projection_row in 0..swir_projection.nrows() {
                            value += swir_row[projection_row] * swir_projection[[projection_row, projection_column]];
                        }
                        output_row[output_index] = value;
                    }
                    wrote_any = true;
                }
                *status_value = if wrote_any {
                    TargetFinalizeStatus::Ok as i32
                } else {
                    TargetFinalizeStatus::NoTargetBands as i32
                };
            })
    });
    Ok(())
}

fn merge_target_sensor_segments_batch_core(
    py: Python<'_>,
    vnir_rows: &ArrayView2<'_, f64>,
    swir_rows: &ArrayView2<'_, f64>,
    vnir_success_values: &ArrayView1<'_, bool>,
    swir_success_values: &ArrayView1<'_, bool>,
    vnir_indices: &[usize],
    swir_indices: &[usize],
    output_width: usize,
    output_flat: &mut [f64],
    status_values: &mut [i32],
) -> Result<(), String> {
    let batch_size = vnir_rows.nrows();
    if swir_rows.nrows() != batch_size
        || vnir_success_values.len() != batch_size
        || swir_success_values.len() != batch_size
    {
        return Err("reconstructed batches and success masks must share the same batch size".to_string());
    }
    if vnir_rows.ncols() != vnir_indices.len() {
        return Err("vnir rows must already be projected to the VNIR target band count".to_string());
    }
    if swir_rows.ncols() != swir_indices.len() {
        return Err("swir rows must already be projected to the SWIR target band count".to_string());
    }
    if output_flat.len() != batch_size * output_width {
        return Err("output_rows must match the batch size and target output width".to_string());
    }
    if status_values.len() != batch_size {
        return Err("status_codes must match the batch size".to_string());
    }
    py.allow_threads(|| {
        output_flat
            .par_chunks_mut(output_width)
            .zip(status_values.par_iter_mut())
            .enumerate()
            .for_each(|(batch_index, (output_row, status_value))| {
                output_row.fill(f64::NAN);
                let mut wrote_any = false;
                if vnir_success_values[batch_index] && !vnir_indices.is_empty() {
                    let projected_row = vnir_rows.row(batch_index);
                    for (projection_column, &output_index) in vnir_indices.iter().enumerate() {
                        output_row[output_index] = projected_row[projection_column];
                    }
                    wrote_any = true;
                }
                if swir_success_values[batch_index] && !swir_indices.is_empty() {
                    let projected_row = swir_rows.row(batch_index);
                    for (projection_column, &output_index) in swir_indices.iter().enumerate() {
                        output_row[output_index] = projected_row[projection_column];
                    }
                    wrote_any = true;
                }
                *status_value = if wrote_any {
                    TargetFinalizeStatus::Ok as i32
                } else {
                    TargetFinalizeStatus::NoTargetBands as i32
                };
            })
    });
    Ok(())
}

#[pyfunction(signature = (
    vnir_reconstructed,
    swir_reconstructed,
    vnir_success,
    swir_success,
    vnir_response_matrix,
    swir_response_matrix,
    vnir_output_indices,
    swir_output_indices,
    output_width
))]
fn finalize_target_sensor_batch(
    py: Python<'_>,
    vnir_reconstructed: PyReadonlyArray2<'_, f64>,
    swir_reconstructed: PyReadonlyArray2<'_, f64>,
    vnir_success: PyReadonlyArray1<'_, bool>,
    swir_success: PyReadonlyArray1<'_, bool>,
    vnir_response_matrix: PyReadonlyArray2<'_, f64>,
    swir_response_matrix: PyReadonlyArray2<'_, f64>,
    vnir_output_indices: PyReadonlyArray1<'_, i64>,
    swir_output_indices: PyReadonlyArray1<'_, i64>,
    output_width: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<i32>>)> {
    let vnir_rows = vnir_reconstructed.as_array();
    let swir_rows = swir_reconstructed.as_array();
    let vnir_success_values = vnir_success.as_array();
    let swir_success_values = swir_success.as_array();
    let vnir_projection = vnir_response_matrix.as_array();
    let swir_projection = swir_response_matrix.as_array();

    let batch_size = vnir_rows.nrows();
    if swir_rows.nrows() != batch_size
        || vnir_success_values.len() != batch_size
        || swir_success_values.len() != batch_size
    {
        return Err(PyValueError::new_err(
            "reconstructed batches and success masks must share the same batch size",
        ));
    }
    if vnir_projection.nrows() != vnir_rows.ncols() {
        return Err(PyValueError::new_err(
            "vnir_response_matrix row count must match the VNIR reconstructed width",
        ));
    }
    if swir_projection.nrows() != swir_rows.ncols() {
        return Err(PyValueError::new_err(
            "swir_response_matrix row count must match the SWIR reconstructed width",
        ));
    }

    let vnir_indices = validate_output_indices(
        &vnir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let swir_indices = validate_output_indices(
        &swir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    if vnir_projection.ncols() != vnir_indices.len() {
        return Err(PyValueError::new_err(
            "vnir_response_matrix column count must match vnir_output_indices",
        ));
    }
    if swir_projection.ncols() != swir_indices.len() {
        return Err(PyValueError::new_err(
            "swir_response_matrix column count must match swir_output_indices",
        ));
    }

    let mut output_flat = vec![f64::NAN; batch_size * output_width];
    let mut status_values = vec![TargetFinalizeStatus::Ok as i32; batch_size];
    finalize_target_sensor_batch_core(
        py,
        &vnir_rows,
        &swir_rows,
        &vnir_success_values,
        &swir_success_values,
        &vnir_projection,
        &swir_projection,
        &vnir_indices,
        &swir_indices,
        output_width,
        &mut output_flat,
        &mut status_values,
    )
    .map_err(PyValueError::new_err)?;

    let output_array = Array2::from_shape_vec((batch_size, output_width), output_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let status_array = Array1::from_vec(status_values);
    Ok((
        output_array.into_pyarray_bound(py).unbind(),
        status_array.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction(signature = (
    vnir_reconstructed,
    swir_reconstructed,
    vnir_success,
    swir_success,
    vnir_response_matrix,
    swir_response_matrix,
    vnir_output_indices,
    swir_output_indices,
    output_width,
    out_output_rows,
    out_status_codes
))]
fn finalize_target_sensor_batch_into(
    py: Python<'_>,
    vnir_reconstructed: PyReadonlyArray2<'_, f64>,
    swir_reconstructed: PyReadonlyArray2<'_, f64>,
    vnir_success: PyReadonlyArray1<'_, bool>,
    swir_success: PyReadonlyArray1<'_, bool>,
    vnir_response_matrix: PyReadonlyArray2<'_, f64>,
    swir_response_matrix: PyReadonlyArray2<'_, f64>,
    vnir_output_indices: PyReadonlyArray1<'_, i64>,
    swir_output_indices: PyReadonlyArray1<'_, i64>,
    output_width: usize,
    mut out_output_rows: PyReadwriteArray2<'_, f64>,
    mut out_status_codes: PyReadwriteArray1<'_, i32>,
) -> PyResult<()> {
    let vnir_rows = vnir_reconstructed.as_array();
    let swir_rows = swir_reconstructed.as_array();
    let vnir_success_values = vnir_success.as_array();
    let swir_success_values = swir_success.as_array();
    let vnir_projection = vnir_response_matrix.as_array();
    let swir_projection = swir_response_matrix.as_array();
    let batch_size = vnir_rows.nrows();
    if swir_rows.nrows() != batch_size
        || vnir_success_values.len() != batch_size
        || swir_success_values.len() != batch_size
    {
        return Err(PyValueError::new_err(
            "reconstructed batches and success masks must share the same batch size",
        ));
    }
    if vnir_projection.nrows() != vnir_rows.ncols() {
        return Err(PyValueError::new_err(
            "vnir_response_matrix row count must match the VNIR reconstructed width",
        ));
    }
    if swir_projection.nrows() != swir_rows.ncols() {
        return Err(PyValueError::new_err(
            "swir_response_matrix row count must match the SWIR reconstructed width",
        ));
    }
    let vnir_indices = validate_output_indices(
        &vnir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let swir_indices = validate_output_indices(
        &swir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    if vnir_projection.ncols() != vnir_indices.len() {
        return Err(PyValueError::new_err(
            "vnir_response_matrix column count must match vnir_output_indices",
        ));
    }
    if swir_projection.ncols() != swir_indices.len() {
        return Err(PyValueError::new_err(
            "swir_response_matrix column count must match swir_output_indices",
        ));
    }
    let mut output_rows = out_output_rows.as_array_mut();
    if output_rows.nrows() != batch_size || output_rows.ncols() != output_width {
        return Err(PyValueError::new_err(
            "out_output_rows must match the batch size and target output width",
        ));
    }
    let output_flat = output_rows
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_output_rows must be contiguous"))?;
    let mut status_codes = out_status_codes.as_array_mut();
    if status_codes.len() != batch_size {
        return Err(PyValueError::new_err(
            "out_status_codes must match the batch size",
        ));
    }
    let status_flat = status_codes
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_status_codes must be contiguous"))?;
    finalize_target_sensor_batch_core(
        py,
        &vnir_rows,
        &swir_rows,
        &vnir_success_values,
        &swir_success_values,
        &vnir_projection,
        &swir_projection,
        &vnir_indices,
        &swir_indices,
        output_width,
        output_flat,
        status_flat,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction(signature = (
    vnir_rows,
    swir_rows,
    vnir_success,
    swir_success,
    vnir_output_indices,
    swir_output_indices,
    output_width
))]
fn merge_target_sensor_segments_batch(
    py: Python<'_>,
    vnir_rows: PyReadonlyArray2<'_, f64>,
    swir_rows: PyReadonlyArray2<'_, f64>,
    vnir_success: PyReadonlyArray1<'_, bool>,
    swir_success: PyReadonlyArray1<'_, bool>,
    vnir_output_indices: PyReadonlyArray1<'_, i64>,
    swir_output_indices: PyReadonlyArray1<'_, i64>,
    output_width: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<i32>>)> {
    let vnir_rows = vnir_rows.as_array();
    let swir_rows = swir_rows.as_array();
    let vnir_success_values = vnir_success.as_array();
    let swir_success_values = swir_success.as_array();
    let batch_size = vnir_rows.nrows();
    let vnir_indices = validate_output_indices(
        &vnir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let swir_indices = validate_output_indices(
        &swir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let mut output_flat = vec![f64::NAN; batch_size * output_width];
    let mut status_values = vec![TargetFinalizeStatus::Ok as i32; batch_size];
    merge_target_sensor_segments_batch_core(
        py,
        &vnir_rows,
        &swir_rows,
        &vnir_success_values,
        &swir_success_values,
        &vnir_indices,
        &swir_indices,
        output_width,
        &mut output_flat,
        &mut status_values,
    )
    .map_err(PyValueError::new_err)?;
    let output_array = Array2::from_shape_vec((batch_size, output_width), output_flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let status_array = Array1::from_vec(status_values);
    Ok((
        output_array.into_pyarray_bound(py).unbind(),
        status_array.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction(signature = (
    vnir_rows,
    swir_rows,
    vnir_success,
    swir_success,
    vnir_output_indices,
    swir_output_indices,
    output_width,
    out_output_rows,
    out_status_codes
))]
fn merge_target_sensor_segments_batch_into(
    py: Python<'_>,
    vnir_rows: PyReadonlyArray2<'_, f64>,
    swir_rows: PyReadonlyArray2<'_, f64>,
    vnir_success: PyReadonlyArray1<'_, bool>,
    swir_success: PyReadonlyArray1<'_, bool>,
    vnir_output_indices: PyReadonlyArray1<'_, i64>,
    swir_output_indices: PyReadonlyArray1<'_, i64>,
    output_width: usize,
    mut out_output_rows: PyReadwriteArray2<'_, f64>,
    mut out_status_codes: PyReadwriteArray1<'_, i32>,
) -> PyResult<()> {
    let vnir_rows = vnir_rows.as_array();
    let swir_rows = swir_rows.as_array();
    let vnir_success_values = vnir_success.as_array();
    let swir_success_values = swir_success.as_array();
    let batch_size = vnir_rows.nrows();
    let vnir_indices = validate_output_indices(
        &vnir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let swir_indices = validate_output_indices(
        &swir_output_indices.as_array().iter().copied().collect::<Vec<_>>(),
        output_width,
    )
    .map_err(PyValueError::new_err)?;
    let mut output_rows = out_output_rows.as_array_mut();
    if output_rows.nrows() != batch_size || output_rows.ncols() != output_width {
        return Err(PyValueError::new_err(
            "out_output_rows must match the batch size and target output width",
        ));
    }
    let output_flat = output_rows
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_output_rows must be contiguous"))?;
    let mut status_codes = out_status_codes.as_array_mut();
    if status_codes.len() != batch_size {
        return Err(PyValueError::new_err(
            "out_status_codes must match the batch size",
        ));
    }
    let status_flat = status_codes
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("out_status_codes must be contiguous"))?;
    merge_target_sensor_segments_batch_core(
        py,
        &vnir_rows,
        &swir_rows,
        &vnir_success_values,
        &swir_success_values,
        &vnir_indices,
        &swir_indices,
        output_width,
        output_flat,
        status_flat,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction(signature = (source_matrix, hyperspectral_rows, neighbor_indices, neighbor_distances, query_values, neighbor_estimator, valid_indices=None))]
fn combine_neighbor_spectra_batch_f32(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f32>,
    hyperspectral_rows: PyReadonlyArray2<'_, f32>,
    neighbor_indices: PyReadonlyArray2<'_, i64>,
    neighbor_distances: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    neighbor_estimator: &str,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    combine_neighbor_spectra_batch_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        neighbor_indices,
        neighbor_distances,
        query_values,
        valid_indices,
        neighbor_estimator,
    )
}

#[pyfunction(signature = (source_matrix, hyperspectral_rows, neighbor_indices, neighbor_distances, query_values, neighbor_estimator, valid_indices=None))]
fn combine_neighbor_spectra_batch_f64(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f64>,
    hyperspectral_rows: PyReadonlyArray2<'_, f64>,
    neighbor_indices: PyReadonlyArray2<'_, i64>,
    neighbor_distances: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    neighbor_estimator: &str,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    combine_neighbor_spectra_batch_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        neighbor_indices,
        neighbor_distances,
        query_values,
        valid_indices,
        neighbor_estimator,
    )
}

#[pyfunction(signature = (candidate_matrix, query_values, candidate_row_indices, k, local_candidate_indices=None, local_candidate_distances=None))]
fn refine_neighbor_rows_batch_f32(
    py: Python<'_>,
    candidate_matrix: PyReadonlyArray2<'_, f32>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f64>>)> {
    refine_neighbor_rows_batch_impl(
        py,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
    )
}

#[pyfunction(signature = (candidate_matrix, query_values, candidate_row_indices, k, local_candidate_indices=None, local_candidate_distances=None))]
fn refine_neighbor_rows_batch_f64(
    py: Python<'_>,
    candidate_matrix: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f64>>)> {
    refine_neighbor_rows_batch_impl(
        py,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
    )
}

#[pyfunction(signature = (
    source_matrix,
    hyperspectral_rows,
    candidate_matrix,
    query_values,
    candidate_row_indices,
    k,
    neighbor_estimator,
    local_candidate_indices=None,
    local_candidate_distances=None,
    valid_indices=None
))]
fn reconstruct_neighbor_spectra_batch_f32(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f32>,
    hyperspectral_rows: PyReadonlyArray2<'_, f32>,
    candidate_matrix: PyReadonlyArray2<'_, f32>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    neighbor_estimator: &str,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    reconstruct_neighbor_spectra_batch_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
        valid_indices,
        neighbor_estimator,
    )
}

#[pyfunction(signature = (
    source_matrix,
    hyperspectral_rows,
    candidate_matrix,
    query_values,
    candidate_row_indices,
    k,
    neighbor_estimator,
    local_candidate_indices=None,
    local_candidate_distances=None,
    valid_indices=None
))]
fn reconstruct_neighbor_spectra_batch_f64(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f64>,
    hyperspectral_rows: PyReadonlyArray2<'_, f64>,
    candidate_matrix: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    neighbor_estimator: &str,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    reconstruct_neighbor_spectra_batch_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
        valid_indices,
        neighbor_estimator,
    )
}

#[pyfunction(signature = (
    source_matrix,
    hyperspectral_rows,
    candidate_matrix,
    query_values,
    candidate_row_indices,
    k,
    neighbor_estimator,
    out_reconstructed,
    out_source_fit_rmse,
    local_candidate_indices=None,
    local_candidate_distances=None,
    valid_indices=None
))]
fn reconstruct_neighbor_spectra_batch_into_f32(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f32>,
    hyperspectral_rows: PyReadonlyArray2<'_, f32>,
    candidate_matrix: PyReadonlyArray2<'_, f32>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    neighbor_estimator: &str,
    out_reconstructed: PyReadwriteArray2<'_, f64>,
    out_source_fit_rmse: PyReadwriteArray1<'_, f64>,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<()> {
    reconstruct_neighbor_spectra_batch_into_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
        valid_indices,
        neighbor_estimator,
        out_reconstructed,
        out_source_fit_rmse,
    )
}

#[pyfunction(signature = (
    source_matrix,
    hyperspectral_rows,
    candidate_matrix,
    query_values,
    candidate_row_indices,
    k,
    neighbor_estimator,
    out_reconstructed,
    out_source_fit_rmse,
    local_candidate_indices=None,
    local_candidate_distances=None,
    valid_indices=None
))]
fn reconstruct_neighbor_spectra_batch_into_f64(
    py: Python<'_>,
    source_matrix: PyReadonlyArray2<'_, f64>,
    hyperspectral_rows: PyReadonlyArray2<'_, f64>,
    candidate_matrix: PyReadonlyArray2<'_, f64>,
    query_values: PyReadonlyArray2<'_, f64>,
    candidate_row_indices: PyReadonlyArray1<'_, i64>,
    k: usize,
    neighbor_estimator: &str,
    out_reconstructed: PyReadwriteArray2<'_, f64>,
    out_source_fit_rmse: PyReadwriteArray1<'_, f64>,
    local_candidate_indices: Option<PyReadonlyArray2<'_, i64>>,
    local_candidate_distances: Option<PyReadonlyArray2<'_, f64>>,
    valid_indices: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<()> {
    reconstruct_neighbor_spectra_batch_into_impl(
        py,
        source_matrix,
        hyperspectral_rows,
        candidate_matrix,
        query_values,
        candidate_row_indices,
        k,
        local_candidate_indices,
        local_candidate_distances,
        valid_indices,
        neighbor_estimator,
        out_reconstructed,
        out_source_fit_rmse,
    )
}

#[pymodule]
fn _mapping_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction_bound!(assemble_full_spectrum_batch, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(finalize_target_sensor_batch, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(finalize_target_sensor_batch_into, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(merge_target_sensor_segments_batch, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(merge_target_sensor_segments_batch_into, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(refine_neighbor_rows_batch_f32, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(refine_neighbor_rows_batch_f64, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(combine_neighbor_spectra_batch_f32, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(combine_neighbor_spectra_batch_f64, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(reconstruct_neighbor_spectra_batch_f32, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(reconstruct_neighbor_spectra_batch_f64, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(reconstruct_neighbor_spectra_batch_into_f32, module)?)?;
    module.add_function(pyo3::wrap_pyfunction_bound!(reconstruct_neighbor_spectra_batch_into_f64, module)?)?;
    Ok(())
}

from __future__ import annotations

from typing import Any

import numpy as np


def _load_native_module() -> Any:
    try:
        from . import _mapping_rust as native
    except ImportError as exc:
        raise RuntimeError(
            "Rust mapping backend is required but could not be imported. Build the spectral_library._mapping_rust extension first."
        ) from exc
    return native


_NATIVE_MODULE = _load_native_module()


def rust_accel_available() -> bool:
    return True


def _validated_native_output_buffer(*, buffer: np.ndarray | None, name: str) -> np.ndarray | None:
    if buffer is None:
        return None
    if not isinstance(buffer, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array when provided.")
    if buffer.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64.")
    if not buffer.flags.writeable:
        raise ValueError(f"{name} must be writeable.")
    if not buffer.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return buffer


def assemble_full_spectrum_batch(
    *,
    vnir: np.ndarray,
    swir: np.ndarray,
) -> np.ndarray:
    return _NATIVE_MODULE.assemble_full_spectrum_batch(
        np.ascontiguousarray(vnir, dtype=np.float64),
        np.ascontiguousarray(swir, dtype=np.float64),
    )


def finalize_target_sensor_batch(
    *,
    vnir_reconstructed: np.ndarray,
    swir_reconstructed: np.ndarray,
    vnir_success: np.ndarray,
    swir_success: np.ndarray,
    vnir_response_matrix: np.ndarray,
    swir_response_matrix: np.ndarray,
    vnir_output_indices: np.ndarray,
    swir_output_indices: np.ndarray,
    output_width: int,
    out_output_rows: np.ndarray | None = None,
    out_status_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_out_output_rows = _validated_native_output_buffer(
        buffer=out_output_rows,
        name="out_output_rows",
    )
    resolved_out_status_codes = None
    if out_status_codes is not None:
        if not isinstance(out_status_codes, np.ndarray):
            raise ValueError("out_status_codes must be a NumPy array when provided.")
        if out_status_codes.dtype != np.int32:
            raise ValueError("out_status_codes must have dtype int32.")
        if not out_status_codes.flags.writeable:
            raise ValueError("out_status_codes must be writeable.")
        if not out_status_codes.flags.c_contiguous:
            raise ValueError("out_status_codes must be C-contiguous.")
        resolved_out_status_codes = out_status_codes

    resolved_vnir = np.ascontiguousarray(vnir_reconstructed, dtype=np.float64)
    resolved_swir = np.ascontiguousarray(swir_reconstructed, dtype=np.float64)
    resolved_vnir_success = np.ascontiguousarray(vnir_success, dtype=bool)
    resolved_swir_success = np.ascontiguousarray(swir_success, dtype=bool)
    resolved_vnir_projection = np.ascontiguousarray(vnir_response_matrix, dtype=np.float64)
    resolved_swir_projection = np.ascontiguousarray(swir_response_matrix, dtype=np.float64)
    resolved_vnir_indices = np.ascontiguousarray(vnir_output_indices, dtype=np.int64)
    resolved_swir_indices = np.ascontiguousarray(swir_output_indices, dtype=np.int64)

    if resolved_out_output_rows is not None or resolved_out_status_codes is not None:
        if resolved_out_output_rows is None or resolved_out_status_codes is None:
            raise ValueError("out_output_rows and out_status_codes must be provided together.")
        _NATIVE_MODULE.finalize_target_sensor_batch_into(
            resolved_vnir,
            resolved_swir,
            resolved_vnir_success,
            resolved_swir_success,
            resolved_vnir_projection,
            resolved_swir_projection,
            resolved_vnir_indices,
            resolved_swir_indices,
            int(output_width),
            resolved_out_output_rows,
            resolved_out_status_codes,
        )
        return resolved_out_output_rows, resolved_out_status_codes

    return _NATIVE_MODULE.finalize_target_sensor_batch(
        resolved_vnir,
        resolved_swir,
        resolved_vnir_success,
        resolved_swir_success,
        resolved_vnir_projection,
        resolved_swir_projection,
        resolved_vnir_indices,
        resolved_swir_indices,
        int(output_width),
    )


def merge_target_sensor_segments_batch(
    *,
    vnir_rows: np.ndarray,
    swir_rows: np.ndarray,
    vnir_success: np.ndarray,
    swir_success: np.ndarray,
    vnir_output_indices: np.ndarray,
    swir_output_indices: np.ndarray,
    output_width: int,
    out_output_rows: np.ndarray | None = None,
    out_status_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_out_output_rows = _validated_native_output_buffer(
        buffer=out_output_rows,
        name="out_output_rows",
    )
    resolved_out_status_codes = None
    if out_status_codes is not None:
        if not isinstance(out_status_codes, np.ndarray):
            raise ValueError("out_status_codes must be a NumPy array when provided.")
        if out_status_codes.dtype != np.int32:
            raise ValueError("out_status_codes must have dtype int32.")
        if not out_status_codes.flags.writeable:
            raise ValueError("out_status_codes must be writeable.")
        if not out_status_codes.flags.c_contiguous:
            raise ValueError("out_status_codes must be C-contiguous.")
        resolved_out_status_codes = out_status_codes

    resolved_vnir_rows = np.ascontiguousarray(vnir_rows, dtype=np.float64)
    resolved_swir_rows = np.ascontiguousarray(swir_rows, dtype=np.float64)
    resolved_vnir_success = np.ascontiguousarray(vnir_success, dtype=bool)
    resolved_swir_success = np.ascontiguousarray(swir_success, dtype=bool)
    resolved_vnir_indices = np.ascontiguousarray(vnir_output_indices, dtype=np.int64)
    resolved_swir_indices = np.ascontiguousarray(swir_output_indices, dtype=np.int64)

    if resolved_out_output_rows is not None or resolved_out_status_codes is not None:
        if resolved_out_output_rows is None or resolved_out_status_codes is None:
            raise ValueError("out_output_rows and out_status_codes must be provided together.")
        _NATIVE_MODULE.merge_target_sensor_segments_batch_into(
            resolved_vnir_rows,
            resolved_swir_rows,
            resolved_vnir_success,
            resolved_swir_success,
            resolved_vnir_indices,
            resolved_swir_indices,
            int(output_width),
            resolved_out_output_rows,
            resolved_out_status_codes,
        )
        return resolved_out_output_rows, resolved_out_status_codes

    return _NATIVE_MODULE.merge_target_sensor_segments_batch(
        resolved_vnir_rows,
        resolved_swir_rows,
        resolved_vnir_success,
        resolved_swir_success,
        resolved_vnir_indices,
        resolved_swir_indices,
        int(output_width),
    )


def refine_neighbor_rows_batch(
    *,
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    candidate_row_indices: np.ndarray,
    local_candidate_indices: np.ndarray | None,
    local_candidate_distances: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.ascontiguousarray(candidate_matrix)
    queries = np.ascontiguousarray(query_values, dtype=np.float64)
    row_indices = np.ascontiguousarray(candidate_row_indices, dtype=np.int64)
    local_indices = None if local_candidate_indices is None else np.ascontiguousarray(local_candidate_indices, dtype=np.int64)
    local_distances = (
        None if local_candidate_distances is None else np.ascontiguousarray(local_candidate_distances, dtype=np.float64)
    )

    if candidates.dtype == np.float32:
        return _NATIVE_MODULE.refine_neighbor_rows_batch_f32(
            candidates,
            queries,
            row_indices,
            int(k),
            local_indices,
            local_distances,
        )

    return _NATIVE_MODULE.refine_neighbor_rows_batch_f64(
        np.ascontiguousarray(candidates, dtype=np.float64),
        queries,
        row_indices,
        int(k),
        local_indices,
        local_distances,
    )


def refine_and_combine_neighbor_spectra_batch(
    *,
    source_matrix: np.ndarray,
    hyperspectral_rows: np.ndarray,
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    candidate_row_indices: np.ndarray,
    k: int,
    neighbor_estimator: str,
    local_candidate_indices: np.ndarray | None,
    local_candidate_distances: np.ndarray | None,
    valid_indices: np.ndarray | None,
    out_reconstructed: np.ndarray | None = None,
    out_source_fit_rmse: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.ascontiguousarray(source_matrix)
    hyperspectral = np.ascontiguousarray(hyperspectral_rows)
    candidates = np.ascontiguousarray(candidate_matrix)
    queries = np.ascontiguousarray(query_values, dtype=np.float64)
    row_indices = np.ascontiguousarray(candidate_row_indices, dtype=np.int64)
    local_indices = None if local_candidate_indices is None else np.ascontiguousarray(local_candidate_indices, dtype=np.int64)
    local_distances = (
        None if local_candidate_distances is None else np.ascontiguousarray(local_candidate_distances, dtype=np.float64)
    )
    resolved_valid_indices = None if valid_indices is None else np.ascontiguousarray(valid_indices, dtype=np.int64)
    resolved_out_reconstructed = _validated_native_output_buffer(
        buffer=out_reconstructed,
        name="out_reconstructed",
    )
    resolved_out_source_fit_rmse = _validated_native_output_buffer(
        buffer=out_source_fit_rmse,
        name="out_source_fit_rmse",
    )

    if resolved_out_reconstructed is not None or resolved_out_source_fit_rmse is not None:
        if resolved_out_reconstructed is None or resolved_out_source_fit_rmse is None:
            raise ValueError("out_reconstructed and out_source_fit_rmse must be provided together.")
        if source.dtype == np.float32 and hyperspectral.dtype == np.float32 and candidates.dtype == np.float32:
            _NATIVE_MODULE.reconstruct_neighbor_spectra_batch_into_f32(
                source,
                hyperspectral,
                candidates,
                queries,
                row_indices,
                int(k),
                str(neighbor_estimator),
                resolved_out_reconstructed,
                resolved_out_source_fit_rmse,
                local_indices,
                local_distances,
                resolved_valid_indices,
            )
        else:
            _NATIVE_MODULE.reconstruct_neighbor_spectra_batch_into_f64(
                np.ascontiguousarray(source, dtype=np.float64),
                np.ascontiguousarray(hyperspectral, dtype=np.float64),
                np.ascontiguousarray(candidates, dtype=np.float64),
                queries,
                row_indices,
                int(k),
                str(neighbor_estimator),
                resolved_out_reconstructed,
                resolved_out_source_fit_rmse,
                local_indices,
                local_distances,
                resolved_valid_indices,
            )
        return resolved_out_reconstructed, resolved_out_source_fit_rmse

    if source.dtype == np.float32 and hyperspectral.dtype == np.float32 and candidates.dtype == np.float32:
        return _NATIVE_MODULE.reconstruct_neighbor_spectra_batch_f32(
            source,
            hyperspectral,
            candidates,
            queries,
            row_indices,
            int(k),
            str(neighbor_estimator),
            local_indices,
            local_distances,
            resolved_valid_indices,
        )

    return _NATIVE_MODULE.reconstruct_neighbor_spectra_batch_f64(
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(hyperspectral, dtype=np.float64),
        np.ascontiguousarray(candidates, dtype=np.float64),
        queries,
        row_indices,
        int(k),
        str(neighbor_estimator),
        local_indices,
        local_distances,
        resolved_valid_indices,
    )


def reconstruct_neighbor_spectra_batch(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    return refine_and_combine_neighbor_spectra_batch(**kwargs)


def combine_neighbor_spectra_batch(
    *,
    source_matrix: np.ndarray,
    hyperspectral_rows: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    query_values: np.ndarray,
    valid_indices: np.ndarray | None,
    neighbor_estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.ascontiguousarray(source_matrix)
    hyperspectral = np.ascontiguousarray(hyperspectral_rows)
    neighbors = np.ascontiguousarray(neighbor_indices, dtype=np.int64)
    distances = np.ascontiguousarray(neighbor_distances, dtype=np.float64)
    queries = np.ascontiguousarray(query_values, dtype=np.float64)
    resolved_valid_indices = None if valid_indices is None else np.ascontiguousarray(valid_indices, dtype=np.int64)

    if source.dtype == np.float32 and hyperspectral.dtype == np.float32:
        return _NATIVE_MODULE.combine_neighbor_spectra_batch_f32(
            source,
            hyperspectral,
            neighbors,
            distances,
            queries,
            str(neighbor_estimator),
            resolved_valid_indices,
        )

    return _NATIVE_MODULE.combine_neighbor_spectra_batch_f64(
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(hyperspectral, dtype=np.float64),
        neighbors,
        distances,
        queries,
        str(neighbor_estimator),
        resolved_valid_indices,
    )

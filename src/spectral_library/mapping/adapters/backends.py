"""KNN and output-store adapters for the mapping package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import duckdb
import numpy as np

from ... import _rustaccel
from ..._version import __version__

from ..engine import core as _core

globals().update({name: getattr(_core, name) for name in dir(_core) if not name.startswith("__")})

del _core

@dataclass
class _ZarrBatchExport:
    root: Any
    reflectance_dataset: Any
    source_fit_rmse_dataset: Any
    sample_id_dataset: Any
    output_columns: tuple[str, ...]
    output_width: int
    chunk_size: int

@dataclass
class _ScipyCkdtreeCacheEntry:
    data: np.ndarray
    index: Any

def _ordered_neighbor_rows(
    distances: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest candidate rows ordered deterministically by distance."""

    neighbor_count = min(int(k), int(candidate_row_indices.size))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})
    if neighbor_count == candidate_row_indices.size:
        local_top = np.arange(candidate_row_indices.size)
    else:
        local_top = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    ordered_local = local_top[np.lexsort((candidate_row_indices[local_top], distances[local_top]))]
    return candidate_row_indices[ordered_local], np.asarray(distances[ordered_local], dtype=np.float64)

def _ordered_neighbor_rows_from_local_distances(
    local_candidate_indices: np.ndarray,
    local_distances: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    query_width: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve exact backend distances without rescoring the candidate matrix."""

    if query_width <= 0:
        raise MappingInputError("At least one valid source band is required for mapping.")
    local_indices = np.asarray(local_candidate_indices, dtype=np.int64)
    resolved_distances = np.asarray(local_distances, dtype=np.float64)
    valid = (local_indices >= 0) & (local_indices < int(candidate_row_indices.size)) & np.isfinite(resolved_distances)
    local_indices = local_indices[valid]
    resolved_distances = resolved_distances[valid]
    if local_indices.size == 0:
        raise PreparedLibraryValidationError("Neighbor search backend returned no candidate rows.")
    unique_local_indices, unique_positions = np.unique(local_indices, return_index=True)
    exact_rmse = resolved_distances[unique_positions] / math.sqrt(float(query_width))
    return _ordered_neighbor_rows(exact_rmse, candidate_row_indices[unique_local_indices], k=k)

def _ordered_neighbor_rows_batch_from_local_distances(
    local_candidate_indices: np.ndarray,
    local_distance_matrix: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    query_width: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch variant of ``_ordered_neighbor_rows_from_local_distances``."""

    local_indices = np.asarray(local_candidate_indices, dtype=np.int64)
    local_distances = np.asarray(local_distance_matrix, dtype=np.float64)
    if local_indices.shape != local_distances.shape:
        raise MappingInputError(
            "Exact local neighbor indices and distances must share the same shape.",
            context={
                "local_index_shape": list(local_indices.shape),
                "local_distance_shape": list(local_distances.shape),
            },
        )
    if local_indices.ndim != 2:
        raise MappingInputError(
            "Exact local neighbor indices must be two-dimensional for batched mapping.",
            context={"local_index_shape": list(local_indices.shape)},
        )
    neighbor_indices_rows: list[np.ndarray] = []
    neighbor_distance_rows: list[np.ndarray] = []
    for row_indices, row_distances in zip(local_indices, local_distances):
        resolved_indices, resolved_distances = _ordered_neighbor_rows_from_local_distances(
            row_indices,
            row_distances,
            candidate_row_indices,
            query_width=query_width,
            k=k,
        )
        neighbor_indices_rows.append(np.asarray(resolved_indices, dtype=np.int64))
        neighbor_distance_rows.append(np.asarray(resolved_distances, dtype=np.float64))
    return (
        np.asarray(neighbor_indices_rows, dtype=np.int64),
        np.asarray(neighbor_distance_rows, dtype=np.float64),
    )

def _load_ckdtree_class() -> type[Any]:
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'scipy_ckdtree backend requires scipy. Install it with `{KNN_BACKEND_INSTALL_HINTS["scipy_ckdtree"]}`.',
            context={"knn_backend": "scipy_ckdtree"},
        ) from exc
    return cKDTree

def _scipy_ckdtree_workers() -> int:
    raw_value = (os.environ.get("SPECTRAL_LIBRARY_SCIPY_WORKERS") or "").strip()
    if not raw_value:
        return 1
    try:
        workers = int(raw_value)
    except ValueError as exc:
        raise MappingInputError(
            "SPECTRAL_LIBRARY_SCIPY_WORKERS must be an integer when set.",
            context={"env_var": "SPECTRAL_LIBRARY_SCIPY_WORKERS", "value": raw_value},
        ) from exc
    if workers == 0 or workers < -1:
        raise MappingInputError(
            "SPECTRAL_LIBRARY_SCIPY_WORKERS must be -1 or a positive integer.",
            context={"env_var": "SPECTRAL_LIBRARY_SCIPY_WORKERS", "value": raw_value},
        )
    return workers

def _load_faiss_module() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'faiss backend requires faiss-cpu. Install it with `{KNN_BACKEND_INSTALL_HINTS["faiss"]}`.',
            context={"knn_backend": "faiss"},
        ) from exc
    return faiss

def _load_pynndescent_class() -> type[Any]:
    try:
        from pynndescent import NNDescent  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'pynndescent backend requires pynndescent. Install it with `{KNN_BACKEND_INSTALL_HINTS["pynndescent"]}`.',
            context={"knn_backend": "pynndescent"},
        ) from exc
    return NNDescent

def _load_scann_ops() -> Any:
    try:
        import scann  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'scann backend requires ScaNN. Install it with `{KNN_BACKEND_INSTALL_HINTS["scann"]}`.',
            context={"knn_backend": "scann"},
        ) from exc
    return scann.scann_ops_pybind

def _load_zarr_module() -> Any:
    try:
        import zarr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires zarr. Install it with `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return zarr

def _default_zarr_compressor() -> Any | None:
    try:
        from numcodecs import Blosc  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

def _load_zarr_vlen_utf8_codec() -> Any:
    try:
        from numcodecs import VLenUTF8  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires numcodecs via `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return VLenUTF8()

def _zarr_utf8_codec() -> Any:
    try:
        from numcodecs import VLenUTF8  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires numcodecs. Install it with `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return VLenUTF8()

def _remove_output_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()

def _temporary_output_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp-{uuid4().hex}"

def _finalize_output_path(temp_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if output_path.exists():
        backup_path = output_path.parent / f".{output_path.name}.bak-{uuid4().hex}"
        output_path.replace(backup_path)
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    else:
        if backup_path is not None:
            _remove_output_path(backup_path)

def _persist_faiss_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    faiss = _load_faiss_module()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(int(candidate_matrix.shape[1]), 32)
    if hasattr(index, "hnsw"):
        index.hnsw.efConstruction = max(40, min(int(candidate_matrix.shape[0]), 320))
    index.add(np.asarray(candidate_matrix, dtype=np.float32))
    faiss.write_index(index, str(output_path))

def _load_persisted_faiss_index(path: Path) -> Any:
    faiss = _load_faiss_module()
    return faiss.read_index(str(path))

def _persist_pynndescent_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    NNDescent = _load_pynndescent_class()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = NNDescent(np.asarray(candidate_matrix, dtype=np.float32), metric="euclidean")
    if hasattr(index, "prepare"):
        index.prepare()
    with output_path.open("wb") as handle:
        pickle.dump(index, handle)

def _load_persisted_pynndescent_index(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)

def _persist_scann_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    searcher = _build_scann_searcher(
        np.asarray(candidate_matrix, dtype=np.float32),
        neighbor_count=min(int(candidate_matrix.shape[0]), 64),
        knn_eps=0.0,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    if hasattr(searcher, "serialize"):
        searcher.serialize(str(output_path))
        return
    raise PreparedLibraryBuildError(
        "scann searcher does not support serialization in this environment.",
        context={"knn_backend": "scann", "path": str(output_path)},
    )

def _load_persisted_scann_index(path: Path) -> Any:
    scann_ops = _load_scann_ops()
    if hasattr(scann_ops, "load_searcher"):
        return scann_ops.load_searcher(str(path))
    raise MappingInputError(
        "scann backend does not support persisted searcher loading in this environment.",
        context={"knn_backend": "scann", "path": str(path)},
    )

def _persist_knn_index(candidate_matrix: np.ndarray, *, backend: str, output_path: Path) -> None:
    handlers = {
        "faiss": _persist_faiss_index,
        "pynndescent": _persist_pynndescent_index,
        "scann": _persist_scann_index,
    }
    handler = handlers.get(backend)
    if handler is not None:
        handler(candidate_matrix, output_path)
        return
    raise PreparedLibraryBuildError(
        "KNN index persistence is not supported for the requested backend.",
        context={"knn_backend": backend, "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS)},
    )

def _load_persisted_knn_index(path: Path, *, backend: str) -> Any:
    handlers = {
        "faiss": _load_persisted_faiss_index,
        "pynndescent": _load_persisted_pynndescent_index,
        "scann": _load_persisted_scann_index,
    }
    handler = handlers.get(backend)
    if handler is not None:
        return handler(path)
    raise MappingInputError(
        "KNN index persistence is not supported for the requested backend.",
        context={"knn_backend": backend, "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS)},
    )

def _normalize_query_matrix(query_values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Normalize single-row and batched queries to a 2D matrix."""

    query_matrix = np.asarray(query_values, dtype=dtype)
    if query_matrix.ndim == 1:
        return query_matrix.reshape(1, -1)
    if query_matrix.ndim != 2:
        raise MappingInputError(
            "Query values must be one-dimensional or two-dimensional.",
            context={"query_shape": list(query_matrix.shape)},
        )
    return query_matrix

def _normalize_local_indices(local_indices: np.ndarray, *, query_count: int) -> np.ndarray:
    """Normalize backend neighbor indices to ``(query_count, k)``."""

    normalized = np.asarray(local_indices, dtype=np.int64)
    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    if normalized.ndim == 1:
        return normalized.reshape(query_count, -1)
    if normalized.ndim != 2:
        raise MappingInputError(
            "Neighbor search backend returned indices with an unsupported shape.",
            context={"local_index_shape": list(normalized.shape)},
        )
    return normalized

def _normalize_local_distances(local_distances: np.ndarray, *, query_count: int) -> np.ndarray:
    """Normalize backend neighbor distances to ``(query_count, k)``."""

    normalized = np.asarray(local_distances, dtype=np.float64)
    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    if normalized.ndim == 1:
        return normalized.reshape(query_count, -1)
    if normalized.ndim != 2:
        raise MappingInputError(
            "Neighbor search backend returned distances with an unsupported shape.",
            context={"local_distance_shape": list(normalized.shape)},
        )
    return normalized

def _build_scann_searcher(
    candidate_matrix: np.ndarray,
    *,
    neighbor_count: int,
    knn_eps: float,
) -> Any:
    """Build a ScaNN searcher tuned for the current candidate set size."""

    scann_ops = _load_scann_ops()
    candidate_count = int(candidate_matrix.shape[0])
    num_leaves = max(1, min(candidate_count, int(round(math.sqrt(candidate_count)))))
    base_leaves_to_search = max(1, min(num_leaves, int(round(math.sqrt(num_leaves))) or 1))
    leaves_to_search = max(
        1,
        min(
            num_leaves,
            int(math.ceil(base_leaves_to_search / (1.0 + max(float(knn_eps), 0.0) * 4.0))),
        ),
    )
    training_sample_size = min(candidate_count, max(32, num_leaves * 10))
    builder = scann_ops.builder(np.asarray(candidate_matrix, dtype=np.float32), neighbor_count, "squared_l2")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=leaves_to_search,
        training_sample_size=training_sample_size,
    )
    # Tiny candidate sets cannot train ScaNN's asymmetric hashing path.
    # Fall back to brute-force scoring so the builder still satisfies ScaNN's
    # requirement that exactly one scoring mode is configured.
    if hasattr(builder, "score_ah") and training_sample_size >= SCANN_MIN_AH_TRAINING_SAMPLE_SIZE:
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    elif hasattr(builder, "score_brute_force"):
        builder = builder.score_brute_force()
    if hasattr(builder, "reorder"):
        builder = builder.reorder(neighbor_count)
    return builder.build()

def _query_faiss_index(index: Any, query_values: np.ndarray, *, k: int) -> np.ndarray:
    _, local_indices = index.search(_normalize_query_matrix(query_values, dtype=np.float32), int(k))
    return np.asarray(local_indices, dtype=np.int64)

def _query_pynndescent_index(index: Any, query_values: np.ndarray, *, k: int, knn_eps: float) -> np.ndarray:
    local_indices, _ = index.query(
        _normalize_query_matrix(query_values, dtype=np.float32),
        k=int(k),
        epsilon=float(knn_eps),
    )
    return np.asarray(local_indices, dtype=np.int64)

def _query_scann_index(index: Any, query_values: np.ndarray, *, k: int) -> np.ndarray:
    query_matrix = _normalize_query_matrix(query_values, dtype=np.float32)
    try:
        search_result = index.search_batched(query_matrix, final_num_neighbors=int(k))
    except TypeError:
        search_result = index.search_batched(query_matrix)
    local_indices = search_result[0] if isinstance(search_result, tuple) else search_result
    return np.asarray(local_indices, dtype=np.int64)

def _query_knn_index(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray:
    """Query a backend index and normalize its output shape."""

    query_count = int(_normalize_query_matrix(query_values, dtype=np.float64).shape[0])
    if knn_backend == "scipy_ckdtree":
        _, local_indices = index.query(
            _normalize_query_matrix(query_values, dtype=np.float64),
            k=int(k),
            eps=float(knn_eps),
            workers=_scipy_ckdtree_workers(),
        )
    elif knn_backend == "faiss":
        local_indices = _query_faiss_index(index, query_values, k=k)
    elif knn_backend == "pynndescent":
        local_indices = _query_pynndescent_index(index, query_values, k=k, knn_eps=knn_eps)
    elif knn_backend == "scann":
        local_indices = _query_scann_index(index, query_values, k=k)
    else:
        raise MappingInputError(
            "knn_backend is not supported.",
            context={
                "knn_backend": knn_backend,
                "supported_knn_backends": list(SUPPORTED_KNN_BACKENDS),
            },
        )
    return _normalize_local_indices(np.asarray(local_indices, dtype=np.int64), query_count=query_count)

def _query_knn_index_with_distances(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Query a backend index and normalize both indices and distances."""

    query_count = int(_normalize_query_matrix(query_values, dtype=np.float64).shape[0])
    if knn_backend != "scipy_ckdtree":
        raise MappingInputError(
            "Distance-returning KNN queries are only supported for scipy_ckdtree.",
            context={"knn_backend": knn_backend},
        )
    local_distances, local_indices = index.query(
        _normalize_query_matrix(query_values, dtype=np.float64),
        k=int(k),
        eps=float(knn_eps),
        workers=_scipy_ckdtree_workers(),
    )
    return (
        _normalize_local_indices(np.asarray(local_indices, dtype=np.int64), query_count=query_count),
        _normalize_local_distances(np.asarray(local_distances, dtype=np.float64), query_count=query_count),
    )

def _query_local_scipy_ckdtree_results(
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    *,
    k: int,
    knn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a temporary SciPy cKDTree and return local indices with distances."""

    neighbor_count = min(int(k), int(candidate_matrix.shape[0]))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})
    cKDTree = _load_ckdtree_class()
    index = cKDTree(np.asarray(candidate_matrix, dtype=np.float64))
    return _query_knn_index_with_distances(
        index,
        query_values,
        k=neighbor_count,
        knn_backend="scipy_ckdtree",
        knn_eps=knn_eps,
    )

def _search_local_neighbor_indices(
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray | None:
    """Return approximate local neighbor indices for one or more query rows.

    Approximate backends are only used to generate a candidate shortlist. The
    final row order is still re-ranked with exact RMSE distances.
    """

    if knn_backend == "numpy":
        return None

    neighbor_count = min(int(k), int(candidate_matrix.shape[0]))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})

    if knn_backend == "scipy_ckdtree":
        cKDTree = _load_ckdtree_class()
        index = cKDTree(np.asarray(candidate_matrix, dtype=np.float64))
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "faiss":
        faiss = _load_faiss_module()
        vector_dim = int(candidate_matrix.shape[1])
        index = faiss.IndexHNSWFlat(vector_dim, 32)
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = max(40, neighbor_count * 8)
            index.hnsw.efSearch = max(
                neighbor_count,
                int(math.ceil(max(32, neighbor_count * 8) / (1.0 + max(float(knn_eps), 0.0) * 4.0))),
            )
        index.add(np.asarray(candidate_matrix, dtype=np.float32))
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "pynndescent":
        NNDescent = _load_pynndescent_class()
        index = NNDescent(np.asarray(candidate_matrix, dtype=np.float32), metric="euclidean")
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "scann":
        index = _build_scann_searcher(candidate_matrix, neighbor_count=neighbor_count, knn_eps=knn_eps)
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    raise MappingInputError(
        "knn_backend is not supported.",
        context={
            "knn_backend": knn_backend,
            "supported_knn_backends": list(SUPPORTED_KNN_BACKENDS),
        },
    )

def _query_persisted_knn_index(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray:
    if knn_backend not in SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS:
        raise MappingInputError(
            "Persisted KNN querying is not supported for the requested backend.",
            context={"knn_backend": knn_backend},
        )
    return _query_knn_index(index, query_values, k=k, knn_backend=knn_backend, knn_eps=knn_eps)

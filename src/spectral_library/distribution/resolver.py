"""Resolve a prepared runtime root for commands that accept an override path."""

from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator

from .._version import __version__
from .runtime import RuntimeDownloadError, download_prepared_library

DEFAULT_PREPARED_RUNTIME_TAG = f"v{__version__}"
_PREPARED_RUNTIME_LOCK_TIMEOUT_SECONDS = 60.0
_PREPARED_RUNTIME_ROOT_ENV = "SPECTRAL_LIBRARY_PREPARED_RUNTIME_ROOT"


def default_prepared_runtime_root() -> Path:
    """Return the default cache location for the published prepared runtime."""
    override_root = (os.environ.get(_PREPARED_RUNTIME_ROOT_ENV) or "").strip()
    if override_root:
        return Path(override_root).expanduser()
    if sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        cache_root = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "spectral-library" / "prepared-runtime" / DEFAULT_PREPARED_RUNTIME_TAG


@contextmanager
def _prepared_runtime_cache_lock(lock_path: Path) -> Iterator[None]:
    deadline = time.monotonic() + _PREPARED_RUNTIME_LOCK_TIMEOUT_SECONDS
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeDownloadError(f"Timed out waiting for prepared runtime cache lock: {lock_path}")
            time.sleep(0.1)
    try:
        os.write(file_descriptor, str(os.getpid()).encode("ascii"))
    finally:
        os.close(file_descriptor)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _cached_prepared_runtime_is_valid(prepared_root: Path) -> bool:
    if not prepared_root.exists():
        return False
    try:
        from ..mapping import validate_prepared_library

        validate_prepared_library(prepared_root, verify_checksums=False)
    except Exception:
        return False
    return True


def _remove_cached_prepared_runtime(prepared_root: Path) -> None:
    if not prepared_root.exists():
        return
    if prepared_root.is_dir():
        shutil.rmtree(prepared_root)
        return
    prepared_root.unlink()


def resolve_prepared_library_root(prepared_root: Path | str | None = None) -> Path:
    """Return a prepared runtime root, downloading the published runtime if needed.

    When *prepared_root* is supplied it is treated as an explicit override and
    returned unchanged. Otherwise the package-matched published runtime is
    cached under :func:`default_prepared_runtime_root`, validated on reuse, and
    downloaded when the cache is missing or stale.
    """
    if prepared_root is not None:
        if isinstance(prepared_root, Path):
            return prepared_root.expanduser()
        prepared_root_text = str(prepared_root).strip()
        if prepared_root_text:
            prepared_root_path = Path(prepared_root_text).expanduser()
            return prepared_root_path

    resolved_root = default_prepared_runtime_root()
    if _cached_prepared_runtime_is_valid(resolved_root):
        return resolved_root

    resolved_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_root.parent / f".{resolved_root.name}.lock"
    with _prepared_runtime_cache_lock(lock_path):
        if _cached_prepared_runtime_is_valid(resolved_root):
            return resolved_root
        _remove_cached_prepared_runtime(resolved_root)
        return download_prepared_library(resolved_root, tag=DEFAULT_PREPARED_RUNTIME_TAG)

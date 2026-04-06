"""Download and verify a prepared spectral-library runtime from a remote URL.

This module backs the ``spectral-library download-prepared-library`` CLI
command.  It supports fetching compressed tarballs from GitHub Releases or
any HTTPS endpoint, verifying the download against an expected SHA-256
digest, and extracting the runtime to a local directory ready for use with
:class:`~spectral_library.mapping.SpectralMapper`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .._version import __version__

GITHUB_REPO = "MarcYin/spectral_library"
RUNTIME_ASSET_PREFIX = "spectral-library-runtime-"
CHUNK_SIZE = 1 << 20  # 1 MiB


class RuntimeDownloadError(Exception):
    """Raised when a runtime download or verification fails."""


def _resolve_latest_release_tag(repo: str) -> str:
    """Query the GitHub API for the latest release tag."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    request = Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeDownloadError(f"Failed to query GitHub releases: {exc}") from exc
    tag = payload.get("tag_name")
    if not tag:
        raise RuntimeDownloadError("GitHub latest release has no tag_name.")
    return str(tag)


def _find_runtime_asset(repo: str, tag: str) -> tuple[str, str | None]:
    """Return (tarball_url, sha256_url) for the runtime asset in a release."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    request = Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeDownloadError(f"Failed to query release {tag}: {exc}") from exc

    assets = payload.get("assets", [])
    tarball_url: str | None = None
    sha256_url: str | None = None
    for asset in assets:
        name = asset.get("name", "")
        download_url = asset.get("browser_download_url", "")
        if name.startswith(RUNTIME_ASSET_PREFIX) and name.endswith(".tar.gz") and not name.endswith(".sha256"):
            tarball_url = download_url
        elif name.startswith(RUNTIME_ASSET_PREFIX) and name.endswith(".tar.gz.sha256"):
            sha256_url = download_url

    if not tarball_url:
        raise RuntimeDownloadError(
            f"Release {tag} has no runtime tarball asset (expected {RUNTIME_ASSET_PREFIX}*.tar.gz). "
            f"Found assets: {[a.get('name') for a in assets]}"
        )
    return tarball_url, sha256_url


def _download_file(url: str, dest: Path, *, label: str = "") -> None:
    """Download a URL to a local file path with progress reporting."""
    request = Request(url, headers={"User-Agent": f"spectral-library/{__version__}"})
    try:
        with urlopen(request, timeout=300) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            downloaded = 0
            with dest.open("wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(f"\r  {label}{mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  {label}{mb:.1f} MB", end="", flush=True)
            print()
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeDownloadError(f"Download failed: {exc}") from exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_expected_digest(sha256_url: str) -> str | None:
    """Fetch the SHA-256 sidecar and return the hex digest, or None."""
    request = Request(sha256_url, headers={"User-Agent": f"spectral-library/{__version__}"})
    try:
        with urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8").strip()
    except (HTTPError, URLError, OSError):
        return None
    # Format: "<hex>  <filename>" or just "<hex>"
    return text.split()[0] if text else None


def download_prepared_library(
    output_root: Path,
    *,
    url: str | None = None,
    tag: str | None = None,
    sha256: str | None = None,
    verify_after_extract: bool = True,
) -> Path:
    """Download and extract a prepared runtime to *output_root*.

    Parameters
    ----------
    output_root:
        Directory where the runtime files will be extracted.
    url:
        Direct URL to a ``.tar.gz`` runtime archive.  When provided,
        GitHub Release discovery is skipped.
    tag:
        GitHub Release tag (e.g. ``v0.2.0``).  When *url* is ``None``
        and *tag* is ``None``, the latest release is used.
    sha256:
        Expected SHA-256 hex digest.  Overrides the sidecar digest
        published alongside the release asset.
    verify_after_extract:
        Run ``validate_prepared_library`` on the extracted runtime.

    Returns
    -------
    Path
        The output root containing the extracted runtime files.
    """
    tarball_url: str
    sha256_url: str | None = None

    if url:
        tarball_url = url
    else:
        repo = GITHUB_REPO
        if tag is None:
            print(f"Resolving latest release from {repo}...")
            tag = _resolve_latest_release_tag(repo)
        print(f"Looking for runtime asset in release {tag}...")
        tarball_url, sha256_url = _find_runtime_asset(repo, tag)

    expected_digest = sha256
    if expected_digest is None and sha256_url:
        print("Fetching expected SHA-256 digest...")
        expected_digest = _fetch_expected_digest(sha256_url)

    with tempfile.TemporaryDirectory(prefix="spectral-library-download-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_name = tarball_url.rsplit("/", 1)[-1] if "/" in tarball_url else "runtime.tar.gz"
        archive_path = tmp_path / archive_name

        print(f"Downloading {tarball_url}")
        _download_file(tarball_url, archive_path, label="downloading: ")

        actual_digest = _sha256_file(archive_path)
        if expected_digest:
            if actual_digest != expected_digest:
                raise RuntimeDownloadError(
                    f"SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
                )
            print(f"SHA-256 verified: {actual_digest}")
        else:
            print(f"SHA-256 (no sidecar to verify against): {actual_digest}")

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        print(f"Extracting to {output_root}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=str(output_root))

    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeDownloadError(f"Extracted archive does not contain manifest.json at {output_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_count = manifest.get("row_count", "?")
    sensors = manifest.get("source_sensors", [])
    print(f"Runtime ready: {row_count} spectra, sensors: {', '.join(sensors)}")

    if verify_after_extract:
        from ..mapping import validate_prepared_library

        print("Validating extracted runtime...")
        validate_prepared_library(output_root, verify_checksums=True)
        print("Validation passed.")

    return output_root

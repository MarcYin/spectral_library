#!/usr/bin/env python3
"""Package a prepared runtime directory into a distributable tarball.

The tarball is suitable for upload as a GitHub Release asset or hosting
on any file server.  The companion CLI command
``spectral-library download-prepared-library`` can fetch, verify, and
extract this archive.

Usage::

    python scripts/package_prepared_runtime.py \
        --prepared-root build/full_library_runtime \
        --output-dir dist

This produces a file like::

    dist/spectral-library-runtime-v0.2.0-77125rows.tar.gz

with a SHA-256 sidecar digest at::

    dist/spectral-library-runtime-v0.2.0-77125rows.tar.gz.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a prepared runtime into a distributable tarball.")
    parser.add_argument("--prepared-root", required=True, help="Path to the prepared runtime directory.")
    parser.add_argument("--output-dir", default="dist", help="Directory for the output tarball (default: dist).")
    args = parser.parse_args(argv)

    prepared_root = Path(args.prepared_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found — is this a valid prepared runtime?", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("package_version", "unknown")
    row_count = manifest.get("row_count", 0)

    archive_stem = f"spectral-library-runtime-v{version}-{row_count}rows"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{archive_stem}.tar.gz"

    print(f"Packaging {prepared_root} -> {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tar:
        for child in sorted(prepared_root.iterdir()):
            if child.name.startswith("."):
                continue
            tar.add(str(child), arcname=child.name)
            print(f"  added {child.name} ({child.stat().st_size:,} bytes)")

    digest = _sha256_file(archive_path)
    digest_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated {archive_path.name} ({size_mb:.1f} MB)")
    print(f"SHA-256: {digest}")
    print(f"Digest:  {digest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json


def _split_archive_url(download_url: str, landing_url: str) -> tuple[str, str]:
    target = download_url or landing_url
    parts = urlsplit(target)
    archive_url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    extract_prefix = parts.fragment.strip("/")
    return archive_url, extract_prefix


def _iter_matching_members(archive: zipfile.ZipFile, extract_prefix: str) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = Path(info.filename)
        relative_parts = path.parts[1:] if len(path.parts) > 1 else ()
        if not relative_parts:
            continue
        relative_path = Path(*relative_parts)
        if extract_prefix and not str(relative_path).startswith(f"{extract_prefix}/") and str(relative_path) != extract_prefix:
            continue
        members.append(info)
    return members


def _relative_destination(info: zipfile.ZipInfo, extract_prefix: str) -> Path:
    path = Path(info.filename)
    relative_parts = path.parts[1:] if len(path.parts) > 1 else ()
    relative_path = Path(*relative_parts)
    if extract_prefix:
        try:
            return relative_path.relative_to(extract_prefix)
        except ValueError as exc:
            raise ValueError(f"Archive member {info.filename} does not match prefix {extract_prefix}") from exc
    return relative_path


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_url, extract_prefix = _split_archive_url(source.download_url, source.landing_url)
    notes: list[str] = []
    artifacts: list[ArtifactRecord] = []
    status = "metadata_only"

    metadata_artifact = write_json(
        output_dir / "github_archive.json",
        {
            "source_id": source.source_id,
            "landing_url": source.landing_url,
            "download_url": source.download_url,
            "archive_url": archive_url,
            "extract_prefix": extract_prefix,
        },
    )
    metadata_artifact.url = archive_url
    artifacts.append(metadata_artifact)

    if fetch_mode == "assets":
        data_root = output_dir / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "archive.zip"
            with urlopen(Request(archive_url, headers={"User-Agent": user_agent}), timeout=300) as response:
                with archive_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)

            with zipfile.ZipFile(archive_path) as archive:
                members = _iter_matching_members(archive, extract_prefix)
                for info in members:
                    destination = data_root / _relative_destination(info, extract_prefix)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source_handle, destination.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                    artifacts.append(
                        ArtifactRecord(
                            artifact_id=destination.stem,
                            kind="data",
                            url=f"{archive_url}#{extract_prefix}" if extract_prefix else archive_url,
                            path=str(destination),
                            media_type="application/octet-stream",
                            size_bytes=destination.stat().st_size,
                            sha256=sha256_file(destination),
                            status="downloaded",
                            note=str(_relative_destination(info, extract_prefix)),
                        )
                    )
        status = "downloaded"
        notes.append(
            f"Extracted {len(artifacts) - 1} files from the GitHub archive"
            + (f" under {extract_prefix}/." if extract_prefix else ".")
        )
    else:
        notes.append("Metadata captured for the GitHub archive without downloading files.")

    finished_at = utc_now_iso()
    return FetchResult(
        source_id=source.source_id,
        source_name=source.name,
        fetch_adapter=source.fetch_adapter,
        fetch_mode=fetch_mode,
        status=status,
        landing_url=source.landing_url,
        started_at=started_at,
        finished_at=finished_at,
        notes=notes,
        artifacts=artifacts,
    )

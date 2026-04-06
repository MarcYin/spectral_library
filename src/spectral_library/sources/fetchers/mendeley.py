from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json


def _extract_dataset_id_and_version(url: str) -> tuple[str, int | None]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "datasets" not in parts:
        raise ValueError(f"Unable to extract Mendeley dataset id from {url}")
    index = parts.index("datasets")
    if index + 1 >= len(parts):
        raise ValueError(f"Unable to extract Mendeley dataset id from {url}")
    dataset_id = parts[index + 1]
    version = None
    if index + 2 < len(parts) and parts[index + 2].isdigit():
        version = int(parts[index + 2])
    return dataset_id, version


def _load_json(url: str, user_agent: str) -> tuple[dict[str, object], str]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        return json.load(response), response.geturl()


def _extract_files(payload: dict[str, object]) -> list[dict[str, object]]:
    files = payload.get("files", [])
    if not isinstance(files, list):
        return []

    extracted: list[dict[str, object]] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        filename = str(file_info.get("filename", ""))
        content_details = file_info.get("content_details", {})
        if not isinstance(content_details, dict):
            continue
        download_url = str(content_details.get("download_url", ""))
        if not filename or not download_url:
            continue
        extracted.append(
            {
                "filename": filename,
                "download_url": download_url,
                "view_url": str(content_details.get("view_url", "")),
                "content_type": str(content_details.get("content_type", "")),
                "size_bytes": int(content_details.get("size", 0) or 0),
                "sha256": str(content_details.get("sha256_hash", "")),
            }
        )
    return extracted


def _download_artifact(url: str, output_dir: Path, filename: str, user_agent: str) -> ArtifactRecord:
    request = Request(url, headers={"User-Agent": user_agent})
    destination = output_dir / filename
    with urlopen(request, timeout=300) as response:
        media_type = response.headers.get_content_type()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    return ArtifactRecord(
        artifact_id=destination.stem,
        kind="data",
        url=url,
        path=str(destination),
        media_type=media_type,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        status="downloaded",
    )


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id, version = _extract_dataset_id_and_version(source.landing_url)
    base_api_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
    api_url = f"{base_api_url}?version={version}" if version is not None else base_api_url
    payload, final_api_url = _load_json(api_url, user_agent)
    files = _extract_files(payload)
    notes: list[str] = []
    if version is not None and not files:
        fallback_payload, fallback_final_url = _load_json(base_api_url, user_agent)
        fallback_files = _extract_files(fallback_payload)
        if fallback_files:
            payload = fallback_payload
            final_api_url = fallback_final_url
            files = fallback_files
            notes.append(
                "Versioned Mendeley API response omitted file metadata; fell back to the public dataset endpoint."
            )

    artifacts: list[ArtifactRecord] = []
    metadata_artifact = write_json(
        output_dir / "mendeley_dataset.json",
        {
            "requested_url": api_url,
            "fallback_url": base_api_url,
            "landing_url": source.landing_url,
            "final_url": final_api_url,
            "dataset_id": dataset_id,
            "requested_version": version,
            "dataset": payload,
            "files": files,
        },
    )
    metadata_artifact.url = final_api_url
    artifacts.append(metadata_artifact)

    notes.append(f"Discovered {len(files)} Mendeley public files for dataset {dataset_id}.")
    status = "metadata_only"
    if fetch_mode == "assets":
        for file_info in files:
            artifacts.append(
                _download_artifact(
                    str(file_info["download_url"]),
                    output_dir,
                    str(file_info["filename"]),
                    user_agent,
                )
            )
        status = "downloaded" if files else "metadata_only"
    else:
        for index, file_info in enumerate(files, start=1):
            artifacts.append(
                ArtifactRecord(
                    artifact_id=f"remote_file_{index}",
                    kind="remote_file",
                    url=str(file_info["download_url"]),
                    path="",
                    media_type=str(file_info["content_type"]),
                    size_bytes=int(file_info["size_bytes"]),
                    sha256=str(file_info["sha256"]),
                    status="discovered",
                    note=str(file_info["filename"]),
                )
            )

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

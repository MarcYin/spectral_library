from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from .http_utils import infer_filename, sanitize_filename


def _extract_record_id(url: str) -> str:
    match = re.search(r"/records/(\d+)", url)
    if not match:
        raise ValueError(f"Unable to extract Zenodo record id from {url}")
    return match.group(1)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment))


def _file_urls(file_item: dict[str, object]) -> list[str]:
    links = file_item.get("links", {})
    if not isinstance(links, dict):
        return []
    urls: list[str] = []
    for key in ("self", "download"):
        value = links.get(key, "")
        if value:
            urls.append(_normalize_url(str(value)))
    return urls


def _filename_from_content_url(url: str) -> str:
    path = unquote(urlsplit(url).path)
    if "/files/" not in path or not path.endswith("/content"):
        return ""
    name = path.split("/files/", 1)[1].rsplit("/content", 1)[0]
    return sanitize_filename(name)


def _download_file(file_url: str, destination: Path, user_agent: str) -> ArtifactRecord:
    file_request = Request(file_url, headers={"User-Agent": user_agent})
    with urlopen(file_request, timeout=300) as file_response:
        media_type = file_response.headers.get_content_type()
        with destination.open("wb") as handle:
            shutil.copyfileobj(file_response, handle)
    return ArtifactRecord(
        artifact_id=destination.stem,
        kind="data",
        url=file_url,
        path=str(destination),
        media_type=media_type,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        status="downloaded",
    )


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_id = _extract_record_id(source.landing_url)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    request = Request(api_url, headers={"User-Agent": user_agent})

    with urlopen(request, timeout=120) as response:
        payload = json.load(response)

    artifacts: list[ArtifactRecord] = []
    record_artifact = write_json(output_dir / "zenodo_record.json", payload)
    record_artifact.url = api_url
    artifacts.append(record_artifact)

    notes: list[str] = []
    status = "metadata_only"
    file_items = payload.get("files", [])
    selected_download_url = _normalize_url(source.download_url) if source.download_url else ""
    matched_selected_file = False

    for index, file_item in enumerate(file_items, start=1):
        file_urls = _file_urls(file_item)
        file_url = file_urls[0] if file_urls else ""
        file_name = str(file_item.get("key") or _filename_from_content_url(file_url) or infer_filename(file_url, f"file-{index}.bin"))
        should_download = fetch_mode == "assets" and file_url and (
            not selected_download_url or selected_download_url in file_urls
        )
        if should_download:
            matched_selected_file = matched_selected_file or bool(selected_download_url)
            destination = output_dir / infer_filename(file_name, file_name)
            status = "downloaded"
            artifacts.append(_download_file(file_url, destination, user_agent))
        else:
            artifacts.append(
                ArtifactRecord(
                    artifact_id=f"remote_file_{index}",
                    kind="remote_file",
                    url=file_url,
                    path="",
                    media_type=file_item.get("mimetype", ""),
                    size_bytes=int(file_item.get("size", 0) or 0),
                    sha256=file_item.get("checksum", ""),
                    status="discovered",
                    note=file_name,
                )
            )

    if fetch_mode == "assets" and selected_download_url and matched_selected_file:
        notes.append("Configured download_url limited asset download to the primary Zenodo file.")
    elif (
        fetch_mode == "assets"
        and selected_download_url
        and "/files/" in urlsplit(selected_download_url).path
        and selected_download_url.endswith("/content")
    ):
        destination_name = _filename_from_content_url(selected_download_url) or infer_filename(
            selected_download_url, f"{source.source_id}.bin"
        )
        destination = output_dir / destination_name
        artifacts.append(_download_file(selected_download_url, destination, user_agent))
        status = "downloaded"
        notes.append("Configured download_url was not listed in the record metadata; downloaded it directly.")
    elif fetch_mode == "assets" and selected_download_url:
        notes.append("Configured download_url did not resolve to a downloadable Zenodo file in the record metadata.")

    if fetch_mode == "metadata":
        notes.append("Zenodo record metadata captured without downloading record files.")

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

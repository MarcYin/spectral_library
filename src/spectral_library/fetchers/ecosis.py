from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from .http_utils import infer_filename


def _extract_package_slug(url: str) -> str:
    marker = "/package/"
    parsed = urlparse(url)
    if marker not in parsed.path:
        raise ValueError(f"Unable to extract EcoSIS package slug from {url}")
    slug = parsed.path.split(marker, 1)[1].strip("/")
    if not slug:
        raise ValueError(f"Unable to extract EcoSIS package slug from {url}")
    return slug


def _normalize_resource_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "http" and parsed.netloc in {"ecosis.org", "data.ecosis.org"}:
        return urlunparse(parsed._replace(scheme="https"))
    return url


def _extract_resources(payload: dict[str, object]) -> list[dict[str, str]]:
    ecosis = payload.get("ecosis", {})
    if not isinstance(ecosis, dict):
        return []

    raw_resources = ecosis.get("resources", [])
    if not isinstance(raw_resources, list):
        return []

    resources: list[dict[str, str]] = []
    for resource in raw_resources:
        if not isinstance(resource, dict):
            continue
        url = _normalize_resource_url(str(resource.get("url", "")))
        if not url:
            continue
        resources.append(
            {
                "name": str(resource.get("name", "") or infer_filename(url, "resource.bin")),
                "url": url,
                "mimetype": str(resource.get("mimetype", "")),
                "type": str(resource.get("type", "")),
            }
        )
    return resources


def _download_artifact(url: str, output_dir: Path, default_name: str, user_agent: str) -> ArtifactRecord:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=300) as response:
        media_type = response.headers.get_content_type()
        filename = infer_filename(response.geturl(), default_name)
        if not Path(filename).suffix and Path(default_name).suffix:
            filename = default_name
        destination = output_dir / filename
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

    slug = _extract_package_slug(source.landing_url)
    api_url = f"https://ecosis.org/api/package/{slug}"
    export_url = f"{api_url}/export?metadata=true"
    request = Request(api_url, headers={"User-Agent": user_agent})

    with urlopen(request, timeout=120) as response:
        final_url = response.geturl()
        payload = json.load(response)

    resources = _extract_resources(payload)
    metadata_artifact = write_json(
        output_dir / "ecosis_package.json",
        {
            "requested_url": api_url,
            "landing_url": source.landing_url,
            "download_url": source.download_url,
            "final_url": final_url,
            "package_slug": slug,
            "export_url": export_url,
            "resource_count": len(resources),
            "resources": resources,
            "package": payload,
        },
    )
    metadata_artifact.url = final_url

    artifacts: list[ArtifactRecord] = [metadata_artifact]
    notes: list[str] = []
    status = "metadata_only"
    selected_urls = {_normalize_resource_url(source.download_url)} if source.download_url else set()
    matched_selected_url = False

    if resources:
        for index, resource in enumerate(resources, start=1):
            resource_url = resource["url"]
            resource_name = resource["name"] or f"resource-{index}.bin"
            should_download = fetch_mode == "assets"
            if should_download:
                if resource_url in selected_urls:
                    matched_selected_url = True
                artifacts.append(_download_artifact(resource_url, output_dir, resource_name, user_agent))
                status = "downloaded"
            else:
                artifacts.append(
                    ArtifactRecord(
                        artifact_id=f"remote_resource_{index}",
                        kind="remote_file",
                        url=resource_url,
                        path="",
                        media_type=resource["mimetype"],
                        size_bytes=0,
                        sha256="",
                        status="discovered",
                        note=resource_name,
                    )
                )

        if fetch_mode == "assets" and selected_urls and not matched_selected_url:
            artifacts.append(_download_artifact(next(iter(selected_urls)), output_dir, f"{source.source_id}.bin", user_agent))
            status = "downloaded"
            notes.append("Configured EcoSIS package asset URL was not listed in package metadata; downloaded it directly.")
        elif fetch_mode == "assets" and selected_urls:
            notes.append("Downloaded all EcoSIS package resources; the configured download_url was included in the package set.")
        elif fetch_mode == "assets":
            downloaded_count = sum(artifact.status == "downloaded" and artifact.kind == "data" for artifact in artifacts)
            notes.append(f"Downloaded {downloaded_count} EcoSIS package resources discovered from package metadata.")
        else:
            notes.append("EcoSIS package metadata captured without downloading package resources.")
    else:
        if fetch_mode == "assets":
            download_url = next(iter(selected_urls), export_url)
            artifacts.append(_download_artifact(download_url, output_dir, f"{slug}.csv", user_agent))
            status = "downloaded"
            if source.download_url:
                notes.append("Downloaded configured EcoSIS package asset URL from the manifest.")
            else:
                notes.append("Downloaded EcoSIS package export because no package resources were listed.")
        else:
            notes.append("EcoSIS package metadata captured without downloading package resources.")

    artifacts.append(
        ArtifactRecord(
            artifact_id="package_export",
            kind="remote_file",
            url=export_url,
            path="",
            media_type="text/csv",
            size_bytes=0,
            sha256="",
            status="discovered",
            note="EcoSIS package export with metadata=true",
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

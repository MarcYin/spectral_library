from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from .http_utils import infer_filename


HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', flags=re.I)
PANGAEA_DOI_RE = re.compile(r"https?://(?:doi\.pangaea\.de|doi\.org)/10\.1594/PANGAEA\.\d+", flags=re.I)
PANGAEA_DOWNLOAD_RE = re.compile(r"https?://download\.pangaea\.de/[^\"']+", flags=re.I)


def _fetch_html(url: str, user_agent: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8", "ignore")
        return response.geturl(), body


def _extract_links(body: str, base_url: str) -> list[str]:
    return [urljoin(base_url, href) for href in HREF_RE.findall(body)]


def _extract_doi_url(links: list[str]) -> str:
    for link in links:
        match = PANGAEA_DOI_RE.search(link)
        if match:
            normalized = match.group(0)
            return normalized.replace("https://doi.org/", "https://doi.pangaea.de/")
    return ""


def _extract_download_url(links: list[str], doi_url: str) -> str:
    for link in links:
        match = PANGAEA_DOWNLOAD_RE.search(link)
        if match:
            return match.group(0)
    if doi_url:
        for link in links:
            if link.startswith(doi_url) and "format=zip" in link:
                return link
    return ""


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)

    landing_final_url, landing_html = _fetch_html(source.landing_url, user_agent)
    landing_links = _extract_links(landing_html, landing_final_url)
    doi_url = _extract_doi_url(landing_links)
    download_url = _extract_download_url(landing_links, doi_url)
    doi_final_url = ""
    doi_links: list[str] = []

    if doi_url:
        doi_final_url, doi_html = _fetch_html(doi_url, user_agent)
        doi_links = _extract_links(doi_html, doi_final_url)
        if not download_url:
            download_url = _extract_download_url(doi_links, doi_final_url)

    metadata_artifact = write_json(
        output_dir / "pangaea_record.json",
        {
            "requested_url": source.landing_url,
            "landing_final_url": landing_final_url,
            "doi_url": doi_url,
            "doi_final_url": doi_final_url,
            "download_url": download_url,
            "landing_links_sample": landing_links[:25],
            "doi_links_sample": doi_links[:25],
        },
    )
    metadata_artifact.url = doi_final_url or landing_final_url

    artifacts: list[ArtifactRecord] = [metadata_artifact]
    notes: list[str] = []
    status = "metadata_only"

    if fetch_mode == "assets" and download_url:
        request = Request(download_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=300) as response:
            final_url = response.geturl()
            media_type = response.headers.get_content_type()
            filename = infer_filename(final_url, f"{source.source_id}.zip")
            if not filename.lower().endswith(".zip") and (
                media_type == "application/zip" or "format=zip" in final_url
            ):
                filename = f"{source.source_id}.zip"
            destination = output_dir / filename
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        status = "downloaded"
        artifacts.append(
            ArtifactRecord(
                artifact_id=destination.stem,
                kind="data",
                url=final_url,
                path=str(destination),
                media_type=media_type,
                size_bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                status="downloaded",
            )
        )
    elif fetch_mode == "assets":
        notes.append("No direct PANGAEA download URL was discovered from the landing page.")
    else:
        notes.append("PANGAEA metadata captured without downloading the dataset archive.")

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

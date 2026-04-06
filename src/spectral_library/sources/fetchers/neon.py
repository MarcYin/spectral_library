from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from .http_utils import infer_filename


def _extract_product_code(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError(f"Unable to extract NEON product code from {url}")
    if parts[-2:] and parts[-2] == "data-products":
        return parts[-1]
    if parts[-3:] and parts[-3] == "api" and parts[-2] == "v0":
        return parts[-1]
    return parts[-1]


def _load_json(url: str, user_agent: str) -> tuple[dict[str, object], str]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        return json.load(response), response.geturl()


def _select_neon_files(files: list[dict[str, object]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        name = str(file_info.get("name", ""))
        url = str(file_info.get("url", ""))
        if not url or not name.endswith(".csv"):
            continue
        if not name.startswith("FSP_"):
            continue
        selected.append(
            {
                "name": name,
                "url": url,
                "size": str(file_info.get("size", "")),
                "md5": str(file_info.get("md5", "")),
            }
        )
    return selected


def _download_artifact(url: str, output_dir: Path, default_name: str, user_agent: str) -> ArtifactRecord:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=300) as response:
        filename = infer_filename(response.geturl(), default_name)
        destination = output_dir / filename
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        media_type = response.headers.get_content_type()
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

    product_code = _extract_product_code(source.landing_url)
    product_api_url = f"https://data.neonscience.org/api/v0/products/{product_code}"
    product_payload, final_product_url = _load_json(product_api_url, user_agent)
    product_data = product_payload.get("data", {})
    site_codes = product_data.get("siteCodes", []) if isinstance(product_data, dict) else []

    discovered_files: list[dict[str, str]] = []
    site_summaries: list[dict[str, object]] = []
    for site in site_codes:
        if not isinstance(site, dict):
            continue
        site_code = str(site.get("siteCode", ""))
        available_urls = site.get("availableDataUrls", [])
        site_file_count = 0
        site_months: list[str] = []
        for data_url in available_urls if isinstance(available_urls, list) else []:
            site_payload, _ = _load_json(str(data_url), user_agent)
            data = site_payload.get("data", {})
            if not isinstance(data, dict):
                continue
            month = str(data.get("month", ""))
            if month:
                site_months.append(month)
            matched = _select_neon_files(data.get("files", []) if isinstance(data.get("files", []), list) else [])
            for file_info in matched:
                file_info["site_code"] = site_code
                file_info["month"] = month
                discovered_files.append(file_info)
            site_file_count += len(matched)
        site_summaries.append(
            {
                "site_code": site_code,
                "months": sorted(set(site_months)),
                "spectra_file_count": site_file_count,
            }
        )

    artifacts: list[ArtifactRecord] = []
    product_artifact = write_json(
        output_dir / "neon_product.json",
        {
            "requested_url": product_api_url,
            "landing_url": source.landing_url,
            "final_url": final_product_url,
            "product_code": product_code,
            "product": product_payload,
        },
    )
    product_artifact.url = final_product_url
    artifacts.append(product_artifact)
    artifacts.append(
        write_json(
            output_dir / "neon_catalog.json",
            {
                "product_code": product_code,
                "source_id": source.source_id,
                "site_count": len(site_summaries),
                "spectra_file_count": len(discovered_files),
                "sites": site_summaries,
                "files": discovered_files,
            },
        )
    )

    status = "metadata_only"
    notes = [f"Discovered {len(discovered_files)} NEON field spectra files across {len(site_summaries)} sites."]
    if fetch_mode == "assets":
        for file_info in discovered_files:
            artifacts.append(_download_artifact(file_info["url"], output_dir, file_info["name"], user_agent))
        status = "downloaded" if discovered_files else "metadata_only"
        if not discovered_files:
            notes.append("NEON product metadata was available, but no field spectra CSV files were listed.")
    else:
        for index, file_info in enumerate(discovered_files, start=1):
            artifacts.append(
                ArtifactRecord(
                    artifact_id=f"remote_file_{index}",
                    kind="remote_file",
                    url=file_info["url"],
                    path="",
                    media_type="text/csv",
                    size_bytes=int(file_info["size"]) if file_info["size"].isdigit() else 0,
                    sha256="",
                    status="discovered",
                    note=f"{file_info['site_code']} {file_info['month']} {file_info['name']}".strip(),
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

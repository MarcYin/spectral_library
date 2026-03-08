from __future__ import annotations

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json


DOWNLOAD_PAGE_URL = "https://speclib.jpl.nasa.gov/download"
SEARCH_URL = "https://speclib.jpl.nasa.gov/ecospeclib_list"
DATA_ROOT_URL = "https://speclib.jpl.nasa.gov/ecospeclibdata/"
DEFAULT_SEARCH_TYPES = [
    "lunar",
    "manmade",
    "meteorites",
    "mineral",
    "nonphotosyntheticvegetation",
    "rock",
    "soil",
    "vegetation",
    "water",
]


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return max(1, int(value))


def _read_list_env(name: str) -> list[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_search_types(html: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for value in re.findall(r"javascript:orderall\('([^']+)'\)", html):
        if value == "spectra" or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values or DEFAULT_SEARCH_TYPES.copy()


def _extract_class_options(html: str) -> list[str]:
    match = re.search(r'<select name="classsel".*?>(.*?)</select>', html, re.S)
    if not match:
        return []
    seen: set[str] = set()
    values: list[str] = []
    for value in re.findall(r'<option value="([^"]+)"', match.group(1)):
        if value == "All" or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_filenames(html: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for value in re.findall(r'value="([^"]+\.txt)"', html):
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _has_more_results(html: str) -> bool:
    return "Additional data is available" in html


def _post_search(search_type: str, class_value: str, maxhits: int, user_agent: str) -> str:
    payload = urlencode(
        {
            "searchtype": search_type,
            "classsel": class_value,
            "subclass": "All",
            "mname": "",
            "xstart": "",
            "xstop": "",
            "maxhits": str(maxhits),
            "wavelength": "Any",
        }
    ).encode()
    request = Request(
        SEARCH_URL,
        data=payload,
        headers={
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://speclib.jpl.nasa.gov/library",
        },
    )
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", "ignore")


def _enumerate_class(search_type: str, class_value: str, user_agent: str) -> tuple[list[str], dict[str, object]]:
    page_size = 100
    maxhits = page_size
    seen_page_signatures: set[tuple[str, ...]] = set()
    ordered_filenames: dict[str, None] = {}
    has_more_results = False

    while True:
        html = _post_search(search_type, class_value, maxhits, user_agent)
        page_filenames = _parse_filenames(html)
        has_more_results = _has_more_results(html)
        page_signature = tuple(page_filenames)
        if not page_filenames or page_signature in seen_page_signatures:
            break
        seen_page_signatures.add(page_signature)
        before_count = len(ordered_filenames)
        for filename in page_filenames:
            ordered_filenames.setdefault(filename, None)
        if len(page_filenames) < page_size or len(ordered_filenames) == before_count:
            break
        maxhits += page_size

    filenames = list(ordered_filenames.keys())
    return filenames, {
        "search_type": search_type,
        "class_value": class_value,
        "file_count": len(filenames),
        "maxhits_used": maxhits,
        "has_more_results": has_more_results,
    }


def _enumerate_catalog(user_agent: str) -> tuple[list[str], dict[str, object]]:
    request = Request(DOWNLOAD_PAGE_URL, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        download_html = response.read().decode("utf-8", "ignore")

    search_types = _parse_search_types(download_html)
    selected_search_types = _read_list_env("ECOSTRESS_SEARCH_TYPES")
    if selected_search_types:
        allowed = set(selected_search_types)
        search_types = [value for value in search_types if value in allowed]
    all_filenames: list[str] = []
    summaries: list[dict[str, object]] = []
    tasks: list[tuple[str, str]] = []
    for search_type in search_types:
        seed_html = _post_search(search_type, "All", 100, user_agent)
        class_values = _extract_class_options(seed_html)
        if not class_values:
            class_values = ["All"]
        for class_value in class_values:
            tasks.append((search_type, class_value))

    worker_count = _read_int_env("ECOSTRESS_ENUMERATION_WORKERS", 8)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_enumerate_class, search_type, class_value, user_agent): (search_type, class_value)
            for search_type, class_value in tasks
        }
        for future in as_completed(future_map):
            filenames, summary = future.result()
            summaries.append(summary)
            all_filenames.extend(filenames)

    summaries.sort(key=lambda item: (str(item["search_type"]), str(item["class_value"])))

    unique_filenames = sorted(dict.fromkeys(all_filenames))
    return unique_filenames, {
        "download_page_url": DOWNLOAD_PAGE_URL,
        "search_url": SEARCH_URL,
        "data_root_url": DATA_ROOT_URL,
        "search_types": search_types,
        "classes": summaries,
        "total_files": len(unique_filenames),
    }


def _download_spectrum(filename: str, output_dir: Path, user_agent: str) -> ArtifactRecord:
    url = urljoin(DATA_ROOT_URL, filename)
    destination = output_dir / Path(filename).name
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=300) as response:
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
        note=filename,
    )


def _resolve_existing_destination(output_dir: Path, filename: str) -> Path | None:
    basename = Path(filename).name
    candidates = [
        output_dir / basename,
        output_dir / "data" / basename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _reuse_spectrum(filename: str, destination: Path) -> ArtifactRecord:
    url = urljoin(DATA_ROOT_URL, filename)
    return ArtifactRecord(
        artifact_id=destination.stem,
        kind="data",
        url=url,
        path=str(destination),
        media_type="text/plain",
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        status="existing",
        note=filename,
    )


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames, catalog_payload = _enumerate_catalog(user_agent)
    metadata_artifact = write_json(
        output_dir / "ecostress_catalog.json",
        {
            "source_id": source.source_id,
            "landing_url": source.landing_url,
            "download_url": source.download_url,
            **catalog_payload,
        },
    )
    metadata_artifact.url = source.landing_url

    artifacts: list[ArtifactRecord] = [metadata_artifact]
    notes = [f"Discovered {len(filenames)} ECOSTRESS spectral text files via the library search endpoints."]
    status = "metadata_only"

    if fetch_mode == "assets" and filenames:
        max_files = os.environ.get("ECOSTRESS_MAX_FILES", "").strip()
        if max_files:
            selected_filenames = filenames[: int(max_files)]
            notes.append(f"ECOSTRESS_MAX_FILES limited the download to the first {len(selected_filenames)} files.")
        else:
            selected_filenames = filenames

        worker_count = _read_int_env("ECOSTRESS_DOWNLOAD_WORKERS", 8)
        downloaded: list[ArtifactRecord] = []
        existing: list[ArtifactRecord] = []
        pending_filenames: list[str] = []
        for filename in selected_filenames:
            existing_path = _resolve_existing_destination(output_dir, filename)
            if existing_path is not None:
                existing.append(_reuse_spectrum(filename, existing_path))
            else:
                pending_filenames.append(filename)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_download_spectrum, filename, output_dir, user_agent): filename
                for filename in pending_filenames
            }
            for future in as_completed(future_map):
                downloaded.append(future.result())

        merged_artifacts = existing + downloaded
        merged_artifacts.sort(key=lambda artifact: artifact.path)
        artifacts.extend(merged_artifacts)
        status = "downloaded"
        if existing:
            notes.append(f"Reused {len(existing)} existing ECOSTRESS spectral text files already present locally.")
        notes.append(f"Downloaded {len(downloaded)} ECOSTRESS spectral text files from ecospeclibdata.")

    elif fetch_mode == "metadata":
        artifacts.append(
            ArtifactRecord(
                artifact_id="ecostress_data_root",
                kind="remote_file",
                url=DATA_ROOT_URL,
                path="",
                media_type="text/plain",
                size_bytes=0,
                sha256="",
                status="discovered",
                note=f"{len(filenames)} spectra discovered",
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

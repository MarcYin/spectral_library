from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from .http_utils import infer_filename, looks_like_download


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_url = source.download_url if fetch_mode == "assets" and source.download_url else source.landing_url
    notes: list[str] = []
    artifacts: list[ArtifactRecord] = []
    status = "metadata_only"

    try:
        response_handle = urlopen(Request(request_url, headers={"User-Agent": user_agent}), timeout=120)
    except HTTPError as exc:
        if request_url == source.landing_url:
            raise
        notes.append(f"Configured download URL returned HTTP {exc.code}; fell back to the landing URL.")
        request_url = source.landing_url
        response_handle = urlopen(Request(request_url, headers={"User-Agent": user_agent}), timeout=120)

    with response_handle as response:
        final_url = response.geturl()
        media_type = response.headers.get_content_type()
        headers = {key: value for key, value in response.headers.items()}
        metadata_artifact = write_json(
            output_dir / "http_response.json",
            {
                "requested_url": request_url,
                "landing_url": source.landing_url,
                "download_url": source.download_url,
                "final_url": final_url,
                "status": response.status,
                "media_type": media_type,
                "headers": headers,
            },
        )
        metadata_artifact.url = final_url
        artifacts.append(metadata_artifact)

        if fetch_mode == "assets" and looks_like_download(final_url, media_type):
            filename = infer_filename(final_url, f"{source.source_id}.bin")
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
            if source.download_url:
                notes.append("Configured download URL resolved to HTML or a non-download response; manual asset resolution still required.")
            else:
                notes.append("URL resolved to an HTML landing page; manual asset resolution still required.")
        else:
            notes.append("Metadata captured without downloading the remote asset.")

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

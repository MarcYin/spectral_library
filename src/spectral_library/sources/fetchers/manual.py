from __future__ import annotations

from pathlib import Path

from .base import FetchResult, utc_now_iso, write_json


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    del fetch_mode
    del user_agent
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_json(
        output_dir / "manual_review.json",
        {
            "source_id": source.source_id,
            "landing_url": source.landing_url,
            "download_url": source.download_url,
            "auth_mode": source.auth_mode,
            "expected_format": source.expected_format,
            "notes": source.notes,
        },
    )
    finished_at = utc_now_iso()
    return FetchResult(
        source_id=source.source_id,
        source_name=source.name,
        fetch_adapter=source.fetch_adapter,
        fetch_mode="metadata",
        status="manual_review",
        landing_url=source.landing_url,
        started_at=started_at,
        finished_at=finished_at,
        notes=["Source requires manual review or authenticated dataset resolution before fetch."],
        artifacts=[artifact],
    )

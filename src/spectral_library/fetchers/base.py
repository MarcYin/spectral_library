from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ArtifactRecord:
    artifact_id: str
    kind: str
    url: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str
    note: str = ""


@dataclass
class FetchResult:
    source_id: str
    source_name: str
    fetch_adapter: str
    fetch_mode: str
    status: str
    landing_url: str
    started_at: str
    finished_at: str
    notes: list[str] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "fetch_adapter": self.fetch_adapter,
            "fetch_mode": self.fetch_mode,
            "status": self.status,
            "landing_url": self.landing_url,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> ArtifactRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return ArtifactRecord(
        artifact_id=path.stem,
        kind="metadata",
        url="",
        path=str(path),
        media_type="application/json",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        status="written",
    )

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    name: str
    section: str
    subsection: str
    spectral_type: str
    coverage: str
    resource_type: str
    provider: str
    landing_url: str
    download_url: str
    fetch_adapter: str
    auth_mode: str
    expected_format: str
    tier: str
    priority: str
    ingest_role: str
    normalization_eligibility: str
    status: str
    notes: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SourceRecord":
        return cls(**{field: row.get(field, "").strip() for field in cls.__dataclass_fields__})

    def to_row(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def to_matrix_row(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "fetch_adapter": self.fetch_adapter,
            "tier": self.tier,
            "status": self.status,
            "auth_mode": self.auth_mode,
            "expected_format": self.expected_format,
        }


def load_manifest(path: Path) -> list[SourceRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [SourceRecord.from_row(row) for row in reader]


def split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_sources(
    records: Iterable[SourceRecord],
    *,
    source_ids: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    adapters: Iterable[str] | None = None,
) -> list[SourceRecord]:
    source_id_set = set(source_ids or [])
    tier_set = set(tiers or [])
    status_set = set(statuses or [])
    adapter_set = set(adapters or [])

    filtered: list[SourceRecord] = []
    for record in records:
        if source_id_set and record.source_id not in source_id_set:
            continue
        if tier_set and record.tier not in tier_set:
            continue
        if status_set and record.status not in status_set:
            continue
        if adapter_set and record.fetch_adapter not in adapter_set:
            continue
        filtered.append(record)
    return filtered


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

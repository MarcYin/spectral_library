from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from .fetchers import get_fetcher
from .fetchers.base import FetchResult, sha256_file, utc_now_iso, write_json
from .manifest import filter_sources, load_manifest


DOC_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".md",
}
SEED_ONLY_ADAPTERS = {"manual_portal", "earthdata_lpdaac", "specchio_client"}
IGNORED_FILENAMES = {".DS_Store", "batch_summary.json"}


def _classify_output_path(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "metadata"
    if path.suffix.lower() in DOC_EXTENSIONS:
        return "docs"
    return "data"


def _write_fetch_result(path: Path, result: FetchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _retarget_artifact_paths(payload: dict[str, object], output_dir: Path) -> None:
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        raw_path = str(artifact.get("path", "") or "")
        if not raw_path:
            continue
        current_path = output_dir / Path(raw_path).name
        artifact["path"] = str(current_path)
        if current_path.exists() and current_path.is_file():
            artifact["size_bytes"] = current_path.stat().st_size
            artifact["sha256"] = sha256_file(current_path)


def _count_bucket_files(source_dir: Path) -> dict[str, int]:
    counts = {"metadata_files": 0, "doc_files": 0, "data_files": 0}
    for bucket_name, key in (("metadata", "metadata_files"), ("docs", "doc_files"), ("data", "data_files")):
        bucket_dir = source_dir / bucket_name
        if bucket_dir.exists():
            counts[key] = sum(1 for path in bucket_dir.rglob("*") if path.is_file())
    return counts


def tidy_source_directory(source_dir: Path) -> dict[str, int]:
    source_dir.mkdir(parents=True, exist_ok=True)
    moved_counts: Counter[str] = Counter()
    for item in sorted(source_dir.iterdir()):
        if item.name in IGNORED_FILENAMES or item.name == "fetch-result.json":
            continue
        if item.is_dir():
            continue
        bucket = _classify_output_path(item)
        destination_dir = source_dir / bucket
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / item.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(item), destination)
        moved_counts[bucket] += 1

    fetch_result_path = source_dir / "fetch-result.json"
    if fetch_result_path.exists():
        payload = json.loads(fetch_result_path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", [])
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                raw_path = str(artifact.get("path", "") or "")
                if not raw_path:
                    continue
                current_name = Path(raw_path).name
                candidates = [
                    source_dir / "metadata" / current_name,
                    source_dir / "docs" / current_name,
                    source_dir / "data" / current_name,
                    source_dir / current_name,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        artifact["path"] = str(candidate)
                        artifact["size_bytes"] = candidate.stat().st_size
                        artifact["sha256"] = sha256_file(candidate)
                        break
        fetch_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = _count_bucket_files(source_dir)
    for bucket_name, key in (("metadata", "metadata_files"), ("docs", "doc_files"), ("data", "data_files")):
        if counts[key] == 0:
            counts[key] = moved_counts[bucket_name]
    return counts


def _copy_seed_directory(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.iterdir()):
        if item.name in IGNORED_FILENAMES:
            continue
        destination = output_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def seed_source_from_existing(source, output_dir: Path, seed_roots: list[Path]) -> bool:
    for seed_root in seed_roots:
        seed_dir = seed_root / source.source_id
        if not seed_dir.exists() or not seed_dir.is_dir():
            continue
        _copy_seed_directory(seed_dir, output_dir)
        fetch_result_path = output_dir / "fetch-result.json"
        if fetch_result_path.exists():
            payload = json.loads(fetch_result_path.read_text(encoding="utf-8"))
        else:
            data_artifacts = []
            for item in sorted(output_dir.iterdir()):
                if item.name in IGNORED_FILENAMES or item.name == "fetch-result.json" or item.is_dir():
                    continue
                data_artifacts.append(
                    {
                        "artifact_id": item.stem,
                        "kind": "data",
                        "url": source.landing_url,
                        "path": str(item),
                        "media_type": "",
                        "size_bytes": item.stat().st_size,
                        "sha256": sha256_file(item),
                        "status": "downloaded",
                        "note": "Seeded local asset",
                    }
                )
            payload = {
                "source_id": source.source_id,
                "source_name": source.name,
                "fetch_adapter": source.fetch_adapter,
                "fetch_mode": "assets",
                "status": "downloaded" if data_artifacts else "metadata_only",
                "landing_url": source.landing_url,
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "notes": [],
                "artifacts": data_artifacts,
            }

        payload["source_id"] = source.source_id
        payload["source_name"] = source.name
        payload["fetch_adapter"] = source.fetch_adapter
        payload["landing_url"] = source.landing_url
        notes = [str(note) for note in payload.get("notes", [])]
        notes.append(f"Copied from local seed directory {seed_dir}.")
        payload["notes"] = list(dict.fromkeys(notes))
        _retarget_artifact_paths(payload, output_dir)
        fetch_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return True
    return False


def _write_error_result(source, output_dir: Path, fetch_mode: str, exc: Exception) -> FetchResult:
    artifact = write_json(
        output_dir / "fetch_error.json",
        {
            "source_id": source.source_id,
            "fetch_mode": fetch_mode,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )
    finished_at = utc_now_iso()
    return FetchResult(
        source_id=source.source_id,
        source_name=source.name,
        fetch_adapter=source.fetch_adapter,
        fetch_mode=fetch_mode,
        status="error",
        landing_url=source.landing_url,
        started_at=finished_at,
        finished_at=finished_at,
        notes=[f"{type(exc).__name__}: {exc}"],
        artifacts=[artifact],
    )


def fetch_batch(
    manifest_path: Path,
    output_root: Path,
    *,
    fetch_mode: str,
    source_ids: list[str] | None = None,
    tiers: list[str] | None = None,
    statuses: list[str] | None = None,
    adapters: list[str] | None = None,
    user_agent: str,
    continue_on_error: bool,
    seed_roots: list[Path] | None = None,
    clean_output: bool,
    tidy_downloads: bool,
) -> dict[str, object]:
    records = filter_sources(
        load_manifest(manifest_path),
        source_ids=source_ids,
        tiers=tiers,
        statuses=statuses,
        adapters=adapters,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    seed_roots = seed_roots or []

    batch_rows: list[dict[str, object]] = []
    for source in records:
        source_dir = output_root / source.source_id
        if clean_output and source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        try:
            seeded = False
            if source.fetch_adapter in SEED_ONLY_ADAPTERS and seed_roots:
                seeded = seed_source_from_existing(source, source_dir, seed_roots)

            if not seeded:
                fetcher = get_fetcher(source.fetch_adapter)
                result = fetcher(source, source_dir, fetch_mode, user_agent)
                _write_fetch_result(source_dir / "fetch-result.json", result)

            tidy_counts = tidy_source_directory(source_dir) if tidy_downloads else {
                "metadata_files": 0,
                "doc_files": 0,
                "data_files": 0,
            }
            fetch_payload = json.loads((source_dir / "fetch-result.json").read_text(encoding="utf-8"))
            batch_rows.append(
                {
                    "source_id": source.source_id,
                    "status": fetch_payload["status"],
                    "fetch_adapter": source.fetch_adapter,
                    "seeded": seeded,
                    **tidy_counts,
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            error_result = _write_error_result(source, source_dir, fetch_mode, exc)
            _write_fetch_result(source_dir / "fetch-result.json", error_result)
            if tidy_downloads:
                tidy_source_directory(source_dir)
            batch_rows.append(
                {
                    "source_id": source.source_id,
                    "status": "error",
                    "fetch_adapter": source.fetch_adapter,
                    "seeded": False,
                    "metadata_files": 1,
                    "doc_files": 0,
                    "data_files": 0,
                }
            )

    summary = {
        "selected_sources": len(records),
        "status_counts": dict(Counter(str(row["status"]) for row in batch_rows)),
        "seeded_sources": sum(1 for row in batch_rows if row["seeded"]),
        "rows": batch_rows,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

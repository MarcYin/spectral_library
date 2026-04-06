from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .manifest import SourceRecord, load_manifest


SOURCE_COLUMNS = list(SourceRecord.__dataclass_fields__.keys())
FETCH_RESULT_COLUMNS = [
    "source_id",
    "source_name",
    "fetch_adapter",
    "fetch_mode",
    "status",
    "landing_url",
    "started_at",
    "finished_at",
    "notes",
]
ARTIFACT_COLUMNS = [
    "source_id",
    "artifact_id",
    "kind",
    "url",
    "path",
    "media_type",
    "size_bytes",
    "sha256",
    "status",
    "note",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _load_fetch_results(results_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fetch_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    if not results_root.exists():
        return fetch_rows, artifact_rows

    for path in sorted(results_root.rglob("fetch-result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetch_rows.append(
            {
                "source_id": payload["source_id"],
                "source_name": payload["source_name"],
                "fetch_adapter": payload["fetch_adapter"],
                "fetch_mode": payload["fetch_mode"],
                "status": payload["status"],
                "landing_url": payload["landing_url"],
                "started_at": payload["started_at"],
                "finished_at": payload["finished_at"],
                "notes": " | ".join(payload.get("notes", [])),
            }
        )
        for artifact in payload.get("artifacts", []):
            artifact_rows.append(
                {
                    "source_id": payload["source_id"],
                    "artifact_id": artifact.get("artifact_id", ""),
                    "kind": artifact.get("kind", ""),
                    "url": artifact.get("url", ""),
                    "path": artifact.get("path", ""),
                    "media_type": artifact.get("media_type", ""),
                    "size_bytes": artifact.get("size_bytes", 0),
                    "sha256": artifact.get("sha256", ""),
                    "status": artifact.get("status", ""),
                    "note": artifact.get("note", ""),
                }
            )
    return fetch_rows, artifact_rows


def assemble_catalog(manifest_path: Path, results_root: Path, output_root: Path) -> dict[str, object]:
    import duckdb

    tabular_dir = output_root / "tabular"
    parquet_dir = output_root / "parquet"
    db_dir = output_root / "db"
    tabular_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(manifest_path)
    source_rows = [source.to_row() for source in sources]
    fetch_rows, artifact_rows = _load_fetch_results(results_root)

    sources_csv = tabular_dir / "sources.csv"
    fetch_csv = tabular_dir / "fetch_results.csv"
    artifacts_csv = tabular_dir / "artifacts.csv"

    _write_csv(sources_csv, SOURCE_COLUMNS, source_rows)
    _write_csv(fetch_csv, FETCH_RESULT_COLUMNS, fetch_rows)
    _write_csv(artifacts_csv, ARTIFACT_COLUMNS, artifact_rows)

    database_path = db_dir / "catalog.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE sources AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(sources_csv)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE fetch_results AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(fetch_csv)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE artifacts AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(artifacts_csv)],
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW source_build_status AS
            SELECT
              s.source_id,
              s.name,
              s.tier,
              s.status AS manifest_status,
              s.fetch_adapter,
              COALESCE(fr.status, 'not_fetched') AS fetch_status,
              COALESCE(fr.fetch_mode, '') AS fetch_mode,
              COUNT(a.artifact_id) AS artifact_count
            FROM sources s
            LEFT JOIN fetch_results fr USING (source_id)
            LEFT JOIN artifacts a USING (source_id)
            GROUP BY ALL
            """
        )
        connection.execute(
            "COPY sources TO ? (FORMAT PARQUET)",
            [str(parquet_dir / "sources.parquet")],
        )
        connection.execute(
            "COPY fetch_results TO ? (FORMAT PARQUET)",
            [str(parquet_dir / "fetch_results.parquet")],
        )
        connection.execute(
            "COPY artifacts TO ? (FORMAT PARQUET)",
            [str(parquet_dir / "artifacts.parquet")],
        )
    finally:
        connection.close()

    summary = {
        "manifest_sources": len(source_rows),
        "fetched_sources": len(fetch_rows),
        "artifact_rows": len(artifact_rows),
        "manifest_status_counts": dict(Counter(row["status"] for row in source_rows)),
        "fetch_status_counts": dict(Counter(row["status"] for row in fetch_rows)),
    }
    summary_path = output_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

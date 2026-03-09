from __future__ import annotations

import csv
import json
from pathlib import Path


GRID_START_NM = 400
GRID_END_NM = 2500
GRID_SIZE = GRID_END_NM - GRID_START_NM + 1


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _nm_columns(connection: object, table_name: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = [row[1] for row in rows]
    return [column for column in columns if column.startswith("nm_")]


def _write_wavelength_grid(path: Path, nm_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["band_index", "band_name", "wavelength_nm"])
        writer.writeheader()
        for band_index, band_name in enumerate(nm_columns):
            writer.writerow(
                {
                    "band_index": band_index,
                    "band_name": band_name,
                    "wavelength_nm": int(band_name.split("_", 1)[1]),
                }
            )


def _write_readme(path: Path, summary: dict[str, object], normalized_root: Path) -> None:
    text = f"""# SIAC Spectral Library

This package is the SIAC-oriented export built from:

- `{normalized_root}`

It keeps the full curated normalized dataset and preserves the shared
`400-2500 nm` / `1 nm` reflectance grid.

Land-cover labels are included only as optional annotations for subsets and
diagnostics. They are not used as a hard inclusion filter.

## Contents

- `tabular/siac_manifest_sources.csv`: manifest rows for sources retained in the package.
- `tabular/siac_spectra_metadata.csv`: spectra metadata with coverage metrics and optional land-cover labels.
- `tabular/siac_normalized_spectra.csv`: normalized reflectance spectra on the SIAC grid.
- `tabular/siac_source_summary.csv`: source-level spectrum counts, label coverage, and coverage statistics.
- `tabular/siac_landcover_summary.csv`: counts by top-level land-cover group for labeled spectra only.
- `tabular/siac_landcover_prototypes.csv`: pooled, source-level, and source-balanced mean spectra for labeled spectra only.
- `tabular/siac_wavelength_grid.csv`: wavelength lookup for the `nm_*` bands.
- `db/siac_spectral_library.duckdb`: queryable DuckDB database with the same tables and a `siac_spectra` view.

## Summary

- total spectra: {summary["total_spectra"]}
- labeled spectra: {summary["classified_spectra"]}
- unlabeled spectra: {summary["unlabeled_spectra"]}
- sources: {summary["source_count"]}
- landcover groups: {summary["landcover_group_count"]}
- prototypes: {summary["prototype_rows"]}
"""
    path.write_text(text, encoding="utf-8")


def build_siac_library(manifest_path: Path, normalized_root: Path, output_root: Path) -> dict[str, object]:
    import duckdb

    metadata_csv = normalized_root / "tabular" / "spectra_metadata.csv"
    spectra_csv = normalized_root / "tabular" / "normalized_spectra.csv"
    labels_csv = normalized_root / "landcover_analysis" / "landcover_labels.csv"
    missing = [path for path in [manifest_path, metadata_csv, spectra_csv, labels_csv] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SIAC export inputs: {missing_text}")

    tabular_dir = output_root / "tabular"
    parquet_dir = output_root / "parquet"
    db_dir = output_root / "db"
    tabular_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    wavelength_grid_csv = tabular_dir / "siac_wavelength_grid.csv"
    database_path = db_dir / "siac_spectral_library.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE manifest_sources AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(manifest_path)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE raw_metadata AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(metadata_csv)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE raw_spectra AS SELECT * FROM read_csv_auto(?, HEADER=TRUE, SAMPLE_SIZE=-1)",
            [str(spectra_csv)],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE raw_labels AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(labels_csv)],
        )

        nm_columns = _nm_columns(connection, "raw_spectra")
        if not nm_columns:
            raise ValueError(f"No normalized wavelength columns found in {spectra_csv}")

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE siac_spectra_metadata AS
            SELECT
              m.source_id,
              m.source_name,
              l.landcover_group,
              l.classification_rule,
              m.ingest_role,
              m.spectrum_id,
              m.sample_name,
              m.input_path,
              m.parser,
              m.native_wavelength_count,
              m.native_min_nm,
              m.native_max_nm,
              m.native_spacing_nm,
              m.value_scale_applied,
              m.normalized_points,
              CAST(m.normalized_points AS DOUBLE) / {GRID_SIZE:.1f} AS coverage_fraction,
              m.metadata_json
            FROM raw_metadata m
            LEFT JOIN raw_labels l USING (source_id, spectrum_id, sample_name)
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE siac_normalized_spectra AS
            SELECT
              s.source_id,
              s.spectrum_id,
              s.sample_name,
              l.landcover_group,
              l.classification_rule,
              {", ".join(_quote_identifier(column) for column in nm_columns)}
            FROM raw_spectra s
            LEFT JOIN raw_labels l USING (source_id, spectrum_id, sample_name)
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW siac_spectra AS
            SELECT
              m.*,
              s.* EXCLUDE (source_id, spectrum_id, sample_name, landcover_group, classification_rule)
            FROM siac_spectra_metadata m
            JOIN siac_normalized_spectra s USING (source_id, spectrum_id, sample_name)
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW siac_labeled_spectra AS
            SELECT *
            FROM siac_spectra
            WHERE landcover_group IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE siac_manifest_sources AS
            SELECT *
            FROM manifest_sources
            WHERE source_id IN (SELECT DISTINCT source_id FROM siac_spectra_metadata)
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE siac_source_summary AS
            SELECT
              m.source_id,
              MIN(m.source_name) AS source_name,
              MIN(ms.provider) AS provider,
              MIN(ms.tier) AS tier,
              MIN(ms.priority) AS priority,
              MIN(ms.ingest_role) AS manifest_ingest_role,
              COUNT(*) AS spectra_count,
              COUNT(*) FILTER (WHERE m.landcover_group IS NOT NULL) AS labeled_spectra_count,
              COUNT(*) FILTER (WHERE m.landcover_group IS NULL) AS unlabeled_spectra_count,
              COUNT(DISTINCT m.landcover_group) AS landcover_group_count,
              AVG(m.coverage_fraction) AS mean_coverage_fraction,
              MIN(m.native_min_nm) AS min_native_nm,
              MAX(m.native_max_nm) AS max_native_nm
            FROM siac_spectra_metadata m
            LEFT JOIN siac_manifest_sources ms USING (source_id)
            GROUP BY m.source_id
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE siac_landcover_summary AS
            SELECT
              landcover_group,
              COUNT(*) AS spectra_count,
              COUNT(DISTINCT source_id) AS source_count,
              AVG(coverage_fraction) AS mean_coverage_fraction
            FROM siac_spectra_metadata
            WHERE landcover_group IS NOT NULL
            GROUP BY landcover_group
            ORDER BY landcover_group
            """
        )

        source_avg_sql = ",\n              ".join(
            f"AVG({_quote_identifier(column)}) AS {_quote_identifier(column)}" for column in nm_columns
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE source_landcover_prototypes AS
            SELECT
              'source' AS prototype_level,
              landcover_group,
              source_id,
              MIN(source_name) AS source_name,
              COUNT(*) AS spectra_count,
              1 AS source_count,
              {source_avg_sql}
            FROM siac_labeled_spectra
            GROUP BY landcover_group, source_id
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE pooled_landcover_prototypes AS
            SELECT
              'pooled' AS prototype_level,
              landcover_group,
              CAST(NULL AS VARCHAR) AS source_id,
              CAST(NULL AS VARCHAR) AS source_name,
              COUNT(*) AS spectra_count,
              COUNT(DISTINCT source_id) AS source_count,
              {source_avg_sql}
            FROM siac_labeled_spectra
            GROUP BY landcover_group
            """
        )
        balanced_avg_sql = ",\n              ".join(
            f"AVG({_quote_identifier(column)}) AS {_quote_identifier(column)}" for column in nm_columns
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE source_balanced_landcover_prototypes AS
            SELECT
              'source_balanced' AS prototype_level,
              landcover_group,
              CAST(NULL AS VARCHAR) AS source_id,
              CAST(NULL AS VARCHAR) AS source_name,
              SUM(spectra_count) AS spectra_count,
              COUNT(*) AS source_count,
              {balanced_avg_sql}
            FROM source_landcover_prototypes
            GROUP BY landcover_group
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE siac_landcover_prototypes AS
            SELECT * FROM pooled_landcover_prototypes
            UNION ALL
            SELECT * FROM source_balanced_landcover_prototypes
            UNION ALL
            SELECT * FROM source_landcover_prototypes
            """
        )

        _write_wavelength_grid(wavelength_grid_csv, nm_columns)
        connection.execute(
            "CREATE OR REPLACE TABLE siac_wavelength_grid AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)",
            [str(wavelength_grid_csv)],
        )

        export_tables = [
            "siac_manifest_sources",
            "siac_spectra_metadata",
            "siac_normalized_spectra",
            "siac_source_summary",
            "siac_landcover_summary",
            "siac_landcover_prototypes",
            "siac_wavelength_grid",
        ]
        for table_name in export_tables:
            connection.execute(
                f"COPY {table_name} TO ? (FORMAT CSV, HEADER)",
                [str(tabular_dir / f"{table_name}.csv")],
            )
            connection.execute(
                f"COPY {table_name} TO ? (FORMAT PARQUET)",
                [str(parquet_dir / f"{table_name}.parquet")],
            )

        total_spectra = connection.execute("SELECT COUNT(*) FROM siac_spectra_metadata").fetchone()[0]
        classified_spectra = connection.execute(
            "SELECT COUNT(*) FROM siac_spectra_metadata WHERE landcover_group IS NOT NULL"
        ).fetchone()[0]
        source_count = connection.execute("SELECT COUNT(*) FROM siac_source_summary").fetchone()[0]
        labeled_source_count = connection.execute(
            "SELECT COUNT(DISTINCT source_id) FROM siac_spectra_metadata WHERE landcover_group IS NOT NULL"
        ).fetchone()[0]
        landcover_counts = connection.execute(
            "SELECT landcover_group, spectra_count FROM siac_landcover_summary ORDER BY landcover_group"
        ).fetchall()
        prototype_rows = connection.execute("SELECT COUNT(*) FROM siac_landcover_prototypes").fetchone()[0]
    finally:
        connection.close()

    summary = {
        "normalized_root": str(normalized_root),
        "manifest_path": str(manifest_path),
        "total_spectra": int(total_spectra),
        "classified_spectra": int(classified_spectra),
        "unlabeled_spectra": int(total_spectra - classified_spectra),
        "source_count": int(source_count),
        "labeled_source_count": int(labeled_source_count),
        "landcover_group_count": len(landcover_counts),
        "landcover_counts": {group: int(count) for group, count in landcover_counts},
        "prototype_rows": int(prototype_rows),
        "grid_start_nm": GRID_START_NM,
        "grid_end_nm": GRID_END_NM,
        "grid_size": GRID_SIZE,
        "output_tables": [
            "siac_manifest_sources",
            "siac_spectra_metadata",
            "siac_normalized_spectra",
            "siac_source_summary",
            "siac_landcover_summary",
            "siac_landcover_prototypes",
            "siac_wavelength_grid",
        ],
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_root / "README.md", summary, normalized_root)
    return summary

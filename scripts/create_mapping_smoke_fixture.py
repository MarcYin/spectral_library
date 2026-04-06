from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


WAVELENGTHS = list(range(400, 2501))
NM_COLUMNS = [f"nm_{wavelength}" for wavelength in WAVELENGTHS]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _spectrum_values(vnir: float, overlap: float, swir: float) -> dict[str, float]:
    row: dict[str, float] = {}
    for wavelength in WAVELENGTHS:
        if wavelength < 800:
            value = vnir
        elif wavelength <= 1000:
            value = overlap
        else:
            value = swir
        row[f"nm_{wavelength}"] = value
    return row


def _band_payload(
    *,
    band_id: str,
    segment: str,
    wavelength_nm: list[float],
    response: list[float],
) -> dict[str, object]:
    return {
        "band_id": band_id,
        "response_definition": {
            "kind": "sampled",
            "wavelength_nm": wavelength_nm,
            "response": response,
        },
        "extensions": {
            "spectral_library": {
                "segment": segment,
            }
        },
    }


def _sensor_payload(*, sensor_id: str, bands: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_type": "rsrf_sensor_definition",
        "schema_version": "1.0.0",
        "sensor_id": sensor_id,
        "bands": bands,
    }


def create_smoke_fixture(output_root: Path) -> dict[str, str]:
    output_root = Path(output_root)
    siac_root = output_root / "siac"
    srf_root = output_root / "srfs"
    prepared_root = output_root / "prepared"

    metadata_rows = [
        {
            "source_id": "smoke_source",
            "spectrum_id": "base",
            "sample_name": "base",
            "source_name": "Smoke Source",
            "landcover_group": "soil",
        },
        {
            "source_id": "smoke_source",
            "spectrum_id": "vnir_high",
            "sample_name": "vnir_high",
            "source_name": "Smoke Source",
            "landcover_group": "soil",
        },
        {
            "source_id": "smoke_source",
            "spectrum_id": "swir_high",
            "sample_name": "swir_high",
            "source_name": "Smoke Source",
            "landcover_group": "soil",
        },
    ]
    spectra_rows = [
        {
            "source_id": "smoke_source",
            "spectrum_id": "base",
            "sample_name": "base",
            **_spectrum_values(0.15, 0.25, 0.25),
        },
        {
            "source_id": "smoke_source",
            "spectrum_id": "vnir_high",
            "sample_name": "vnir_high",
            **_spectrum_values(0.80, 0.40, 0.20),
        },
        {
            "source_id": "smoke_source",
            "spectrum_id": "swir_high",
            "sample_name": "swir_high",
            **_spectrum_values(0.10, 0.90, 0.90),
        },
    ]

    _write_csv(
        siac_root / "tabular" / "siac_spectra_metadata.csv",
        ["source_id", "spectrum_id", "sample_name", "source_name", "landcover_group"],
        metadata_rows,
    )
    _write_csv(
        siac_root / "tabular" / "siac_normalized_spectra.csv",
        ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
        spectra_rows,
    )

    sensor_a = _sensor_payload(
        sensor_id="sensor_a",
        bands=[
            _band_payload(
                band_id="blue",
                segment="vnir",
                wavelength_nm=[445.0, 450.0, 455.0],
                response=[0.2, 1.0, 0.2],
            ),
            _band_payload(
                band_id="swir",
                segment="swir",
                wavelength_nm=[1595.0, 1600.0, 1605.0],
                response=[0.2, 1.0, 0.2],
            ),
        ],
    )
    sensor_b = _sensor_payload(
        sensor_id="sensor_b",
        bands=[
            _band_payload(
                band_id="target_vnir",
                segment="vnir",
                wavelength_nm=[495.0, 500.0, 505.0],
                response=[0.2, 1.0, 0.2],
            ),
            _band_payload(
                band_id="target_swir",
                segment="swir",
                wavelength_nm=[1695.0, 1700.0, 1705.0],
                response=[0.2, 1.0, 0.2],
            ),
        ],
    )
    srf_root.mkdir(parents=True, exist_ok=True)
    (srf_root / "sensor_a.json").write_text(json.dumps(sensor_a, indent=2) + "\n", encoding="utf-8")
    (srf_root / "sensor_b.json").write_text(json.dumps(sensor_b, indent=2) + "\n", encoding="utf-8")

    single_query_path = output_root / "query_single.csv"
    _write_csv(
        single_query_path,
        ["band_id", "reflectance"],
        [
            {"band_id": "blue", "reflectance": 0.80},
            {"band_id": "swir", "reflectance": 0.20},
        ],
    )

    batch_query_path = output_root / "query_batch.csv"
    _write_csv(
        batch_query_path,
        ["sample_id", "exclude_row_id", "blue", "swir", "valid_swir"],
        [
            {
                "sample_id": "alpha",
                "exclude_row_id": "smoke_source:vnir_high:vnir_high",
                "blue": 0.80,
                "swir": 0.20,
                "valid_swir": "true",
            },
            {
                "sample_id": "beta",
                "exclude_row_id": "smoke_source:swir_high:swir_high",
                "blue": 0.80,
                "swir": 0.90,
                "valid_swir": "false",
            },
        ],
    )

    summary = {
        "fixture_root": str(output_root),
        "siac_root": str(siac_root),
        "srf_root": str(srf_root),
        "prepared_root": str(prepared_root),
        "single_query_csv": str(single_query_path),
        "batch_query_csv": str(batch_query_path),
    }
    summary_path = output_root / "fixture_paths.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a minimal SIAC and SRF fixture for package smoke tests.")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    summary = create_smoke_fixture(Path(args.output_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

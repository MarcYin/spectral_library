from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from spectral_library import PreparedLibraryValidationError, benchmark_mapping, validate_prepared_library
from spectral_library.mapping import SUPPORTED_KNN_BACKENDS, SUPPORTED_NEIGHBOR_ESTIMATORS


def _parse_repeated_csv(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if text and text not in normalized:
                normalized.append(text)
    return normalized


def _ordered_sensor_pairs(sensor_ids: list[str]) -> list[tuple[str, str]]:
    return [(source, target) for source in sensor_ids for target in sensor_ids if source != target]


def _threshold_value(thresholds: dict[str, object], *, scenario_key: str, metric_key: str) -> float | None:
    scenario_thresholds = thresholds.get("scenarios", {})
    if isinstance(scenario_thresholds, dict):
        scenario_payload = scenario_thresholds.get(scenario_key, {})
        if isinstance(scenario_payload, dict) and metric_key in scenario_payload:
            return float(scenario_payload[metric_key])
    defaults = thresholds.get("defaults", {})
    if isinstance(defaults, dict) and metric_key in defaults:
        return float(defaults[metric_key])
    return None


def _metric_value(report: dict[str, object], metric_key: str) -> float:
    value: object = report
    for part in metric_key.split("."):
        if not isinstance(value, dict):
            raise KeyError(metric_key)
        value = value[part]
    return float(value)


def _scenario_key(*, source_sensor: str, target_sensor: str, neighbor_estimator: str, knn_backend: str, k: int) -> str:
    return f"{source_sensor}->{target_sensor}|{neighbor_estimator}|{knn_backend}|{k}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmarks(
    prepared_root: Path,
    *,
    source_sensors: list[str],
    neighbor_estimators: list[str],
    knn_backends: list[str],
    k_values: list[int],
    test_fraction: float,
    max_test_rows: int | None,
    random_seed: int,
    output_root: Path,
    thresholds_path: Path | None,
    fail_on_thresholds: bool,
) -> int:
    manifest = validate_prepared_library(prepared_root, verify_checksums=False)
    available_sources = list(manifest.source_sensors)
    if not source_sensors:
        source_sensors = available_sources
    unknown_sensors = sorted(sensor_id for sensor_id in source_sensors if sensor_id not in available_sources)
    if unknown_sensors:
        raise PreparedLibraryValidationError(
            "Requested benchmark sensors are not present in the prepared runtime.",
            context={"unknown_sensors": unknown_sensors, "available_sensors": available_sources},
        )

    thresholds: dict[str, object] = {}
    if thresholds_path:
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    per_run_reports: list[dict[str, object]] = []

    for source_sensor, target_sensor in _ordered_sensor_pairs(source_sensors):
        for neighbor_estimator in neighbor_estimators:
            for knn_backend in knn_backends:
                for k in k_values:
                    scenario = _scenario_key(
                        source_sensor=source_sensor,
                        target_sensor=target_sensor,
                        neighbor_estimator=neighbor_estimator,
                        knn_backend=knn_backend,
                        k=k,
                    )
                    report = benchmark_mapping(
                        prepared_root,
                        source_sensor,
                        target_sensor,
                        k=k,
                        test_fraction=test_fraction,
                        max_test_rows=max_test_rows,
                        random_seed=random_seed,
                        neighbor_estimator=neighbor_estimator,
                        knn_backend=knn_backend,
                    )
                    report["scenario_key"] = scenario
                    report["prepared_root"] = str(prepared_root)
                    run_path = runs_root / f"{scenario.replace('|', '__').replace('->', '_to_')}.json"
                    run_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    per_run_reports.append(report)

                    row = {
                        "scenario_key": scenario,
                        "source_sensor": source_sensor,
                        "target_sensor": target_sensor,
                        "neighbor_estimator": neighbor_estimator,
                        "knn_backend": knn_backend,
                        "k": int(k),
                        "target_mean_mae": float(report["target_sensor"]["retrieval"]["mean_mae"]),  # type: ignore[index]
                        "target_mean_rmse": float(report["target_sensor"]["retrieval"]["mean_rmse"]),  # type: ignore[index]
                        "full_mean_mae": float(report["full_spectrum"]["retrieval"]["mean_mae"]),  # type: ignore[index]
                        "full_mean_rmse": float(report["full_spectrum"]["retrieval"]["mean_rmse"]),  # type: ignore[index]
                        "train_rows": int(report["train_rows"]),
                        "test_rows": int(report["test_rows"]),
                    }
                    summary_rows.append(row)

                    for metric_key in (
                        "target_sensor.retrieval.mean_mae",
                        "target_sensor.retrieval.mean_rmse",
                        "full_spectrum.retrieval.mean_mae",
                    ):
                        limit = _threshold_value(thresholds, scenario_key=scenario, metric_key=metric_key)
                        if limit is None:
                            continue
                        actual = _metric_value(report, metric_key)
                        if actual > limit:
                            failures.append(
                                {
                                    "scenario_key": scenario,
                                    "metric": metric_key,
                                    "actual": actual,
                                    "limit": limit,
                                }
                            )

    summary_payload = {
        "prepared_root": str(prepared_root),
        "source_sensors": source_sensors,
        "neighbor_estimators": neighbor_estimators,
        "knn_backends": knn_backends,
        "k_values": k_values,
        "test_fraction": float(test_fraction),
        "max_test_rows": None if max_test_rows is None else int(max_test_rows),
        "random_seed": int(random_seed),
        "run_count": len(summary_rows),
        "threshold_failures": failures,
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_root / "summary.csv", summary_rows)
    (output_root / "reports.json").write_text(json.dumps(per_run_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failures and fail_on_thresholds:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run held-out full-library mapping benchmarks.")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--source-sensor", action="append", default=[])
    parser.add_argument("--neighbor-estimator", action="append", default=[])
    parser.add_argument("--knn-backend", action="append", default=[], choices=list(SUPPORTED_KNN_BACKENDS))
    parser.add_argument("--k", action="append", default=[])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-test-rows", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--fail-on-thresholds", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    source_sensors = _parse_repeated_csv(list(args.source_sensor))
    neighbor_estimators = _parse_repeated_csv(list(args.neighbor_estimator)) or ["simplex_mixture"]
    unknown_estimators = [name for name in neighbor_estimators if name not in SUPPORTED_NEIGHBOR_ESTIMATORS]
    if unknown_estimators:
        raise SystemExit(f"Unsupported neighbor estimators: {unknown_estimators}")
    knn_backends = _parse_repeated_csv(list(args.knn_backend)) or ["numpy"]
    k_values = [int(value) for value in (_parse_repeated_csv(list(args.k)) or ["10"])]
    exit_code = run_benchmarks(
        Path(args.prepared_root),
        source_sensors=source_sensors,
        neighbor_estimators=neighbor_estimators,
        knn_backends=knn_backends,
        k_values=k_values,
        test_fraction=float(args.test_fraction),
        max_test_rows=(int(args.max_test_rows) if int(args.max_test_rows) > 0 else None),
        random_seed=int(args.random_seed),
        output_root=Path(args.output_root),
        thresholds_path=Path(args.thresholds) if args.thresholds else None,
        fail_on_thresholds=bool(args.fail_on_thresholds),
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

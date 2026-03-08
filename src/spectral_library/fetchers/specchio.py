from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json


def _load_jpype_module():
    import jpype

    return jpype


def _load_query_config(value: str | None) -> list[dict[str, str]]:
    if not value:
        return []

    text = value.strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        payload = json.loads(text)
    else:
        payload = json.loads(Path(text).read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        payload = payload.get("conditions", [])

    if not isinstance(payload, list):
        raise ValueError("SPECCHIO_QUERY_JSON must resolve to a list of query conditions.")

    conditions: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("SPECCHIO query conditions must be objects.")
        if not {"attribute", "operator", "value"}.issubset(item):
            raise ValueError("SPECCHIO query conditions must contain attribute, operator, and value.")
        conditions.append(
            {
                "attribute": str(item["attribute"]),
                "operator": str(item["operator"]),
                "value": str(item["value"]),
            }
        )
    return conditions


def _safe_len(value) -> int:
    if hasattr(value, "size"):
        return int(value.size())
    return len(list(value))


def _to_list(value) -> list:
    if isinstance(value, list):
        return value
    return list(value)


def _runtime_snapshot(source, query_conditions: list[dict[str, str]]) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "landing_url": source.landing_url,
        "client_jar": os.environ.get("SPECCHIO_CLIENT_JAR", ""),
        "jvm_path": os.environ.get("SPECCHIO_JVM_PATH", ""),
        "java_home": os.environ.get("SPECCHIO_JAVA_HOME", ""),
        "server_index": os.environ.get("SPECCHIO_SERVER_INDEX", "0"),
        "order_by": os.environ.get("SPECCHIO_ORDER_BY", "Acquisition Time"),
        "query_conditions": query_conditions,
    }


def _parse_server_index(value: object) -> int:
    try:
        index = int(str(value))
    except ValueError as exc:
        raise ValueError("SPECCHIO_SERVER_INDEX must be an integer.") from exc
    if index < 0:
        raise ValueError("SPECCHIO_SERVER_INDEX must be zero or greater.")
    return index


def _descriptor_name(descriptor) -> str:
    if hasattr(descriptor, "getDataSourceName"):
        return str(descriptor.getDataSourceName())
    return str(descriptor)


def _write_space_csv(space, output_dir: Path, index: int) -> ArtifactRecord:
    wavelengths = _to_list(space.getAverageWavelengths())
    vectors = [_to_list(vector) for vector in _to_list(space.getVectorsAsArray())]
    spectrum_ids = _to_list(space.getSpectrumIds())
    unit = ""
    if hasattr(space, "getMeasurementUnit") and space.getMeasurementUnit() is not None:
        unit = str(space.getMeasurementUnit().getUnitName())

    path = output_dir / f"space_{index:03d}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spectrum_id", "measurement_unit", *wavelengths])
        for spectrum_id, vector in zip(spectrum_ids, vectors):
            writer.writerow([spectrum_id, unit, *vector])

    return ArtifactRecord(
        artifact_id=path.stem,
        kind="data",
        url="",
        path=str(path),
        media_type="text/csv",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        status="downloaded",
        note=f"{len(vectors)} spectra",
    )


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    del user_agent
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)

    query_conditions = _load_query_config(os.environ.get("SPECCHIO_QUERY_JSON"))
    runtime_payload = _runtime_snapshot(source, query_conditions)

    client_jar = runtime_payload["client_jar"]
    if not client_jar:
        artifact = write_json(output_dir / "specchio_runtime.json", runtime_payload)
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
            notes=["SPECCHIO requires SPECCHIO_CLIENT_JAR and query configuration before automated access can run."],
            artifacts=[artifact],
        )

    if fetch_mode == "assets" and not query_conditions:
        artifact = write_json(output_dir / "specchio_runtime.json", runtime_payload)
        finished_at = utc_now_iso()
        return FetchResult(
            source_id=source.source_id,
            source_name=source.name,
            fetch_adapter=source.fetch_adapter,
            fetch_mode=fetch_mode,
            status="manual_review",
            landing_url=source.landing_url,
            started_at=started_at,
            finished_at=finished_at,
            notes=["SPECCHIO assets mode requires SPECCHIO_QUERY_JSON with at least one query condition."],
            artifacts=[artifact],
        )

    jpype = _load_jpype_module()
    if runtime_payload["java_home"]:
        os.environ["JAVA_HOME"] = str(runtime_payload["java_home"])
    if not jpype.isJVMStarted():
        jvm_path = runtime_payload["jvm_path"] or jpype.getDefaultJVMPath()
        jpype.startJVM(jvm_path, "-ea", f"-Djava.class.path={client_jar}")

    ch = jpype.JPackage("ch")
    specchio_pkg = ch.specchio
    client_pkg = specchio_pkg.client
    queries_pkg = specchio_pkg.queries

    client_factory = client_pkg.SPECCHIOClientFactory.getInstance()
    descriptors = _to_list(client_factory.getAllServerDescriptors())
    if not descriptors:
        raise ValueError("SPECCHIO returned no server descriptors.")
    server_index = _parse_server_index(runtime_payload["server_index"])
    if server_index >= len(descriptors):
        raise ValueError(f"SPECCHIO_SERVER_INDEX {server_index} is out of range for {len(descriptors)} descriptors.")
    descriptor = descriptors[server_index]
    specchio_client = client_factory.createClient(descriptor)

    query = queries_pkg.Query()
    applied_conditions: list[dict[str, str]] = []
    attribute_hash = specchio_client.getAttributesNameHash()
    for condition in query_conditions:
        attr = attribute_hash.get(condition["attribute"])
        if attr is None:
            raise ValueError(f"SPECCHIO attribute not found: {condition['attribute']}")
        query_condition = queries_pkg.EAVQueryConditionObject(attr)
        query_condition.setValue(condition["value"])
        query_condition.setOperator(condition["operator"])
        query.add_condition(query_condition)
        applied_conditions.append(condition)

    ids = specchio_client.getSpectrumIdsMatchingQuery(query) if applied_conditions else []
    ids_count = _safe_len(ids) if applied_conditions else 0
    spaces = specchio_client.getSpaces(ids, runtime_payload["order_by"]) if applied_conditions else []
    spaces_count = len(_to_list(spaces)) if applied_conditions else 0

    metadata_artifact = write_json(
        output_dir / "specchio_runtime.json",
        {
            **runtime_payload,
            "descriptor_name": _descriptor_name(descriptor),
            "descriptor_names": [_descriptor_name(item) for item in descriptors],
            "ids_count": ids_count,
            "spaces_count": spaces_count,
            "applied_conditions": applied_conditions,
        },
    )

    if fetch_mode == "metadata":
        finished_at = utc_now_iso()
        return FetchResult(
            source_id=source.source_id,
            source_name=source.name,
            fetch_adapter=source.fetch_adapter,
            fetch_mode=fetch_mode,
            status="metadata_only",
            landing_url=source.landing_url,
            started_at=started_at,
            finished_at=finished_at,
            notes=[f"Connected to SPECCHIO descriptor {_descriptor_name(descriptor)} and resolved {ids_count} spectra."],
            artifacts=[metadata_artifact],
        )

    artifacts: list[ArtifactRecord] = [metadata_artifact]
    for index, space in enumerate(_to_list(spaces), start=1):
        loaded_space = specchio_client.loadSpace(space)
        artifacts.append(_write_space_csv(loaded_space, output_dir, index))

    finished_at = utc_now_iso()
    return FetchResult(
        source_id=source.source_id,
        source_name=source.name,
        fetch_adapter=source.fetch_adapter,
        fetch_mode=fetch_mode,
        status="downloaded" if len(artifacts) > 1 else "metadata_only",
        landing_url=source.landing_url,
        started_at=started_at,
        finished_at=finished_at,
        notes=[f"Loaded {ids_count} spectra across {spaces_count} SPECCHIO spaces."],
        artifacts=artifacts,
    )

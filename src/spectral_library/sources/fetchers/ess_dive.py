from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json


def _extract_doi(url: str) -> str:
    doi_match = re.search(r"10\.5440/\d+", url)
    if doi_match:
        return doi_match.group(0)

    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "biblio" in parts:
        index = parts.index("biblio")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return f"10.5440/{parts[index + 1]}"
    raise ValueError(f"Unable to extract ESS-DIVE DOI from {url}")


def _load_json(url: str, user_agent: str) -> tuple[dict[str, object], str]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        return json.load(response), response.geturl()


def _load_text(url: str, user_agent: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="ignore"), response.geturl()


def _parse_sysmeta(xml_text: str) -> dict[str, object]:
    def _match(pattern: str) -> str:
        match = re.search(pattern, xml_text)
        return match.group(1) if match else ""

    size_text = _match(r"<size>(.*?)</size>")
    return {
        "identifier": _match(r"<identifier>(.*?)</identifier>"),
        "format_id": _match(r"<formatId>(.*?)</formatId>"),
        "file_name": _match(r"<fileName>(.*?)</fileName>"),
        "size_bytes": int(size_text) if size_text.isdigit() else 0,
    }


def _is_metadata_member(member: dict[str, object], metadata_identifier: str) -> bool:
    format_id = str(member.get("format_id", ""))
    file_name = str(member.get("file_name", ""))
    identifier = str(member.get("identifier", ""))
    if identifier == metadata_identifier:
        return True
    if format_id.startswith("https://eml.ecoinformatics.org"):
        return True
    if "resource" in format_id.lower() and "map" in format_id.lower():
        return True
    if file_name.endswith(".xml") or file_name.endswith(".rdf"):
        return True
    return False


def _download_artifact(url: str, output_dir: Path, filename: str, user_agent: str) -> ArtifactRecord:
    request = Request(url, headers={"User-Agent": user_agent})
    destination = output_dir / filename
    with urlopen(request, timeout=300) as response:
        media_type = response.headers.get_content_type()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    return ArtifactRecord(
        artifact_id=destination.stem,
        kind="data",
        url=url,
        path=str(destination),
        media_type=media_type,
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        status="downloaded",
    )


def fetch(source, output_dir: Path, fetch_mode: str, user_agent: str) -> FetchResult:
    started_at = utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)

    doi = _extract_doi(source.landing_url)
    solr_url = (
        "https://cn.dataone.org/cn/v2/query/solr/"
        f"?q={quote(doi)}&fl=id,identifier,seriesId,formatId,title,documents&rows=20&wt=json"
    )
    solr_payload, final_solr_url = _load_json(solr_url, user_agent)
    docs = solr_payload.get("response", {}).get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"No ESS-DIVE package found for {doi}")

    package = docs[0]
    metadata_identifier = str(package.get("id", "") or package.get("identifier", ""))
    document_ids = [str(value) for value in package.get("documents", []) if str(value)]
    members: list[dict[str, object]] = []
    for document_id in document_ids:
        sysmeta_url = f"https://cn.dataone.org/cn/v2/meta/{quote(document_id, safe='')}"
        sysmeta_text, final_meta_url = _load_text(sysmeta_url, user_agent)
        member = _parse_sysmeta(sysmeta_text)
        member["sysmeta_url"] = final_meta_url
        member["resolve_url"] = f"https://cn.dataone.org/cn/v2/resolve/{quote(document_id, safe='')}"
        member["is_metadata"] = _is_metadata_member(member, metadata_identifier)
        members.append(member)

    artifacts: list[ArtifactRecord] = []
    metadata_artifact = write_json(
        output_dir / "ess_dive_package.json",
        {
            "requested_url": solr_url,
            "landing_url": source.landing_url,
            "final_url": final_solr_url,
            "doi": doi,
            "package": package,
            "members": members,
        },
    )
    metadata_artifact.url = final_solr_url
    artifacts.append(metadata_artifact)

    data_members = [member for member in members if not member["is_metadata"]]
    notes = [f"Discovered {len(data_members)} ESS-DIVE package members for {doi}."]
    status = "metadata_only"
    if fetch_mode == "assets":
        for member in data_members:
            file_name = str(member.get("file_name", "")) or f"{member['identifier']}.bin"
            artifacts.append(
                _download_artifact(
                    str(member["resolve_url"]),
                    output_dir,
                    file_name,
                    user_agent,
                )
            )
        status = "downloaded" if data_members else "metadata_only"
    else:
        for index, member in enumerate(data_members, start=1):
            artifacts.append(
                ArtifactRecord(
                    artifact_id=f"remote_file_{index}",
                    kind="remote_file",
                    url=str(member["resolve_url"]),
                    path="",
                    media_type=str(member.get("format_id", "")),
                    size_bytes=int(member.get("size_bytes", 0)),
                    sha256="",
                    status="discovered",
                    note=str(member.get("file_name", "")),
                )
            )

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

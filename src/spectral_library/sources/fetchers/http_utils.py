from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


DOWNLOAD_SUFFIXES = {
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".bz2",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".xlsx",
    ".xls",
    ".nc",
    ".pdf",
    ".hdr",
    ".sli",
    ".asd",
}


def sanitize_filename(name: str) -> str:
    safe = name.replace("/", "_").replace("\\", "_").strip()
    return safe or "artifact.bin"


def infer_filename(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        name = fallback
    return sanitize_filename(name)


def looks_like_download(url: str, media_type: str) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in DOWNLOAD_SUFFIXES:
        return True
    media_type = media_type.lower()
    return not media_type.startswith("text/html")

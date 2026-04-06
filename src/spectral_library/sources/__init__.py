"""Source acquisition and catalog-building package."""

from .catalog import assemble_catalog
from .fetch import fetch_batch, seed_source_from_existing, tidy_source_directory
from .manifest import SourceRecord, filter_sources, load_manifest, manifest_sha256, split_csv_arg

__all__ = [
    "SourceRecord",
    "assemble_catalog",
    "fetch_batch",
    "filter_sources",
    "load_manifest",
    "manifest_sha256",
    "seed_source_from_existing",
    "split_csv_arg",
    "tidy_source_directory",
]

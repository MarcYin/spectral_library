from __future__ import annotations

from . import ecosis, ecostress, ess_dive, github_archive, manual, mendeley, neon, pangaea, specchio, static_http, zenodo

FETCHERS = {
    "static_http": static_http.fetch,
    "ecostress_web": ecostress.fetch,
    "github_archive": github_archive.fetch,
    "zenodo_api": zenodo.fetch,
    "ecosis_package": ecosis.fetch,
    "ess_dive_dataone": ess_dive.fetch,
    "mendeley_public": mendeley.fetch,
    "neon_api": neon.fetch,
    "pangaea": pangaea.fetch,
    "specchio_client": specchio.fetch,
    "manual_portal": manual.fetch,
    "earthdata_lpdaac": manual.fetch,
}


def get_fetcher(name: str):
    if name not in FETCHERS:
        raise KeyError(f"Unsupported fetch adapter: {name}")
    return FETCHERS[name]

"""Backward-compatible wrapper package for fetcher adapters."""

from .base import *  # noqa: F401,F403
from .ecosis import *  # noqa: F401,F403
from .ecostress import *  # noqa: F401,F403
from .ess_dive import *  # noqa: F401,F403
from .github_archive import *  # noqa: F401,F403
from .manual import *  # noqa: F401,F403
from .mendeley import *  # noqa: F401,F403
from .neon import *  # noqa: F401,F403
from .pangaea import *  # noqa: F401,F403
from .specchio import *  # noqa: F401,F403
from .static_http import *  # noqa: F401,F403
from .zenodo import *  # noqa: F401,F403
from ..sources.fetchers import FETCHERS, get_fetcher

__all__ = ["FETCHERS", "get_fetcher"]

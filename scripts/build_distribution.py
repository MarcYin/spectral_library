from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _remove_stale_package_artifacts(dist_dir: Path) -> None:
    for pattern in ("spectral_library-*.whl", "spectral_library-*.tar.gz"):
        for artifact in dist_dir.glob(pattern):
            artifact.unlink()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(exist_ok=True)
    _remove_stale_package_artifacts(dist_dir)
    filtered_sys_path: list[str] = []
    for entry in sys.path:
        resolved = repo_root if not entry else Path(entry).resolve()
        if resolved == repo_root:
            continue
        filtered_sys_path.append(entry)
    sys.path[:] = filtered_sys_path
    runpy.run_module("build", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

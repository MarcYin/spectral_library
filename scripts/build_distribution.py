from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
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

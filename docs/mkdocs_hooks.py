from __future__ import annotations

import shutil
from pathlib import Path

from mkdocs.structure.files import File, Files


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_GLOB_ROOTS = (
    ("examples/official_mapping", {"README.md"}),
)
COPIED_FILES = ("LICENSE", "CNAME")
LINK_REWRITES = {
    "../examples/": "examples/",
}


def on_page_markdown(markdown: str, **_: object) -> str:
    rewritten = markdown
    for source, target in LINK_REWRITES.items():
        rewritten = rewritten.replace(source, target)
    return rewritten


def on_files(files: Files, config: dict[str, object], **_: object) -> Files:
    site_dir = str(config["site_dir"])
    use_directory_urls = bool(config["use_directory_urls"])

    for root_name, excluded_names in STATIC_GLOB_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in excluded_names:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            files.append(
                File(
                    relative,
                    src_dir=str(REPO_ROOT),
                    dest_dir=site_dir,
                    use_directory_urls=use_directory_urls,
                )
            )
    return files


def on_post_build(config: dict[str, object], **_: object) -> None:
    site_dir = Path(str(config["site_dir"]))

    for file_name in COPIED_FILES:
        source = REPO_ROOT / file_name
        if source.exists():
            destination = site_dir / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

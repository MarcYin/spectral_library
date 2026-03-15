from __future__ import annotations

import argparse
import html
import os
import re
import shutil
from pathlib import Path

try:
    import markdown
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scripts/build_docs_site.py requires the optional Markdown dependency. "
        'Install it with `python3 -m pip install markdown` or `python3 -m pip install ".[dev]"`.'
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "docs-site"
SITE_CSS_RELATIVE = Path("assets") / "site.css"
LOCAL_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]
NAV_ITEMS = (
    ("Docs Home", Path("docs/index.html")),
    ("Quickstart", Path("docs/mapping_quickstart.html")),
    ("Official Examples", Path("docs/official_sensor_examples.html")),
    ("CLI Reference", Path("docs/cli_reference.html")),
    ("Python API", Path("docs/python_api_reference.html")),
    ("Runtime Contract", Path("docs/prepared_runtime_contract.html")),
    ("Changelog", Path("CHANGELOG.html")),
    ("Release Notes", Path("docs/releases/0.1.0.html")),
    ("README", Path("README.html")),
)


def _markdown_sources() -> list[Path]:
    sources = {
        REPO_ROOT / "README.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "RELEASE.md",
    }
    sources.update(path for path in (REPO_ROOT / "docs").rglob("*.md"))
    sources.update(path for path in (REPO_ROOT / "examples").rglob("*.md"))
    return sorted(path for path in sources if path.exists())


def _static_roots() -> tuple[Path, ...]:
    return (
        REPO_ROOT / "docs" / "assets",
        REPO_ROOT / "examples",
        REPO_ROOT / "tests",
    )


def _output_path_for(source_path: Path, output_root: Path) -> Path:
    relative = source_path.relative_to(REPO_ROOT)
    return output_root / relative.with_suffix(".html")


def _rewrite_local_markdown_links(text: str, *, source_path: Path, output_root: Path) -> str:
    source_output = _output_path_for(source_path, output_root)

    def replacer(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        target_path, anchor = (target.split("#", 1) + [""])[:2]
        if not target_path:
            return match.group(0)
        if not target_path.endswith(".md"):
            return match.group(0)

        resolved_markdown = (source_path.parent / target_path).resolve()
        try:
            target_relative = resolved_markdown.relative_to(REPO_ROOT)
        except ValueError:
            return match.group(0)

        target_output = output_root / target_relative.with_suffix(".html")
        rewritten = Path(os.path.relpath(target_output, source_output.parent)).as_posix()
        if anchor:
            rewritten = f"{rewritten}#{anchor}"
        return f"[{label}]({rewritten})"

    return LOCAL_LINK_PATTERN.sub(replacer, text)


def _page_title(markdown_text: str, source_path: Path) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return source_path.stem.replace("_", " ").title()


def _nav_html(current_output: Path, output_root: Path) -> str:
    items: list[str] = []
    for label, relative_target in NAV_ITEMS:
        href = Path(os.path.relpath(output_root / relative_target, current_output.parent)).as_posix()
        items.append(f'<a href="{html.escape(href)}">{html.escape(label)}</a>')
    return "".join(f"<li>{item}</li>" for item in items)


def _render_html(markdown_text: str, *, source_path: Path, output_root: Path) -> str:
    rewritten = _rewrite_local_markdown_links(markdown_text, source_path=source_path, output_root=output_root)
    body = markdown.markdown(rewritten, extensions=MARKDOWN_EXTENSIONS)
    title = _page_title(markdown_text, source_path)
    output_path = _output_path_for(source_path, output_root)
    stylesheet_href = Path(os.path.relpath(output_root / SITE_CSS_RELATIVE, output_path.parent)).as_posix()
    nav_html = _nav_html(output_path, output_root)
    source_label = source_path.relative_to(REPO_ROOT).as_posix()
    home_href = Path(os.path.relpath(output_root / "docs/index.html", output_path.parent)).as_posix()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} | spectral-library</title>
  <link rel="stylesheet" href="{html.escape(stylesheet_href)}">
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand-wrap">
        <div class="eyebrow">GitHub Pages</div>
        <div class="brand">
          <a href="{html.escape(home_href)}">spectral-library docs</a>
        </div>
        <p class="tagline">Prepared-runtime mapping, official sensor examples, and release docs in one site.</p>
      </div>
      <ul class="nav">
        {nav_html}
      </ul>
    </aside>
    <main class="content">
      <div class="source-path">{html.escape(source_label)}</div>
      {body}
    </main>
  </div>
</body>
</html>
"""


def _write_site_css(output_root: Path) -> None:
    css = """body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: #1f2933;
  background: linear-gradient(180deg, #f6f4ed 0%, #ffffff 35%);
}

.layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  box-sizing: border-box;
  padding: 2rem 1.5rem;
  background:
    radial-gradient(circle at top left, rgba(255, 212, 96, 0.16), transparent 28%),
    linear-gradient(180deg, #17324d 0%, #10243a 100%);
  color: #f5efe5;
}

.brand-wrap {
  display: grid;
  gap: 0.75rem;
}

.eyebrow {
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #d9e2ec;
}

.brand a {
  color: #f5efe5;
  text-decoration: none;
  font-size: 1.35rem;
  font-weight: 700;
}

.tagline {
  margin: 0;
  line-height: 1.5;
  color: #d9e2ec;
}

.nav {
  list-style: none;
  padding: 0;
  margin: 2rem 0 0;
}

.nav li + li {
  margin-top: 0.75rem;
}

.nav a {
  color: #f5efe5;
  text-decoration: none;
}

.nav a:hover {
  text-decoration: underline;
}

.content {
  box-sizing: border-box;
  max-width: 980px;
  padding: 2.5rem min(4vw, 3rem) 4rem;
}

.source-path {
  margin-bottom: 1rem;
  font-family: "Courier New", monospace;
  font-size: 0.85rem;
  color: #52606d;
}

h1, h2, h3 {
  color: #102a43;
  line-height: 1.15;
}

h1 {
  font-size: clamp(2rem, 3.5vw, 3.2rem);
}

h2 {
  margin-top: 2rem;
  border-top: 1px solid #d9e2ec;
  padding-top: 1.25rem;
}

p, li {
  line-height: 1.7;
}

a {
  color: #0f609b;
}

code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  background: #f0f4f8;
  padding: 0.1rem 0.3rem;
  border-radius: 0.25rem;
}

pre {
  overflow-x: auto;
  padding: 1rem;
  background: #102a43;
  color: #f0f4f8;
  border-radius: 0.5rem;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0 1.5rem;
  background: #ffffff;
}

th, td {
  border: 1px solid #d9e2ec;
  padding: 0.65rem 0.8rem;
  text-align: left;
  vertical-align: top;
}

img {
  max-width: 100%;
  height: auto;
  border-radius: 0.4rem;
  box-shadow: 0 12px 28px rgba(16, 42, 67, 0.12);
}

@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .sidebar {
    padding-bottom: 1rem;
  }

  .content {
    padding-top: 1.5rem;
  }
}
"""
    css_path = output_root / SITE_CSS_RELATIVE
    css_path.parent.mkdir(parents=True, exist_ok=True)
    css_path.write_text(css, encoding="utf-8")


def _copy_static_assets(output_root: Path) -> None:
    for root in _static_roots():
        if not root.exists():
            continue
        destination = output_root / root.relative_to(REPO_ROOT)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(root, destination)

    for file_name in ("LICENSE", "CNAME"):
        source = REPO_ROOT / file_name
        if source.exists():
            shutil.copy2(source, output_root / file_name)


def build_docs_site(output_root: Path) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _copy_static_assets(output_root)
    _write_site_css(output_root)

    for source_path in _markdown_sources():
        output_path = _output_path_for(source_path, output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_text = source_path.read_text(encoding="utf-8")
        output_path.write_text(
            _render_html(markdown_text, source_path=source_path, output_root=output_root),
            encoding="utf-8",
        )

    index_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=docs/index.html">
  <title>spectral-library docs</title>
</head>
<body>
  <p><a href="docs/index.html">Go to spectral-library docs</a></p>
</body>
</html>
"""
    (output_root / "index.html").write_text(index_html, encoding="utf-8")
    (output_root / "404.html").write_text(index_html, encoding="utf-8")
    (output_root / ".nojekyll").write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static documentation site for GitHub Pages.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    build_docs_site(Path(args.output_root))
    print(f"Built documentation site at {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

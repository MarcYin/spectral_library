from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_docs_site.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docs-pages.yml"
PACKAGE_CHECKS_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "package-checks.yml"


def _load_docs_site_module():
    spec = importlib.util.spec_from_file_location("build_docs_site", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocsSiteBuildTests(unittest.TestCase):
    def test_build_docs_site_generates_expected_pages_and_assets(self) -> None:
        module = _load_docs_site_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "site"
            module.build_docs_site(output_root)

            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "404.html").exists())
            self.assertTrue((output_root / ".nojekyll").exists())
            self.assertTrue((output_root / "assets" / "site.css").exists())
            self.assertTrue((output_root / "README.html").exists())
            self.assertTrue((output_root / "docs" / "index.html").exists())
            self.assertTrue((output_root / "docs" / "official_sensor_examples.html").exists())
            self.assertTrue((output_root / "docs" / "assets" / "official_sensor_selected_bands.png").exists())
            self.assertTrue((output_root / "examples" / "official_mapping" / "README.html").exists())

            readme_html = (output_root / "README.html").read_text(encoding="utf-8")
            official_html = (output_root / "docs" / "official_sensor_examples.html").read_text(encoding="utf-8")

            self.assertIn('href="docs/index.html"', readme_html)
            self.assertIn('href="../examples/official_mapping/README.html"', official_html)
            self.assertIn("spectral-library prepare-mapping-library", official_html)
            self.assertIn("--siac-root examples/official_mapping/siac", official_html)

    def test_build_docs_site_rewrites_links_outside_repo_root(self) -> None:
        module = _load_docs_site_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "site"
            original_cwd = Path.cwd()
            try:
                os.chdir(Path(tmpdir))
                module.build_docs_site(output_root)
            finally:
                os.chdir(original_cwd)

            readme_html = (output_root / "README.html").read_text(encoding="utf-8")
            official_html = (output_root / "docs" / "official_sensor_examples.html").read_text(encoding="utf-8")

            self.assertIn('href="docs/index.html"', readme_html)
            self.assertNotIn('href="docs/index.md"', readme_html)
            self.assertIn('href="examples/official_mapping/README.html"', readme_html)
            self.assertNotIn('href="examples/official_mapping/README.md"', readme_html)
            self.assertIn('href="../examples/official_mapping/README.html"', official_html)
            self.assertNotIn('href="../examples/official_mapping/README.md"', official_html)

    def test_docs_workflows_are_present(self) -> None:
        docs_workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        package_checks_workflow = PACKAGE_CHECKS_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("actions/deploy-pages@v4", docs_workflow)
        self.assertIn("actions/upload-pages-artifact@v3", docs_workflow)
        self.assertIn("actions/configure-pages@v5", docs_workflow)
        self.assertIn('python -m pip install -e ".[docs]"', docs_workflow)
        self.assertIn("python scripts/build_docs_site.py --output-root build/docs-site", docs_workflow)
        self.assertIn('python -m pip install -e ".[docs]"', package_checks_workflow)
        self.assertIn("python scripts/build_docs_site.py --output-root build/docs-site", package_checks_workflow)


if __name__ == "__main__":
    unittest.main()

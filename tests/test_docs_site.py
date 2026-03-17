from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG_PATH = REPO_ROOT / "mkdocs.yml"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docs-pages.yml"
PACKAGE_CHECKS_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "package-checks.yml"
FULL_LIBRARY_BENCHMARK_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "full-library-benchmarks.yml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-package.yml"
SECURITY_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "security-checks.yml"
CODEQL_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "codeql.yml"
DEPENDABOT_CONFIG_PATH = REPO_ROOT / ".github" / "dependabot.yml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


class DocsSiteBuildTests(unittest.TestCase):
    def test_mkdocs_build_generates_expected_pages_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "site"
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "mkdocs",
                    "build",
                    "--clean",
                    "--config-file",
                    str(MKDOCS_CONFIG_PATH),
                    "--site-dir",
                    str(output_root),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "example_bundle.html").exists())
            self.assertTrue((output_root / "cli_reference.html").exists())
            self.assertTrue((output_root / "internal_overview.html").exists())
            self.assertTrue((output_root / "official_sensor_examples.html").exists())
            self.assertTrue((output_root / "theory.html").exists())
            self.assertTrue((output_root / "release_process.html").exists())
            self.assertTrue((output_root / "security_provenance.html").exists())
            self.assertFalse((output_root / "scale_factor_verification.html").exists())
            self.assertFalse((output_root / "metadata_only_actual_links.html").exists())
            self.assertTrue((output_root / "assets" / "official_sensor_selected_bands.png").exists())
            self.assertTrue((output_root / "assets" / "official_holdout_batch_examples.png").exists())
            self.assertTrue((output_root / "assets" / "official_holdout_neighbor_overlays.png").exists())
            self.assertTrue((output_root / "assets" / "official_estimator_comparison.png").exists())
            self.assertTrue((output_root / "assets" / "mkdocs-extra.css").exists())
            self.assertTrue((output_root / "examples" / "official_mapping" / "official_source_manifest.json").exists())
            self.assertTrue((output_root / "examples" / "official_mapping" / "results" / "selected" / "landsat8_to_sentinel2a_holdout_neighbor_review.csv").exists())
            self.assertFalse((output_root / "examples" / "official_mapping" / "siac").exists())
            self.assertFalse((output_root / "tests").exists())
            self.assertTrue((output_root / "LICENSE").exists())
            self.assertTrue((output_root / ".nojekyll").exists())

            index_html = (output_root / "index.html").read_text(encoding="utf-8")
            cli_html = (output_root / "cli_reference.html").read_text(encoding="utf-8")
            python_html = (output_root / "python_api_reference.html").read_text(encoding="utf-8")
            official_html = (output_root / "official_sensor_examples.html").read_text(encoding="utf-8")
            theory_html = (output_root / "theory.html").read_text(encoding="utf-8")
            security_html = (output_root / "security_provenance.html").read_text(encoding="utf-8")

            self.assertIn("Map between sensors", index_html)
            self.assertIn("--json-logs", cli_html)
            self.assertIn("distance_weighted_mean", cli_html)
            self.assertIn("simplex_mixture", cli_html)
            self.assertIn("scipy_ckdtree", cli_html)
            self.assertIn("faiss", cli_html)
            self.assertIn("pynndescent", cli_html)
            self.assertIn("scann", cli_html)
            self.assertIn("--knn-index-backend", cli_html)
            self.assertIn("spectral-library[knn]", cli_html)
            self.assertIn("spectral-library[knn-faiss]", cli_html)
            self.assertIn("spectral-library[knn-pynndescent]", cli_html)
            self.assertIn("spectral-library[knn-scann]", cli_html)
            self.assertIn("command_failed", cli_html)
            self.assertIn("elapsed_ms", cli_html)
            self.assertIn("per-segment query values", python_html)
            self.assertIn("knn_backend", python_html)
            self.assertIn("knn_eps", python_html)
            self.assertIn("knn_index_artifacts", python_html)
            self.assertIn("confidence_score", python_html)
            self.assertIn("confidence_policy", python_html)
            self.assertIn("manual_review", python_html)
            self.assertIn("neighbor-review-output", cli_html)
            self.assertIn("spectral-library-internal", cli_html)
            self.assertIn("baseline_deltas", cli_html)
            self.assertIn('href="example_bundle.html"', official_html)
            self.assertIn('href="examples/official_mapping/results/metrics/pairwise_band_metrics.csv"', official_html)
            self.assertIn('href="examples/official_mapping/results/metrics/neighbor_estimator_holdout_comparison.csv"', official_html)
            self.assertIn("held-out reconstruction", official_html)
            self.assertIn("77,125 library rows", official_html)
            self.assertIn("external full SIAC", official_html)
            self.assertIn('href="examples/official_mapping/results/selected/landsat8_to_sentinel2a_holdout_batch.csv"', official_html)
            self.assertIn('href="examples/official_mapping/results/selected/landsat8_to_sentinel2a_holdout_neighbor_review.csv"', official_html)
            self.assertIn("Neighbor Overlay Diagnostic", official_html)
            self.assertIn("Held-Out Estimator Comparison", official_html)
            self.assertIn("visible-to-NIR retrieval", official_html)
            self.assertIn("NIR-to-SWIR retrieval", official_html)
            self.assertIn("exclude_row_id", official_html)
            self.assertIn("simplex_mixture", official_html)
            self.assertIn("B8A", official_html)
            self.assertNotIn('href="../examples/', official_html)
            self.assertIn("Mathematical Foundations", official_html)
            self.assertIn("root-mean-square Euclidean distance", theory_html)
            self.assertIn("scipy_ckdtree", theory_html)
            self.assertIn("faiss", theory_html)
            self.assertIn("pynndescent", theory_html)
            self.assertIn("scann", theory_html)
            self.assertIn("full-feature ANN indexes", theory_html)
            self.assertIn("not a calibrated", theory_html)
            self.assertIn("manual_review", theory_html)
            self.assertIn("Pre-sorting the full library rows", theory_html)
            self.assertIn("two independent nearest-neighbor", theory_html)
            self.assertIn("arithmatex", theory_html)
            self.assertIn("Security and Provenance", security_html)
            self.assertIn("pip-audit", security_html)
            self.assertIn("CodeQL", security_html)
            self.assertIn("CycloneDX", security_html)
            self.assertIn("attestations", security_html)
            self.assertIn("dependabot.yml", security_html)
            self.assertIn("immutable SHAs", security_html)
            self.assertNotIn("/Users/fengyin/Documents/spectral_library", official_html)

    def test_mkdocs_configuration_and_workflows_are_present(self) -> None:
        mkdocs_config = MKDOCS_CONFIG_PATH.read_text(encoding="utf-8")
        docs_workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        package_checks_workflow = PACKAGE_CHECKS_WORKFLOW_PATH.read_text(encoding="utf-8")
        full_benchmark_workflow = FULL_LIBRARY_BENCHMARK_WORKFLOW_PATH.read_text(encoding="utf-8")
        release_workflow = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")
        security_workflow = SECURITY_WORKFLOW_PATH.read_text(encoding="utf-8")
        codeql_workflow = CODEQL_WORKFLOW_PATH.read_text(encoding="utf-8")
        dependabot_config = DEPENDABOT_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("exclude_docs:", mkdocs_config)
        self.assertIn("scale_factor_verification.md", mkdocs_config)
        self.assertIn("hooks:", mkdocs_config)
        self.assertIn("docs/mkdocs_hooks.py", mkdocs_config)
        self.assertIn("name: material", mkdocs_config)
        self.assertIn("navigation.tabs", mkdocs_config)
        self.assertIn("Mathematical Foundations: theory.md", mkdocs_config)
        self.assertIn("Example Bundle: example_bundle.md", mkdocs_config)
        self.assertIn("Security and Provenance: security_provenance.md", mkdocs_config)
        self.assertIn("Internal Docs Overview: internal_overview.md", mkdocs_config)
        self.assertIn("use_directory_urls: false", mkdocs_config)
        self.assertIn("python -m mkdocs build --clean --config-file mkdocs.yml --site-dir build/docs-site", docs_workflow)
        self.assertIn("actions/deploy-pages@", docs_workflow)
        self.assertIn('python -m pip install -e ".[docs]"', docs_workflow)
        self.assertIn("python -m mkdocs build --clean --config-file mkdocs.yml --site-dir build/docs-site", package_checks_workflow)
        self.assertIn("optional-knn-smoke:", package_checks_workflow)
        self.assertIn("knn-faiss", package_checks_workflow)
        self.assertIn("knn-pynndescent", package_checks_workflow)
        self.assertIn("knn-scann", package_checks_workflow)
        self.assertIn("full-library-benchmarks", full_benchmark_workflow)
        self.assertIn("FULL_LIBRARY_PREPARED_ROOT", full_benchmark_workflow)
        self.assertIn("run_full_library_benchmarks.py", full_benchmark_workflow)
        self.assertIn("actions/attest-build-provenance@", release_workflow)
        self.assertIn("actions/attest-sbom@", release_workflow)
        self.assertIn("spectral-library-wheel.sbom.cdx.json", release_workflow)
        self.assertIn("spectral-library-sdist.sbom.cdx.json", release_workflow)
        self.assertIn("packages-dir: upload-dist/", release_workflow)
        self.assertIn("actions/dependency-review-action@", security_workflow)
        self.assertIn("pip-audit", security_workflow)
        self.assertIn("github/codeql-action/init@", codeql_workflow)
        self.assertIn("security-extended,security-and-quality", codeql_workflow)
        self.assertIn("package-ecosystem: \"github-actions\"", dependabot_config)
        self.assertIn("package-ecosystem: \"pip\"", dependabot_config)

        pinned_action_pattern = re.compile(r"uses:\s+[-A-Za-z0-9_./]+@[0-9a-f]{40}\b")
        for workflow_path in sorted(WORKFLOW_DIR.glob("*.yml")):
            workflow_text = workflow_path.read_text(encoding="utf-8")
            uses_lines = [
                line
                for line in workflow_text.splitlines()
                if line.lstrip().startswith("- uses:") or line.lstrip().startswith("uses:")
            ]
            self.assertTrue(uses_lines, msg=f"No uses lines found in {workflow_path.name}")
            for line in uses_lines:
                self.assertRegex(line, pinned_action_pattern, msg=f"Unpinned action in {workflow_path.name}: {line}")


if __name__ == "__main__":
    unittest.main()

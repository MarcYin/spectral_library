from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from spectral_library.fetchers import get_fetcher
from spectral_library.fetchers.base import ArtifactRecord, FetchResult, sha256_file, utc_now_iso, write_json
from spectral_library.fetchers.ecostress import (
    DEFAULT_SEARCH_TYPES as ECOSTRESS_DEFAULT_SEARCH_TYPES,
    _enumerate_catalog as ecostress_enumerate_catalog,
    _enumerate_class as ecostress_enumerate_class,
    _extract_class_options as ecostress_extract_class_options,
    _has_more_results as ecostress_has_more_results,
    _parse_filenames as ecostress_parse_filenames,
    _parse_search_types as ecostress_parse_search_types,
    _read_list_env as ecostress_read_list_env,
    _read_int_env as ecostress_read_int_env,
    fetch as fetch_ecostress,
)
from spectral_library.fetchers.ecosis import (
    _extract_package_slug,
    _extract_resources,
    _normalize_resource_url,
    fetch as fetch_ecosis,
)
from spectral_library.fetchers.github_archive import (
    _iter_matching_members,
    _relative_destination,
    fetch as fetch_github_archive,
)
from spectral_library.fetchers.http_utils import infer_filename, looks_like_download, sanitize_filename
from spectral_library.fetchers.manual import fetch as fetch_manual
from spectral_library.fetchers.pangaea import fetch as fetch_pangaea
from spectral_library.fetchers.specchio import _load_query_config, fetch as fetch_specchio
from spectral_library.fetchers.static_http import fetch as fetch_static_http
from spectral_library.fetchers.zenodo import (
    _extract_record_id,
    _file_urls,
    _filename_from_content_url,
    fetch as fetch_zenodo,
)
from spectral_library.manifest import SourceRecord


class FakeHeaders(dict):
    def get_content_type(self) -> str:
        return self.get("Content-Type", "application/octet-stream").split(";", 1)[0]


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, url: str, status: int = 200, headers: dict[str, str] | None = None) -> None:
        super().__init__(body)
        self._url = url
        self.status = status
        self.headers = FakeHeaders(headers or {"Content-Type": "application/octet-stream"})

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def make_source(**overrides: str) -> SourceRecord:
    values = {
        "source_id": "test_source",
        "name": "Test Source",
        "section": "direct_public",
        "subsection": "soil",
        "spectral_type": "Soil",
        "coverage": "400-2500 nm",
        "resource_type": "Direct library",
        "provider": "zenodo",
        "landing_url": "https://zenodo.org/records/12345",
        "download_url": "",
        "fetch_adapter": "zenodo_api",
        "auth_mode": "public",
        "expected_format": "zip_csv",
        "tier": "tier1",
        "priority": "high",
        "ingest_role": "primary_raw",
        "normalization_eligibility": "eligible_full",
        "status": "planned",
        "notes": "note",
    }
    values.update(overrides)
    return SourceRecord(**values)


class FakeSpecchioDescriptor:
    def __init__(self, name: str) -> None:
        self.name = name

    def getDataSourceName(self) -> str:
        return self.name


class FakeMeasurementUnit:
    def __init__(self, name: str) -> None:
        self.name = name

    def getUnitName(self) -> str:
        return self.name


class FakeSpecchioSpace:
    def __init__(self, wavelengths, vectors, spectrum_ids, unit: str = "reflectance") -> None:
        self._wavelengths = wavelengths
        self._vectors = vectors
        self._spectrum_ids = spectrum_ids
        self._unit = FakeMeasurementUnit(unit)

    def getAverageWavelengths(self):
        return self._wavelengths

    def getVectorsAsArray(self):
        return self._vectors

    def getSpectrumIds(self):
        return self._spectrum_ids

    def getMeasurementUnit(self):
        return self._unit


class FakeSpecchioCondition:
    def __init__(self, attribute) -> None:
        self.attribute = attribute
        self.operator = ""
        self.value = ""

    def setValue(self, value) -> None:
        self.value = value

    def setOperator(self, operator) -> None:
        self.operator = operator


class FakeSpecchioQuery:
    def __init__(self) -> None:
        self.conditions = []

    def add_condition(self, condition) -> None:
        self.conditions.append(condition)


class FakeSpecchioClient:
    def __init__(self, spaces=None) -> None:
        self._spaces = spaces or []
        self.query = None
        self.order_by = ""

    def getAttributesNameHash(self):
        return {"Species": "species_attr", "Campaign": "campaign_attr"}

    def getSpectrumIdsMatchingQuery(self, query):
        self.query = query
        return [101, 102]

    def getSpaces(self, ids, order_by):
        del ids
        self.order_by = order_by
        return self._spaces

    def loadSpace(self, space):
        return space


class FakeSpecchioFactory:
    def __init__(self, client: FakeSpecchioClient) -> None:
        self.client = client
        self.descriptors = [FakeSpecchioDescriptor("Primary"), FakeSpecchioDescriptor("Secondary")]

    def getAllServerDescriptors(self):
        return self.descriptors

    def createClient(self, descriptor):
        self.descriptor = descriptor
        return self.client


class FakeSpecchioClientFactoryClass:
    def __init__(self, factory: FakeSpecchioFactory) -> None:
        self.factory = factory

    def getInstance(self):
        return self.factory


class FakeSpecchioQueriesPackage:
    Query = FakeSpecchioQuery
    EAVQueryConditionObject = FakeSpecchioCondition


class FakeSpecchioClientPackage:
    def __init__(self, factory: FakeSpecchioFactory) -> None:
        self.SPECCHIOClientFactory = FakeSpecchioClientFactoryClass(factory)


class FakeSpecchioPackage:
    def __init__(self, factory: FakeSpecchioFactory) -> None:
        self.client = FakeSpecchioClientPackage(factory)
        self.queries = FakeSpecchioQueriesPackage()


class FakeJPackageRoot:
    def __init__(self, factory: FakeSpecchioFactory) -> None:
        self.specchio = FakeSpecchioPackage(factory)


class FakeJPype:
    def __init__(self, factory: FakeSpecchioFactory) -> None:
        self.factory = factory
        self.started = False
        self.start_args = ()

    def isJVMStarted(self) -> bool:
        return self.started

    def getDefaultJVMPath(self) -> str:
        return "/fake/libjvm.so"

    def startJVM(self, *args) -> None:
        self.started = True
        self.start_args = args

    def JPackage(self, name: str):
        if name != "ch":
            raise ValueError(name)
        return FakeJPackageRoot(self.factory)


class BaseUtilityTests(unittest.TestCase):
    def test_fetch_result_to_dict_and_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "meta.json"
            artifact = write_json(output, {"name": "value"})
            result = FetchResult(
                source_id="src",
                source_name="Source",
                fetch_adapter="manual_portal",
                fetch_mode="metadata",
                status="manual_review",
                landing_url="https://example.com",
                started_at="2024-01-01T00:00:00+00:00",
                finished_at="2024-01-01T00:00:01+00:00",
                notes=["one"],
                artifacts=[artifact],
            )
            payload = result.to_dict()

            self.assertEqual(payload["source_id"], "src")
            self.assertEqual(payload["artifacts"][0]["artifact_id"], "meta")
            self.assertEqual(artifact.media_type, "application/json")
            self.assertEqual(artifact.sha256, sha256_file(output))

    def test_sha256_file_and_utc_now_iso(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(b"abc").hexdigest(),
            )
            self.assertTrue(utc_now_iso().endswith("+00:00"))


class HttpUtilsTests(unittest.TestCase):
    def test_filename_helpers(self) -> None:
        self.assertEqual(sanitize_filename(" a/b\\c "), "a_b_c")
        self.assertEqual(sanitize_filename("   "), "artifact.bin")
        self.assertEqual(infer_filename("https://example.com/data/file.zip", "fallback.bin"), "file.zip")
        self.assertEqual(infer_filename("https://example.com/", "fallback.bin"), "fallback.bin")

    def test_looks_like_download(self) -> None:
        self.assertTrue(looks_like_download("https://example.com/file.zip", "text/html"))
        self.assertTrue(looks_like_download("https://example.com/page", "application/json"))
        self.assertFalse(looks_like_download("https://example.com/page", "text/html; charset=utf-8"))


class FetcherRegistryTests(unittest.TestCase):
    def test_get_fetcher_known_and_unknown(self) -> None:
        self.assertIs(get_fetcher("manual_portal"), fetch_manual)
        self.assertIs(get_fetcher("ecostress_web"), fetch_ecostress)
        self.assertIs(get_fetcher("github_archive"), fetch_github_archive)
        self.assertIs(get_fetcher("pangaea"), fetch_pangaea)
        self.assertIs(get_fetcher("specchio_client"), fetch_specchio)
        with self.assertRaises(KeyError):
            get_fetcher("missing")


class ManualFetcherTests(unittest.TestCase):
    def test_manual_fetch_writes_review_record(self) -> None:
        source = make_source(
            fetch_adapter="manual_portal",
            landing_url="https://example.com/manual",
            download_url="https://example.com/manual/download.csv",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_manual(source, Path(tmpdir), "assets", "ua")
            self.assertEqual(result.status, "manual_review")
            self.assertEqual(result.fetch_mode, "metadata")
            self.assertEqual(len(result.artifacts), 1)
            self.assertTrue((Path(tmpdir) / "manual_review.json").exists())
            payload = json.loads((Path(tmpdir) / "manual_review.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["download_url"], "https://example.com/manual/download.csv")


class EcosisFetcherTests(unittest.TestCase):
    def test_ecosis_helpers(self) -> None:
        self.assertEqual(
            _extract_package_slug("https://ecosis.org/package/test-package"),
            "test-package",
        )
        self.assertEqual(
            _normalize_resource_url("http://data.ecosis.org/dataset/test.csv"),
            "https://data.ecosis.org/dataset/test.csv",
        )
        self.assertEqual(
            _extract_resources(
                {
                    "ecosis": {
                        "resources": [
                            {
                                "name": "spectra.csv",
                                "url": "http://data.ecosis.org/dataset/test.csv",
                                "mimetype": "text/csv",
                                "type": "upload",
                            }
                        ]
                    }
                }
            ),
            [
                {
                    "name": "spectra.csv",
                    "url": "https://data.ecosis.org/dataset/test.csv",
                    "mimetype": "text/csv",
                    "type": "upload",
                }
            ],
        )
        with self.assertRaises(ValueError):
            _extract_package_slug("https://ecosis.org/dataset/test")
        with self.assertRaises(ValueError):
            _extract_package_slug("https://ecosis.org/package/")
        self.assertEqual(_extract_resources({"ecosis": []}), [])
        self.assertEqual(_extract_resources({"ecosis": {"resources": {}}}), [])
        self.assertEqual(
            _extract_resources(
                {
                    "ecosis": {
                        "resources": [
                            None,
                            {"name": "blank", "url": ""},
                            {
                                "url": "https://data.ecosis.org/dataset/test/resource.csv",
                                "mimetype": "text/csv",
                                "type": "upload",
                            },
                        ]
                    }
                }
            )[0]["name"],
            "resource.csv",
        )

    @patch("spectral_library.fetchers.ecosis.urlopen")
    def test_ecosis_fetch_captures_metadata(self, mock_urlopen) -> None:
        payload = {
            "ecosis": {
                "package_id": "package-1",
                "package_title": "Test Package",
                "resources": [
                    {
                        "name": "spectra.csv",
                        "url": "http://data.ecosis.org/dataset/package-1/resource/abc/download/spectra.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                    {
                        "name": "metadata.csv",
                        "url": "https://data.ecosis.org/dataset/package-1/resource/def/download/metadata.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                ],
            }
        }
        mock_urlopen.return_value = FakeResponse(
            json.dumps(payload).encode("utf-8"),
            url="https://ecosis.org/api/package/test",
            headers={"Content-Type": "application/json"},
        )
        source = make_source(
            fetch_adapter="ecosis_package",
            provider="ecosis",
            landing_url="https://ecosis.org/package/test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecosis(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(result.fetch_mode, "metadata")
            self.assertEqual(result.artifacts[0].url, "https://ecosis.org/api/package/test")
            self.assertEqual(len(result.artifacts), 4)
            self.assertEqual(result.artifacts[1].url, "https://data.ecosis.org/dataset/package-1/resource/abc/download/spectra.csv")
            self.assertTrue((Path(tmpdir) / "ecosis_package.json").exists())
            package_payload = json.loads((Path(tmpdir) / "ecosis_package.json").read_text(encoding="utf-8"))
            self.assertEqual(package_payload["resource_count"], 2)
            self.assertEqual(package_payload["export_url"], "https://ecosis.org/api/package/test/export?metadata=true")

    @patch("spectral_library.fetchers.ecosis.urlopen")
    def test_ecosis_fetch_downloads_package_resources(self, mock_urlopen) -> None:
        payload = {
            "ecosis": {
                "package_id": "package-1",
                "package_title": "Test Package",
                "resources": [
                    {
                        "name": "spectra.csv",
                        "url": "http://data.ecosis.org/dataset/package-1/resource/abc/download/spectra.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                    {
                        "name": "metadata.csv",
                        "url": "https://data.ecosis.org/dataset/package-1/resource/def/download/metadata.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                ],
            }
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://ecosis.org/api/package/test",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"wavelength,reflectance\n400,0.1\n",
                url="https://data.ecosis.org/dataset/package-1/resource/abc/download/spectra.csv",
                headers={"Content-Type": "text/csv"},
            ),
            FakeResponse(
                b"id,site\n1,a\n",
                url="https://data.ecosis.org/dataset/package-1/resource/def/download/metadata.csv",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            fetch_adapter="ecosis_package",
            provider="ecosis",
            landing_url="https://ecosis.org/package/test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecosis(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 4)
            self.assertTrue((Path(tmpdir) / "spectra.csv").exists())
            self.assertTrue((Path(tmpdir) / "metadata.csv").exists())
            self.assertIn("Downloaded 2 EcoSIS package resources", result.notes[0])

    @patch("spectral_library.fetchers.ecosis.urlopen")
    def test_ecosis_fetch_downloads_configured_asset_url(self, mock_urlopen) -> None:
        payload = {
            "ecosis": {
                "package_id": "package-1",
                "package_title": "Test Package",
                "resources": [
                    {
                        "name": "spectra.csv",
                        "url": "https://data.ecosis.org/download/test.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                    {
                        "name": "metadata.csv",
                        "url": "https://data.ecosis.org/download/meta.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    },
                ],
            }
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://ecosis.org/api/package/test",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"wavelength,reflectance\n400,0.1\n",
                url="https://data.ecosis.org/download/test.csv",
                headers={"Content-Type": "text/csv"},
            ),
            FakeResponse(
                b"id,site\n1,a\n",
                url="https://data.ecosis.org/download/meta.csv",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            fetch_adapter="ecosis_package",
            provider="ecosis",
            landing_url="https://ecosis.org/package/test",
            download_url="https://data.ecosis.org/download/test.csv",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecosis(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 4)
            self.assertEqual(result.artifacts[1].url, "https://data.ecosis.org/download/test.csv")
            self.assertEqual(result.artifacts[2].url, "https://data.ecosis.org/download/meta.csv")
            self.assertTrue((Path(tmpdir) / "test.csv").exists())
            self.assertTrue((Path(tmpdir) / "meta.csv").exists())
            self.assertIn("configured", result.notes[0].lower())

    @patch("spectral_library.fetchers.ecosis.urlopen")
    def test_ecosis_fetch_falls_back_to_package_export(self, mock_urlopen) -> None:
        payload = {"ecosis": {"package_id": "package-1", "package_title": "Test Package", "resources": []}}
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://ecosis.org/api/package/test",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"wavelength,reflectance\n400,0.1\n",
                url="https://ecosis.org/api/package/test/export?metadata=true",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            fetch_adapter="ecosis_package",
            provider="ecosis",
            landing_url="https://ecosis.org/package/test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecosis(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 3)
            self.assertTrue((Path(tmpdir) / "test.csv").exists())
            self.assertIn("package export", result.notes[0].lower())

    @patch("spectral_library.fetchers.ecosis.urlopen")
    def test_ecosis_fetch_downloads_selected_url_not_listed_in_resources(self, mock_urlopen) -> None:
        payload = {
            "ecosis": {
                "package_id": "package-1",
                "package_title": "Test Package",
                "resources": [
                    {
                        "name": "metadata.csv",
                        "url": "https://data.ecosis.org/download/meta.csv",
                        "mimetype": "text/csv",
                        "type": "upload",
                    }
                ],
            }
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://ecosis.org/api/package/test",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"id,site\n1,a\n",
                url="https://data.ecosis.org/download/meta.csv",
                headers={"Content-Type": "text/csv"},
            ),
            FakeResponse(
                b"wavelength,reflectance\n400,0.1\n",
                url="https://data.ecosis.org/download/extra.csv",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            fetch_adapter="ecosis_package",
            provider="ecosis",
            landing_url="https://ecosis.org/package/test",
            download_url="https://data.ecosis.org/download/extra.csv",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecosis(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertTrue((Path(tmpdir) / "meta.csv").exists())
            self.assertTrue((Path(tmpdir) / "extra.csv").exists())
            self.assertIn("not listed", result.notes[0].lower())


class EcostressFetcherTests(unittest.TestCase):
    def test_ecostress_helpers(self) -> None:
        download_html = """
        <a href="javascript:orderall('water');">All Water</a>
        <a href="javascript:orderall('spectra');">ALL Spectra</a>
        <a href="javascript:orderall('vegetation');">All Vegetation</a>
        """
        search_html = """
        <select name="classsel">
          <option value="All">All</option>
          <option value="distilledwater">Distilled Water</option>
          <option value="distilledwater">Distilled Water</option>
          <option value="snow">Snow</option>
        </select>
        <input type="checkbox" value="water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt" />
        <font color="#0000FF">Additional data is available by increasing the "Next xxxx Set&gt;"</font>
        """

        self.assertEqual(ecostress_parse_search_types(download_html), ["water", "vegetation"])
        self.assertEqual(ecostress_extract_class_options(search_html), ["distilledwater", "snow"])
        self.assertEqual(
            ecostress_parse_filenames(search_html),
            ["water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt"],
        )
        self.assertTrue(ecostress_has_more_results(search_html))
        self.assertEqual(ecostress_parse_search_types("<html></html>"), ECOSTRESS_DEFAULT_SEARCH_TYPES)
        self.assertEqual(ecostress_extract_class_options("<html></html>"), [])
        self.assertEqual(
            ecostress_parse_filenames(
                '<input value="a.txt" /><input value="a.txt" /><input value="b.txt" />'
            ),
            ["a.txt", "b.txt"],
        )

    @patch("spectral_library.fetchers.ecostress._post_search")
    def test_ecostress_enumerate_class_handles_more_results(self, mock_post_search) -> None:
        first_page = "".join(f'<input value="f{value}.txt" />' for value in range(1, 101))
        second_page = "".join(f'<input value="f{value}.txt" />' for value in range(100, 103))
        mock_post_search.side_effect = [
            first_page + '<font color="#0000FF">Additional data is available by increasing the "Next xxxx Set&gt;"</font>',
            second_page,
        ]

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(ecostress_read_int_env("ECOSTRESS_DOWNLOAD_WORKERS", 3), 3)
            self.assertEqual(ecostress_read_list_env("ECOSTRESS_SEARCH_TYPES"), [])
        with patch.dict("os.environ", {"ECOSTRESS_DOWNLOAD_WORKERS": "0"}, clear=True):
            self.assertEqual(ecostress_read_int_env("ECOSTRESS_DOWNLOAD_WORKERS", 3), 1)
        with patch.dict("os.environ", {"ECOSTRESS_SEARCH_TYPES": "water, vegetation"}, clear=True):
            self.assertEqual(ecostress_read_list_env("ECOSTRESS_SEARCH_TYPES"), ["water", "vegetation"])

        filenames, summary = ecostress_enumerate_class("water", "distilledwater", "ua")
        self.assertEqual(len(filenames), 102)
        self.assertEqual(filenames[:3], ["f1.txt", "f2.txt", "f3.txt"])
        self.assertEqual(filenames[-3:], ["f100.txt", "f101.txt", "f102.txt"])
        self.assertEqual(summary["maxhits_used"], 200)
        self.assertFalse(summary["has_more_results"])

    @patch("spectral_library.fetchers.ecostress._post_search")
    def test_ecostress_enumerate_class_collects_paged_windows(self, mock_post_search) -> None:
        def make_page(start: int, stop: int, has_more: bool = True) -> str:
            inputs = "".join(f'<input value="f{value}.txt" />' for value in range(start, stop + 1))
            suffix = (
                '<font color="#0000FF">Additional data is available by increasing the "Next xxxx Set&gt;"</font>'
                if has_more
                else ""
            )
            return inputs + suffix

        mock_post_search.side_effect = [
            make_page(1, 100),
            make_page(100, 200),
            make_page(200, 235),
        ]

        filenames, summary = ecostress_enumerate_class("vegetation", "tree", "ua")

        self.assertEqual(len(filenames), 235)
        self.assertEqual(filenames[:3], ["f1.txt", "f2.txt", "f3.txt"])
        self.assertEqual(filenames[-3:], ["f233.txt", "f234.txt", "f235.txt"])
        self.assertEqual(summary["maxhits_used"], 300)
        self.assertTrue(summary["has_more_results"])

    @patch("spectral_library.fetchers.ecostress._post_search")
    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_enumerate_catalog_falls_back_to_all_class(self, mock_urlopen, mock_post_search) -> None:
        mock_urlopen.return_value = FakeResponse(
            b"<a href=\"javascript:orderall('water');\">All Water</a>",
            url="https://speclib.jpl.nasa.gov/download",
            headers={"Content-Type": "text/html"},
        )
        mock_post_search.side_effect = ['<input value="water.txt" />', '<input value="water.txt" />']

        filenames, payload = ecostress_enumerate_catalog("ua")

        self.assertEqual(filenames, ["water.txt"])
        self.assertEqual(payload["search_types"], ["water"])
        self.assertEqual(payload["classes"][0]["class_value"], "All")

    @patch("spectral_library.fetchers.ecostress._post_search")
    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_enumerate_catalog_honors_selected_search_types(self, mock_urlopen, mock_post_search) -> None:
        mock_urlopen.return_value = FakeResponse(
            b"""
            <a href=\"javascript:orderall('water');\">All Water</a>
            <a href=\"javascript:orderall('vegetation');\">All Vegetation</a>
            """,
            url="https://speclib.jpl.nasa.gov/download",
            headers={"Content-Type": "text/html"},
        )
        mock_post_search.side_effect = ['<input value="water.txt" />', '<input value="water.txt" />']

        with patch.dict("os.environ", {"ECOSTRESS_SEARCH_TYPES": "water"}, clear=False):
            filenames, payload = ecostress_enumerate_catalog("ua")

        self.assertEqual(filenames, ["water.txt"])
        self.assertEqual(payload["search_types"], ["water"])

    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_fetch_metadata_mode(self, mock_urlopen) -> None:
        download_html = b"""
        <html>
          <a href=\"javascript:orderall('water');\">All Water</a>
          <a href=\"javascript:orderall('spectra');\">ALL Spectra</a>
        </html>
        """
        all_html = b"""
        <div id=\"spectral_list\">
          <select name=\"classsel\">
            <option value=\"All\">All</option>
            <option value=\"distilledwater\">Distilled Water</option>
          </select>
        </div>
        """
        class_html = b"""
        <div id=\"spectral_list\">
          <input type=\"checkbox\" value=\"water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt\" />
        </div>
        """
        mock_urlopen.side_effect = [
            FakeResponse(download_html, url="https://speclib.jpl.nasa.gov/download", headers={"Content-Type": "text/html"}),
            FakeResponse(all_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(class_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
        ]
        source = make_source(
            source_id="ecostress_v1",
            name="ECOSTRESS Spectral Library v1.0",
            provider="jpl_nasa",
            fetch_adapter="ecostress_web",
            landing_url="https://speclib.jpl.nasa.gov/",
            download_url="https://speclib.jpl.nasa.gov/download",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_ecostress(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(result.artifacts[0].artifact_id, "ecostress_catalog")
            self.assertEqual(result.artifacts[1].artifact_id, "ecostress_data_root")
            catalog = json.loads((Path(tmpdir) / "ecostress_catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(catalog["total_files"], 1)
            self.assertEqual(catalog["search_types"], ["water"])

    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_fetch_assets_mode_downloads_files(self, mock_urlopen) -> None:
        download_html = b"""
        <html>
          <a href=\"javascript:orderall('water');\">All Water</a>
        </html>
        """
        all_html = b"""
        <div id=\"spectral_list\">
          <select name=\"classsel\">
            <option value=\"All\">All</option>
            <option value=\"distilledwater\">Distilled Water</option>
          </select>
        </div>
        """
        class_html = b"""
        <div id=\"spectral_list\">
          <input type=\"checkbox\" value=\"water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt\" />
        </div>
        """
        file_body = b"Name: Distilled Water\\n14.0\\t3.8\\n"
        mock_urlopen.side_effect = [
            FakeResponse(download_html, url="https://speclib.jpl.nasa.gov/download", headers={"Content-Type": "text/html"}),
            FakeResponse(all_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(class_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(file_body, url="https://speclib.jpl.nasa.gov/ecospeclibdata/water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt", headers={"Content-Type": "text/plain"}),
        ]
        source = make_source(
            source_id="ecostress_v1",
            name="ECOSTRESS Spectral Library v1.0",
            provider="jpl_nasa",
            fetch_adapter="ecostress_web",
            landing_url="https://speclib.jpl.nasa.gov/",
            download_url="https://speclib.jpl.nasa.gov/download",
        )

        env = {"ECOSTRESS_DOWNLOAD_WORKERS": "1", "ECOSTRESS_MAX_FILES": "1"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", env, clear=False):
            result = fetch_ecostress(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertTrue((Path(tmpdir) / "water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt").exists())
            self.assertIn("ECOSTRESS_MAX_FILES", result.notes[1])
            self.assertIn("Downloaded 1 ECOSTRESS spectral text files", result.notes[-1])

    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_fetch_assets_mode_without_max_file_limit(self, mock_urlopen) -> None:
        download_html = b"""
        <html>
          <a href=\"javascript:orderall('water');\">All Water</a>
        </html>
        """
        all_html = b"""
        <div id=\"spectral_list\">
          <select name=\"classsel\">
            <option value=\"All\">All</option>
            <option value=\"distilledwater\">Distilled Water</option>
          </select>
        </div>
        """
        class_html = b"""
        <div id=\"spectral_list\">
          <input type=\"checkbox\" value=\"water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt\" />
        </div>
        """
        file_body = b"Name: Distilled Water\\n14.0\\t3.8\\n"
        mock_urlopen.side_effect = [
            FakeResponse(download_html, url="https://speclib.jpl.nasa.gov/download", headers={"Content-Type": "text/html"}),
            FakeResponse(all_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(class_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(file_body, url="https://speclib.jpl.nasa.gov/ecospeclibdata/water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt", headers={"Content-Type": "text/plain"}),
        ]
        source = make_source(
            source_id="ecostress_v1",
            name="ECOSTRESS Spectral Library v1.0",
            provider="jpl_nasa",
            fetch_adapter="ecostress_web",
            landing_url="https://speclib.jpl.nasa.gov/",
            download_url="https://speclib.jpl.nasa.gov/download",
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"ECOSTRESS_DOWNLOAD_WORKERS": "1"}, clear=False):
            result = fetch_ecostress(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            self.assertNotIn("ECOSTRESS_MAX_FILES", " ".join(result.notes))

    @patch("spectral_library.fetchers.ecostress.urlopen")
    def test_ecostress_fetch_assets_mode_reuses_existing_tidied_files(self, mock_urlopen) -> None:
        download_html = b"""
        <html>
          <a href=\"javascript:orderall('water');\">All Water</a>
        </html>
        """
        all_html = b"""
        <div id=\"spectral_list\">
          <select name=\"classsel\">
            <option value=\"All\">All</option>
            <option value=\"distilledwater\">Distilled Water</option>
          </select>
        </div>
        """
        class_html = b"""
        <div id=\"spectral_list\">
          <input type=\"checkbox\" value=\"water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt\" />
        </div>
        """
        mock_urlopen.side_effect = [
            FakeResponse(download_html, url="https://speclib.jpl.nasa.gov/download", headers={"Content-Type": "text/html"}),
            FakeResponse(all_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
            FakeResponse(class_html, url="https://speclib.jpl.nasa.gov/ecospeclib_list", headers={"Content-Type": "text/html"}),
        ]
        source = make_source(
            source_id="ecostress_v1",
            name="ECOSTRESS Spectral Library v1.0",
            provider="jpl_nasa",
            fetch_adapter="ecostress_web",
            landing_url="https://speclib.jpl.nasa.gov/",
            download_url="https://speclib.jpl.nasa.gov/download",
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"ECOSTRESS_DOWNLOAD_WORKERS": "1"}, clear=False):
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            destination = data_dir / "water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt"
            destination.write_text("Name: Distilled Water\n14.0\t3.8\n", encoding="utf-8")

            result = fetch_ecostress(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            self.assertEqual(result.artifacts[1].status, "existing")
            self.assertEqual(Path(result.artifacts[1].path), destination)
            self.assertIn("Reused 1 existing ECOSTRESS spectral text files", " ".join(result.notes))


class GitHubArchiveFetcherTests(unittest.TestCase):
    def test_github_archive_helpers_cover_member_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("repo-root/surface/", "")
                archive.writestr("README.md", "root")
                archive.writestr("repo-root/other/file.txt", "skip")
                archive.writestr("repo-root/surface/veg/file_a.hdr", "hdr")
            with zipfile.ZipFile(archive_path) as archive:
                members = _iter_matching_members(archive, "surface")
                self.assertEqual([member.filename for member in members], ["repo-root/surface/veg/file_a.hdr"])
                self.assertEqual(str(_relative_destination(members[0], "surface")), "veg/file_a.hdr")
                with self.assertRaises(ValueError):
                    _relative_destination(zipfile.ZipInfo("repo-root/other/file.txt"), "surface")

    @patch("spectral_library.fetchers.github_archive.urlopen")
    def test_github_archive_fetch_metadata_mode(self, mock_urlopen) -> None:
        source = make_source(
            source_id="emit_l2a_surface",
            name="EMIT L2A surface repository (emit-sds-l2a/surface)",
            provider="github",
            fetch_adapter="github_archive",
            landing_url="https://github.com/emit-sds/emit-sds-l2a",
            download_url="https://codeload.github.com/emit-sds/emit-sds-l2a/zip/refs/heads/develop#surface",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_github_archive(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(len(result.artifacts), 1)
            self.assertFalse(mock_urlopen.called)
            payload = json.loads((Path(tmpdir) / "github_archive.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["archive_url"], "https://codeload.github.com/emit-sds/emit-sds-l2a/zip/refs/heads/develop")
            self.assertEqual(payload["extract_prefix"], "surface")

    @patch("spectral_library.fetchers.github_archive.urlopen")
    def test_github_archive_fetch_assets_mode_extracts_only_surface_prefix(self, mock_urlopen) -> None:
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("emit-sds-l2a-develop/surface/veg/file_a.hdr", "hdr")
            archive.writestr("emit-sds-l2a-develop/surface/veg/file_b.json", "{\"k\": 1}")
            archive.writestr("emit-sds-l2a-develop/data/kurudz_0.1nm.dat", "skip me")
            archive.writestr("emit-sds-l2a-develop/README.md", "skip me too")
        archive_buffer.seek(0)
        mock_urlopen.return_value = FakeResponse(
            archive_buffer.read(),
            url="https://codeload.github.com/emit-sds/emit-sds-l2a/zip/refs/heads/develop",
            headers={"Content-Type": "application/zip"},
        )
        source = make_source(
            source_id="emit_l2a_surface",
            name="EMIT L2A surface repository (emit-sds-l2a/surface)",
            provider="github",
            fetch_adapter="github_archive",
            landing_url="https://github.com/emit-sds/emit-sds-l2a",
            download_url="https://codeload.github.com/emit-sds/emit-sds-l2a/zip/refs/heads/develop#surface",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_github_archive(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 3)
            self.assertTrue((Path(tmpdir) / "data" / "veg" / "file_a.hdr").exists())
            self.assertTrue((Path(tmpdir) / "data" / "veg" / "file_b.json").exists())
            self.assertFalse((Path(tmpdir) / "data" / "kurudz_0.1nm.dat").exists())
            self.assertIn("Extracted 2 files", result.notes[0])


class StaticHttpFetcherTests(unittest.TestCase):
    @patch("spectral_library.fetchers.static_http.urlopen")
    def test_static_http_metadata_mode(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(
            b"ignored",
            url="https://example.com/file.zip",
            headers={"Content-Type": "application/zip"},
        )
        source = make_source(fetch_adapter="static_http", landing_url="https://example.com/file.zip")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_static_http(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(len(result.artifacts), 1)
            self.assertIn("Metadata captured", result.notes[0])

    @patch("spectral_library.fetchers.static_http.urlopen")
    def test_static_http_assets_mode_downloads_file(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(
            b"payload",
            url="https://example.com/archive.zip",
            headers={"Content-Type": "application/zip"},
        )
        source = make_source(fetch_adapter="static_http", landing_url="https://example.com/archive.zip")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_static_http(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            self.assertTrue((Path(tmpdir) / "archive.zip").exists())

    @patch("spectral_library.fetchers.static_http.urlopen")
    def test_static_http_assets_mode_html_only_adds_note(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(
            b"<html></html>",
            url="https://example.com/landing",
            headers={"Content-Type": "text/html"},
        )
        source = make_source(fetch_adapter="static_http", landing_url="https://example.com/landing")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_static_http(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(len(result.artifacts), 1)
            self.assertIn("manual asset resolution", result.notes[0])

    @patch("spectral_library.fetchers.static_http.urlopen")
    def test_static_http_assets_mode_prefers_download_url(self, mock_urlopen) -> None:
        response = FakeResponse(
            b"payload",
            url="https://downloads.example.com/direct.csv",
            headers={"Content-Type": "text/csv"},
        )
        mock_urlopen.return_value = response
        source = make_source(
            fetch_adapter="static_http",
            landing_url="https://example.com/landing",
            download_url="https://downloads.example.com/direct.csv",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_static_http(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.artifacts[1].url, "https://downloads.example.com/direct.csv")
            self.assertTrue((Path(tmpdir) / "direct.csv").exists())
            requested_url = json.loads((Path(tmpdir) / "http_response.json").read_text(encoding="utf-8"))["requested_url"]
            self.assertEqual(requested_url, "https://downloads.example.com/direct.csv")

    @patch("spectral_library.fetchers.static_http.urlopen")
    def test_static_http_falls_back_when_download_url_is_forbidden(self, mock_urlopen) -> None:
        forbidden = HTTPError(
            url="https://downloads.example.com/direct.zip",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )
        landing_response = FakeResponse(
            b"<html></html>",
            url="https://example.com/landing",
            headers={"Content-Type": "text/html"},
        )
        mock_urlopen.side_effect = [forbidden, landing_response]
        source = make_source(
            fetch_adapter="static_http",
            landing_url="https://example.com/landing",
            download_url="https://downloads.example.com/direct.zip",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_static_http(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertIn("HTTP 403", result.notes[0])
            self.assertIn("manual asset resolution", result.notes[1])
            requested_url = json.loads((Path(tmpdir) / "http_response.json").read_text(encoding="utf-8"))["requested_url"]
            self.assertEqual(requested_url, "https://example.com/landing")


class SpecchioFetcherTests(unittest.TestCase):
    def test_specchio_helper_functions_cover_edge_cases(self) -> None:
        fake_jpype_module = object()
        with patch.dict(sys.modules, {"jpype": fake_jpype_module}):
            from spectral_library.fetchers.specchio import _descriptor_name, _load_jpype_module, _parse_server_index, _safe_len, _to_list

            descriptor = object()
            self.assertIs(_load_jpype_module(), fake_jpype_module)
            self.assertEqual(_load_query_config("   "), [])
            self.assertEqual(_safe_len(type("Sized", (), {"size": lambda self: 3})()), 3)
            self.assertEqual(_to_list((1, 2)), [1, 2])
            self.assertEqual(_descriptor_name(descriptor), str(descriptor))
            self.assertEqual(_parse_server_index("2"), 2)

        with self.assertRaises(ValueError):
            _load_query_config('{"conditions": "invalid"}')
        with self.assertRaises(ValueError):
            _load_query_config('[1]')
        with self.assertRaises(ValueError):
            from spectral_library.fetchers.specchio import _parse_server_index

            _parse_server_index("abc")
        with self.assertRaises(ValueError):
            from spectral_library.fetchers.specchio import _parse_server_index

            _parse_server_index("-1")

    def test_load_query_config_accepts_json_and_file(self) -> None:
        inline_conditions = _load_query_config(
            '[{"attribute": "Species", "operator": "=", "value": "oak"}]'
        )
        self.assertEqual(
            inline_conditions,
            [{"attribute": "Species", "operator": "=", "value": "oak"}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            query_path = Path(tmpdir) / "query.json"
            query_path.write_text(
                json.dumps({"conditions": [{"attribute": "Campaign", "operator": "=", "value": "demo"}]}),
                encoding="utf-8",
            )
            file_conditions = _load_query_config(str(query_path))

        self.assertEqual(
            file_conditions,
            [{"attribute": "Campaign", "operator": "=", "value": "demo"}],
        )

    def test_load_query_config_rejects_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            _load_query_config('[{"attribute": "Species", "value": "oak"}]')

    def test_specchio_fetch_without_client_jar_requires_manual_review(self) -> None:
        source = make_source(
            source_id="specchio_portal",
            name="SPECCHIO",
            provider="specchio",
            fetch_adapter="specchio_client",
            landing_url="https://specchio.ch/",
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {}, clear=True):
            result = fetch_specchio(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "manual_review")
            self.assertEqual(result.fetch_mode, "metadata")
            runtime_payload = json.loads((Path(tmpdir) / "specchio_runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_payload["client_jar"], "")
            self.assertEqual(runtime_payload["query_conditions"], [])

    def test_specchio_assets_mode_requires_query_conditions(self) -> None:
        source = make_source(
            source_id="specchio_portal",
            name="SPECCHIO",
            provider="specchio",
            fetch_adapter="specchio_client",
            landing_url="https://specchio.ch/",
        )
        env = {"SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", env, clear=True):
            result = fetch_specchio(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "manual_review")
            self.assertIn("SPECCHIO_QUERY_JSON", result.notes[0])

    def test_specchio_metadata_mode_captures_runtime_details(self) -> None:
        source = make_source(
            source_id="specchio_portal",
            name="SPECCHIO",
            provider="specchio",
            fetch_adapter="specchio_client",
            landing_url="https://specchio.ch/",
        )
        fake_factory = FakeSpecchioFactory(FakeSpecchioClient())
        fake_jpype = FakeJPype(fake_factory)
        env = {
            "SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar",
            "SPECCHIO_SERVER_INDEX": "1",
            "SPECCHIO_ORDER_BY": "Spectrum Name",
            "SPECCHIO_JAVA_HOME": "/tmp/java-home",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", env, clear=True):
            with patch("spectral_library.fetchers.specchio._load_jpype_module", return_value=fake_jpype):
                result = fetch_specchio(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertTrue(fake_jpype.started)
            self.assertIn("-Djava.class.path=/tmp/specchio-client.jar", fake_jpype.start_args)
            self.assertEqual(os.environ["JAVA_HOME"], "/tmp/java-home")
            runtime_payload = json.loads((Path(tmpdir) / "specchio_runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_payload["descriptor_name"], "Secondary")
            self.assertEqual(runtime_payload["descriptor_names"], ["Primary", "Secondary"])
            self.assertEqual(runtime_payload["ids_count"], 0)
            self.assertEqual(runtime_payload["spaces_count"], 0)
            self.assertIn("Secondary", result.notes[0])

    def test_specchio_fetch_rejects_missing_attributes_and_bad_descriptors(self) -> None:
        source = make_source(
            source_id="specchio_portal",
            name="SPECCHIO",
            provider="specchio",
            fetch_adapter="specchio_client",
            landing_url="https://specchio.ch/",
        )
        env = {
            "SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar",
            "SPECCHIO_QUERY_JSON": json.dumps(
                [{"attribute": "Missing Attribute", "operator": "=", "value": "oak"}]
            ),
        }
        fake_factory = FakeSpecchioFactory(FakeSpecchioClient())
        fake_jpype = FakeJPype(fake_factory)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", env, clear=True):
            with patch("spectral_library.fetchers.specchio._load_jpype_module", return_value=fake_jpype):
                with self.assertRaisesRegex(ValueError, "attribute not found"):
                    fetch_specchio(source, Path(tmpdir), "assets", "ua")

        bad_index_env = {
            "SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar",
            "SPECCHIO_SERVER_INDEX": "9",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", bad_index_env, clear=True):
            with patch("spectral_library.fetchers.specchio._load_jpype_module", return_value=fake_jpype):
                with self.assertRaisesRegex(ValueError, "out of range"):
                    fetch_specchio(source, Path(tmpdir), "metadata", "ua")

        empty_factory = FakeSpecchioFactory(FakeSpecchioClient())
        empty_factory.descriptors = []
        empty_jpype = FakeJPype(empty_factory)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar"}, clear=True):
            with patch("spectral_library.fetchers.specchio._load_jpype_module", return_value=empty_jpype):
                with self.assertRaisesRegex(ValueError, "no server descriptors"):
                    fetch_specchio(source, Path(tmpdir), "metadata", "ua")

    def test_specchio_assets_mode_writes_space_csv(self) -> None:
        source = make_source(
            source_id="specchio_portal",
            name="SPECCHIO",
            provider="specchio",
            fetch_adapter="specchio_client",
            landing_url="https://specchio.ch/",
        )
        spaces = [
            FakeSpecchioSpace(
                wavelengths=[400, 401],
                vectors=[[0.1, 0.2], [0.3, 0.4]],
                spectrum_ids=[9001, 9002],
            )
        ]
        fake_factory = FakeSpecchioFactory(FakeSpecchioClient(spaces=spaces))
        fake_jpype = FakeJPype(fake_factory)
        env = {
            "SPECCHIO_CLIENT_JAR": "/tmp/specchio-client.jar",
            "SPECCHIO_QUERY_JSON": json.dumps(
                [{"attribute": "Species", "operator": "=", "value": "oak"}]
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", env, clear=True):
            with patch("spectral_library.fetchers.specchio._load_jpype_module", return_value=fake_jpype):
                result = fetch_specchio(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            csv_path = Path(tmpdir) / "space_001.csv"
            self.assertTrue(csv_path.exists())
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "spectrum_id,measurement_unit,400,401")
            self.assertIn("9001,reflectance,0.1,0.2", lines[1])
            self.assertEqual(fake_factory.client.order_by, "Acquisition Time")
            self.assertEqual(len(fake_factory.client.query.conditions), 1)
            self.assertEqual(fake_factory.client.query.conditions[0].attribute, "species_attr")


class ZenodoFetcherTests(unittest.TestCase):
    def test_extract_record_id(self) -> None:
        self.assertEqual(_extract_record_id("https://zenodo.org/records/12345"), "12345")
        self.assertEqual(_extract_record_id("https://zenodo.org/records/12345/files/file.pdf?download=1"), "12345")
        self.assertEqual(
            _file_urls({"links": {"self": "https://zenodo.org/api/records/1/files/a file.csv/content"}}),
            ["https://zenodo.org/api/records/1/files/a%20file.csv/content"],
        )
        self.assertEqual(_file_urls({"links": []}), [])
        self.assertEqual(_filename_from_content_url("https://zenodo.org/api/records/1/files/a%20file.csv/content"), "a file.csv")
        self.assertEqual(_filename_from_content_url("https://zenodo.org/api/records/1"), "")
        with self.assertRaises(ValueError):
            _extract_record_id("https://zenodo.org/files/12345")
        with self.assertRaises(ValueError):
            _extract_record_id("https://zenodo.org/records/not-a-number")

    @patch("spectral_library.fetchers.zenodo.urlopen")
    def test_zenodo_fetch_metadata_mode(self, mock_urlopen) -> None:
        payload = {
            "files": [
                {
                    "key": "a.csv",
                    "checksum": "abc",
                    "size": 5,
                    "mimetype": "text/csv",
                    "links": {"self": "https://zenodo.org/api/files/a.csv"},
                }
            ]
        }
        mock_urlopen.return_value = FakeResponse(
            json.dumps(payload).encode("utf-8"),
            url="https://zenodo.org/api/records/12345",
            headers={"Content-Type": "application/json"},
        )
        source = make_source(landing_url="https://zenodo.org/records/12345")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_zenodo(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(len(result.artifacts), 2)
            self.assertEqual(result.artifacts[1].status, "discovered")
            self.assertIn("metadata captured", result.notes[0].lower())

    @patch("spectral_library.fetchers.zenodo.urlopen")
    def test_zenodo_fetch_assets_mode_downloads_files(self, mock_urlopen) -> None:
        payload = {
            "files": [
                {
                    "key": "spectra.csv",
                    "checksum": "sha256:ignored",
                    "size": 7,
                    "mimetype": "text/csv",
                    "links": {"download": "https://zenodo.org/api/files/spectra data.csv"},
                }
            ]
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://zenodo.org/api/records/12345",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"1,2,3\n",
                url="https://zenodo.org/api/files/spectra%20data.csv",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(landing_url="https://zenodo.org/records/12345")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_zenodo(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            self.assertTrue((Path(tmpdir) / "spectra.csv").exists())

    @patch("spectral_library.fetchers.zenodo.urlopen")
    def test_zenodo_fetch_assets_mode_prefers_download_url(self, mock_urlopen) -> None:
        payload = {
            "files": [
                {
                    "key": "spectra.csv",
                    "checksum": "sha256:ignored",
                    "size": 7,
                    "mimetype": "text/csv",
                    "links": {"download": "https://zenodo.org/api/records/12345/files/spectra.csv/content"},
                },
                {
                    "key": "metadata.csv",
                    "checksum": "sha256:ignored",
                    "size": 8,
                    "mimetype": "text/csv",
                    "links": {"download": "https://zenodo.org/api/records/12345/files/metadata.csv/content"},
                },
            ]
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://zenodo.org/api/records/12345",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"1,2,3\n",
                url="https://zenodo.org/api/records/12345/files/spectra.csv/content",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            landing_url="https://zenodo.org/records/12345",
            download_url="https://zenodo.org/api/records/12345/files/spectra.csv/content",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_zenodo(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 3)
            self.assertEqual(result.artifacts[1].url, "https://zenodo.org/api/records/12345/files/spectra.csv/content")
            self.assertEqual(result.artifacts[2].status, "discovered")
            self.assertTrue((Path(tmpdir) / "spectra.csv").exists())
            self.assertIn("limited asset download", result.notes[0].lower())

    @patch("spectral_library.fetchers.zenodo.urlopen")
    def test_zenodo_fetch_downloads_selected_content_url_not_listed_in_record(self, mock_urlopen) -> None:
        payload = {
            "files": [
                {
                    "key": "metadata.csv",
                    "checksum": "sha256:ignored",
                    "size": 8,
                    "mimetype": "text/csv",
                    "links": {"download": "https://zenodo.org/api/records/12345/files/metadata.csv/content"},
                }
            ]
        }
        mock_urlopen.side_effect = [
            FakeResponse(
                json.dumps(payload).encode("utf-8"),
                url="https://zenodo.org/api/records/12345",
                headers={"Content-Type": "application/json"},
            ),
            FakeResponse(
                b"1,2,3\n",
                url="https://zenodo.org/api/records/12345/files/primary%20spectra.csv/content",
                headers={"Content-Type": "text/csv"},
            ),
        ]
        source = make_source(
            landing_url="https://zenodo.org/records/12345",
            download_url="https://zenodo.org/api/records/12345/files/primary spectra.csv/content",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_zenodo(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertTrue((Path(tmpdir) / "primary spectra.csv").exists())
            self.assertIn("downloaded it directly", result.notes[0].lower())

    @patch("spectral_library.fetchers.zenodo.urlopen")
    def test_zenodo_fetch_reports_unmatched_noncontent_download_url(self, mock_urlopen) -> None:
        payload = {"files": []}
        mock_urlopen.return_value = FakeResponse(
            json.dumps(payload).encode("utf-8"),
            url="https://zenodo.org/api/records/12345",
            headers={"Content-Type": "application/json"},
        )
        source = make_source(
            landing_url="https://zenodo.org/records/12345",
            download_url="https://zenodo.org/records/12345",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_zenodo(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertIn("did not resolve", result.notes[0].lower())


class PangaeaFetcherTests(unittest.TestCase):
    @patch("spectral_library.fetchers.pangaea.urlopen")
    def test_pangaea_fetch_metadata_mode(self, mock_urlopen) -> None:
        landing_html = b'<html><a href="https://doi.pangaea.de/10.1594/PANGAEA.948492">doi</a></html>'
        doi_html = b'<html><a href="https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip">zip</a></html>'
        mock_urlopen.side_effect = [
            FakeResponse(landing_html, url="https://example.com/gloria", headers={"Content-Type": "text/html"}),
            FakeResponse(doi_html, url="https://doi.pangaea.de/10.1594/PANGAEA.948492", headers={"Content-Type": "text/html"}),
        ]
        source = make_source(
            source_id="gloria",
            fetch_adapter="pangaea",
            provider="pangaea",
            landing_url="https://example.com/gloria",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_pangaea(source, Path(tmpdir), "metadata", "ua")

            self.assertEqual(result.status, "metadata_only")
            self.assertEqual(result.artifacts[0].artifact_id, "pangaea_record")
            self.assertIn("metadata captured", result.notes[0].lower())

    @patch("spectral_library.fetchers.pangaea.urlopen")
    def test_pangaea_fetch_assets_mode_downloads_archive(self, mock_urlopen) -> None:
        landing_html = b'<html><a href="https://doi.pangaea.de/10.1594/PANGAEA.886287">doi</a></html>'
        doi_html = b'<html><a href="?format=zip">zip</a></html>'
        mock_urlopen.side_effect = [
            FakeResponse(landing_html, url="https://example.com/seaswir", headers={"Content-Type": "text/html"}),
            FakeResponse(doi_html, url="https://doi.pangaea.de/10.1594/PANGAEA.886287", headers={"Content-Type": "text/html"}),
            FakeResponse(b'zip-bytes', url="https://doi.pangaea.de/10.1594/PANGAEA.886287?format=zip", headers={"Content-Type": "application/zip"}),
        ]
        source = make_source(
            source_id="seaswir",
            fetch_adapter="pangaea",
            provider="pangaea",
            landing_url="https://example.com/seaswir",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fetch_pangaea(source, Path(tmpdir), "assets", "ua")

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(len(result.artifacts), 2)
            self.assertTrue((Path(tmpdir) / "seaswir.zip").exists())


if __name__ == "__main__":
    unittest.main()

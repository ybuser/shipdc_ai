from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shipdc_ai.manifest import MANIFEST_HEADER  # noqa: E402


EXPECTED_HEADER = [
    "asset_id",
    "source_group",
    "source_name",
    "source_url",
    "download_date",
    "local_path",
    "asset_type",
    "license_or_rights",
    "zone_type",
    "confuser_tag",
    "use_purpose",
    "sha256",
    "notes",
]


class ManifestHeaderTest(TestCase):
    def test_manifest_constant_matches_required_header(self) -> None:
        self.assertEqual(MANIFEST_HEADER, EXPECTED_HEADER)

    def test_master_manifest_contains_only_header_initially(self) -> None:
        manifest_path = ROOT / "02_processed" / "manifests" / "master_manifest.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        self.assertEqual(rows, [EXPECTED_HEADER])

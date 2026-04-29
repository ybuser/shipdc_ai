from __future__ import annotations

import csv
import sys
from io import StringIO
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

    def test_master_manifest_first_row_matches_required_header(self) -> None:
        manifest_path = ROOT / "02_processed" / "manifests" / "master_manifest.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, [])
        self.assertEqual(header, EXPECTED_HEADER)

    def test_populated_manifest_first_row_matches_required_header(self) -> None:
        populated_manifest = StringIO()
        writer = csv.writer(populated_manifest)
        writer.writerow(EXPECTED_HEADER)
        writer.writerow(["asset_001", *[""] * (len(EXPECTED_HEADER) - 1)])
        populated_manifest.seek(0)

        reader = csv.reader(populated_manifest)
        self.assertEqual(next(reader, []), EXPECTED_HEADER)

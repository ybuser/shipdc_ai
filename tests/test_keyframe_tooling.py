from __future__ import annotations

import base64
import csv
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.extract_keyframes import (  # noqa: E402
    KEYFRAME_MANIFEST_HEADER,
    hash_file,
    run_extraction,
    write_keyframe_manifest,
)
from scripts.validate_keyframe_manifest import validate_keyframe_manifest  # noqa: E402
from shipdc_ai.manifest import MANIFEST_HEADER  # noqa: E402


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYPgPAAEDAQD8yZ2+AAAAAElFTkSuQmCC"
)


def write_master_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def master_image_row(asset_id: str, local_path: str) -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "source_group": "shiplike",
        "source_name": "NOAA",
        "source_url": "https://example.invalid/public",
        "download_date": "2026-04-26",
        "local_path": local_path,
        "asset_type": "image",
        "license_or_rights": "public review",
        "zone_type": "control_room",
        "confuser_tag": "",
        "use_purpose": "shiplike_background",
        "sha256": "",
        "notes": "test public still image",
    }


def write_png(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_1X1)
    return path


class KeyframeToolingTest(TestCase):
    def test_keyframe_manifest_reports_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "bad_keyframes.csv"
            manifest.write_text("asset_id,frame_path\nSHIP_TEST_001,missing.png\n", encoding="utf-8")

            result = validate_keyframe_manifest(manifest, root=root)

        self.assertTrue(any("missing required columns" in error for error in result.errors))

    def test_sha256_validation_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = write_png(root, "02_processed/frames/ship_like/NOAA/20260426/SHIP_TEST_001/still_000000.png")
            manifest = root / "02_processed" / "manifests" / "ship_like_keyframes.csv"
            good_row = {
                "asset_id": "SHIP_TEST_001",
                "source_name": "NOAA",
                "source_asset_path": "01_raw/source.png",
                "frame_path": frame.relative_to(root).as_posix(),
                "timestamp_s": "",
                "frame_index": "0",
                "width": "1",
                "height": "1",
                "extraction_fps": "",
                "sha256": hash_file(frame),
                "notes": "still_image_candidate; not_video_keyframe",
            }
            write_keyframe_manifest(manifest, [good_row])
            self.assertEqual(validate_keyframe_manifest(manifest, root=root).errors, [])

            bad_row = dict(good_row)
            bad_row["sha256"] = "0" * 64
            write_keyframe_manifest(manifest, [bad_row])
            result = validate_keyframe_manifest(manifest, root=root)

        self.assertTrue(any("sha256 mismatch" in error for error in result.errors))

    def test_missing_frame_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "02_processed" / "manifests" / "ship_like_keyframes.csv"
            row = {
                "asset_id": "SHIP_TEST_001",
                "source_name": "NOAA",
                "source_asset_path": "01_raw/source.png",
                "frame_path": "02_processed/frames/ship_like/NOAA/20260426/SHIP_TEST_001/missing.png",
                "timestamp_s": "",
                "frame_index": "0",
                "width": "",
                "height": "",
                "extraction_fps": "",
                "sha256": "",
                "notes": "still_image_candidate; not_video_keyframe",
            }
            write_keyframe_manifest(manifest, [row])
            result = validate_keyframe_manifest(manifest, root=root)

        self.assertTrue(any("frame file does not exist" in error for error in result.errors))

    def test_still_image_candidate_is_copied_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_relative = "01_raw/02_shiplike/NOAA/downloads/source.png"
            write_png(root, source_relative)
            master_manifest = root / "02_processed" / "manifests" / "master_manifest.csv"
            output_manifest = root / "02_processed" / "manifests" / "ship_like_keyframes.csv"
            output_root = root / "02_processed" / "frames" / "ship_like"
            write_master_manifest(master_manifest, [master_image_row("SHIP_TEST_001", source_relative)])

            summary = run_extraction(
                manifest_path=master_manifest,
                output_root=output_root,
                output_manifest=output_manifest,
                asset_prefixes=["SHIP_TEST"],
                root=root,
            )

            copied = output_root / "NOAA" / "20260426" / "SHIP_TEST_001" / "still_000000.png"
            copied_exists = copied.is_file()
            with output_manifest.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(summary.processed_assets, 1)
        self.assertTrue(copied_exists)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_path"], "02_processed/frames/ship_like/NOAA/20260426/SHIP_TEST_001/still_000000.png")
        self.assertEqual(rows[0]["timestamp_s"], "")
        self.assertEqual(rows[0]["extraction_fps"], "")
        self.assertEqual(rows[0]["frame_index"], "0")
        self.assertEqual(rows[0]["width"], "1")
        self.assertEqual(rows[0]["height"], "1")
        self.assertIn("not_video_keyframe", rows[0]["notes"])

    def test_manifest_merge_preserves_unselected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_relative = "01_raw/02_shiplike/NOAA/downloads/source.png"
            write_png(root, source_relative)
            master_manifest = root / "02_processed" / "manifests" / "master_manifest.csv"
            output_manifest = root / "02_processed" / "manifests" / "ship_like_keyframes.csv"
            output_root = root / "02_processed" / "frames" / "ship_like"
            write_master_manifest(master_manifest, [master_image_row("SHIP_TEST_001", source_relative)])

            old_frame = write_png(root, "02_processed/frames/ship_like/DVIDS/20260426/SHIP_OLD_001/still_000000.png")
            existing_rows = [
                {
                    "asset_id": "SHIP_OLD_001",
                    "source_name": "DVIDS",
                    "source_asset_path": "01_raw/old.png",
                    "frame_path": old_frame.relative_to(root).as_posix(),
                    "timestamp_s": "",
                    "frame_index": "0",
                    "width": "1",
                    "height": "1",
                    "extraction_fps": "",
                    "sha256": hash_file(old_frame),
                    "notes": "preserve",
                },
                {
                    "asset_id": "SHIP_TEST_001",
                    "source_name": "NOAA",
                    "source_asset_path": "01_raw/stale.png",
                    "frame_path": "02_processed/frames/ship_like/NOAA/stale/still_000000.png",
                    "timestamp_s": "",
                    "frame_index": "0",
                    "width": "",
                    "height": "",
                    "extraction_fps": "",
                    "sha256": "",
                    "notes": "replace",
                },
            ]
            write_keyframe_manifest(output_manifest, existing_rows)

            run_extraction(
                manifest_path=master_manifest,
                output_root=output_root,
                output_manifest=output_manifest,
                asset_prefixes=["SHIP_TEST"],
                root=root,
            )
            with output_manifest.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([row["asset_id"] for row in rows], ["SHIP_OLD_001", "SHIP_TEST_001"])
        self.assertEqual(rows[0]["notes"], "preserve")
        self.assertEqual(rows[1]["source_asset_path"], source_relative)
        self.assertNotEqual(rows[1]["notes"], "replace")
        self.assertEqual(KEYFRAME_MANIFEST_HEADER, list(rows[1].keys()))

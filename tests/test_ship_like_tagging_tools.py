from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_review_gallery import parse_args, run_gallery  # noqa: E402
from scripts.make_ship_like_tagging_sheet import (  # noqa: E402
    KEYFRAME_INPUT_HEADER,
    TAGGING_SHEET_HEADER,
    build_tagging_rows,
    run_tagging_sheet,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def keyframe_row(
    *,
    asset_id: str,
    source_name: str,
    frame_path: str,
    frame_index: str = "0",
    timestamp_s: str = "",
) -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "source_name": source_name,
        "source_asset_path": f"01_raw/02_shiplike/{source_name}/downloads/source.jpg",
        "frame_path": frame_path,
        "timestamp_s": timestamp_s,
        "frame_index": frame_index,
        "width": "640",
        "height": "360",
        "extraction_fps": "1",
        "sha256": "a" * 64,
        "notes": "video_keyframe",
    }


class ShipLikeTaggingToolsTest(TestCase):
    def test_tagging_sheet_header_defaults_and_stable_frame_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_manifest = root / "02_processed" / "manifests" / "ship_like_keyframes.csv"
            output_sheet = root / "02_processed" / "manifests" / "ship_like_tagging_sheet.csv"
            rows = [
                keyframe_row(
                    asset_id="SHIP_TEST_001",
                    source_name="NOAA",
                    frame_path="02_processed/frames/ship_like/NOAA/SHIP_TEST_001/frame_000000.jpg",
                )
            ]
            write_csv(input_manifest, KEYFRAME_INPUT_HEADER, rows)

            summary = run_tagging_sheet(input_manifest, output_sheet)
            header, tagging_rows = read_csv(output_sheet)

        self.assertEqual(summary.written_rows, 1)
        self.assertEqual(header, TAGGING_SHEET_HEADER)
        self.assertEqual(tagging_rows[0]["asset_id"], "SHIP_TEST_001")
        self.assertEqual(
            tagging_rows[0]["frame_path"],
            "02_processed/frames/ship_like/NOAA/SHIP_TEST_001/frame_000000.jpg",
        )
        self.assertTrue(tagging_rows[0]["frame_id"].startswith("SHIP_TEST_001_"))
        self.assertEqual(tagging_rows[0]["zone_l1"], "exterior_or_unknown")
        self.assertEqual(tagging_rows[0]["confuser_tags"], "none")
        self.assertEqual(tagging_rows[0]["use_purpose"], "exclude")
        self.assertEqual(tagging_rows[0]["split"], "exclude")
        self.assertEqual(tagging_rows[0]["quality_flag"], "good")
        self.assertEqual(tagging_rows[0]["reviewer_notes"], "")
        self.assertEqual(build_tagging_rows(rows)[0]["frame_id"], tagging_rows[0]["frame_id"])

    def test_tagging_sheet_rejects_absolute_frame_path(self) -> None:
        rows = [
            keyframe_row(
                asset_id="SHIP_TEST_001",
                source_name="NOAA",
                frame_path="C:/frames/frame_000000.jpg",
            )
        ]

        with self.assertRaisesRegex(ValueError, "repository-relative"):
            build_tagging_rows(rows)

    def test_review_gallery_uses_relative_paths_and_limit_per_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_sheet = root / "02_processed" / "manifests" / "ship_like_tagging_sheet.csv"
            output_html = root / "02_processed" / "review" / "ship_like_gallery.html"
            base_row = {
                "frame_id": "SHIP_TEST_001_a",
                "asset_id": "SHIP_TEST_001",
                "source_name": "NOAA",
                "frame_path": "02_processed/frames/ship_like/NOAA/SHIP_TEST_001/frame_000000.jpg",
                "timestamp_s": "0",
                "frame_index": "0",
                "width": "640",
                "height": "360",
                "zone_l1": "exterior_or_unknown",
                "confuser_tags": "none",
                "use_purpose": "exclude",
                "split": "exclude",
                "quality_flag": "good",
                "reviewer_notes": "",
                "sha256": "a" * 64,
            }
            second_row = dict(base_row)
            second_row["frame_id"] = "SHIP_TEST_001_b"
            second_row["frame_path"] = "02_processed/frames/ship_like/NOAA/SHIP_TEST_001/frame_000001.jpg"
            second_row["timestamp_s"] = "1"
            second_row["frame_index"] = "1"
            write_csv(input_sheet, TAGGING_SHEET_HEADER, [base_row, second_row])

            summary = run_gallery(
                input_sheet=input_sheet,
                output_html=output_html,
                limit_per_asset=1,
                root=root,
            )
            content = output_html.read_text(encoding="utf-8")

        self.assertEqual(summary.total_rows, 2)
        self.assertEqual(summary.displayed_rows, 1)
        self.assertIn("../frames/ship_like/NOAA/SHIP_TEST_001/frame_000000.jpg", content)
        self.assertIn("SHIP_TEST_001_a", content)
        self.assertNotIn("SHIP_TEST_001_b", content)

    def test_review_gallery_parse_args_accepts_preferred_and_legacy_flags(self) -> None:
        preferred = parse_args(
            [
                "--input",
                "input.csv",
                "--output",
                "gallery.html",
            ]
        )
        legacy = parse_args(
            [
                "--manifest",
                "legacy_input.csv",
                "--out",
                "legacy_gallery.html",
            ]
        )

        self.assertEqual(preferred.input, Path("input.csv"))
        self.assertEqual(preferred.output, Path("gallery.html"))
        self.assertEqual(legacy.input, Path("legacy_input.csv"))
        self.assertEqual(legacy.output, Path("legacy_gallery.html"))

    def test_review_gallery_cli_accepts_preferred_and_legacy_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_sheet = root / "ship_like_tagging_sheet.csv"
            preferred_output = root / "preferred_gallery.html"
            legacy_output = root / "legacy_gallery.html"
            row = {
                "frame_id": "SHIP_TEST_001_a",
                "asset_id": "SHIP_TEST_001",
                "source_name": "NOAA",
                "frame_path": "02_processed/frames/ship_like/NOAA/SHIP_TEST_001/frame_000000.jpg",
                "timestamp_s": "0",
                "frame_index": "0",
                "width": "640",
                "height": "360",
                "zone_l1": "exterior_or_unknown",
                "confuser_tags": "none",
                "use_purpose": "exclude",
                "split": "exclude",
                "quality_flag": "good",
                "reviewer_notes": "",
                "sha256": "a" * 64,
            }
            write_csv(input_sheet, TAGGING_SHEET_HEADER, [row])
            script = ROOT / "scripts" / "make_review_gallery.py"

            preferred = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_sheet),
                    "--output",
                    str(preferred_output),
                    "--limit-per-asset",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            legacy = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(input_sheet),
                    "--out",
                    str(legacy_output),
                    "--limit-per-asset",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(preferred.returncode, 0, preferred.stderr)
            self.assertEqual(legacy.returncode, 0, legacy.stderr)
            self.assertTrue(preferred_output.is_file())
            self.assertTrue(legacy_output.is_file())

from __future__ import annotations

import base64
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_donor_crop_bank import (  # noqa: E402
    DONOR_CROP_MANIFEST_HEADER,
    UnknownClassMappingError,
    YoloBox,
    class_map_from_source_config,
    convert_yolo_box,
    hash_file,
    parse_args,
    resolve_cli_output_paths,
    run_build_crop_bank,
)
from scripts.validate_donor_crop_manifest import validate_donor_crop_manifest  # noqa: E402
from shipdc_ai.manifest import MANIFEST_HEADER  # noqa: E402


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYPgPAAEDAQD8yZ2+AAAAAElFTkSuQmCC"
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def donor_master_row(source_name: str) -> dict[str, str]:
    return {
        "asset_id": f"DONOR_{source_name.upper()}_001",
        "source_group": "donor",
        "source_name": source_name,
        "source_url": "https://example.invalid/public",
        "download_date": "2026-05-06",
        "local_path": f"01_raw/01_donor/{source_name}/downloads/source.zip",
        "asset_type": "zip",
        "license_or_rights": "public_domain",
        "zone_type": "donor_fire",
        "confuser_tag": "",
        "use_purpose": "donor_train",
        "sha256": "",
        "notes": "public test donor",
    }


class DonorCropBankToolsTest(TestCase):
    def test_cli_explicit_output_options_resolve(self) -> None:
        root = Path("repo")
        args = parse_args(
            [
                "--output-images",
                "custom/images",
                "--preview-dir",
                "custom/preview",
                "--output-manifest",
                "custom/donor_crop_manifest.csv",
            ]
        )

        output_paths = resolve_cli_output_paths(args, root)

        self.assertEqual(output_paths.output_images_dir, root / "custom/images")
        self.assertEqual(output_paths.preview_dir, root / "custom/preview")
        self.assertEqual(output_paths.output_manifest, root / "custom/donor_crop_manifest.csv")

    def test_cli_out_dir_and_out_manifest_aliases_resolve(self) -> None:
        root = Path("repo")
        args = parse_args(
            [
                "--out-dir",
                "02_processed/crops/fire_smoke_donor_v1",
                "--out-manifest",
                "02_processed/manifests/donor_crop_manifest_v1.csv",
            ]
        )

        output_paths = resolve_cli_output_paths(args, root)

        self.assertEqual(
            output_paths.output_images_dir,
            root / "02_processed/crops/fire_smoke_donor_v1/images",
        )
        self.assertEqual(
            output_paths.preview_dir,
            root / "02_processed/crops/fire_smoke_donor_v1/preview",
        )
        self.assertEqual(
            output_paths.output_manifest,
            root / "02_processed/manifests/donor_crop_manifest_v1.csv",
        )

    def test_cli_out_dir_does_not_override_explicit_output_dirs(self) -> None:
        root = Path("repo")
        args = parse_args(
            [
                "--out-dir",
                "derived/root",
                "--output-images",
                "explicit/images",
                "--preview-dir",
                "explicit/preview",
            ]
        )

        output_paths = resolve_cli_output_paths(args, root)

        self.assertEqual(output_paths.output_images_dir, root / "explicit/images")
        self.assertEqual(output_paths.preview_dir, root / "explicit/preview")

    def test_unknown_numeric_metadata_class_map_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "dataset"
            source_root.mkdir()
            (source_root / "data.yaml").write_text("names: ['0', '1']\n", encoding="utf-8")

            with self.assertRaisesRegex(UnknownClassMappingError, "class mapping unknown"):
                class_map_from_source_config(
                    {
                        "source_name": "IndoorFireSmoke",
                        "metadata_names_file": "data.yaml",
                        "class_map": {},
                    },
                    source_root,
                )

    def test_yolo_normalized_bbox_converts_and_tiny_boxes_skip_with_note(self) -> None:
        converted, note = convert_yolo_box(
            YoloBox("0", 0.5, 0.5, 0.5, 0.5, line_number=3),
            image_width=100,
            image_height=80,
            min_crop_width=16,
            min_crop_height=16,
        )

        self.assertIsNone(note)
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted.xyxy, (25, 20, 75, 60))
        self.assertEqual(converted.width, 50)
        self.assertEqual(converted.height, 40)
        self.assertIn("yolo_normalized_bbox", converted.notes)

        tiny, tiny_note = convert_yolo_box(
            YoloBox("0", 0.5, 0.5, 0.01, 0.01, line_number=4),
            image_width=100,
            image_height=80,
            min_crop_width=16,
            min_crop_height=16,
        )

        self.assertIsNone(tiny)
        self.assertIn("tiny_box", tiny_note or "")

    def test_validate_donor_crop_manifest_success_and_hash_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            crop_path = root / "02_processed" / "crops" / "fire_smoke_donor_v1" / "images" / "fire" / "crop.png"
            source_image = root / "01_raw" / "source.jpg"
            source_label = root / "01_raw" / "source.txt"
            manifest = root / "02_processed" / "manifests" / "donor_crop_manifest_v1.csv"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            source_image.parent.mkdir(parents=True, exist_ok=True)
            crop_path.write_bytes(PNG_1X1)
            source_image.write_bytes(b"source")
            source_label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
            row = {
                "crop_id": "crop_001",
                "source_name": "TestSource",
                "source_image_path": "01_raw/source.jpg",
                "source_label_path": "01_raw/source.txt",
                "crop_path": "02_processed/crops/fire_smoke_donor_v1/images/fire/crop.png",
                "class_name": "fire",
                "bbox_xyxy": "[0,0,1,1]",
                "crop_width": "1",
                "crop_height": "1",
                "sha256": hash_file(crop_path),
                "notes": "yolo_normalized_bbox",
            }
            write_csv(manifest, DONOR_CROP_MANIFEST_HEADER, [row])

            self.assertEqual(validate_donor_crop_manifest(manifest, root=root).errors, [])

            bad_row = dict(row)
            bad_row["sha256"] = "0" * 64
            write_csv(manifest, DONOR_CROP_MANIFEST_HEADER, [bad_row])
            result = validate_donor_crop_manifest(manifest, root=root)

        self.assertTrue(any("sha256 mismatch" in error for error in result.errors))

    @skipUnless(importlib.util.find_spec("PIL") is not None, "Pillow is not installed")
    def test_build_crop_bank_writes_manifest_with_repo_relative_paths(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_name = "TestDonor"
            dataset_root = root / "01_raw" / "01_donor" / source_name / "unzipped"
            images_dir = dataset_root / "train" / "images"
            labels_dir = dataset_root / "train" / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), (255, 0, 0)).save(images_dir / "sample.jpg")
            (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            master_manifest = root / "02_processed" / "manifests" / "master_manifest.csv"
            write_csv(master_manifest, MANIFEST_HEADER, [donor_master_row(source_name)])
            config = {
                "master_manifest": "02_processed/manifests/master_manifest.csv",
                "output_images_dir": "02_processed/crops/fire_smoke_donor_v1/images",
                "preview_dir": "02_processed/crops/fire_smoke_donor_v1/preview",
                "output_manifest": "02_processed/manifests/donor_crop_manifest_v1.csv",
                "min_crop_width": 2,
                "min_crop_height": 2,
                "preview_max_per_class": 2,
                "preview_size_px": 16,
                "sources": [
                    {
                        "source_name": source_name,
                        "enabled": True,
                        "root": f"01_raw/01_donor/{source_name}/unzipped",
                        "splits": ["train"],
                        "image_dir_name": "images",
                        "label_dir_name": "labels",
                        "class_map": {"0": "fire"},
                    }
                ],
            }
            config_path = root / "configs" / "donor_crop_v1.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            summary = run_build_crop_bank(config_path=config_path, overwrite=True, root=root)
            output_manifest = root / "02_processed" / "manifests" / "donor_crop_manifest_v1.csv"
            with output_manifest.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(summary.crops_written, 1)
        self.assertEqual(rows[0]["class_name"], "fire")
        self.assertNotIn("\\", rows[0]["crop_path"])
        self.assertFalse(Path(rows[0]["crop_path"]).is_absolute())

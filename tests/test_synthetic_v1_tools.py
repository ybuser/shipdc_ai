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

from scripts.build_synth_v1 import (  # noqa: E402
    SYNTH_MANIFEST_HEADER,
    build_generation_schedule,
    hash_file,
    load_backgrounds,
    parse_args,
    resolve_build_options,
    run_build_synth_v1,
)
from scripts.summarize_synth_v1 import SUMMARY_CSV, SUMMARY_TXT, run_summary  # noqa: E402
from scripts.validate_synth_manifest import validate_synth_manifest  # noqa: E402


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYPgPAAEDAQD8yZ2+AAAAAElFTkSuQmCC"
)

BACKGROUND_HEADER = [
    "frame_id",
    "asset_id",
    "source_name",
    "frame_path",
    "timestamp_s",
    "frame_index",
    "width",
    "height",
    "zone_l1",
    "confuser_tags",
    "use_purpose",
    "split",
    "quality_flag",
    "reviewer_notes",
    "sha256",
]

DONOR_HEADER = [
    "crop_id",
    "source_name",
    "source_image_path",
    "source_label_path",
    "crop_path",
    "class_name",
    "bbox_xyxy",
    "crop_width",
    "crop_height",
    "sha256",
    "notes",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def background_row(frame_id: str, frame_path: str, **overrides: str) -> dict[str, str]:
    row = {
        "frame_id": frame_id,
        "asset_id": "SHIP_TEST_001",
        "source_name": "NOAA",
        "frame_path": frame_path,
        "timestamp_s": "0",
        "frame_index": "0",
        "width": "64",
        "height": "48",
        "zone_l1": "engine_room",
        "confuser_tags": "none",
        "use_purpose": "synth_background",
        "split": "train",
        "quality_flag": "good",
        "reviewer_notes": "",
        "sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def donor_row(crop_id: str, crop_path: str, class_name: str, source_name: str = "DFire") -> dict[str, str]:
    return {
        "crop_id": crop_id,
        "source_name": source_name,
        "source_image_path": "01_raw/source.jpg",
        "source_label_path": "01_raw/source.txt",
        "crop_path": crop_path,
        "class_name": class_name,
        "bbox_xyxy": "[0,0,10,10]",
        "crop_width": "10",
        "crop_height": "10",
        "sha256": "b" * 64,
        "notes": "test",
    }


class SyntheticV1ToolsTest(TestCase):
    def test_cli_aliases_resolve_through_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "synth_v1.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "background_manifest": "default/backgrounds.csv",
                        "donor_manifest": "default/donors.csv",
                        "output_dir": "default/out",
                        "output_manifest": "default/manifest.csv",
                        "max_samples_total": 170,
                        "preview_count": 50,
                        "seed": 20260426,
                        "class_ids": {"fire": 0, "smoke": 1},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config),
                    "--background-manifest",
                    "custom/backgrounds.csv",
                    "--donor-crop-manifest",
                    "custom/donors.csv",
                    "--out-dir",
                    "custom/out",
                    "--out-manifest",
                    "custom/synth.csv",
                    "--max-samples-total",
                    "4",
                    "--preview-count",
                    "2",
                    "--seed",
                    "99",
                ]
            )

            options = resolve_build_options(args, root=root)

        self.assertEqual(options.background_manifest, root / "custom/backgrounds.csv")
        self.assertEqual(options.donor_manifest, root / "custom/donors.csv")
        self.assertEqual(options.output_dir, root / "custom/out")
        self.assertEqual(options.output_manifest, root / "custom/synth.csv")
        self.assertEqual(options.max_samples_total, 4)
        self.assertEqual(options.preview_count, 2)
        self.assertEqual(options.seed, 99)

    def test_background_filter_excludes_with_human_and_non_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = root / "02_processed" / "frames" / "good.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(PNG_1X1)
            manifest = root / "02_processed" / "manifests" / "ship_like_background_train.csv"
            rows = [
                background_row("keep", "02_processed/frames/good.jpg"),
                background_row("human", "02_processed/frames/good.jpg", reviewer_notes="with_human"),
                background_row("val", "02_processed/frames/good.jpg", split="val"),
                background_row("bad_quality", "02_processed/frames/good.jpg", quality_flag="bad"),
            ]
            write_csv(manifest, BACKGROUND_HEADER, rows)

            backgrounds = load_backgrounds(manifest, root=root)

        self.assertEqual([background.frame_id for background in backgrounds], ["keep"])

    def test_generation_schedule_balances_classes_before_repeating_backgrounds(self) -> None:
        backgrounds = [
            background_row_obj("bg1"),
            background_row_obj("bg2"),
            background_row_obj("bg3"),
        ]

        schedule = build_generation_schedule(backgrounds, max_samples_total=6)

        self.assertEqual(
            [(background.frame_id, class_name) for background, class_name in schedule],
            [
                ("bg1", "fire"),
                ("bg1", "smoke"),
                ("bg2", "fire"),
                ("bg2", "smoke"),
                ("bg3", "fire"),
                ("bg3", "smoke"),
            ],
        )

    def test_validate_synth_manifest_success_and_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "02_processed" / "synth" / "v1" / "images" / "sample.jpg"
            label = root / "02_processed" / "synth" / "v1" / "labels" / "sample.txt"
            manifest = root / "02_processed" / "manifests" / "synth_v1_manifest.csv"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(PNG_1X1)
            label.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
            row = synth_manifest_row(image, label, root, class_name="fire", class_id="0")
            write_csv(manifest, SYNTH_MANIFEST_HEADER, [row])

            self.assertEqual(validate_synth_manifest(manifest, root=root).errors, [])

            label.write_text("1 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
            result = validate_synth_manifest(manifest, root=root)

        self.assertTrue(any("label class id does not match" in error for error in result.errors))

    def test_summary_writes_csv_and_paper_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "02_processed" / "synth" / "v1" / "images" / "sample.jpg"
            label = root / "02_processed" / "synth" / "v1" / "labels" / "sample.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(PNG_1X1)
            label.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
            manifest = root / "02_processed" / "manifests" / "synth_v1_manifest.csv"
            write_csv(manifest, SYNTH_MANIFEST_HEADER, [synth_manifest_row(image, label, root)])
            out_dir = root / "04_papers" / "seminar" / "tables"

            csv_path, txt_path, row_count = run_summary(manifest=manifest, out_dir=out_dir)
            txt_content = txt_path.read_text(encoding="utf-8")
            csv_exists = csv_path.is_file()

        self.assertEqual(row_count, 1)
        self.assertEqual(csv_path.name, SUMMARY_CSV)
        self.assertEqual(txt_path.name, SUMMARY_TXT)
        self.assertIn("synthetic positive proxies", txt_content)
        self.assertTrue(csv_exists)

    @skipUnless(importlib.util.find_spec("PIL") is not None, "Pillow is not installed")
    def test_build_synth_v1_writes_manifest_labels_and_previews(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            background = root / "02_processed" / "frames" / "ship_like" / "bg.jpg"
            fire_crop = root / "02_processed" / "crops" / "fire.png"
            smoke_crop = root / "02_processed" / "crops" / "smoke.png"
            background.parent.mkdir(parents=True)
            fire_crop.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), (20, 30, 40)).save(background)
            Image.new("RGB", (14, 12), (255, 40, 20)).save(fire_crop)
            Image.new("RGB", (12, 14), (160, 160, 160)).save(smoke_crop)

            background_manifest = root / "02_processed" / "manifests" / "ship_like_background_train.csv"
            donor_manifest = root / "02_processed" / "manifests" / "donor_crop_manifest_v1.csv"
            output_manifest = root / "02_processed" / "manifests" / "synth_v1_manifest.csv"
            config_path = root / "configs" / "synth_v1.json"
            write_csv(
                background_manifest,
                BACKGROUND_HEADER,
                [background_row("bg1", "02_processed/frames/ship_like/bg.jpg")],
            )
            write_csv(
                donor_manifest,
                DONOR_HEADER,
                [
                    donor_row("fire1", "02_processed/crops/fire.png", "fire", "DFire"),
                    donor_row("smoke1", "02_processed/crops/smoke.png", "smoke", "CQU_FireSmokeYOLO"),
                ],
            )
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "background_manifest": str(background_manifest),
                        "donor_manifest": str(donor_manifest),
                        "output_dir": str(root / "02_processed" / "synth" / "v1"),
                        "output_manifest": str(output_manifest),
                        "max_samples_total": 2,
                        "preview_count": 2,
                        "seed": 7,
                        "class_ids": {"fire": 0, "smoke": 1},
                        "min_scale": 0.2,
                        "max_scale": 0.2,
                        "opacity_min": 0.8,
                        "opacity_max": 0.8,
                        "blur_radius_min": 0.0,
                        "blur_radius_max": 0.0,
                        "avoid_edge_margin_ratio": 0.05,
                        "min_bbox_width_px": 4,
                        "min_bbox_height_px": 4,
                    }
                ),
                encoding="utf-8",
            )

            summary = run_build_synth_v1(config_path=config_path, overwrite=True, root=root)
            rows = read_csv(output_manifest)
            result = validate_synth_manifest(output_manifest, root=root)
            preview_exists = (
                root / "02_processed" / "synth" / "v1" / "previews" / "synth_v1_000001_fire_preview.jpg"
            ).is_file()

        self.assertEqual(summary.samples_written, 2)
        self.assertEqual([row["class_name"] for row in rows], ["fire", "smoke"])
        self.assertEqual(result.errors, [])
        self.assertTrue(preview_exists)


def background_row_obj(frame_id: str):
    from scripts.build_synth_v1 import BackgroundFrame

    return BackgroundFrame(
        frame_id=frame_id,
        asset_id="asset",
        frame_path="02_processed/frames/bg.jpg",
        zone_l1="engine_room",
    )


def synth_manifest_row(
    image: Path,
    label: Path,
    root: Path,
    *,
    class_name: str = "fire",
    class_id: str = "0",
) -> dict[str, str]:
    return {
        "synth_id": "synth_v1_000001_fire",
        "image_path": image.relative_to(root).as_posix(),
        "label_path": label.relative_to(root).as_posix(),
        "background_frame_id": "bg1",
        "background_asset_id": "SHIP_TEST_001",
        "background_frame_path": "02_processed/frames/bg.jpg",
        "background_zone_l1": "engine_room",
        "donor_crop_id": "crop1",
        "donor_source_name": "DFire",
        "donor_crop_path": "02_processed/crops/crop.png",
        "class_name": class_name,
        "class_id": class_id,
        "bbox_xywh_norm": "[0.5,0.5,0.25,0.25]",
        "generation_seed": "1",
        "image_width": "1",
        "image_height": "1",
        "sha256": hash_file(image),
        "quality_flag": "synthetic_proxy",
        "notes": "public_data_only; synthetic_positive_proxy; not_real_shipboard_positive_label",
    }

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fast_detector_dataset import parse_args as parse_dataset_args  # noqa: E402
from scripts.build_fast_detector_dataset import resolve_dataset_root  # noqa: E402
from scripts.build_fast_detector_dataset import run_build_fast_detector_dataset  # noqa: E402
from scripts.build_runtime_mp4_inputs import parse_args as parse_runtime_inputs_args  # noqa: E402
from scripts.summarize_fast_nvr_run import run_summary as run_runtime_summary  # noqa: E402
from scripts.summarize_yolo_predictions import summarize_hard_negative, summarize_synthetic_proxy  # noqa: E402


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

SYNTH_HEADER = [
    "synth_id",
    "image_path",
    "label_path",
    "background_frame_id",
    "background_asset_id",
    "background_frame_path",
    "background_zone_l1",
    "donor_crop_id",
    "donor_source_name",
    "donor_crop_path",
    "class_name",
    "class_id",
    "bbox_xywh_norm",
    "generation_seed",
    "image_width",
    "image_height",
    "sha256",
    "quality_flag",
    "notes",
]

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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")


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
        "sha256": "0" * 64,
        "notes": "test",
    }


def synth_row(synth_id: str, image_path: str, label_path: str, class_name: str, class_id: int) -> dict[str, str]:
    return {
        "synth_id": synth_id,
        "image_path": image_path,
        "label_path": label_path,
        "background_frame_id": "bg1",
        "background_asset_id": "SHIP_DVIDS_001",
        "background_frame_path": "02_processed/frames/bg.jpg",
        "background_zone_l1": "engine_room",
        "donor_crop_id": "crop1",
        "donor_source_name": "DFire",
        "donor_crop_path": "02_processed/crops/crop.png",
        "class_name": class_name,
        "class_id": str(class_id),
        "bbox_xywh_norm": "[0.5,0.5,0.25,0.25]",
        "generation_seed": "1",
        "image_width": "10",
        "image_height": "10",
        "sha256": "1" * 64,
        "quality_flag": "synthetic_proxy",
        "notes": "public_data_only",
    }


def background_row(frame_id: str, frame_path: str, split: str, quality: str = "good") -> dict[str, str]:
    return {
        "frame_id": frame_id,
        "asset_id": "SHIP_TEST_001",
        "source_name": "DVIDS",
        "frame_path": frame_path,
        "timestamp_s": "0",
        "frame_index": "0",
        "width": "10",
        "height": "10",
        "zone_l1": "engine_room",
        "confuser_tags": "none",
        "use_purpose": "background",
        "split": split,
        "quality_flag": quality,
        "reviewer_notes": "",
        "sha256": "2" * 64,
    }


class FastDetectorDatasetToolsTest(TestCase):
    def test_detector_dataset_cli_aliases(self) -> None:
        args = parse_dataset_args(
            [
                "--donor-crop-manifest",
                "manifests/donors.csv",
                "--out-dir",
                "03_experiments/detector_datasets",
                "--dataset-name",
                "demo",
            ]
        )

        self.assertEqual(args.donor_manifest, Path("manifests/donors.csv"))
        self.assertEqual(args.output_dir, Path("03_experiments/detector_datasets"))
        self.assertEqual(args.dataset_name, "demo")

    def test_detector_dataset_root_accepts_parent_or_final_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_name = "fire_smoke_fast_v1"
            parent_output_dir = root / "03_experiments" / "detector_datasets"
            final_output_dir = parent_output_dir / dataset_name

            from_parent = resolve_dataset_root(parent_output_dir, dataset_name)
            from_final = resolve_dataset_root(final_output_dir, dataset_name)

        self.assertEqual(from_parent, from_final)
        self.assertEqual(from_parent.name, dataset_name)
        self.assertNotEqual(from_final, final_output_dir / dataset_name)

    def test_runtime_mp4_cli_alias(self) -> None:
        args = parse_runtime_inputs_args(["--out-dir", "03_experiments/runtime_inputs/demo"])

        self.assertEqual(args.output_dir, Path("03_experiments/runtime_inputs/demo"))

    def test_build_fast_detector_dataset_writes_yolo_outputs_and_eval_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifests = root / "02_processed" / "manifests"

            donor_rows = []
            for class_name in ["fire", "smoke"]:
                for index in range(2):
                    crop_path = f"02_processed/crops/{class_name}_{index}.jpg"
                    touch_image(root / crop_path)
                    donor_rows.append(donor_row(f"{class_name}_{index}", crop_path, class_name))
            weird_donor_header = ["\ufeffcrop_id", " source_name ", *DONOR_HEADER[2:]]
            weird_donor_rows = []
            for row in donor_rows:
                weird = dict(row)
                weird["\ufeffcrop_id"] = weird.pop("crop_id")
                weird[" source_name "] = weird.pop("source_name")
                weird_donor_rows.append(weird)
            donor_manifest = manifests / "donor_crop_manifest_v1.csv"
            write_csv(donor_manifest, weird_donor_header, weird_donor_rows)

            synth_rows = []
            for class_name, class_id in [("fire", 0), ("smoke", 1)]:
                for index in range(2):
                    image_path = f"02_processed/synth/v1/images/{class_name}_{index}.jpg"
                    label_path = f"02_processed/synth/v1/labels/{class_name}_{index}.txt"
                    touch_image(root / image_path)
                    (root / label_path).parent.mkdir(parents=True, exist_ok=True)
                    (root / label_path).write_text(f"{class_id} 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
                    synth_rows.append(synth_row(f"synth_{class_name}_{index}", image_path, label_path, class_name, class_id))
            synth_manifest = manifests / "synth_v1_manifest.csv"
            write_csv(synth_manifest, SYNTH_HEADER, synth_rows)

            train_bg_path = "02_processed/frames/train_bg.jpg"
            val_bg_path = "02_processed/frames/val_bg.jpg"
            hard_bg_path = "02_processed/frames/hard_bg.jpg"
            for image_path in [train_bg_path, val_bg_path, hard_bg_path]:
                touch_image(root / image_path)
            background_train = manifests / "ship_like_background_train.csv"
            write_csv(
                background_train,
                BACKGROUND_HEADER,
                [
                    background_row("train_keep", train_bg_path, "train"),
                    background_row("train_bad", train_bg_path, "train", "bad"),
                    background_row("train_wrong_split", train_bg_path, "val"),
                ],
            )
            background_val = manifests / "ship_like_background_val.csv"
            write_csv(background_val, BACKGROUND_HEADER, [background_row("val_keep", val_bg_path, "val")])
            hard_negative = manifests / "ship_like_hard_negative_eval.csv"
            write_csv(hard_negative, BACKGROUND_HEADER, [background_row("hard_keep", hard_bg_path, "test")])

            summary = run_build_fast_detector_dataset(
                donor_manifest=donor_manifest,
                synth_manifest=synth_manifest,
                background_train=background_train,
                background_val=background_val,
                hard_negative=hard_negative,
                output_dir=root / "03_experiments" / "detector_datasets",
                dataset_name="demo",
                max_donor_crops=4,
                seed=7,
                overwrite=True,
                root=root,
            )
            dataset_root = root / "03_experiments" / "detector_datasets" / "demo"
            donor_train_labels = list((dataset_root / "labels" / "train").glob("donor_train_fire*.txt"))
            hard_eval_rows = read_csv(dataset_root / "eval_hard_negative_manifest.csv")
            data_yaml_exists = (dataset_root / "data.yaml").is_file()
            paper_summary_exists = (dataset_root / "dataset_summary_for_paper.txt").is_file()
            donor_label_text = donor_train_labels[0].read_text(encoding="utf-8")

        self.assertEqual(summary.train_images, 5)
        self.assertEqual(summary.val_images, 5)
        self.assertEqual(summary.eval_synth_val_images, 2)
        self.assertEqual(summary.eval_hard_negative_images, 1)
        self.assertTrue(data_yaml_exists)
        self.assertTrue(paper_summary_exists)
        self.assertEqual(donor_label_text, "0 0.500000 0.500000 0.980000 0.980000\n")
        self.assertFalse(Path(hard_eval_rows[0]["image_path"]).is_absolute())
        self.assertNotIn("\\", hard_eval_rows[0]["image_path"])

    def test_prediction_summaries_write_synthetic_and_hard_negative_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            pred_dir = root / "pred"
            gt_dir = root / "gt"
            for directory in [image_dir, pred_dir, gt_dir]:
                directory.mkdir(parents=True)
            touch_image(image_dir / "fire.jpg")
            touch_image(image_dir / "smoke.jpg")
            (gt_dir / "fire.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (gt_dir / "smoke.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (pred_dir / "fire.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
            (pred_dir / "smoke.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

            synthetic_row = summarize_synthetic_proxy(
                image_dir=image_dir,
                pred_label_dir=pred_dir,
                gt_label_dir=gt_dir,
                out=root / "synthetic.csv",
                iou_threshold=0.5,
                conf_threshold=0.25,
            )
            hard_row = summarize_hard_negative(
                image_dir=image_dir,
                pred_label_dir=pred_dir,
                out=root / "hard.csv",
                conf_threshold=0.25,
            )

        self.assertEqual(synthetic_row["hit_rate"], "0.500000")
        self.assertEqual(synthetic_row["fire_hit_rate"], "1.000000")
        self.assertEqual(synthetic_row["smoke_hit_rate"], "0.000000")
        self.assertEqual(hard_row["false_alarm_frame_rate"], "1.000000")

    def test_runtime_summary_writes_table6_and_paper_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "03_experiments" / "runtime_logs" / "run"
            out_dir = root / "04_papers" / "seminar" / "tables"
            write_csv(
                run_dir / "metrics.csv",
                [
                    "config_path",
                    "model_path",
                    "channel_count",
                    "elapsed_wall_clock_s",
                    "frames_processed",
                    "analyzed_fps",
                    "predicted_frames",
                    "total_pred_boxes",
                    "alarm_count",
                    "sample_fps",
                    "conf",
                    "temporal_k",
                    "temporal_window_s",
                    "device",
                    "processing_mode",
                    "notes",
                ],
                [
                    {
                        "config_path": "configs/nvr_4stream_fast.json",
                        "model_path": "03_experiments/onnx/model.onnx",
                        "channel_count": "4",
                        "elapsed_wall_clock_s": "2.0",
                        "frames_processed": "40",
                        "analyzed_fps": "20.0",
                        "predicted_frames": "6",
                        "total_pred_boxes": "8",
                        "alarm_count": "1",
                        "sample_fps": "5.0",
                        "conf": "0.25",
                        "temporal_k": "3",
                        "temporal_window_s": "10.0",
                        "device": "cpu",
                        "processing_mode": "sequential_channel_emulated_4stream",
                        "notes": "public_mp4_emulator",
                    }
                ],
            )
            write_csv(
                run_dir / "channel_metrics.csv",
                [
                    "channel_id",
                    "display_name",
                    "expected_role",
                    "zone_l1",
                    "source_path",
                    "frames_processed",
                    "elapsed_wall_clock_s",
                    "analyzed_fps",
                    "predicted_frames",
                    "total_pred_boxes",
                    "alarm_count",
                    "first_alarm_timestamp_s",
                    "vid_stride",
                    "processing_mode",
                ],
                [
                    {
                        "channel_id": "ch01",
                        "display_name": "Synthetic positive proxy",
                        "expected_role": "synth_positive",
                        "zone_l1": "engine_room",
                        "source_path": "03_experiments/runtime_inputs/ch01.mp4",
                        "frames_processed": "10",
                        "elapsed_wall_clock_s": "0.5",
                        "analyzed_fps": "20.0",
                        "predicted_frames": "6",
                        "total_pred_boxes": "8",
                        "alarm_count": "1",
                        "first_alarm_timestamp_s": "0.4",
                        "vid_stride": "1",
                        "processing_mode": "sequential_channel_emulated_4stream",
                    }
                ],
            )

            table_path, txt_path, row_count = run_runtime_summary(run_dir=run_dir, out_dir=out_dir)
            table_exists = table_path.is_file()
            txt_exists = txt_path.is_file()
            txt_content = txt_path.read_text(encoding="utf-8")

        self.assertEqual(row_count, 2)
        self.assertTrue(table_exists)
        self.assertTrue(txt_exists)
        self.assertIn("not real CCTV", txt_content)

"""Summarize YOLO label predictions for preliminary detector tables."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_CLASS_IDS = {0: "fire", 1: "smoke"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def image_files(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_yolo_label_file(path: Path, *, prediction: bool, conf_threshold: float = 0.0) -> list[YoloBox]:
    if not path.is_file():
        return []
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in {5, 6}:
            raise ValueError(f"{path}:{line_number}: expected YOLO class x y w h [conf]")
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
            confidence = float(parts[5]) if len(parts) == 6 else 1.0
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: nonnumeric YOLO value") from error
        if prediction and confidence < conf_threshold:
            continue
        boxes.append(YoloBox(class_id, x_center, y_center, width, height, confidence))
    return boxes


def yolo_xywh_to_xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    half_width = box.width / 2.0
    half_height = box.height / 2.0
    return (
        box.x_center - half_width,
        box.y_center - half_height,
        box.x_center + half_width,
        box.y_center + half_height,
    )


def box_iou(left: YoloBox, right: YoloBox) -> float:
    left_x1, left_y1, left_x2, left_y2 = yolo_xywh_to_xyxy(left)
    right_x1, right_y1, right_x2, right_y2 = yolo_xywh_to_xyxy(right)
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def summarize_hard_negative(
    *,
    image_dir: Path,
    pred_label_dir: Path,
    out: Path,
    conf_threshold: float,
) -> dict[str, str]:
    if not pred_label_dir.is_dir():
        raise FileNotFoundError(f"prediction label directory does not exist: {pred_label_dir}")
    images = image_files(image_dir)
    frames_with_prediction = 0
    total_pred_boxes = 0
    for image_path in images:
        pred_path = pred_label_dir / f"{image_path.stem}.txt"
        pred_boxes = parse_yolo_label_file(pred_path, prediction=True, conf_threshold=conf_threshold)
        if pred_boxes:
            frames_with_prediction += 1
            total_pred_boxes += len(pred_boxes)

    row = {
        "task": "hard_negative",
        "total_images": str(len(images)),
        "frames_with_prediction": str(frames_with_prediction),
        "total_pred_boxes": str(total_pred_boxes),
        "false_alarm_frame_rate": format_rate(frames_with_prediction, len(images)),
        "conf_threshold": f"{conf_threshold:.6f}",
    }
    write_csv(
        out,
        [
            "task",
            "total_images",
            "frames_with_prediction",
            "total_pred_boxes",
            "false_alarm_frame_rate",
            "conf_threshold",
        ],
        [row],
    )
    return row


def image_class_hit(
    *,
    gt_boxes: list[YoloBox],
    pred_boxes: list[YoloBox],
    class_id: int,
    iou_threshold: float,
) -> bool:
    class_gt = [box for box in gt_boxes if box.class_id == class_id]
    class_pred = [box for box in pred_boxes if box.class_id == class_id]
    return any(box_iou(pred_box, gt_box) >= iou_threshold for pred_box in class_pred for gt_box in class_gt)


def summarize_synthetic_proxy(
    *,
    image_dir: Path,
    pred_label_dir: Path,
    gt_label_dir: Path,
    out: Path,
    iou_threshold: float,
    conf_threshold: float,
) -> dict[str, str]:
    if not pred_label_dir.is_dir():
        raise FileNotFoundError(f"prediction label directory does not exist: {pred_label_dir}")
    if not gt_label_dir.is_dir():
        raise FileNotFoundError(f"ground-truth label directory does not exist: {gt_label_dir}")

    images = image_files(image_dir)
    hit_images = 0
    class_totals = {0: 0, 1: 0}
    class_hits = {0: 0, 1: 0}

    for image_path in images:
        gt_path = gt_label_dir / f"{image_path.stem}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"ground-truth label missing for image: {image_path}")
        pred_path = pred_label_dir / f"{image_path.stem}.txt"
        gt_boxes = parse_yolo_label_file(gt_path, prediction=False)
        pred_boxes = parse_yolo_label_file(pred_path, prediction=True, conf_threshold=conf_threshold)

        image_hit = False
        for class_id in TARGET_CLASS_IDS:
            if any(box.class_id == class_id for box in gt_boxes):
                class_totals[class_id] += 1
                if image_class_hit(
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    class_id=class_id,
                    iou_threshold=iou_threshold,
                ):
                    class_hits[class_id] += 1
                    image_hit = True
        if image_hit:
            hit_images += 1

    row = {
        "task": "synthetic_proxy",
        "total_images": str(len(images)),
        "hit_images": str(hit_images),
        "hit_rate": format_rate(hit_images, len(images)),
        "fire_images": str(class_totals[0]),
        "fire_hit_images": str(class_hits[0]),
        "fire_hit_rate": format_rate(class_hits[0], class_totals[0]),
        "smoke_images": str(class_totals[1]),
        "smoke_hit_images": str(class_hits[1]),
        "smoke_hit_rate": format_rate(class_hits[1], class_totals[1]),
        "iou_threshold": f"{iou_threshold:.6f}",
        "conf_threshold": f"{conf_threshold:.6f}",
    }
    write_csv(
        out,
        [
            "task",
            "total_images",
            "hit_images",
            "hit_rate",
            "fire_images",
            "fire_hit_images",
            "fire_hit_rate",
            "smoke_images",
            "smoke_hit_images",
            "smoke_hit_rate",
            "iou_threshold",
            "conf_threshold",
        ],
        [row],
    )
    return row


def read_single_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"summary CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        reader.fieldnames = [(fieldname or "").strip().lstrip("\ufeff").strip() for fieldname in reader.fieldnames or []]
        rows = list(reader)
    if not rows:
        raise ValueError(f"summary CSV has no data rows: {path}")
    return rows[0]


def summarize_combined(*, out: Path) -> None:
    synthetic_path = out.parent / "table5_synthetic_proxy_eval.csv"
    hard_negative_path = out.parent / "table5_hard_negative_eval.csv"
    synthetic = read_single_row(synthetic_path)
    hard_negative = read_single_row(hard_negative_path)
    rows = [
        {
            "task": "synthetic_proxy",
            "primary_metric": "hit_rate",
            "value": synthetic.get("hit_rate", ""),
            "total_images": synthetic.get("total_images", ""),
            "notes": "held_out_synthetic_proxy",
        },
        {
            "task": "hard_negative",
            "primary_metric": "false_alarm_frame_rate",
            "value": hard_negative.get("false_alarm_frame_rate", ""),
            "total_images": hard_negative.get("total_images", ""),
            "notes": "public_ship_like_hard_negative",
        },
    ]
    write_csv(out, ["task", "primary_metric", "value", "total_images", "notes"], rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["hard_negative", "synthetic_proxy", "combined"],
        required=True,
        help="Summary task to run.",
    )
    parser.add_argument("--image-dir", type=Path, help="Image directory to evaluate.")
    parser.add_argument("--pred-label-dir", type=Path, help="YOLO prediction label directory.")
    parser.add_argument("--gt-label-dir", type=Path, help="YOLO ground-truth label directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for synthetic proxy hits.")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Prediction confidence threshold.")
    return parser.parse_args(argv)


def require_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if value is None:
        option_name = name.replace("_", "-")
        raise ValueError(f"--{option_name} is required for task {args.task}")
    return resolve_cli_path(value, ROOT)


def main() -> int:
    args = parse_args()
    try:
        out = resolve_cli_path(args.out, ROOT)
        if args.task == "hard_negative":
            row = summarize_hard_negative(
                image_dir=require_path(args, "image_dir"),
                pred_label_dir=require_path(args, "pred_label_dir"),
                out=out,
                conf_threshold=args.conf_threshold,
            )
            print(f"PASS hard-negative images summarized: {row['total_images']}")
            print(f"PASS false alarm frame rate: {row['false_alarm_frame_rate']}")
        elif args.task == "synthetic_proxy":
            row = summarize_synthetic_proxy(
                image_dir=require_path(args, "image_dir"),
                pred_label_dir=require_path(args, "pred_label_dir"),
                gt_label_dir=require_path(args, "gt_label_dir"),
                out=out,
                iou_threshold=args.iou_threshold,
                conf_threshold=args.conf_threshold,
            )
            print(f"PASS synthetic proxy images summarized: {row['total_images']}")
            print(f"PASS hit rate: {row['hit_rate']}")
        elif args.task == "combined":
            summarize_combined(out=out)
            print(f"PASS combined preliminary detection table: {out}")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

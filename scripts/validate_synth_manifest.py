"""Validate the synthetic v1 manifest and referenced YOLO labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_synth_v1 import (  # noqa: E402
    CSV_MANIFEST_ENCODING,
    DEFAULT_CLASS_IDS,
    SYNTH_MANIFEST_HEADER,
    format_missing_columns_error,
    hash_file,
    normalize_csv_reader_fieldnames,
    normalize_repo_relative,
    resolve_cli_path,
)


@dataclass
class SynthManifestValidation:
    rows: int = 0
    errors: list[str] = field(default_factory=list)
    class_counts: Counter[str] = field(default_factory=Counter)
    background_zone_counts: Counter[str] = field(default_factory=Counter)
    background_asset_counts: Counter[str] = field(default_factory=Counter)
    donor_source_counts: Counter[str] = field(default_factory=Counter)


def resolve_manifest_path(value: str, root: Path, *, field_name: str) -> Path | None:
    if not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    try:
        return root / normalize_repo_relative(value, field_name=field_name)
    except ValueError:
        return None


def parse_bbox(value: str, *, row_number: int, result: SynthManifestValidation) -> tuple[float, float, float, float] | None:
    stripped = value.strip()
    parsed: object
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            result.errors.append(f"row {row_number}: bbox_xywh_norm is not valid JSON")
            return None
    else:
        parsed = stripped.split()

    if not isinstance(parsed, list) or len(parsed) != 4:
        result.errors.append(f"row {row_number}: bbox_xywh_norm must contain four values")
        return None
    try:
        values = tuple(float(item) for item in parsed)
    except (TypeError, ValueError):
        result.errors.append(f"row {row_number}: bbox_xywh_norm contains nonnumeric values")
        return None
    if any(not math.isfinite(value) for value in values):
        result.errors.append(f"row {row_number}: bbox_xywh_norm contains nonfinite values")
        return None
    x_center, y_center, width, height = values
    if not all(0.0 <= value <= 1.0 for value in values):
        result.errors.append(f"row {row_number}: bbox_xywh_norm values must be within [0,1]")
    if width <= 0.0 or height <= 0.0:
        result.errors.append(f"row {row_number}: bbox width and height must be greater than zero")
    if x_center - width / 2.0 < -1e-6 or x_center + width / 2.0 > 1.0 + 1e-6:
        result.errors.append(f"row {row_number}: bbox x extent falls outside the image")
    if y_center - height / 2.0 < -1e-6 or y_center + height / 2.0 > 1.0 + 1e-6:
        result.errors.append(f"row {row_number}: bbox y extent falls outside the image")
    return values


def parse_label_file(
    path: Path,
    *,
    row_number: int,
    result: SynthManifestValidation,
) -> tuple[int, tuple[float, float, float, float]] | None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        result.errors.append(f"row {row_number}: label file must contain exactly one YOLO bbox: {path}")
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        result.errors.append(f"row {row_number}: label file must contain class_id plus four bbox values: {path}")
        return None
    try:
        class_id = int(parts[0])
        bbox = tuple(float(value) for value in parts[1:5])
    except ValueError:
        result.errors.append(f"row {row_number}: label file contains nonnumeric YOLO values: {path}")
        return None
    if class_id not in DEFAULT_CLASS_IDS.values():
        result.errors.append(f"row {row_number}: label class id must be 0 or 1: {path}")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in bbox):
        result.errors.append(f"row {row_number}: label bbox values must be finite and within [0,1]: {path}")
    return class_id, bbox


def validate_synth_manifest(path: Path, *, root: Path = ROOT) -> SynthManifestValidation:
    result = SynthManifestValidation()
    if not path.is_file():
        result.errors.append(f"manifest does not exist: {path}")
        return result

    with path.open("r", newline="", encoding=CSV_MANIFEST_ENCODING) as csv_file:
        reader = csv.DictReader(csv_file)
        raw_fieldnames, normalized_fieldnames = normalize_csv_reader_fieldnames(reader)
        missing = [column for column in SYNTH_MANIFEST_HEADER if column not in normalized_fieldnames]
        if missing:
            result.errors.append(
                format_missing_columns_error(
                    "manifest",
                    missing=missing,
                    required_columns=SYNTH_MANIFEST_HEADER,
                    normalized_actual_columns=normalized_fieldnames,
                    raw_actual_columns=raw_fieldnames,
                )
            )
            return result
        rows = list(reader)

    seen_synth_ids: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        result.rows += 1
        synth_id = row.get("synth_id", "").strip()
        if not synth_id:
            result.errors.append(f"row {row_number}: synth_id is required")
        elif synth_id in seen_synth_ids:
            result.errors.append(f"row {row_number}: duplicate synth_id also seen at row {seen_synth_ids[synth_id]}")
        else:
            seen_synth_ids[synth_id] = row_number

        class_name = row.get("class_name", "").strip().lower()
        if class_name not in DEFAULT_CLASS_IDS:
            result.errors.append(f"row {row_number}: class_name must be fire or smoke: {class_name!r}")
        else:
            result.class_counts[class_name] += 1

        try:
            class_id = int(row.get("class_id", "").strip())
        except ValueError:
            result.errors.append(f"row {row_number}: class_id must be an integer")
            class_id = -1
        if class_id not in DEFAULT_CLASS_IDS.values():
            result.errors.append(f"row {row_number}: class_id must be 0 or 1")
        if class_name in DEFAULT_CLASS_IDS and class_id != DEFAULT_CLASS_IDS[class_name]:
            result.errors.append(f"row {row_number}: class_id {class_id} does not match class_name {class_name}")

        bbox = parse_bbox(row.get("bbox_xywh_norm", ""), row_number=row_number, result=result)

        image_path = resolve_manifest_path(row.get("image_path", ""), root, field_name="image_path")
        label_path = resolve_manifest_path(row.get("label_path", ""), root, field_name="label_path")
        if image_path is None:
            result.errors.append(f"row {row_number}: image_path is invalid")
        elif not image_path.is_file():
            result.errors.append(f"row {row_number}: image_path does not exist: {row.get('image_path', '')}")
        if label_path is None:
            result.errors.append(f"row {row_number}: label_path is invalid")
        elif not label_path.is_file():
            result.errors.append(f"row {row_number}: label_path does not exist: {row.get('label_path', '')}")

        sha256 = row.get("sha256", "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            result.errors.append(f"row {row_number}: sha256 must be 64 lowercase hex characters")
        elif image_path and image_path.is_file() and hash_file(image_path) != sha256:
            result.errors.append(f"row {row_number}: sha256 mismatch for {row.get('image_path', '')}")

        if label_path and label_path.is_file():
            label = parse_label_file(label_path, row_number=row_number, result=result)
            if label:
                label_class_id, label_bbox = label
                if label_class_id != class_id:
                    result.errors.append(f"row {row_number}: label class id does not match manifest class_id")
                if bbox and any(abs(label_value - manifest_value) > 1e-6 for label_value, manifest_value in zip(label_bbox, bbox)):
                    result.errors.append(f"row {row_number}: label bbox does not match manifest bbox_xywh_norm")

        result.background_zone_counts[row.get("background_zone_l1", "").strip() or "unknown"] += 1
        result.background_asset_counts[row.get("background_asset_id", "").strip() or "unknown"] += 1
        result.donor_source_counts[row.get("donor_source_name", "").strip() or "unknown"] += 1

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "synth_v1_manifest.csv",
        help="Synthetic v1 manifest CSV path.",
    )
    return parser.parse_args(argv)


def print_counts(name: str, counter: Counter[str]) -> None:
    print(f"{name}:")
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def main() -> int:
    args = parse_args()
    result = validate_synth_manifest(resolve_cli_path(args.manifest, ROOT), root=ROOT)
    if result.errors:
        print(f"FAIL synth manifest rows checked: {result.rows}")
        for error in result.errors:
            print(f"  {error}")
        if result.rows:
            print_counts("Counts by class_name", result.class_counts)
            print_counts("Counts by background_zone_l1", result.background_zone_counts)
            print_counts("Counts by background_asset_id", result.background_asset_counts)
            print_counts("Counts by donor_source_name", result.donor_source_counts)
        return 1

    print(f"PASS synth manifest rows checked: {result.rows}")
    print_counts("Counts by class_name", result.class_counts)
    print_counts("Counts by background_zone_l1", result.background_zone_counts)
    print_counts("Counts by background_asset_id", result.background_asset_counts)
    print_counts("Counts by donor_source_name", result.donor_source_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

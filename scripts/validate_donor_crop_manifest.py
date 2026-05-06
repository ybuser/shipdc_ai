"""Validate the synthetic v1 donor crop manifest and referenced crop files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_donor_crop_bank import (  # noqa: E402
    DONOR_CROP_MANIFEST_HEADER,
    TARGET_CLASS_NAMES,
    hash_file,
    normalize_repo_relative,
    resolve_cli_path,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass
class DonorCropManifestValidation:
    rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def read_image_dimensions(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as file_obj:
        header = file_obj.read(32)

    if header.startswith(PNG_SIGNATURE) and len(header) >= 24:
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    if header[:2] == b"\xff\xd8":
        return read_jpeg_dimensions(path)
    if header[:6] in {b"GIF87a", b"GIF89a"} and len(header) >= 10:
        width = int.from_bytes(header[6:8], "little")
        height = int.from_bytes(header[8:10], "little")
        return width, height
    if header[:2] == b"BM" and len(header) >= 26:
        width = int.from_bytes(header[18:22], "little", signed=True)
        height = abs(int.from_bytes(header[22:26], "little", signed=True))
        return width, height
    return None, None


def read_jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as file_obj:
        if file_obj.read(2) != b"\xff\xd8":
            return None, None
        while True:
            marker_start = file_obj.read(1)
            if not marker_start:
                return None, None
            if marker_start != b"\xff":
                continue

            marker = file_obj.read(1)
            while marker == b"\xff":
                marker = file_obj.read(1)
            if not marker:
                return None, None

            marker_code = marker[0]
            if marker_code == 0xD9 or 0xD0 <= marker_code <= 0xD7:
                continue

            length_bytes = file_obj.read(2)
            if len(length_bytes) != 2:
                return None, None
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                return None, None

            if marker_code in JPEG_SOF_MARKERS:
                segment = file_obj.read(5)
                if len(segment) != 5:
                    return None, None
                height = int.from_bytes(segment[1:3], "big")
                width = int.from_bytes(segment[3:5], "big")
                return width, height
            file_obj.seek(segment_length - 2, 1)


def parse_positive_int(value: str, *, field_name: str, row_number: int, result: DonorCropManifestValidation) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        result.errors.append(f"row {row_number}: {field_name} is not an integer: {value!r}")
        return None
    if parsed <= 0:
        result.errors.append(f"row {row_number}: {field_name} must be greater than zero")
        return None
    return parsed


def parse_bbox(value: str, *, row_number: int, result: DonorCropManifestValidation) -> tuple[int, int, int, int] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        result.errors.append(f"row {row_number}: bbox_xyxy is not JSON: {value!r}")
        return None
    if not isinstance(parsed, list) or len(parsed) != 4 or not all(isinstance(item, int) for item in parsed):
        result.errors.append(f"row {row_number}: bbox_xyxy must be a JSON list of four integers")
        return None
    x1, y1, x2, y2 = parsed
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        result.errors.append(f"row {row_number}: bbox_xyxy has invalid geometry: {value}")
        return None
    return x1, y1, x2, y2


def validate_repo_path(value: str, *, field_name: str, row_number: int, result: DonorCropManifestValidation) -> str | None:
    if "\\" in value:
        result.errors.append(f"row {row_number}: {field_name} must use forward slashes")
        return None
    try:
        return normalize_repo_relative(value, field_name=field_name)
    except ValueError as error:
        result.errors.append(f"row {row_number}: {error}")
        return None


def validate_donor_crop_manifest(path: Path, *, root: Path = ROOT) -> DonorCropManifestValidation:
    result = DonorCropManifestValidation()
    if not path.exists():
        result.errors.append(f"manifest does not exist: {path}")
        return result
    if not path.is_file():
        result.errors.append(f"manifest path is not a file: {path}")
        return result

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if fieldnames != DONOR_CROP_MANIFEST_HEADER:
            result.errors.append(f"header mismatch: found {fieldnames!r}")
            return result
        rows = list(reader)

    seen_crop_ids: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        result.rows += 1
        crop_id = row.get("crop_id", "").strip()
        if not crop_id:
            result.errors.append(f"row {row_number}: crop_id is required")
        elif crop_id in seen_crop_ids:
            result.errors.append(f"row {row_number}: duplicate crop_id also seen at row {seen_crop_ids[crop_id]}")
        else:
            seen_crop_ids[crop_id] = row_number

        class_name = row.get("class_name", "").strip()
        if class_name not in TARGET_CLASS_NAMES:
            result.errors.append(f"row {row_number}: class_name must be fire or smoke: {class_name!r}")

        source_image_rel = validate_repo_path(
            row.get("source_image_path", ""), field_name="source_image_path", row_number=row_number, result=result
        )
        source_label_rel = validate_repo_path(
            row.get("source_label_path", ""), field_name="source_label_path", row_number=row_number, result=result
        )
        crop_rel = validate_repo_path(row.get("crop_path", ""), field_name="crop_path", row_number=row_number, result=result)

        for field_name, relative_path in [
            ("source_image_path", source_image_rel),
            ("source_label_path", source_label_rel),
            ("crop_path", crop_rel),
        ]:
            if not relative_path:
                continue
            full_path = root / relative_path
            if not full_path.is_file():
                result.errors.append(f"row {row_number}: {field_name} does not exist: {relative_path}")

        crop_width = parse_positive_int(row.get("crop_width", ""), field_name="crop_width", row_number=row_number, result=result)
        crop_height = parse_positive_int(
            row.get("crop_height", ""), field_name="crop_height", row_number=row_number, result=result
        )
        bbox = parse_bbox(row.get("bbox_xyxy", ""), row_number=row_number, result=result)
        if bbox and crop_width and crop_height:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            if bbox_width != crop_width or bbox_height != crop_height:
                result.errors.append(
                    f"row {row_number}: crop dimensions {crop_width}x{crop_height} do not match bbox {bbox_width}x{bbox_height}"
                )

        sha256 = row.get("sha256", "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            result.errors.append(f"row {row_number}: sha256 must be 64 lowercase hex characters")
        elif crop_rel:
            crop_path = root / crop_rel
            if crop_path.is_file():
                actual_hash = hash_file(crop_path)
                if actual_hash != sha256:
                    result.errors.append(f"row {row_number}: sha256 mismatch for {crop_rel}")

                if crop_width and crop_height:
                    actual_width, actual_height = read_image_dimensions(crop_path)
                    if actual_width is None or actual_height is None:
                        result.warnings.append(f"row {row_number}: crop image dimensions could not be read")
                    elif (actual_width, actual_height) != (crop_width, crop_height):
                        result.errors.append(
                            f"row {row_number}: manifest dimensions {crop_width}x{crop_height} "
                            f"do not match crop file {actual_width}x{actual_height}"
                        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "donor_crop_manifest_v1.csv",
        help="Donor crop manifest CSV path.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = validate_donor_crop_manifest(resolve_cli_path(args.manifest, ROOT), root=ROOT)
    if result.errors:
        print(f"FAIL donor crop manifest rows checked: {result.rows}")
        for error in result.errors:
            print(f"  {error}")
        return 1
    print(f"PASS donor crop manifest rows checked: {result.rows}")
    for warning in result.warnings:
        print(f"WARN {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

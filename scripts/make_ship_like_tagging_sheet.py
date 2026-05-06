"""Create a reviewer tagging sheet for public ship-like keyframes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]

KEYFRAME_INPUT_HEADER = [
    "asset_id",
    "source_name",
    "source_asset_path",
    "frame_path",
    "timestamp_s",
    "frame_index",
    "width",
    "height",
    "extraction_fps",
    "sha256",
    "notes",
]

TAGGING_SHEET_HEADER = [
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

ALLOWED_ZONE_L1 = [
    "engine_room",
    "bridge_control",
    "control_room",
    "passageway_ladder",
    "service_maintenance_support",
    "exterior_or_unknown",
]

ALLOWED_USE_PURPOSE = [
    "background_train",
    "background_val",
    "hard_negative_eval",
    "qualitative_figure",
    "synth_background",
    "exclude",
]

ALLOWED_SPLIT = [
    "train",
    "val",
    "test",
    "qualitative",
    "exclude",
]

ALLOWED_QUALITY_FLAG = [
    "good",
    "low_light",
    "blurred",
    "duplicate",
    "occluded",
    "ambiguous",
    "exclude",
]

DEFAULT_LABELS = {
    "zone_l1": "exterior_or_unknown",
    "confuser_tags": "none",
    "use_purpose": "exclude",
    "split": "exclude",
    "quality_flag": "good",
    "reviewer_notes": "",
}

FRAME_ID_HASH_LENGTH = 16


@dataclass(frozen=True)
class TaggingSheetSummary:
    input_manifest: Path
    output_sheet: Path
    written_rows: int


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def normalize_frame_path(value: str) -> str:
    stripped = value.strip().replace("\\", "/")
    if not stripped:
        raise ValueError("empty frame_path")

    windows_path = PureWindowsPath(stripped)
    if windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"frame_path must be repository-relative: {value}")

    posix_path = PurePosixPath(stripped)
    if posix_path.is_absolute():
        raise ValueError(f"frame_path must be repository-relative: {value}")
    if any(part == ".." for part in posix_path.parts):
        raise ValueError(f"frame_path escapes repository root: {value}")

    return posix_path.as_posix()


def sanitize_frame_id_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return cleaned.upper() if cleaned else "SHIP_LIKE_FRAME"


def stable_frame_id(asset_id: str, frame_path: str) -> str:
    digest = hashlib.sha256(frame_path.encode("utf-8")).hexdigest()[:FRAME_ID_HASH_LENGTH]
    return f"{sanitize_frame_id_prefix(asset_id)}_{digest}"


def read_keyframe_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"input manifest does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input manifest path is not a file: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in KEYFRAME_INPUT_HEADER if column not in fieldnames]
        if missing:
            raise ValueError(f"missing required keyframe columns: {', '.join(missing)}")
        return list(reader)


def build_tagging_row(keyframe_row: dict[str, str]) -> dict[str, str]:
    frame_path = normalize_frame_path(keyframe_row.get("frame_path", ""))
    asset_id = keyframe_row.get("asset_id", "").strip()
    row = {
        "frame_id": stable_frame_id(asset_id, frame_path),
        "asset_id": asset_id,
        "source_name": keyframe_row.get("source_name", "").strip(),
        "frame_path": frame_path,
        "timestamp_s": keyframe_row.get("timestamp_s", "").strip(),
        "frame_index": keyframe_row.get("frame_index", "").strip(),
        "width": keyframe_row.get("width", "").strip(),
        "height": keyframe_row.get("height", "").strip(),
        "sha256": keyframe_row.get("sha256", "").strip(),
    }
    row.update(DEFAULT_LABELS)
    return {column: row.get(column, "") for column in TAGGING_SHEET_HEADER}


def build_tagging_rows(keyframe_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [build_tagging_row(row) for row in keyframe_rows]
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=2):
        frame_id = row["frame_id"]
        if frame_id in seen:
            raise ValueError(f"duplicate frame_id at output rows {seen[frame_id]} and {index}: {frame_id}")
        seen[frame_id] = index
    return rows


def write_tagging_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TAGGING_SHEET_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_tagging_sheet(input_manifest: Path, output_sheet: Path) -> TaggingSheetSummary:
    keyframe_rows = read_keyframe_rows(input_manifest)
    tagging_rows = build_tagging_rows(keyframe_rows)
    write_tagging_sheet(output_sheet, tagging_rows)
    return TaggingSheetSummary(
        input_manifest=input_manifest,
        output_sheet=output_sheet,
        written_rows=len(tagging_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_keyframes.csv",
        help="Input ship-like keyframe manifest CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_tagging_sheet.csv",
        help="Output reviewer tagging sheet CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_tagging_sheet(
            input_manifest=resolve_cli_path(args.input, ROOT),
            output_sheet=resolve_cli_path(args.output, ROOT),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS input manifest: {summary.input_manifest}")
    print(f"PASS output sheet: {summary.output_sheet}")
    print(f"PASS rows written: {summary.written_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

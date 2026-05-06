"""Validate the derived ship-like keyframe manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_keyframes import KEYFRAME_MANIFEST_HEADER, hash_file  # noqa: E402


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    asset_counts: Counter[str]
    source_counts: Counter[str]


def resolve_manifest_frame_path(root: Path, value: str) -> tuple[Path | None, str | None]:
    stripped = value.strip()
    if not stripped:
        return None, "empty frame_path"

    path = Path(stripped)
    if path.is_absolute():
        return None, f"frame_path must be repository-relative: {stripped}"

    resolved_root = root.resolve()
    resolved_path = (root / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return None, f"frame_path escapes repository root: {stripped}"
    return resolved_path, None


def validate_keyframe_manifest(path: Path, *, root: Path = ROOT) -> ValidationResult:
    errors: list[str] = []
    asset_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    if not path.exists():
        return ValidationResult([f"manifest does not exist: {path}"], asset_counts, source_counts)
    if not path.is_file():
        return ValidationResult([f"manifest path is not a file: {path}"], asset_counts, source_counts)

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in KEYFRAME_MANIFEST_HEADER if column not in fieldnames]
        if missing:
            errors.append(f"missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            asset_id = row.get("asset_id", "").strip()
            source_name = row.get("source_name", "").strip()
            if asset_id:
                asset_counts[asset_id] += 1
            if source_name:
                source_counts[source_name] += 1

            frame_path, path_error = resolve_manifest_frame_path(root, row.get("frame_path", ""))
            if path_error is not None:
                errors.append(f"line {line_number}: {path_error}")
                continue
            if frame_path is None:
                errors.append(f"line {line_number}: empty frame_path")
                continue
            if not frame_path.is_file():
                errors.append(f"line {line_number}: frame file does not exist: {row.get('frame_path', '')}")
                continue

            expected_hash = row.get("sha256", "").strip()
            if expected_hash:
                actual_hash = hash_file(frame_path)
                if actual_hash != expected_hash:
                    errors.append(
                        "line "
                        f"{line_number}: sha256 mismatch for {row.get('frame_path', '')}: "
                        f"expected {expected_hash}, found {actual_hash}"
                    )

    return ValidationResult(errors, asset_counts, source_counts)


def print_counts(label: str, counts: Counter[str]) -> None:
    print(label)
    if not counts:
        print("  none")
        return
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_keyframes.csv",
        help="Derived keyframe manifest CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    result = validate_keyframe_manifest(manifest_path, root=ROOT)
    if result.errors:
        print(f"FAIL keyframe manifest: {manifest_path}")
        for error in result.errors:
            print(f"  {error}")
    else:
        print(f"PASS keyframe manifest: {manifest_path}")

    print_counts("Counts by asset_id:", result.asset_counts)
    print_counts("Counts by source_name:", result.source_counts)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

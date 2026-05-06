"""Build a public fire/smoke donor crop bank for synthetic v1 data prep."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shipdc_ai.manifest import MANIFEST_HEADER, validate_manifest_header  # noqa: E402


DONOR_CROP_MANIFEST_HEADER = [
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

TARGET_CLASS_NAMES = {"fire", "smoke"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CHUNK_SIZE = 1024 * 1024
NON_PUBLIC_RIGHTS_MARKERS = {
    "classified",
    "restricted",
    "registration_required",
    "private_operational",
    "cctv",
    "rtsp",
}


class UnknownClassMappingError(ValueError):
    """Raised when a source has bbox labels but no verified fire/smoke class map."""


@dataclass(frozen=True)
class SourceSplit:
    source_name: str
    split: str
    images_dir: Path
    labels_dir: Path


@dataclass(frozen=True)
class YoloBox:
    class_id: str
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int
    extra_fields: bool = False


@dataclass(frozen=True)
class ConvertedBox:
    xyxy: tuple[int, int, int, int]
    width: int
    height: int
    notes: list[str]


@dataclass
class CropBuildSummary:
    output_manifest: Path
    output_images_dir: Path
    preview_dir: Path
    sources_seen: int = 0
    sources_built: int = 0
    labels_seen: int = 0
    images_seen: int = 0
    boxes_seen: int = 0
    crops_written: int = 0
    skipped_boxes: int = 0
    skipped_labels: int = 0
    skip_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CliOutputPaths:
    output_images_dir: Path | None
    preview_dir: Path | None
    output_manifest: Path | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def resolve_cli_output_paths(args: argparse.Namespace, root: Path) -> CliOutputPaths:
    output_images_dir = resolve_cli_path(args.output_images, root) if args.output_images else None
    preview_dir = resolve_cli_path(args.preview_dir, root) if args.preview_dir else None
    if args.out_dir:
        out_dir = resolve_cli_path(args.out_dir, root)
        if output_images_dir is None:
            output_images_dir = out_dir / "images"
        if preview_dir is None:
            preview_dir = out_dir / "preview"
    output_manifest = resolve_cli_path(args.output_manifest, root) if args.output_manifest else None
    return CliOutputPaths(
        output_images_dir=output_images_dir,
        preview_dir=preview_dir,
        output_manifest=output_manifest,
    )


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a JSON object: {path}")
    return config


def normalize_repo_relative(value: str, *, field_name: str = "path") -> str:
    stripped = value.strip().replace("\\", "/")
    if not stripped:
        raise ValueError(f"{field_name} is empty")

    windows_path = PureWindowsPath(stripped)
    if windows_path.drive or windows_path.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative: {value}")

    posix_path = PurePosixPath(stripped)
    if posix_path.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative: {value}")
    if any(part == ".." for part in posix_path.parts):
        raise ValueError(f"{field_name} escapes repository root: {value}")
    return posix_path.as_posix()


def path_from_config(config: dict[str, Any], key: str, root: Path) -> Path:
    value = config.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config key is required: {key}")
    return root / normalize_repo_relative(value, field_name=key)


def repo_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository root: {path}") from error


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def normalize_class_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in TARGET_CLASS_NAMES:
        raise ValueError(f"unknown normalized class name: {value!r}; expected fire or smoke")
    return normalized


def normalize_class_map(raw_map: Any, source_name: str) -> dict[str, str]:
    if not isinstance(raw_map, dict) or not raw_map:
        raise UnknownClassMappingError(
            f"class mapping unknown for {source_name}; set class_map with YOLO ids mapped to fire/smoke"
        )

    normalized: dict[str, str] = {}
    for raw_key, raw_value in raw_map.items():
        key_text = str(raw_key).strip()
        if not re.fullmatch(r"\d+", key_text):
            raise ValueError(f"class_map key for {source_name} must be an integer id: {raw_key!r}")
        normalized[key_text] = normalize_class_name(str(raw_value))
    return normalized


def parse_yaml_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        if stripped.startswith("[") and stripped.endswith("]"):
            items = stripped[1:-1].split(",")
            return [item.strip().strip("\"'") for item in items if item.strip()]
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed_dict: dict[str, str] = {}
            for item in stripped[1:-1].split(","):
                if ":" not in item:
                    return stripped.strip("\"'")
                key, raw_value = item.split(":", 1)
                parsed_dict[key.strip().strip("\"'")] = raw_value.strip().strip("\"'")
            return parsed_dict
        return stripped.strip("\"'")


def read_yolo_names_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("names:"):
            continue

        _, remainder = line.split(":", 1)
        parsed = parse_yaml_scalar(remainder)
        if isinstance(parsed, list):
            return {str(class_id): str(name) for class_id, name in enumerate(parsed)}
        if isinstance(parsed, dict):
            return {str(class_id): str(name) for class_id, name in parsed.items()}
        if parsed:
            return {}

        names: dict[str, str] = {}
        next_index = 0
        for child_line in lines[index + 1 :]:
            if child_line and not child_line[0].isspace():
                break
            match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", child_line)
            if match:
                names[match.group(1)] = match.group(2).strip("\"'")
                continue
            list_match = re.match(r"\s*-\s*(.+?)\s*$", child_line)
            if list_match:
                names[str(next_index)] = list_match.group(1).strip("\"'")
                next_index += 1
        return names
    return {}


def class_map_from_source_config(source_config: dict[str, Any], source_root: Path) -> dict[str, str]:
    source_name = str(source_config.get("source_name", "source_unknown"))
    raw_map = source_config.get("class_map", {})
    if raw_map:
        return normalize_class_map(raw_map, source_name)

    metadata_file = source_config.get("metadata_names_file", "")
    if isinstance(metadata_file, str) and metadata_file.strip():
        metadata_path = source_root / normalize_repo_relative(metadata_file, field_name="metadata_names_file")
        names = read_yolo_names_file(metadata_path)
        if names:
            try:
                return normalize_class_map(names, source_name)
            except ValueError as error:
                raise UnknownClassMappingError(
                    f"class mapping unknown for {source_name}; metadata names in {metadata_path} "
                    f"are {names!r}, not fire/smoke"
                ) from error

    raise UnknownClassMappingError(
        f"class mapping unknown for {source_name}; set class_map with YOLO ids mapped to fire/smoke"
    )


def read_master_manifest(path: Path) -> list[dict[str, str]]:
    errors = validate_manifest_header(path)
    if errors:
        raise ValueError("; ".join(errors))
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def manifest_rows_for_source(rows: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("source_group", "").strip().lower() == "donor"
        and row.get("source_name", "").strip() == source_name
    ]


def assert_public_donor_source(master_rows: list[dict[str, str]], source_name: str) -> None:
    rows = manifest_rows_for_source(master_rows, source_name)
    if not rows:
        raise ValueError(f"source {source_name} has no donor rows in the master manifest")

    for row in rows:
        rights = row.get("license_or_rights", "").strip().lower()
        notes = row.get("notes", "").strip().lower()
        combined = f"{rights} {notes}"
        if any(marker in combined for marker in NON_PUBLIC_RIGHTS_MARKERS):
            raise ValueError(
                f"source {source_name} is not allowed for donor crops because manifest row "
                f"{row.get('asset_id', '')} is not public-data-only"
            )


def source_root_from_config(source_config: dict[str, Any], root: Path) -> Path:
    root_value = source_config.get("root", "")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError(f"source root is required for {source_config.get('source_name', '')}")
    return root / normalize_repo_relative(root_value, field_name="source root")


def source_splits(source_config: dict[str, Any], root: Path) -> list[SourceSplit]:
    source_name = str(source_config.get("source_name", "source_unknown"))
    source_root = source_root_from_config(source_config, root)
    splits = source_config.get("splits", [])
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"source {source_name} must define one or more splits")

    image_dir_name = str(source_config.get("image_dir_name", "images"))
    label_dir_name = str(source_config.get("label_dir_name", "labels"))
    discovered: list[SourceSplit] = []
    for split in splits:
        split_name = str(split)
        images_dir = source_root / split_name / image_dir_name
        labels_dir = source_root / split_name / label_dir_name
        if images_dir.is_dir() and labels_dir.is_dir():
            discovered.append(
                SourceSplit(
                    source_name=source_name,
                    split=split_name,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                )
            )
    return discovered


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def count_labels(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".txt")


def build_image_index(images_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for image_path in images_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = image_path.stem
        if stem in indexed:
            ambiguous.add(stem)
            continue
        indexed[stem] = image_path
    if ambiguous:
        examples = ", ".join(sorted(ambiguous)[:5])
        raise ValueError(f"ambiguous image stems in {images_dir}: {examples}")
    return indexed


def iter_label_files(labels_dir: Path) -> list[Path]:
    return sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def parse_yolo_label_line(line: str, label_path: Path, line_number: int) -> tuple[YoloBox | None, str | None]:
    stripped = line.strip()
    if not stripped:
        return None, None

    parts = stripped.split()
    if len(parts) < 5:
        return None, f"{label_path}:{line_number}: invalid_yolo_field_count"
    class_id = parts[0]
    if not re.fullmatch(r"\d+", class_id):
        return None, f"{label_path}:{line_number}: invalid_class_id"
    try:
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
    except ValueError:
        return None, f"{label_path}:{line_number}: nonnumeric_yolo_bbox"

    return (
        YoloBox(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            line_number=line_number,
            extra_fields=len(parts) > 5,
        ),
        None,
    )


def parse_yolo_label_file(label_path: Path) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    notes: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        box, note = parse_yolo_label_line(line, label_path, line_number)
        if note:
            notes.append(note)
        if box:
            boxes.append(box)
    return boxes, notes


def convert_yolo_box(
    box: YoloBox,
    *,
    image_width: int,
    image_height: int,
    min_crop_width: int,
    min_crop_height: int,
) -> tuple[ConvertedBox | None, str | None]:
    values = [box.x_center, box.y_center, box.width, box.height]
    if any(not math.isfinite(value) for value in values):
        return None, f"line {box.line_number}: nonfinite_yolo_bbox"
    if box.width <= 0 or box.height <= 0:
        return None, f"line {box.line_number}: nonpositive_yolo_bbox"

    x1_float = (box.x_center - box.width / 2.0) * image_width
    y1_float = (box.y_center - box.height / 2.0) * image_height
    x2_float = (box.x_center + box.width / 2.0) * image_width
    y2_float = (box.y_center + box.height / 2.0) * image_height

    clipped_x1 = max(0.0, min(float(image_width), x1_float))
    clipped_y1 = max(0.0, min(float(image_height), y1_float))
    clipped_x2 = max(0.0, min(float(image_width), x2_float))
    clipped_y2 = max(0.0, min(float(image_height), y2_float))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None, f"line {box.line_number}: bbox_outside_image"

    x1 = max(0, min(image_width, math.floor(clipped_x1)))
    y1 = max(0, min(image_height, math.floor(clipped_y1)))
    x2 = max(0, min(image_width, math.ceil(clipped_x2)))
    y2 = max(0, min(image_height, math.ceil(clipped_y2)))
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < min_crop_width or crop_height < min_crop_height:
        return (
            None,
            f"line {box.line_number}: tiny_box {crop_width}x{crop_height} below "
            f"{min_crop_width}x{min_crop_height}",
        )

    notes = ["yolo_normalized_bbox"]
    if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1_float, y1_float, x2_float, y2_float):
        notes.append("bbox_clipped_to_image")
    if box.extra_fields:
        notes.append("extra_label_fields_ignored")
    return ConvertedBox(xyxy=(x1, y1, x2, y2), width=crop_width, height=crop_height, notes=notes), None


def crop_id_for(
    *,
    source_name: str,
    class_name: str,
    source_image_path: str,
    source_label_path: str,
    line_number: int,
    bbox_xyxy: tuple[int, int, int, int],
) -> str:
    digest_input = "|".join(
        [
            "donor_crop_v1",
            source_name,
            class_name,
            source_image_path,
            source_label_path,
            str(line_number),
            ",".join(str(value) for value in bbox_xyxy),
        ]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    source_slug = sanitize_segment(source_name, "source").lower()
    return f"{source_slug}_{class_name}_{digest}"


def load_pillow_modules() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required only for scripts/build_donor_crop_bank.py. "
            "Install it in the data-prep environment with `python -m pip install Pillow` "
            "and rerun. Do not add Pillow to project runtime dependencies."
        ) from error
    return Image, ImageOps


def source_configs_to_build(
    config: dict[str, Any],
    selected_sources: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("config must define sources")

    to_build: list[dict[str, Any]] = []
    skipped: list[str] = []
    known_names = {str(source.get("source_name", "")) for source in raw_sources if isinstance(source, dict)}
    if selected_sources:
        unknown = sorted(selected_sources - known_names)
        if unknown:
            raise ValueError(f"unknown configured source(s): {', '.join(unknown)}")

    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("each source config must be a JSON object")
        source_name = str(source.get("source_name", "")).strip()
        if selected_sources:
            if source_name in selected_sources:
                to_build.append(source)
            continue
        if source.get("enabled", False):
            to_build.append(source)
        else:
            skipped.append(source_name)
    return to_build, skipped


def write_crop_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DONOR_CROP_MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_preview_crop(
    crop: Any,
    *,
    class_name: str,
    crop_id: str,
    preview_dir: Path,
    preview_counts: dict[str, int],
    max_per_class: int,
    preview_size_px: int,
) -> None:
    if preview_counts.get(class_name, 0) >= max_per_class:
        return
    target_dir = preview_dir / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    preview = crop.copy()
    preview.thumbnail((preview_size_px, preview_size_px))
    preview.convert("RGB").save(target_dir / f"{crop_id}.jpg", format="JPEG", quality=85)
    preview_counts[class_name] = preview_counts.get(class_name, 0) + 1


def build_rows_for_split(
    split_source: SourceSplit,
    *,
    class_map: dict[str, str],
    output_images_dir: Path,
    preview_dir: Path,
    min_crop_width: int,
    min_crop_height: int,
    preview_max_per_class: int,
    preview_size_px: int,
    max_crops_for_source: int | None,
    source_crop_count: int,
    overwrite: bool,
    root: Path,
    image_module: Any,
    image_ops_module: Any,
    preview_counts: dict[str, int],
    summary: CropBuildSummary,
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    image_index = build_image_index(split_source.images_dir)
    label_files = iter_label_files(split_source.labels_dir)
    summary.labels_seen += len(label_files)
    summary.images_seen += len(image_index)

    for label_path in label_files:
        if max_crops_for_source is not None and source_crop_count >= max_crops_for_source:
            break

        image_path = image_index.get(label_path.stem)
        if image_path is None:
            summary.skipped_labels += 1
            summary.skip_notes.append(f"{label_path}: no matching image")
            continue

        boxes, parse_notes = parse_yolo_label_file(label_path)
        summary.skip_notes.extend(parse_notes)
        summary.skipped_boxes += len(parse_notes)
        if not boxes:
            continue

        try:
            with image_module.open(image_path) as opened_image:
                image = image_ops_module.exif_transpose(opened_image).convert("RGB")
        except OSError as error:
            summary.skipped_labels += 1
            summary.skip_notes.append(f"{image_path}: cannot_open_image {error}")
            continue

        image_width, image_height = image.size
        source_image_path = repo_relative(image_path, root)
        source_label_path = repo_relative(label_path, root)
        for box in boxes:
            if max_crops_for_source is not None and source_crop_count >= max_crops_for_source:
                break

            summary.boxes_seen += 1
            class_name = class_map.get(box.class_id)
            if class_name is None:
                raise UnknownClassMappingError(
                    f"unknown class id {box.class_id!r} in {source_label_path}:{box.line_number}; "
                    f"configure class_map for {split_source.source_name}"
                )

            converted, skip_note = convert_yolo_box(
                box,
                image_width=image_width,
                image_height=image_height,
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
            )
            if converted is None:
                summary.skipped_boxes += 1
                summary.skip_notes.append(f"{source_label_path}:{skip_note}")
                continue

            crop_id = crop_id_for(
                source_name=split_source.source_name,
                class_name=class_name,
                source_image_path=source_image_path,
                source_label_path=source_label_path,
                line_number=box.line_number,
                bbox_xyxy=converted.xyxy,
            )
            crop_rel_dir = Path(class_name) / sanitize_segment(split_source.source_name, "source")
            crop_path = output_images_dir / crop_rel_dir / f"{crop_id}.png"
            if crop_path.exists() and not overwrite:
                raise FileExistsError(f"crop already exists; pass --overwrite to replace: {crop_path}")
            crop_path.parent.mkdir(parents=True, exist_ok=True)

            crop = image.crop(converted.xyxy)
            crop.save(crop_path, format="PNG")
            write_preview_crop(
                crop,
                class_name=class_name,
                crop_id=crop_id,
                preview_dir=preview_dir,
                preview_counts=preview_counts,
                max_per_class=preview_max_per_class,
                preview_size_px=preview_size_px,
            )

            row = {
                "crop_id": crop_id,
                "source_name": split_source.source_name,
                "source_image_path": source_image_path,
                "source_label_path": source_label_path,
                "crop_path": repo_relative(crop_path, root),
                "class_name": class_name,
                "bbox_xyxy": json.dumps(list(converted.xyxy), separators=(",", ":")),
                "crop_width": str(converted.width),
                "crop_height": str(converted.height),
                "sha256": hash_file(crop_path),
                "notes": "; ".join(converted.notes),
            }
            rows.append(row)
            source_crop_count += 1
            summary.crops_written += 1
    return rows, source_crop_count


def run_build_crop_bank(
    *,
    config_path: Path,
    manifest_path: Path | None = None,
    output_images_dir: Path | None = None,
    preview_dir: Path | None = None,
    output_manifest: Path | None = None,
    selected_sources: set[str] | None = None,
    max_crops_per_source: int | None = None,
    overwrite: bool = False,
    root: Path = ROOT,
) -> CropBuildSummary:
    config = load_config(config_path)
    manifest = manifest_path or path_from_config(config, "master_manifest", root)
    images_dir = output_images_dir or path_from_config(config, "output_images_dir", root)
    preview_output_dir = preview_dir or path_from_config(config, "preview_dir", root)
    manifest_output = output_manifest or path_from_config(config, "output_manifest", root)

    if manifest_output.exists() and not overwrite:
        raise FileExistsError(f"output manifest already exists; pass --overwrite to replace: {manifest_output}")

    min_crop_width = int(config.get("min_crop_width", 16))
    min_crop_height = int(config.get("min_crop_height", 16))
    preview_max_per_class = int(config.get("preview_max_per_class", 50))
    preview_size_px = int(config.get("preview_size_px", 192))
    if min_crop_width <= 0 or min_crop_height <= 0:
        raise ValueError("min_crop_width and min_crop_height must be greater than zero")

    source_configs, skipped_sources = source_configs_to_build(config, selected_sources)
    master_rows = read_master_manifest(manifest)
    image_module, image_ops_module = load_pillow_modules()

    summary = CropBuildSummary(
        output_manifest=manifest_output,
        output_images_dir=images_dir,
        preview_dir=preview_output_dir,
        sources_seen=len(source_configs) + len(skipped_sources),
    )
    for skipped_source in skipped_sources:
        summary.skip_notes.append(f"{skipped_source}: skipped because enabled=false")

    manifest_rows: list[dict[str, str]] = []
    preview_counts: dict[str, int] = {}
    seen_crop_ids: set[str] = set()

    for source_config in source_configs:
        source_name = str(source_config.get("source_name", "")).strip()
        if not source_name:
            raise ValueError("source_name is required")
        assert_public_donor_source(master_rows, source_name)
        source_root = source_root_from_config(source_config, root)
        class_map = class_map_from_source_config(source_config, source_root)
        splits = source_splits(source_config, root)
        if not splits:
            raise FileNotFoundError(f"no YOLO image/label split directories found for {source_name}: {source_root}")

        summary.sources_built += 1
        source_crop_count = 0
        for split_source in splits:
            split_rows, source_crop_count = build_rows_for_split(
                split_source,
                class_map=class_map,
                output_images_dir=images_dir,
                preview_dir=preview_output_dir,
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
                preview_max_per_class=preview_max_per_class,
                preview_size_px=preview_size_px,
                max_crops_for_source=max_crops_per_source,
                source_crop_count=source_crop_count,
                overwrite=overwrite,
                root=root,
                image_module=image_module,
                image_ops_module=image_ops_module,
                preview_counts=preview_counts,
                summary=summary,
            )
            for row in split_rows:
                crop_id = row["crop_id"]
                if crop_id in seen_crop_ids:
                    raise ValueError(f"duplicate crop_id generated: {crop_id}")
                seen_crop_ids.add(crop_id)
                manifest_rows.append(row)

    if not manifest_rows:
        raise RuntimeError("no donor crops were written")
    write_crop_manifest(manifest_output, manifest_rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "donor_crop_v1.json",
        help="Donor crop config JSON path.",
    )
    parser.add_argument("--manifest", type=Path, help="Override master manifest CSV path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Convenience output root. Supplies images/ and preview/ unless explicit output dirs are provided.",
    )
    parser.add_argument("--output-images", type=Path, help="Override crop image output directory.")
    parser.add_argument("--preview-dir", type=Path, help="Override preview output directory.")
    parser.add_argument(
        "--output-manifest",
        "--out-manifest",
        dest="output_manifest",
        type=Path,
        help="Override donor crop manifest CSV path.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Build only this configured source_name. Repeat for multiple sources. Overrides enabled=false.",
    )
    parser.add_argument(
        "--max-crops-per-source",
        type=positive_int,
        help="Optional cap for smoke-test builds. Omit for the full crop bank.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop files and output manifest.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output_paths = resolve_cli_output_paths(args, ROOT)
    try:
        summary = run_build_crop_bank(
            config_path=resolve_cli_path(args.config, ROOT),
            manifest_path=resolve_cli_path(args.manifest, ROOT) if args.manifest else None,
            output_images_dir=output_paths.output_images_dir,
            preview_dir=output_paths.preview_dir,
            output_manifest=output_paths.output_manifest,
            selected_sources=set(args.source) if args.source else None,
            max_crops_per_source=args.max_crops_per_source,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, UnknownClassMappingError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS sources built: {summary.sources_built}")
    print(f"PASS labels seen: {summary.labels_seen}")
    print(f"PASS boxes seen: {summary.boxes_seen}")
    print(f"PASS crops written: {summary.crops_written}")
    print(f"PASS skipped boxes: {summary.skipped_boxes}")
    print(f"PASS skipped labels: {summary.skipped_labels}")
    print(f"PASS output images: {summary.output_images_dir}")
    print(f"PASS preview: {summary.preview_dir}")
    print(f"PASS output manifest: {summary.output_manifest}")
    if summary.skip_notes:
        print("NOTES first skipped items:")
        for note in summary.skip_notes[:20]:
            print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

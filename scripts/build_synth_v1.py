"""Build synthetic v1 public-data-only positive proxy images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024

SYNTH_MANIFEST_HEADER = [
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

TARGET_CLASS_NAMES = ("fire", "smoke")
DEFAULT_CLASS_IDS = {"fire": 0, "smoke": 1}
BACKGROUND_REQUIRED_COLUMNS = {
    "frame_id",
    "asset_id",
    "frame_path",
    "zone_l1",
    "use_purpose",
    "split",
    "quality_flag",
}
DONOR_REQUIRED_COLUMNS = {"crop_id", "source_name", "crop_path", "class_name"}


@dataclass(frozen=True)
class BackgroundFrame:
    frame_id: str
    asset_id: str
    frame_path: str
    zone_l1: str


@dataclass(frozen=True)
class DonorCrop:
    crop_id: str
    source_name: str
    crop_path: str
    class_name: str
    class_id: int


@dataclass(frozen=True)
class SynthConfig:
    min_scale: float
    max_scale: float
    opacity_min: float
    opacity_max: float
    blur_radius_min: float
    blur_radius_max: float
    avoid_edge_margin_ratio: float
    min_bbox_width_px: int
    min_bbox_height_px: int
    class_ids: dict[str, int]


@dataclass(frozen=True)
class ResolvedBuildOptions:
    config_path: Path
    background_manifest: Path
    donor_manifest: Path
    output_dir: Path
    output_manifest: Path
    max_samples_total: int
    preview_count: int
    seed: int
    overwrite: bool
    synth_config: SynthConfig


@dataclass
class SynthBuildSummary:
    output_dir: Path
    output_manifest: Path
    backgrounds_available: int = 0
    donors_available: dict[str, int] = field(default_factory=dict)
    samples_written: int = 0
    previews_written: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    background_zone_counts: Counter[str] = field(default_factory=Counter)
    donor_source_counts: Counter[str] = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a JSON object: {path}")
    return loaded


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def resolve_path_value(value: str | Path, root: Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    return resolve_cli_path(path, root)


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


def resolve_data_path(value: str, root: Path, *, field_name: str = "path") -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / normalize_repo_relative(value, field_name=field_name)


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
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def string_contains_with_human(row: dict[str, str]) -> bool:
    notes = f"{row.get('reviewer_notes', '')} {row.get('review_note', '')}".lower()
    return "with_human" in notes


def validate_background_manifest_name(path: Path) -> None:
    lowered = path.name.lower()
    if "hard_negative_eval" in lowered:
        raise ValueError("ship_like_hard_negative_eval.csv must never be used as a synthetic background manifest")
    if "background_val" in lowered or "background_test" in lowered:
        raise ValueError("validation/test background manifests must never be used for synthetic generation")


def load_backgrounds(path: Path, *, root: Path = ROOT) -> list[BackgroundFrame]:
    validate_background_manifest_name(path)
    if not path.is_file():
        raise FileNotFoundError(f"background manifest does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(BACKGROUND_REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"background manifest missing required columns: {', '.join(missing)}")
        rows = list(reader)

    backgrounds: list[BackgroundFrame] = []
    for row in rows:
        if row.get("use_purpose", "").strip().lower() != "synth_background":
            continue
        if row.get("split", "").strip().lower() != "train":
            continue
        if row.get("quality_flag", "").strip().lower() != "good":
            continue
        if string_contains_with_human(row):
            continue

        frame_path = row.get("frame_path", "").strip()
        if not frame_path:
            continue
        resolve_data_path(frame_path, root, field_name="frame_path")
        backgrounds.append(
            BackgroundFrame(
                frame_id=row.get("frame_id", "").strip(),
                asset_id=row.get("asset_id", "").strip(),
                frame_path=frame_path.replace("\\", "/"),
                zone_l1=row.get("zone_l1", "").strip() or "unknown",
            )
        )
    return backgrounds


def class_ids_from_config(config: dict[str, Any]) -> dict[str, int]:
    raw_class_ids = config.get("class_ids", DEFAULT_CLASS_IDS)
    if not isinstance(raw_class_ids, dict):
        raise ValueError("config class_ids must be an object")

    class_ids: dict[str, int] = {}
    for class_name in TARGET_CLASS_NAMES:
        raw_value = raw_class_ids.get(class_name)
        if raw_value is None:
            raise ValueError(f"config class_ids must include {class_name}")
        class_ids[class_name] = int(raw_value)
    if class_ids != DEFAULT_CLASS_IDS:
        raise ValueError("synthetic v1 class ids are fixed: fire=0, smoke=1")
    return class_ids


def synth_config_from_dict(config: dict[str, Any]) -> SynthConfig:
    class_ids = class_ids_from_config(config)
    synth_config = SynthConfig(
        min_scale=float(config.get("min_scale", 0.12)),
        max_scale=float(config.get("max_scale", 0.42)),
        opacity_min=float(config.get("opacity_min", 0.55)),
        opacity_max=float(config.get("opacity_max", 0.90)),
        blur_radius_min=float(config.get("blur_radius_min", 0.0)),
        blur_radius_max=float(config.get("blur_radius_max", 1.2)),
        avoid_edge_margin_ratio=float(config.get("avoid_edge_margin_ratio", 0.08)),
        min_bbox_width_px=int(config.get("min_bbox_width_px", 12)),
        min_bbox_height_px=int(config.get("min_bbox_height_px", 12)),
        class_ids=class_ids,
    )
    if synth_config.min_scale <= 0 or synth_config.max_scale < synth_config.min_scale:
        raise ValueError("min_scale must be greater than zero and max_scale must be >= min_scale")
    if not (0 < synth_config.opacity_min <= synth_config.opacity_max <= 1):
        raise ValueError("opacity_min and opacity_max must satisfy 0 < min <= max <= 1")
    if synth_config.blur_radius_min < 0 or synth_config.blur_radius_max < synth_config.blur_radius_min:
        raise ValueError("blur radius bounds must satisfy 0 <= min <= max")
    if not (0 <= synth_config.avoid_edge_margin_ratio < 0.5):
        raise ValueError("avoid_edge_margin_ratio must be >= 0 and < 0.5")
    if synth_config.min_bbox_width_px <= 0 or synth_config.min_bbox_height_px <= 0:
        raise ValueError("min bbox dimensions must be greater than zero")
    return synth_config


def load_donors(path: Path, *, class_ids: dict[str, int], root: Path = ROOT) -> dict[str, list[DonorCrop]]:
    if not path.is_file():
        raise FileNotFoundError(f"donor crop manifest does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(DONOR_REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"donor manifest missing required columns: {', '.join(missing)}")
        rows = list(reader)

    donors: dict[str, list[DonorCrop]] = {class_name: [] for class_name in TARGET_CLASS_NAMES}
    for row in rows:
        class_name = normalize_class_name(row.get("class_name", ""))
        if class_name not in donors:
            continue
        crop_path = row.get("crop_path", "").strip()
        if not crop_path:
            continue
        resolve_data_path(crop_path, root, field_name="crop_path")
        donors[class_name].append(
            DonorCrop(
                crop_id=row.get("crop_id", "").strip(),
                source_name=row.get("source_name", "").strip(),
                crop_path=crop_path.replace("\\", "/"),
                class_name=class_name,
                class_id=class_ids[class_name],
            )
        )
    return donors


def target_class_counts(max_samples_total: int) -> dict[str, int]:
    return {
        "fire": (max_samples_total + 1) // 2,
        "smoke": max_samples_total // 2,
    }


def build_generation_schedule(
    backgrounds: list[BackgroundFrame],
    *,
    max_samples_total: int,
) -> list[tuple[BackgroundFrame, str]]:
    targets = target_class_counts(max_samples_total)
    counts = Counter()
    schedule: list[tuple[BackgroundFrame, str]] = []
    while len(schedule) < max_samples_total:
        wrote_this_cycle = False
        for background in backgrounds:
            for class_name in TARGET_CLASS_NAMES:
                if counts[class_name] >= targets[class_name]:
                    continue
                schedule.append((background, class_name))
                counts[class_name] += 1
                wrote_this_cycle = True
                if len(schedule) >= max_samples_total:
                    return schedule
        if not wrote_this_cycle:
            break
    return schedule


def load_pillow_modules() -> tuple[Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required only for scripts/build_synth_v1.py in the data-prep environment. "
            "Install it in .venv_synth or with `python -m pip install Pillow`. "
            "Do not add Pillow to project runtime dependencies."
        ) from error
    return Image, ImageChops, ImageDraw, ImageFilter, ImageOps


def resample_lanczos(image_module: Any) -> Any:
    if hasattr(image_module, "Resampling"):
        return image_module.Resampling.LANCZOS
    return image_module.LANCZOS


def choose_scaled_size(
    donor_width: int,
    donor_height: int,
    background_width: int,
    background_height: int,
    *,
    rng: random.Random,
    config: SynthConfig,
) -> tuple[int, int] | None:
    margin_x = int(round(background_width * config.avoid_edge_margin_ratio))
    margin_y = int(round(background_height * config.avoid_edge_margin_ratio))
    margin_x = min(margin_x, max(0, (background_width - config.min_bbox_width_px) // 2))
    margin_y = min(margin_y, max(0, (background_height - config.min_bbox_height_px) // 2))
    available_width = background_width - 2 * margin_x
    available_height = background_height - 2 * margin_y
    if available_width <= 0 or available_height <= 0:
        return None

    target_long_side = rng.uniform(config.min_scale, config.max_scale) * min(background_width, background_height)
    scale = target_long_side / max(donor_width, donor_height)
    new_width = max(1, int(round(donor_width * scale)))
    new_height = max(1, int(round(donor_height * scale)))

    if new_width < config.min_bbox_width_px or new_height < config.min_bbox_height_px:
        scale_up = max(config.min_bbox_width_px / new_width, config.min_bbox_height_px / new_height)
        new_width = max(1, int(math.ceil(new_width * scale_up)))
        new_height = max(1, int(math.ceil(new_height * scale_up)))

    if new_width > available_width or new_height > available_height:
        scale_down = min(available_width / new_width, available_height / new_height)
        new_width = max(1, int(math.floor(new_width * scale_down)))
        new_height = max(1, int(math.floor(new_height * scale_down)))

    if new_width < config.min_bbox_width_px or new_height < config.min_bbox_height_px:
        return None
    return new_width, new_height


def choose_position(
    background_width: int,
    background_height: int,
    paste_width: int,
    paste_height: int,
    *,
    rng: random.Random,
    config: SynthConfig,
) -> tuple[int, int]:
    margin_x = int(round(background_width * config.avoid_edge_margin_ratio))
    margin_y = int(round(background_height * config.avoid_edge_margin_ratio))
    margin_x = min(margin_x, max(0, (background_width - paste_width) // 2))
    margin_y = min(margin_y, max(0, (background_height - paste_height) // 2))
    max_x = max(margin_x, background_width - paste_width - margin_x)
    max_y = max(margin_y, background_height - paste_height - margin_y)
    x = rng.randint(margin_x, max_x) if max_x > margin_x else margin_x
    y = rng.randint(margin_y, max_y) if max_y > margin_y else margin_y
    return x, y


def feather_alpha(alpha: Any, *, image_module: Any, image_chops_module: Any, image_filter_module: Any) -> Any:
    width, height = alpha.size
    feather_px = max(1, int(round(min(width, height) * 0.035)))
    if width <= feather_px * 2 or height <= feather_px * 2:
        return alpha
    edge_mask = image_module.new("L", (width, height), 0)
    edge_mask.paste(255, (feather_px, feather_px, width - feather_px, height - feather_px))
    edge_mask = edge_mask.filter(image_filter_module.GaussianBlur(feather_px))
    return image_chops_module.multiply(alpha, edge_mask)


def prepare_donor_crop(
    donor_image: Any,
    *,
    target_size: tuple[int, int],
    opacity: float,
    blur_radius: float,
    image_module: Any,
    image_chops_module: Any,
    image_filter_module: Any,
) -> Any:
    resized = donor_image.resize(target_size, resample_lanczos(image_module))
    if blur_radius > 0.05:
        resized = resized.filter(image_filter_module.GaussianBlur(blur_radius))
    red, green, blue, alpha = resized.split()
    alpha = feather_alpha(
        alpha,
        image_module=image_module,
        image_chops_module=image_chops_module,
        image_filter_module=image_filter_module,
    )
    if opacity < 1.0:
        alpha = alpha.point(lambda value: int(value * opacity))
    return image_module.merge("RGBA", (red, green, blue, alpha))


def bbox_to_yolo(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    return (
        (x + width / 2.0) / image_width,
        (y + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def format_bbox_values(values: tuple[float, float, float, float]) -> tuple[str, str, str, str]:
    return tuple(f"{value:.6f}" for value in values)


def write_label_file(path: Path, class_id: int, bbox: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox_text = " ".join(format_bbox_values(bbox))
    path.write_text(f"{class_id} {bbox_text}\n", encoding="utf-8")


def write_preview(
    image: Any,
    *,
    preview_path: Path,
    class_name: str,
    box_xywh: tuple[int, int, int, int],
    image_draw_module: Any,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    draw = image_draw_module.Draw(preview)
    x, y, width, height = box_xywh
    color = (255, 72, 48) if class_name == "fire" else (48, 168, 255)
    line_width = max(2, int(round(min(preview.size) / 240)))
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=line_width)
    preview.save(preview_path, format="JPEG", quality=88)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SYNTH_MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def config_value(
    args: argparse.Namespace,
    config: dict[str, Any],
    cli_name: str,
    config_name: str,
    default: Any = None,
) -> Any:
    value = getattr(args, cli_name)
    if value is not None:
        return value
    return config.get(config_name, default)


def resolve_build_options(args: argparse.Namespace, *, root: Path = ROOT) -> ResolvedBuildOptions:
    config_path = resolve_cli_path(args.config, root)
    config = load_config(config_path)
    synth_config = synth_config_from_dict(config)

    background_manifest = resolve_path_value(
        config_value(args, config, "background_manifest", "background_manifest"),
        root,
    )
    donor_manifest = resolve_path_value(
        config_value(args, config, "donor_manifest", "donor_manifest"),
        root,
    )
    output_dir = resolve_path_value(config_value(args, config, "output_dir", "output_dir"), root)
    output_manifest = resolve_path_value(
        config_value(args, config, "output_manifest", "output_manifest"),
        root,
    )
    max_samples_total = int(config_value(args, config, "max_samples_total", "max_samples_total", 170))
    preview_count = int(config_value(args, config, "preview_count", "preview_count", 50))
    seed = int(config_value(args, config, "seed", "seed", 20260426))
    if max_samples_total <= 0:
        raise ValueError("max_samples_total must be greater than zero")
    if preview_count < 0:
        raise ValueError("preview_count must be zero or greater")
    return ResolvedBuildOptions(
        config_path=config_path,
        background_manifest=background_manifest,
        donor_manifest=donor_manifest,
        output_dir=output_dir,
        output_manifest=output_manifest,
        max_samples_total=max_samples_total,
        preview_count=preview_count,
        seed=seed,
        overwrite=args.overwrite,
        synth_config=synth_config,
    )


def assert_outputs_writable(options: ResolvedBuildOptions) -> None:
    if options.output_manifest.exists() and not options.overwrite:
        raise FileExistsError(f"output manifest already exists; pass --overwrite to replace: {options.output_manifest}")
    for child_dir in ["images", "labels", "previews"]:
        output_child = options.output_dir / child_dir
        if not output_child.exists():
            continue
        if options.overwrite:
            continue
        existing_files = [path for path in output_child.rglob("*") if path.is_file()]
        if existing_files:
            raise FileExistsError(
                f"output directory already contains files; pass --overwrite to replace matching names: {output_child}"
            )


def run_build_synth_v1(
    *,
    config_path: Path | None = None,
    background_manifest: Path | None = None,
    donor_manifest: Path | None = None,
    output_dir: Path | None = None,
    output_manifest: Path | None = None,
    max_samples_total: int | None = None,
    preview_count: int | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    root: Path = ROOT,
) -> SynthBuildSummary:
    args = argparse.Namespace(
        config=config_path or (root / "configs" / "synth_v1.json"),
        background_manifest=background_manifest,
        donor_manifest=donor_manifest,
        output_dir=output_dir,
        output_manifest=output_manifest,
        max_samples_total=max_samples_total,
        preview_count=preview_count,
        seed=seed,
        overwrite=overwrite,
    )
    options = resolve_build_options(args, root=root)
    assert_outputs_writable(options)

    backgrounds = load_backgrounds(options.background_manifest, root=root)
    if not backgrounds:
        raise RuntimeError("no eligible train synth_background rows found after filtering")

    donors_by_class = load_donors(options.donor_manifest, class_ids=options.synth_config.class_ids, root=root)
    targets = target_class_counts(options.max_samples_total)
    for class_name, target_count in targets.items():
        if target_count > 0 and not donors_by_class[class_name]:
            raise RuntimeError(f"no usable donor crops available for class {class_name}")

    rng = random.Random(options.seed)
    backgrounds = list(backgrounds)
    rng.shuffle(backgrounds)
    for class_name in TARGET_CLASS_NAMES:
        rng.shuffle(donors_by_class[class_name])
    schedule = build_generation_schedule(backgrounds, max_samples_total=options.max_samples_total)

    image_module, image_chops_module, image_draw_module, image_filter_module, image_ops_module = load_pillow_modules()
    image_dir = options.output_dir / "images"
    label_dir = options.output_dir / "labels"
    preview_dir = options.output_dir / "previews"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if options.preview_count:
        preview_dir.mkdir(parents=True, exist_ok=True)

    summary = SynthBuildSummary(
        output_dir=options.output_dir,
        output_manifest=options.output_manifest,
        backgrounds_available=len(backgrounds),
        donors_available={class_name: len(donors_by_class[class_name]) for class_name in TARGET_CLASS_NAMES},
    )
    donor_offsets = {class_name: 0 for class_name in TARGET_CLASS_NAMES}
    rows: list[dict[str, str]] = []

    for index, (background, class_name) in enumerate(schedule, start=1):
        donor_list = donors_by_class[class_name]
        donor = donor_list[donor_offsets[class_name] % len(donor_list)]
        donor_offsets[class_name] += 1
        sample_seed = rng.randrange(0, 2**32)
        sample_rng = random.Random(sample_seed)

        background_path = resolve_data_path(background.frame_path, root, field_name="background_frame_path")
        donor_path = resolve_data_path(donor.crop_path, root, field_name="donor_crop_path")
        with image_module.open(background_path) as opened_background:
            background_image = image_ops_module.exif_transpose(opened_background).convert("RGB")
        with image_module.open(donor_path) as opened_donor:
            donor_image = image_ops_module.exif_transpose(opened_donor).convert("RGBA")

        background_width, background_height = background_image.size
        donor_width, donor_height = donor_image.size
        scaled_size = choose_scaled_size(
            donor_width,
            donor_height,
            background_width,
            background_height,
            rng=sample_rng,
            config=options.synth_config,
        )
        if scaled_size is None:
            summary.notes.append(f"{background.frame_id}: skipped too small for {donor.crop_id}")
            continue

        opacity = sample_rng.uniform(options.synth_config.opacity_min, options.synth_config.opacity_max)
        blur_radius = sample_rng.uniform(options.synth_config.blur_radius_min, options.synth_config.blur_radius_max)
        prepared_crop = prepare_donor_crop(
            donor_image,
            target_size=scaled_size,
            opacity=opacity,
            blur_radius=blur_radius,
            image_module=image_module,
            image_chops_module=image_chops_module,
            image_filter_module=image_filter_module,
        )
        paste_width, paste_height = prepared_crop.size
        paste_x, paste_y = choose_position(
            background_width,
            background_height,
            paste_width,
            paste_height,
            rng=sample_rng,
            config=options.synth_config,
        )

        composited = background_image.convert("RGBA")
        composited.alpha_composite(prepared_crop, (paste_x, paste_y))
        final_image = composited.convert("RGB")

        synth_id = f"synth_v1_{index:06d}_{class_name}"
        image_path = image_dir / f"{synth_id}.jpg"
        label_path = label_dir / f"{synth_id}.txt"
        if not options.overwrite and (image_path.exists() or label_path.exists()):
            raise FileExistsError(f"synthetic output already exists; pass --overwrite to replace: {synth_id}")

        final_image.save(image_path, format="JPEG", quality=92)
        bbox = bbox_to_yolo(paste_x, paste_y, paste_width, paste_height, background_width, background_height)
        write_label_file(label_path, donor.class_id, bbox)

        if summary.previews_written < options.preview_count:
            preview_path = preview_dir / f"{synth_id}_preview.jpg"
            write_preview(
                final_image,
                preview_path=preview_path,
                class_name=class_name,
                box_xywh=(paste_x, paste_y, paste_width, paste_height),
                image_draw_module=image_draw_module,
            )
            summary.previews_written += 1

        bbox_values = [round(value, 6) for value in bbox]
        notes = [
            "public_data_only",
            "synthetic_positive_proxy",
            "not_real_shipboard_positive_label",
            f"run_seed={options.seed}",
            f"opacity={opacity:.3f}",
            f"blur_radius={blur_radius:.3f}",
        ]
        row = {
            "synth_id": synth_id,
            "image_path": repo_relative(image_path, root),
            "label_path": repo_relative(label_path, root),
            "background_frame_id": background.frame_id,
            "background_asset_id": background.asset_id,
            "background_frame_path": background.frame_path,
            "background_zone_l1": background.zone_l1,
            "donor_crop_id": donor.crop_id,
            "donor_source_name": donor.source_name,
            "donor_crop_path": donor.crop_path,
            "class_name": class_name,
            "class_id": str(donor.class_id),
            "bbox_xywh_norm": json.dumps(bbox_values, separators=(",", ":")),
            "generation_seed": str(sample_seed),
            "image_width": str(background_width),
            "image_height": str(background_height),
            "sha256": hash_file(image_path),
            "quality_flag": "synthetic_proxy",
            "notes": "; ".join(notes),
        }
        rows.append(row)
        summary.samples_written += 1
        summary.class_counts[class_name] += 1
        summary.background_zone_counts[background.zone_l1] += 1
        summary.donor_source_counts[donor.source_name] += 1

    if not rows:
        raise RuntimeError("no synthetic samples were written")
    write_manifest(options.output_manifest, rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "synth_v1.json",
        help="Synthetic v1 config JSON path.",
    )
    parser.add_argument("--background-manifest", type=Path, help="Ship-like train background manifest CSV path.")
    parser.add_argument(
        "--donor-manifest",
        "--donor-crop-manifest",
        dest="donor_manifest",
        type=Path,
        help="Donor crop manifest CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        help="Synthetic output directory. Images, labels, and previews are written below it.",
    )
    parser.add_argument(
        "--output-manifest",
        "--out-manifest",
        dest="output_manifest",
        type=Path,
        help="Synthetic manifest CSV path.",
    )
    parser.add_argument("--max-samples-total", type=positive_int, help="Maximum synthetic samples to generate.")
    parser.add_argument("--preview-count", type=nonnegative_int, help="Number of QA preview images to write.")
    parser.add_argument("--seed", type=int, help="Base deterministic generation seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite matching synthetic v1 outputs.")
    return parser.parse_args(argv)


def print_counter(name: str, counter: Counter[str]) -> None:
    print(f"PASS {name}:")
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def main() -> int:
    args = parse_args()
    try:
        summary = run_build_synth_v1(
            config_path=resolve_cli_path(args.config, ROOT),
            background_manifest=resolve_cli_path(args.background_manifest, ROOT) if args.background_manifest else None,
            donor_manifest=resolve_cli_path(args.donor_manifest, ROOT) if args.donor_manifest else None,
            output_dir=resolve_cli_path(args.output_dir, ROOT) if args.output_dir else None,
            output_manifest=resolve_cli_path(args.output_manifest, ROOT) if args.output_manifest else None,
            max_samples_total=args.max_samples_total,
            preview_count=args.preview_count,
            seed=args.seed,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS backgrounds available: {summary.backgrounds_available}")
    for class_name in TARGET_CLASS_NAMES:
        print(f"PASS donor crops available {class_name}: {summary.donors_available.get(class_name, 0)}")
    print(f"PASS samples written: {summary.samples_written}")
    print(f"PASS previews written: {summary.previews_written}")
    print(f"PASS output dir: {summary.output_dir}")
    print(f"PASS output manifest: {summary.output_manifest}")
    print_counter("class counts", summary.class_counts)
    print_counter("background zone counts", summary.background_zone_counts)
    print_counter("donor source counts", summary.donor_source_counts)
    if summary.notes:
        print("NOTES first skipped items:")
        for note in summary.notes[:20]:
            print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

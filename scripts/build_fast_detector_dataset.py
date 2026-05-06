"""Build a fast YOLO detector dataset for seminar-only feasibility evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
CSV_MANIFEST_ENCODING = "utf-8-sig"
TARGET_CLASS_NAMES = ("fire", "smoke")
CLASS_IDS = {"fire": 0, "smoke": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DONOR_REQUIRED_COLUMNS = {"crop_id", "source_name", "crop_path", "class_name"}
SYNTH_REQUIRED_COLUMNS = {
    "synth_id",
    "image_path",
    "label_path",
    "class_name",
    "class_id",
    "bbox_xywh_norm",
    "background_zone_l1",
    "donor_source_name",
}
BACKGROUND_REQUIRED_COLUMNS = {
    "frame_id",
    "asset_id",
    "source_name",
    "frame_path",
    "split",
    "quality_flag",
    "zone_l1",
}

EVAL_MANIFEST_HEADER = [
    "eval_id",
    "task",
    "image_path",
    "label_path",
    "source_id",
    "source_name",
    "class_name",
    "class_id",
    "bbox_xywh_norm",
    "zone_l1",
    "notes",
]


@dataclass
class FastDatasetSummary:
    dataset_root: Path
    train_images: int = 0
    val_images: int = 0
    eval_synth_val_images: int = 0
    eval_hard_negative_images: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    source_type_counts: Counter[str] = field(default_factory=Counter)
    source_name_counts: Counter[str] = field(default_factory=Counter)
    class_counts: Counter[str] = field(default_factory=Counter)
    negative_counts: Counter[str] = field(default_factory=Counter)


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def normalize_csv_fieldname(value: str | None) -> str:
    return (value or "").strip().lstrip("\ufeff").strip()


def normalize_csv_reader_fieldnames(reader: csv.DictReader) -> tuple[list[str], list[str]]:
    raw_fieldnames = list(reader.fieldnames or [])
    normalized_fieldnames = [normalize_csv_fieldname(fieldname) for fieldname in raw_fieldnames]
    reader.fieldnames = normalized_fieldnames
    return raw_fieldnames, normalized_fieldnames


def format_missing_columns_error(
    source_name: str,
    *,
    missing: list[str],
    required_columns: list[str],
    normalized_actual_columns: list[str],
    raw_actual_columns: list[str],
) -> str:
    normalized_actual = ", ".join(normalized_actual_columns) if normalized_actual_columns else "(none)"
    return (
        f"{source_name} missing required columns: {', '.join(missing)}; "
        f"required columns: {', '.join(required_columns)}; "
        f"normalized actual columns: {normalized_actual}; "
        f"raw actual columns repr: {raw_actual_columns!r}"
    )


def read_csv_rows(path: Path, *, required_columns: set[str], source_name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} does not exist: {path}")
    with path.open("r", newline="", encoding=CSV_MANIFEST_ENCODING) as csv_file:
        reader = csv.DictReader(csv_file)
        raw_fieldnames, normalized_fieldnames = normalize_csv_reader_fieldnames(reader)
        missing = [column for column in sorted(required_columns) if column not in normalized_fieldnames]
        if missing:
            raise ValueError(
                format_missing_columns_error(
                    source_name,
                    missing=missing,
                    required_columns=sorted(required_columns),
                    normalized_actual_columns=normalized_fieldnames,
                    raw_actual_columns=raw_fieldnames,
                )
            )
        return list(reader)


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


def sanitize_segment(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def resolve_dataset_root(output_dir: Path, dataset_name: str) -> Path:
    dataset_segment = sanitize_segment(dataset_name, "fast_detector_dataset")
    if output_dir.name == dataset_name or output_dir.name == dataset_segment:
        return output_dir.resolve()
    return (output_dir / dataset_segment).resolve()


def normalize_class_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def stable_digest(seed: int, identifier: str, namespace: str) -> str:
    payload = f"{namespace}:{seed}:{identifier}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_rows_by_id(
    rows: list[dict[str, str]],
    *,
    id_field: str,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ordered = sorted(rows, key=lambda row: stable_digest(seed, row.get(id_field, ""), namespace))
    if len(ordered) <= 1:
        return ordered, []
    train_count = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
    return ordered[:train_count], ordered[train_count:]


def select_balanced_donors(rows: list[dict[str, str]], *, max_total: int, seed: int) -> list[dict[str, str]]:
    by_class: dict[str, list[dict[str, str]]] = {class_name: [] for class_name in TARGET_CLASS_NAMES}
    for row in rows:
        class_name = normalize_class_name(row.get("class_name", ""))
        if class_name in by_class:
            by_class[class_name].append(row)

    for class_name in TARGET_CLASS_NAMES:
        by_class[class_name].sort(
            key=lambda row: stable_digest(seed, row.get("crop_id", ""), f"donor_select_{class_name}")
        )

    selected: list[dict[str, str]] = []
    offsets = {class_name: 0 for class_name in TARGET_CLASS_NAMES}
    counts = Counter()
    while len(selected) < max_total:
        available = [class_name for class_name in TARGET_CLASS_NAMES if offsets[class_name] < len(by_class[class_name])]
        if not available:
            break
        class_name = min(available, key=lambda name: (counts[name], name))
        row = by_class[class_name][offsets[class_name]]
        offsets[class_name] += 1
        counts[class_name] += 1
        selected.append(row)
    return selected


def safe_clear_dataset_root(dataset_root: Path, root: Path) -> None:
    if not dataset_root.exists():
        return
    allowed_parent = (root / "03_experiments" / "detector_datasets").resolve()
    resolved_target = dataset_root.resolve()
    try:
        resolved_target.relative_to(allowed_parent)
    except ValueError as error:
        raise ValueError(f"refusing to overwrite dataset outside {allowed_parent}: {dataset_root}") from error
    for child in dataset_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def assert_dataset_writable(dataset_root: Path, *, overwrite: bool, root: Path) -> None:
    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"dataset already exists; pass --overwrite to replace: {dataset_root}")
        safe_clear_dataset_root(dataset_root, root)
    dataset_root.mkdir(parents=True, exist_ok=True)


def copy_image(source_path: Path, destination_dir: Path, filename_stem: str) -> Path:
    suffix = source_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / f"{filename_stem}{suffix}"
    shutil.copy2(source_path, destination_path)
    return destination_path


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_label(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def add_positive(
    *,
    source_image: Path,
    source_label: Path | None,
    image_dir: Path,
    label_dir: Path,
    stem: str,
    class_name: str,
    source_type: str,
    source_name: str,
    split: str,
    summary: FastDatasetSummary,
    root: Path,
) -> tuple[Path, Path]:
    if not source_image.is_file():
        raise FileNotFoundError(f"image does not exist: {source_image}")
    image_path = copy_image(source_image, image_dir, stem)
    label_path = label_path_for_image(image_path, label_dir)
    if source_label is None:
        class_id = CLASS_IDS[class_name]
        write_label(label_path, f"{class_id} 0.500000 0.500000 0.980000 0.980000\n")
    else:
        if not source_label.is_file():
            raise FileNotFoundError(f"label does not exist: {source_label}")
        copy_label(source_label, label_path)

    summary.split_counts[split] += 1
    summary.source_type_counts[source_type] += 1
    summary.source_name_counts[source_name or source_type] += 1
    summary.class_counts[class_name] += 1
    if split == "train":
        summary.train_images += 1
    elif split == "val":
        summary.val_images += 1
    repo_relative(image_path, root)
    repo_relative(label_path, root)
    return image_path, label_path


def add_negative(
    *,
    source_image: Path,
    image_dir: Path,
    label_dir: Path,
    stem: str,
    source_type: str,
    source_name: str,
    split: str,
    summary: FastDatasetSummary,
    root: Path,
) -> tuple[Path, Path]:
    if not source_image.is_file():
        raise FileNotFoundError(f"image does not exist: {source_image}")
    image_path = copy_image(source_image, image_dir, stem)
    label_path = label_path_for_image(image_path, label_dir)
    write_label(label_path, "")

    summary.split_counts[split] += 1
    summary.source_type_counts[source_type] += 1
    summary.source_name_counts[source_name or source_type] += 1
    summary.negative_counts[split] += 1
    if split == "train":
        summary.train_images += 1
    elif split == "val":
        summary.val_images += 1
    elif split == "eval_hard_negative":
        summary.eval_hard_negative_images += 1
    repo_relative(image_path, root)
    repo_relative(label_path, root)
    return image_path, label_path


def eligible_background_rows(rows: list[dict[str, str]], *, split: str | None) -> list[dict[str, str]]:
    eligible: list[dict[str, str]] = []
    for row in rows:
        if split is not None and row.get("split", "").strip().lower() != split:
            continue
        if row.get("quality_flag", "").strip().lower() != "good":
            continue
        if not row.get("frame_path", "").strip():
            continue
        eligible.append(row)
    return eligible


def write_data_yaml(dataset_root: Path) -> None:
    path_value = dataset_root.resolve().as_posix()
    lines = [
        f"path: {json.dumps(path_value)}",
        'train: "images/train"',
        'val: "images/val"',
        'test: "eval/synth_val/images"',
        "names:",
        '  0: "fire"',
        '  1: "smoke"',
        "",
    ]
    (dataset_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def summary_table_rows(summary: FastDatasetSummary) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {"section": "total", "name": "train_images", "count": str(summary.train_images)},
        {"section": "total", "name": "val_images", "count": str(summary.val_images)},
        {"section": "total", "name": "eval_synth_val_images", "count": str(summary.eval_synth_val_images)},
        {
            "section": "total",
            "name": "eval_hard_negative_images",
            "count": str(summary.eval_hard_negative_images),
        },
    ]
    for section, counter in [
        ("split", summary.split_counts),
        ("source_type", summary.source_type_counts),
        ("source_name", summary.source_name_counts),
        ("class_name", summary.class_counts),
        ("negative_image_count", summary.negative_counts),
    ]:
        for name, count in sorted(counter.items()):
            rows.append({"section": section, "name": name, "count": str(count)})
    return rows


def write_summary_csv(path: Path, summary: FastDatasetSummary) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "name", "count"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_table_rows(summary))


def write_eval_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVAL_MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_paper_summary(path: Path, summary: FastDatasetSummary) -> None:
    lines = [
        "Fast detector dataset summary for seminar paper",
        "",
        f"Dataset root: {summary.dataset_root}",
        f"Training images: {summary.train_images}",
        f"Validation images: {summary.val_images}",
        f"Held-out synthetic proxy evaluation images: {summary.eval_synth_val_images}",
        f"Hard-negative evaluation images: {summary.eval_hard_negative_images}",
        "",
        "Class ids: fire=0, smoke=1.",
        "Donor crop positives use full-image proxy boxes (0.5 0.5 0.98 0.98), so they are useful for fast detector warm-start evidence only.",
        "Synthetic held-out validation uses public ship-like backgrounds with public donor fire/smoke overlays; it is not real shipboard positive evidence.",
        "Hard-negative frames are public ship-like proxy frames and are excluded from training.",
        "Generated detector datasets live under 03_experiments and are intentionally ignored by Git.",
        "Claim boundary: public-data-only preliminary feasibility, not a SOTA detector study and not real CCTV validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"PASS {title}:")
    if not counter:
        print("  (none): 0")
    for name, count in sorted(counter.items()):
        print(f"  {name}: {count}")


def run_build_fast_detector_dataset(
    *,
    donor_manifest: Path,
    synth_manifest: Path,
    background_train: Path,
    background_val: Path,
    hard_negative: Path,
    output_dir: Path,
    dataset_name: str,
    max_donor_crops: int,
    seed: int,
    overwrite: bool = False,
    root: Path = ROOT,
) -> FastDatasetSummary:
    dataset_root = resolve_dataset_root(output_dir, dataset_name)
    assert_dataset_writable(dataset_root, overwrite=overwrite, root=root)

    train_image_dir = dataset_root / "images" / "train"
    train_label_dir = dataset_root / "labels" / "train"
    val_image_dir = dataset_root / "images" / "val"
    val_label_dir = dataset_root / "labels" / "val"
    eval_synth_image_dir = dataset_root / "eval" / "synth_val" / "images"
    eval_synth_label_dir = dataset_root / "eval" / "synth_val" / "labels"
    eval_hard_image_dir = dataset_root / "eval" / "hard_negative" / "images"
    eval_hard_label_dir = dataset_root / "eval" / "hard_negative" / "labels"
    for directory in [
        train_image_dir,
        train_label_dir,
        val_image_dir,
        val_label_dir,
        eval_synth_image_dir,
        eval_synth_label_dir,
        eval_hard_image_dir,
        eval_hard_label_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    summary = FastDatasetSummary(dataset_root=dataset_root)
    eval_synth_rows: list[dict[str, str]] = []
    eval_hard_rows: list[dict[str, str]] = []

    donor_rows = read_csv_rows(donor_manifest, required_columns=DONOR_REQUIRED_COLUMNS, source_name="donor manifest")
    selected_donors = select_balanced_donors(donor_rows, max_total=max_donor_crops, seed=seed)
    donors_by_class: dict[str, list[dict[str, str]]] = {class_name: [] for class_name in TARGET_CLASS_NAMES}
    for row in selected_donors:
        class_name = normalize_class_name(row.get("class_name", ""))
        if class_name in donors_by_class:
            donors_by_class[class_name].append(row)

    for class_name in TARGET_CLASS_NAMES:
        donor_train, donor_val = split_rows_by_id(
            donors_by_class[class_name],
            id_field="crop_id",
            seed=seed,
            namespace=f"donor_split_{class_name}",
        )
        for split, split_rows, image_dir, label_dir in [
            ("train", donor_train, train_image_dir, train_label_dir),
            ("val", donor_val, val_image_dir, val_label_dir),
        ]:
            for index, row in enumerate(split_rows, start=1):
                source_image = resolve_data_path(row.get("crop_path", ""), root, field_name="crop_path")
                stem = sanitize_segment(f"donor_{split}_{class_name}_{index:05d}_{row.get('crop_id', '')}", "donor")
                add_positive(
                    source_image=source_image,
                    source_label=None,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    stem=stem,
                    class_name=class_name,
                    source_type="donor_crop",
                    source_name=row.get("source_name", "").strip(),
                    split=split,
                    summary=summary,
                    root=root,
                )

    synth_rows = read_csv_rows(synth_manifest, required_columns=SYNTH_REQUIRED_COLUMNS, source_name="synthetic manifest")
    synth_by_class: dict[str, list[dict[str, str]]] = {class_name: [] for class_name in TARGET_CLASS_NAMES}
    for row in synth_rows:
        class_name = normalize_class_name(row.get("class_name", ""))
        if class_name in synth_by_class:
            synth_by_class[class_name].append(row)

    for class_name in TARGET_CLASS_NAMES:
        synth_train, synth_val = split_rows_by_id(
            synth_by_class[class_name],
            id_field="synth_id",
            seed=seed,
            namespace=f"synth_split_{class_name}",
        )
        for split, split_rows, image_dir, label_dir in [
            ("train", synth_train, train_image_dir, train_label_dir),
            ("val", synth_val, val_image_dir, val_label_dir),
        ]:
            for index, row in enumerate(split_rows, start=1):
                source_image = resolve_data_path(row.get("image_path", ""), root, field_name="image_path")
                source_label = resolve_data_path(row.get("label_path", ""), root, field_name="label_path")
                stem = sanitize_segment(f"synth_{split}_{class_name}_{index:05d}_{row.get('synth_id', '')}", "synth")
                copied_image, copied_label = add_positive(
                    source_image=source_image,
                    source_label=source_label,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    stem=stem,
                    class_name=class_name,
                    source_type="synthetic_proxy",
                    source_name=row.get("donor_source_name", "").strip(),
                    split=split,
                    summary=summary,
                    root=root,
                )
                if split == "val":
                    eval_stem = sanitize_segment(f"eval_synth_{class_name}_{index:05d}_{row.get('synth_id', '')}", "eval_synth")
                    eval_image = copy_image(source_image, eval_synth_image_dir, eval_stem)
                    eval_label = label_path_for_image(eval_image, eval_synth_label_dir)
                    copy_label(source_label, eval_label)
                    summary.eval_synth_val_images += 1
                    eval_synth_rows.append(
                        {
                            "eval_id": row.get("synth_id", "").strip(),
                            "task": "synthetic_proxy",
                            "image_path": repo_relative(eval_image, root),
                            "label_path": repo_relative(eval_label, root),
                            "source_id": row.get("synth_id", "").strip(),
                            "source_name": "synth_v1",
                            "class_name": class_name,
                            "class_id": row.get("class_id", "").strip(),
                            "bbox_xywh_norm": row.get("bbox_xywh_norm", "").strip(),
                            "zone_l1": row.get("background_zone_l1", "").strip(),
                            "notes": "held_out_synthetic_proxy_eval; public_data_only",
                        }
                    )
                    repo_relative(copied_image, root)
                    repo_relative(copied_label, root)

    background_train_rows = eligible_background_rows(
        read_csv_rows(background_train, required_columns=BACKGROUND_REQUIRED_COLUMNS, source_name="background train manifest"),
        split="train",
    )
    for index, row in enumerate(background_train_rows, start=1):
        source_image = resolve_data_path(row.get("frame_path", ""), root, field_name="frame_path")
        stem = sanitize_segment(f"bg_train_{index:05d}_{row.get('frame_id', '')}", "bg_train")
        add_negative(
            source_image=source_image,
            image_dir=train_image_dir,
            label_dir=train_label_dir,
            stem=stem,
            source_type="ship_like_background_train",
            source_name=row.get("source_name", "").strip(),
            split="train",
            summary=summary,
            root=root,
        )

    background_val_rows = eligible_background_rows(
        read_csv_rows(background_val, required_columns=BACKGROUND_REQUIRED_COLUMNS, source_name="background val manifest"),
        split="val",
    )
    for index, row in enumerate(background_val_rows, start=1):
        source_image = resolve_data_path(row.get("frame_path", ""), root, field_name="frame_path")
        stem = sanitize_segment(f"bg_val_{index:05d}_{row.get('frame_id', '')}", "bg_val")
        add_negative(
            source_image=source_image,
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            stem=stem,
            source_type="ship_like_background_val",
            source_name=row.get("source_name", "").strip(),
            split="val",
            summary=summary,
            root=root,
        )

    hard_negative_rows = eligible_background_rows(
        read_csv_rows(hard_negative, required_columns=BACKGROUND_REQUIRED_COLUMNS, source_name="hard-negative manifest"),
        split=None,
    )
    for index, row in enumerate(hard_negative_rows, start=1):
        source_image = resolve_data_path(row.get("frame_path", ""), root, field_name="frame_path")
        stem = sanitize_segment(f"hardneg_{index:05d}_{row.get('frame_id', '')}", "hardneg")
        image_path, label_path = add_negative(
            source_image=source_image,
            image_dir=eval_hard_image_dir,
            label_dir=eval_hard_label_dir,
            stem=stem,
            source_type="ship_like_hard_negative_eval",
            source_name=row.get("source_name", "").strip(),
            split="eval_hard_negative",
            summary=summary,
            root=root,
        )
        eval_hard_rows.append(
            {
                "eval_id": row.get("frame_id", "").strip(),
                "task": "hard_negative",
                "image_path": repo_relative(image_path, root),
                "label_path": repo_relative(label_path, root),
                "source_id": row.get("asset_id", "").strip(),
                "source_name": row.get("source_name", "").strip(),
                "class_name": "",
                "class_id": "",
                "bbox_xywh_norm": "",
                "zone_l1": row.get("zone_l1", "").strip(),
                "notes": "hard_negative_eval_only; public_data_only; excluded_from_training",
            }
        )

    write_data_yaml(dataset_root)
    write_summary_csv(dataset_root / "dataset_summary.csv", summary)
    write_eval_manifest(dataset_root / "eval_synth_val_manifest.csv", eval_synth_rows)
    write_eval_manifest(dataset_root / "eval_hard_negative_manifest.csv", eval_hard_rows)
    write_paper_summary(dataset_root / "dataset_summary_for_paper.txt", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor-manifest",
        "--donor-crop-manifest",
        dest="donor_manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "donor_crop_manifest_v1.csv",
        help="Donor crop manifest CSV path.",
    )
    parser.add_argument(
        "--synth-manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "synth_v1_manifest.csv",
        help="Synthetic v1 manifest CSV path.",
    )
    parser.add_argument(
        "--background-train",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_background_train.csv",
        help="Ship-like train background manifest CSV path.",
    )
    parser.add_argument(
        "--background-val",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_background_val.csv",
        help="Ship-like validation background manifest CSV path.",
    )
    parser.add_argument(
        "--hard-negative",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_hard_negative_eval.csv",
        help="Hard-negative eval manifest CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=ROOT / "03_experiments" / "detector_datasets",
        help="Parent detector-datasets directory or the final dataset root.",
    )
    parser.add_argument("--dataset-name", default="fast_seminar_v1", help="Detector dataset directory name.")
    parser.add_argument(
        "--max-donor-crops",
        type=nonnegative_int,
        default=2000,
        help="Maximum donor crops to include, balanced by fire/smoke if possible.",
    )
    parser.add_argument("--seed", type=int, default=20260507, help="Deterministic split and sampling seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing generated dataset directory.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        summary = run_build_fast_detector_dataset(
            donor_manifest=resolve_cli_path(args.donor_manifest, ROOT),
            synth_manifest=resolve_cli_path(args.synth_manifest, ROOT),
            background_train=resolve_cli_path(args.background_train, ROOT),
            background_val=resolve_cli_path(args.background_val, ROOT),
            hard_negative=resolve_cli_path(args.hard_negative, ROOT),
            output_dir=resolve_cli_path(args.output_dir, ROOT),
            dataset_name=args.dataset_name,
            max_donor_crops=args.max_donor_crops,
            seed=args.seed,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS canonical dataset root: {summary.dataset_root.resolve()}")
    print(f"PASS train images: {summary.train_images}")
    print(f"PASS val images: {summary.val_images}")
    print(f"PASS eval synth val images: {summary.eval_synth_val_images}")
    print(f"PASS eval hard-negative images: {summary.eval_hard_negative_images}")
    print_counter("counts by split", summary.split_counts)
    print_counter("counts by source type", summary.source_type_counts)
    print_counter("counts by source", summary.source_name_counts)
    print_counter("counts by class", summary.class_counts)
    print_counter("negative image count by split", summary.negative_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

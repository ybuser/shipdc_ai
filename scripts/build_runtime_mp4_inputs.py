"""Build four normalized MP4 inputs for the fast seminar NVR emulator."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CSV_MANIFEST_ENCODING = "utf-8-sig"
FRAME_SIZE = (640, 360)
CONFIG_RELATIVE_PATH = Path("configs") / "nvr_4stream_fast.json"

SYNTH_REQUIRED_COLUMNS = {"synth_id", "image_path", "class_name", "background_zone_l1", "notes"}
BACKGROUND_REQUIRED_COLUMNS = {
    "frame_id",
    "asset_id",
    "source_name",
    "frame_path",
    "split",
    "quality_flag",
    "zone_l1",
    "reviewer_notes",
}

STREAM_MANIFEST_HEADER = [
    "channel_id",
    "display_name",
    "expected_role",
    "zone_l1",
    "frame_index",
    "sequence_frame_path",
    "mp4_path",
    "source_manifest",
    "source_row_id",
    "source_image_path",
    "notes",
]


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    display_name: str
    expected_role: str
    zone_l1: str
    source_manifest: str
    source_id_field: str
    image_field: str
    rows: list[dict[str, str]]
    mp4_name: str


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


def load_pillow_modules() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required only for scripts/build_runtime_mp4_inputs.py in the experiment environment. "
            "Install it in .venv_train or .venv_synth; do not add Pillow to project runtime dependencies."
        ) from error
    return Image, ImageOps


def has_with_human_note(row: dict[str, str]) -> bool:
    text = " ".join(
        [
            row.get("reviewer_notes", ""),
            row.get("review_note", ""),
            row.get("notes", ""),
        ]
    ).lower()
    return "with_human" in text


def prefer_non_human(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    non_human = [row for row in rows if not has_with_human_note(row)]
    return non_human if non_human else rows


def good_rows(
    rows: list[dict[str, str]],
    *,
    split: str | None,
    image_field: str,
    require_quality_good: bool = True,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if split is not None and row.get("split", "").strip().lower() != split:
            continue
        quality = row.get("quality_flag", "").strip().lower()
        if require_quality_good and quality and quality != "good":
            continue
        if not row.get(image_field, "").strip():
            continue
        selected.append(row)
    return selected


def cycle_rows(rows: list[dict[str, str]], frame_count: int, *, id_field: str) -> list[dict[str, str]]:
    if not rows:
        raise RuntimeError("no eligible rows available for runtime stream")
    ordered = sorted(rows, key=lambda row: row.get(id_field, ""))
    return [ordered[index % len(ordered)] for index in range(frame_count)]


def safe_clear_output_dir(output_dir: Path, root: Path) -> None:
    if not output_dir.exists():
        return
    allowed_parent = (root / "03_experiments").resolve()
    resolved_target = output_dir.resolve()
    try:
        resolved_target.relative_to(allowed_parent)
    except ValueError as error:
        raise ValueError(f"refusing to overwrite runtime inputs outside {allowed_parent}: {output_dir}") from error
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def assert_outputs_writable(output_dir: Path, *, config_path: Path, overwrite: bool, root: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"runtime output directory already exists; pass --overwrite: {output_dir}")
        safe_clear_output_dir(output_dir, root)
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"runtime config already exists; pass --overwrite: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_normalized_frame(
    *,
    source_path: Path,
    destination_path: Path,
    image_module: Any,
    image_ops_module: Any,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"source image does not exist: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with image_module.open(source_path) as opened_image:
        image = image_ops_module.exif_transpose(opened_image).convert("RGB")
        normalized = image_ops_module.pad(image, FRAME_SIZE, color=(0, 0, 0), centering=(0.5, 0.5))
        normalized.save(destination_path, format="JPEG", quality=88)


def build_mp4(*, ffmpeg: str, fps: float, sequence_dir: Path, output_mp4: Path, overwrite: bool) -> None:
    if output_mp4.exists() and not overwrite:
        raise FileExistsError(f"MP4 already exists; pass --overwrite: {output_mp4}")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        ffmpeg,
        overwrite_flag,
        "-framerate",
        f"{fps:g}",
        "-i",
        str(sequence_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:g}",
        str(output_mp4),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg failed for {output_mp4}: {message}")


def write_streams_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=STREAM_MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_config(
    *,
    channels: list[ChannelSpec],
    output_dir: Path,
    fps: float,
    seconds: float,
    config_path: Path,
    root: Path,
) -> None:
    config_channels = []
    for channel in channels:
        mp4_path = output_dir / channel.mp4_name
        config_channels.append(
            {
                "channel_id": channel.channel_id,
                "display_name": channel.display_name,
                "source_path": repo_relative(mp4_path, root),
                "expected_role": channel.expected_role,
                "zone_l1": channel.zone_l1,
                "fps": fps,
                "enabled": True,
            }
        )
    config = {
        "name": "fast_4stream_public_proxy_emulator",
        "generated_by": "scripts/build_runtime_mp4_inputs.py",
        "public_data_only": True,
        "emulator_note": "Four local MP4 files emulate NVR inputs for a preliminary seminar experiment; this is not real CCTV or RTSP.",
        "seconds": seconds,
        "channels": config_channels,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def build_channel_sequence(
    *,
    channel: ChannelSpec,
    output_dir: Path,
    fps: float,
    frame_count: int,
    ffmpeg: str,
    overwrite: bool,
    root: Path,
    image_module: Any,
    image_ops_module: Any,
) -> list[dict[str, str]]:
    selected_rows = cycle_rows(channel.rows, frame_count, id_field=channel.source_id_field)
    sequence_dir = output_dir / "sequences" / channel.channel_id
    mp4_path = output_dir / channel.mp4_name
    manifest_rows: list[dict[str, str]] = []
    for frame_index, row in enumerate(selected_rows, start=1):
        sequence_path = sequence_dir / f"frame_{frame_index:06d}.jpg"
        source_path_text = row.get(channel.image_field, "").strip()
        source_path = resolve_data_path(source_path_text, root, field_name=channel.image_field)
        write_normalized_frame(
            source_path=source_path,
            destination_path=sequence_path,
            image_module=image_module,
            image_ops_module=image_ops_module,
        )
        manifest_rows.append(
            {
                "channel_id": channel.channel_id,
                "display_name": channel.display_name,
                "expected_role": channel.expected_role,
                "zone_l1": channel.zone_l1,
                "frame_index": str(frame_index),
                "sequence_frame_path": repo_relative(sequence_path, root),
                "mp4_path": repo_relative(mp4_path, root),
                "source_manifest": channel.source_manifest,
                "source_row_id": row.get(channel.source_id_field, "").strip(),
                "source_image_path": source_path_text.replace("\\", "/"),
                "notes": "public_data_only; normalized_to_640x360",
            }
        )
    build_mp4(ffmpeg=ffmpeg, fps=fps, sequence_dir=sequence_dir, output_mp4=mp4_path, overwrite=overwrite)
    return manifest_rows


def run_build_runtime_mp4_inputs(
    *,
    synth_manifest: Path,
    background_train: Path,
    background_val: Path,
    hard_negative: Path,
    output_dir: Path,
    ffmpeg: str,
    fps: float,
    seconds: float,
    overwrite: bool = False,
    root: Path = ROOT,
) -> tuple[Path, Path, int]:
    frame_count = max(1, int(round(fps * seconds)))
    config_path = root / CONFIG_RELATIVE_PATH
    assert_outputs_writable(output_dir, config_path=config_path, overwrite=overwrite, root=root)

    synth_rows = prefer_non_human(
        good_rows(
            read_csv_rows(synth_manifest, required_columns=SYNTH_REQUIRED_COLUMNS, source_name="synthetic manifest"),
            split=None,
            image_field="image_path",
            require_quality_good=False,
        )
    )
    train_background_rows = prefer_non_human(
        good_rows(
            read_csv_rows(
                background_train,
                required_columns=BACKGROUND_REQUIRED_COLUMNS,
                source_name="background train manifest",
            ),
            split="train",
            image_field="frame_path",
        )
    )
    engine_rows = [row for row in train_background_rows if row.get("zone_l1", "").strip().lower() == "engine_room"]
    if engine_rows:
        train_background_rows = engine_rows
    hard_negative_rows = prefer_non_human(
        good_rows(
            read_csv_rows(hard_negative, required_columns=BACKGROUND_REQUIRED_COLUMNS, source_name="hard-negative manifest"),
            split=None,
            image_field="frame_path",
        )
    )
    val_background_rows = prefer_non_human(
        good_rows(
            read_csv_rows(
                background_val,
                required_columns=BACKGROUND_REQUIRED_COLUMNS,
                source_name="background val manifest",
            ),
            split="val",
            image_field="frame_path",
        )
    )

    channels = [
        ChannelSpec(
            channel_id="ch01",
            display_name="Synthetic positive proxy",
            expected_role="synth_positive",
            zone_l1="engine_room",
            source_manifest=repo_relative(synth_manifest, root),
            source_id_field="synth_id",
            image_field="image_path",
            rows=synth_rows,
            mp4_name="ch01_synth_positive.mp4",
        ),
        ChannelSpec(
            channel_id="ch02",
            display_name="Engine-room background",
            expected_role="engine_background",
            zone_l1="engine_room",
            source_manifest=repo_relative(background_train, root),
            source_id_field="frame_id",
            image_field="frame_path",
            rows=train_background_rows,
            mp4_name="ch02_engine_background.mp4",
        ),
        ChannelSpec(
            channel_id="ch03",
            display_name="Hard-negative confuser",
            expected_role="hard_negative_confuser",
            zone_l1="ship_like_eval",
            source_manifest=repo_relative(hard_negative, root),
            source_id_field="frame_id",
            image_field="frame_path",
            rows=hard_negative_rows,
            mp4_name="ch03_hard_negative_confuser.mp4",
        ),
        ChannelSpec(
            channel_id="ch04",
            display_name="Validation background",
            expected_role="val_background",
            zone_l1="ship_like_background",
            source_manifest=repo_relative(background_val, root),
            source_id_field="frame_id",
            image_field="frame_path",
            rows=val_background_rows,
            mp4_name="ch04_val_background.mp4",
        ),
    ]

    image_module, image_ops_module = load_pillow_modules()
    manifest_rows: list[dict[str, str]] = []
    for channel in channels:
        manifest_rows.extend(
            build_channel_sequence(
                channel=channel,
                output_dir=output_dir,
                fps=fps,
                frame_count=frame_count,
                ffmpeg=ffmpeg,
                overwrite=overwrite,
                root=root,
                image_module=image_module,
                image_ops_module=image_ops_module,
            )
        )

    streams_manifest = output_dir / "runtime_streams_manifest.csv"
    write_streams_manifest(streams_manifest, manifest_rows)
    write_runtime_config(
        channels=channels,
        output_dir=output_dir,
        fps=fps,
        seconds=seconds,
        config_path=config_path,
        root=root,
    )
    return streams_manifest, config_path, len(manifest_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=ROOT / "03_experiments" / "runtime_inputs" / "fast_4stream",
        help="Output directory for JPEG sequences and MP4 streams.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path or command name.")
    parser.add_argument("--fps", type=positive_float, default=5.0, help="Output MP4 frame rate.")
    parser.add_argument("--seconds", type=positive_float, default=20.0, help="Duration for each MP4 stream.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated runtime inputs and config.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        streams_manifest, config_path, row_count = run_build_runtime_mp4_inputs(
            synth_manifest=resolve_cli_path(args.synth_manifest, ROOT),
            background_train=resolve_cli_path(args.background_train, ROOT),
            background_val=resolve_cli_path(args.background_val, ROOT),
            hard_negative=resolve_cli_path(args.hard_negative, ROOT),
            output_dir=resolve_cli_path(args.output_dir, ROOT),
            ffmpeg=args.ffmpeg,
            fps=args.fps,
            seconds=args.seconds,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS runtime stream manifest rows: {row_count}")
    print(f"PASS runtime streams manifest: {streams_manifest}")
    print(f"PASS NVR config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

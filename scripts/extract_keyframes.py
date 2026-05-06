"""Extract ship-like keyframe candidates from manifested public assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shipdc_ai.manifest import validate_manifest_header  # noqa: E402


KEYFRAME_MANIFEST_HEADER = [
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

CHUNK_SIZE = 1024 * 1024
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


@dataclass(frozen=True)
class ExtractionSummary:
    selected_assets: int
    processed_assets: int
    skipped_assets: int
    written_rows: int
    output_manifest: Path


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def sanitize_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def batch_from_download_date(value: str) -> str:
    stripped = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return stripped.replace("-", "")
    if re.fullmatch(r"\d{8}", stripped):
        return stripped
    return "unbatched"


def repo_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_asset_path(local_path: str, root: Path) -> Path:
    path = Path(local_path)
    if path.is_absolute():
        return path
    return root / path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def format_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def read_master_manifest(path: Path) -> list[dict[str, str]]:
    errors = validate_manifest_header(path)
    if errors:
        raise ValueError("; ".join(errors))
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def select_assets(rows: list[dict[str, str]], prefixes: list[str]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        asset_id = row.get("asset_id", "")
        asset_type = row.get("asset_type", "").strip().lower()
        if asset_type not in {"video", "image"}:
            continue
        if any(asset_id.startswith(prefix) for prefix in prefixes):
            selected.append(row)
    return selected


def read_existing_keyframe_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if fieldnames != KEYFRAME_MANIFEST_HEADER:
            raise ValueError(f"keyframe manifest header mismatch: found {fieldnames!r}")
        return list(reader)


def write_keyframe_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=KEYFRAME_MANIFEST_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("asset_id", ""), row.get("frame_path", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({column: row.get(column, "") for column in KEYFRAME_MANIFEST_HEADER})
    return deduped


def merge_keyframe_rows(
    existing_rows: list[dict[str, str]],
    processed_asset_ids: set[str],
    new_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    preserved = [row for row in existing_rows if row.get("asset_id", "") not in processed_asset_ids]
    merged = preserved + new_rows
    return dedupe_rows(merged)


def asset_output_dir(row: dict[str, str], output_root: Path) -> Path:
    source = sanitize_segment(row.get("source_name", ""), "source_unknown")
    batch = batch_from_download_date(row.get("download_date", ""))
    asset_id = sanitize_segment(row.get("asset_id", ""), "asset_unknown")
    return output_root / source / batch / asset_id


def dimension_fields(path: Path) -> tuple[str, str]:
    width, height = read_image_dimensions(path)
    return (str(width) if width is not None else "", str(height) if height is not None else "")


def build_manifest_row(
    *,
    source_row: dict[str, str],
    source_path: Path,
    frame_path: Path,
    frame_index: int,
    timestamp_s: str,
    extraction_fps: str,
    notes: str,
    root: Path,
) -> dict[str, str]:
    width, height = dimension_fields(frame_path)
    return {
        "asset_id": source_row.get("asset_id", ""),
        "source_name": source_row.get("source_name", ""),
        "source_asset_path": repo_relative(source_path, root),
        "frame_path": repo_relative(frame_path, root),
        "timestamp_s": timestamp_s,
        "frame_index": str(frame_index),
        "width": width,
        "height": height,
        "extraction_fps": extraction_fps,
        "sha256": hash_file(frame_path),
        "notes": notes,
    }


def copy_still_candidate(
    row: dict[str, str],
    *,
    output_root: Path,
    overwrite: bool,
    root: Path,
) -> tuple[list[dict[str, str]], bool]:
    source_path = resolve_asset_path(row.get("local_path", ""), root)
    if not source_path.is_file():
        raise FileNotFoundError(f"source image not found for {row.get('asset_id', '')}: {source_path}")

    target_dir = asset_output_dir(row, output_root)
    existing_stills = sorted(target_dir.glob("still_000000.*")) if target_dir.exists() else []
    if existing_stills and not overwrite:
        print(f"SKIP existing still candidate: {row.get('asset_id', '')}")
        return [], False

    if overwrite:
        for candidate in existing_stills:
            candidate.unlink()

    suffix = source_path.suffix.lower() or ".image"
    frame_path = target_dir / f"still_000000{suffix}"
    if frame_path.exists() and not overwrite:
        print(f"SKIP existing still candidate: {row.get('asset_id', '')}")
        return [], False

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, frame_path)
    manifest_row = build_manifest_row(
        source_row=row,
        source_path=source_path,
        frame_path=frame_path,
        frame_index=0,
        timestamp_s="",
        extraction_fps="",
        notes="still_image_candidate; not_video_keyframe",
        root=root,
    )
    return [manifest_row], True


def resolve_ffmpeg(provided: Path | None) -> Path | None:
    if provided is not None:
        return provided if provided.is_file() else None
    discovered = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    return Path(discovered) if discovered else None


def parse_frame_index(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d+)\.jpg", path.name)
    if not match:
        raise ValueError(f"unexpected frame filename: {path.name}")
    return int(match.group(1))


def extract_video_keyframes(
    row: dict[str, str],
    *,
    output_root: Path,
    ffmpeg_path: Path,
    fps: float,
    max_per_asset: int | None,
    overwrite: bool,
    root: Path,
) -> tuple[list[dict[str, str]], bool]:
    source_path = resolve_asset_path(row.get("local_path", ""), root)
    if not source_path.is_file():
        raise FileNotFoundError(f"source video not found for {row.get('asset_id', '')}: {source_path}")

    target_dir = asset_output_dir(row, output_root)
    existing_frames = sorted(target_dir.glob("frame_*.jpg")) if target_dir.exists() else []
    if existing_frames and not overwrite:
        print(f"SKIP existing extracted frames: {row.get('asset_id', '')}")
        return [], False

    if overwrite:
        for frame in existing_frames:
            frame.unlink()

    target_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = target_dir / "frame_%06d.jpg"
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-vf",
        f"fps={format_number(fps)}",
    ]
    if max_per_asset is not None:
        command.extend(["-frames:v", str(max_per_asset)])
    command.extend(["-start_number", "0", str(output_pattern)])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg extraction failed for {row.get('asset_id', '')}: {message}")

    generated_frames = sorted(target_dir.glob("frame_*.jpg"))
    if not generated_frames:
        raise RuntimeError(f"ffmpeg produced no frames for {row.get('asset_id', '')}")

    extraction_fps = format_number(fps)
    manifest_rows: list[dict[str, str]] = []
    for frame_path in generated_frames:
        frame_index = parse_frame_index(frame_path)
        timestamp_s = format_number(frame_index / fps)
        manifest_rows.append(
            build_manifest_row(
                source_row=row,
                source_path=source_path,
                frame_path=frame_path,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                extraction_fps=extraction_fps,
                notes="video_keyframe",
                root=root,
            )
        )
    return manifest_rows, True


def run_extraction(
    *,
    manifest_path: Path,
    output_root: Path,
    output_manifest: Path,
    asset_prefixes: list[str],
    fps: float = 1.0,
    max_per_asset: int | None = None,
    ffmpeg: Path | None = None,
    overwrite: bool = False,
    root: Path = ROOT,
) -> ExtractionSummary:
    rows = read_master_manifest(manifest_path)
    selected_rows = select_assets(rows, asset_prefixes)
    video_rows = [row for row in selected_rows if row.get("asset_type", "").strip().lower() == "video"]
    ffmpeg_path = resolve_ffmpeg(ffmpeg) if video_rows else None
    if video_rows and ffmpeg_path is None:
        raise RuntimeError("ffmpeg.exe is required for selected video assets; provide --ffmpeg or add it to PATH")

    existing_rows = read_existing_keyframe_rows(output_manifest)
    new_rows: list[dict[str, str]] = []
    processed_asset_ids: set[str] = set()
    skipped_assets = 0

    for row in selected_rows:
        asset_type = row.get("asset_type", "").strip().lower()
        if asset_type == "image":
            rows_for_asset, processed = copy_still_candidate(
                row,
                output_root=output_root,
                overwrite=overwrite,
                root=root,
            )
        elif asset_type == "video":
            if ffmpeg_path is None:
                raise RuntimeError("ffmpeg.exe is required for selected video assets")
            rows_for_asset, processed = extract_video_keyframes(
                row,
                output_root=output_root,
                ffmpeg_path=ffmpeg_path,
                fps=fps,
                max_per_asset=max_per_asset,
                overwrite=overwrite,
                root=root,
            )
        else:
            continue

        if processed:
            processed_asset_ids.add(row.get("asset_id", ""))
            new_rows.extend(rows_for_asset)
        else:
            skipped_assets += 1

    if processed_asset_ids:
        merged_rows = merge_keyframe_rows(existing_rows, processed_asset_ids, new_rows)
        write_keyframe_manifest(output_manifest, merged_rows)

    return ExtractionSummary(
        selected_assets=len(selected_rows),
        processed_assets=len(processed_asset_ids),
        skipped_assets=skipped_assets,
        written_rows=len(new_rows),
        output_manifest=output_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "master_manifest.csv",
        help="Master manifest CSV path.",
    )
    parser.add_argument(
        "--asset-prefix",
        action="append",
        required=True,
        help="Asset ID prefix to select. Repeat for multiple prefixes, for example SHIP_DVIDS.",
    )
    parser.add_argument("--fps", type=positive_float, default=1.0, help="Video extraction FPS. Default: 1.")
    parser.add_argument("--max-per-asset", type=positive_int, help="Maximum frames to extract per video asset.")
    parser.add_argument("--ffmpeg", type=Path, help="Path to ffmpeg.exe. Defaults to ffmpeg on PATH.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames for selected assets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "02_processed" / "frames" / "ship_like",
        help="Root directory for extracted frames.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_keyframes.csv",
        help="Derived keyframe manifest CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_extraction(
            manifest_path=resolve_cli_path(args.manifest, ROOT),
            output_root=resolve_cli_path(args.output_root, ROOT),
            output_manifest=resolve_cli_path(args.output_manifest, ROOT),
            asset_prefixes=args.asset_prefix,
            fps=args.fps,
            max_per_asset=args.max_per_asset,
            ffmpeg=resolve_cli_path(args.ffmpeg, ROOT) if args.ffmpeg else None,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS selected assets: {summary.selected_assets}")
    print(f"PASS processed assets: {summary.processed_assets}")
    print(f"PASS skipped assets: {summary.skipped_assets}")
    print(f"PASS new manifest rows: {summary.written_rows}")
    print(f"PASS output manifest: {summary.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the fast four-stream MP4 NVR emulator with an Ultralytics YOLO model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_MODE = "sequential_channel_emulated_4stream"


@dataclass(frozen=True)
class RuntimeChannel:
    channel_id: str
    display_name: str
    source_path: Path
    expected_role: str
    zone_l1: str
    fps: float | None = None


@dataclass
class ChannelRunResult:
    metrics: dict[str, str]
    events: list[dict[str, str]]


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


def load_yolo_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required only in .venv_train for scripts/run_fast_4stream_nvr.py. "
            "Do not add Ultralytics or Torch to project runtime dependencies."
        ) from error
    return YOLO(str(model_path))


def load_channels(config_path: Path, *, root: Path) -> list[RuntimeChannel]:
    if not config_path.is_file():
        raise FileNotFoundError(f"NVR config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("NVR config root must be a JSON object")

    raw_channels = config.get("channels")
    if raw_channels is None:
        raw_channels = config.get("nvr", {}).get("cameras", [])
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ValueError("NVR config must contain a non-empty channels list")

    channels: list[RuntimeChannel] = []
    for index, raw_channel in enumerate(raw_channels, start=1):
        if not isinstance(raw_channel, dict):
            raise ValueError(f"channel {index} must be an object")
        if raw_channel.get("enabled", True) is False:
            continue
        channel_id = str(raw_channel.get("channel_id") or raw_channel.get("camera_id") or f"ch{index:02d}").strip()
        display_name = str(raw_channel.get("display_name") or channel_id).strip()
        source_value = str(raw_channel.get("source_path") or raw_channel.get("mp4_source_path") or "").strip()
        if not source_value:
            raise ValueError(f"channel {channel_id}: source_path is required")
        source_path = resolve_data_path(source_value, root, field_name="source_path")
        expected_role = str(raw_channel.get("expected_role") or raw_channel.get("zone_type") or "unknown").strip()
        zone_l1 = str(raw_channel.get("zone_l1") or raw_channel.get("zone_type") or "unknown").strip()
        raw_fps = raw_channel.get("fps") or raw_channel.get("fps_hint")
        fps = float(raw_fps) if raw_fps is not None else None
        channels.append(
            RuntimeChannel(
                channel_id=channel_id,
                display_name=display_name,
                source_path=source_path,
                expected_role=expected_role,
                zone_l1=zone_l1,
                fps=fps,
            )
        )
    if not channels:
        raise ValueError("NVR config has no enabled channels")
    return channels


def assert_run_dir_writable(run_dir: Path, *, overwrite: bool) -> None:
    known_outputs = [
        run_dir / "metrics.csv",
        run_dir / "channel_metrics.csv",
        run_dir / "events.csv",
    ]
    snapshot_dir = run_dir / "snapshots"
    if not overwrite:
        existing = [path for path in known_outputs if path.exists()]
        if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            existing.append(snapshot_dir)
        if existing:
            raise FileExistsError(f"run outputs already exist; pass --overwrite: {existing[0]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)


def result_confidences(result: Any, *, conf_threshold: float) -> list[float]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    raw_conf = getattr(boxes, "conf", None)
    if raw_conf is None:
        try:
            return [1.0] * len(boxes)
        except TypeError:
            return []
    try:
        values = raw_conf.tolist()
    except AttributeError:
        values = list(raw_conf)
    return [float(value) for value in values if float(value) >= conf_threshold]


def save_snapshot(result: Any, snapshot_path: Path) -> str:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.save(filename=str(snapshot_path))
        return str(snapshot_path)
    except Exception:
        return ""


def prediction_stream(model: Any, *, channel: RuntimeChannel, conf: float, device: str, vid_stride: int) -> Any:
    if not channel.source_path.is_file():
        raise FileNotFoundError(f"MP4 source does not exist for {channel.channel_id}: {channel.source_path}")
    return model.predict(
        source=str(channel.source_path),
        stream=True,
        conf=conf,
        device=device,
        vid_stride=vid_stride,
        verbose=False,
    )


def run_channel(
    *,
    model: Any,
    channel: RuntimeChannel,
    run_dir: Path,
    sample_fps: float,
    conf: float,
    temporal_k: int,
    temporal_window_s: float,
    device: str,
    root: Path,
) -> ChannelRunResult:
    source_fps = channel.fps or sample_fps
    vid_stride = max(1, int(round(source_fps / sample_fps))) if source_fps > 0 else 1
    snapshots_dir = run_dir / "snapshots"
    observations: deque[tuple[float, float]] = deque()
    alarm_active = False
    first_suspect_saved = False
    first_alarm_saved = False
    first_alarm_timestamp = ""
    frames_processed = 0
    predicted_frames = 0
    total_pred_boxes = 0
    alarm_count = 0
    events: list[dict[str, str]] = []

    started = time.perf_counter()
    for result in prediction_stream(model, channel=channel, conf=conf, device=device, vid_stride=vid_stride):
        frames_processed += 1
        timestamp_s = (frames_processed - 1) / sample_fps
        while observations and observations[0][0] < timestamp_s - temporal_window_s:
            observations.popleft()

        confidences = result_confidences(result, conf_threshold=conf)
        max_confidence = max(confidences) if confidences else 0.0
        snapshot_path = ""
        if confidences:
            predicted_frames += 1
            total_pred_boxes += len(confidences)
            observations.append((timestamp_s, max_confidence))
            if not first_suspect_saved:
                suspect_path = snapshots_dir / f"{channel.channel_id}_first_suspect.jpg"
                snapshot_path = save_snapshot(result, suspect_path)
                first_suspect_saved = bool(snapshot_path)
            events.append(
                {
                    "channel_id": channel.channel_id,
                    "event_type": "suspect",
                    "frame_index": str(frames_processed),
                    "timestamp_s": f"{timestamp_s:.3f}",
                    "confidence": f"{max_confidence:.6f}",
                    "boxes_in_frame": str(len(confidences)),
                    "window_suspect_count": str(len(observations)),
                    "snapshot_path": repo_relative(Path(snapshot_path), root) if snapshot_path else "",
                    "notes": "prediction_above_conf",
                }
            )

        if len(observations) >= temporal_k:
            if not alarm_active:
                alarm_count += 1
                if not first_alarm_timestamp:
                    first_alarm_timestamp = f"{timestamp_s:.3f}"
                alarm_snapshot_path = ""
                if not first_alarm_saved:
                    alarm_path = snapshots_dir / f"{channel.channel_id}_first_alarm.jpg"
                    alarm_snapshot_path = save_snapshot(result, alarm_path)
                    first_alarm_saved = bool(alarm_snapshot_path)
                events.append(
                    {
                        "channel_id": channel.channel_id,
                        "event_type": "alarm",
                        "frame_index": str(frames_processed),
                        "timestamp_s": f"{timestamp_s:.3f}",
                        "confidence": f"{max_confidence:.6f}",
                        "boxes_in_frame": str(len(confidences)),
                        "window_suspect_count": str(len(observations)),
                        "snapshot_path": repo_relative(Path(alarm_snapshot_path), root) if alarm_snapshot_path else "",
                        "notes": f"temporal_k_in_{temporal_window_s:g}s_window",
                    }
                )
                alarm_active = True
        else:
            alarm_active = False

    elapsed = time.perf_counter() - started
    analyzed_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    metrics = {
        "channel_id": channel.channel_id,
        "display_name": channel.display_name,
        "expected_role": channel.expected_role,
        "zone_l1": channel.zone_l1,
        "source_path": repo_relative(channel.source_path, root),
        "frames_processed": str(frames_processed),
        "elapsed_wall_clock_s": f"{elapsed:.6f}",
        "analyzed_fps": f"{analyzed_fps:.6f}",
        "predicted_frames": str(predicted_frames),
        "total_pred_boxes": str(total_pred_boxes),
        "alarm_count": str(alarm_count),
        "first_alarm_timestamp_s": first_alarm_timestamp,
        "vid_stride": str(vid_stride),
        "processing_mode": PROCESSING_MODE,
    }
    return ChannelRunResult(metrics=metrics, events=events)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_fast_4stream_nvr(
    *,
    config: Path,
    model_path: Path,
    run_dir: Path,
    sample_fps: float,
    conf: float,
    temporal_k: int,
    temporal_window_s: float,
    device: str,
    overwrite: bool = False,
    root: Path = ROOT,
) -> tuple[Path, Path, Path]:
    if not model_path.is_file():
        raise FileNotFoundError(f"model does not exist: {model_path}")
    assert_run_dir_writable(run_dir, overwrite=overwrite)
    channels = load_channels(config, root=root)
    model = load_yolo_model(model_path)

    started = time.perf_counter()
    channel_rows: list[dict[str, str]] = []
    event_rows: list[dict[str, str]] = []
    for channel in channels:
        result = run_channel(
            model=model,
            channel=channel,
            run_dir=run_dir,
            sample_fps=sample_fps,
            conf=conf,
            temporal_k=temporal_k,
            temporal_window_s=temporal_window_s,
            device=device,
            root=root,
        )
        channel_rows.append(result.metrics)
        event_rows.extend(result.events)
    elapsed = time.perf_counter() - started

    for event_id, row in enumerate(event_rows, start=1):
        row["event_id"] = f"event_{event_id:06d}"

    frames_processed = sum(int(row["frames_processed"]) for row in channel_rows)
    predicted_frames = sum(int(row["predicted_frames"]) for row in channel_rows)
    total_pred_boxes = sum(int(row["total_pred_boxes"]) for row in channel_rows)
    alarm_count = sum(int(row["alarm_count"]) for row in channel_rows)
    analyzed_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    metrics_rows = [
        {
            "config_path": repo_relative(config, root),
            "model_path": str(model_path),
            "channel_count": str(len(channels)),
            "elapsed_wall_clock_s": f"{elapsed:.6f}",
            "frames_processed": str(frames_processed),
            "analyzed_fps": f"{analyzed_fps:.6f}",
            "predicted_frames": str(predicted_frames),
            "total_pred_boxes": str(total_pred_boxes),
            "alarm_count": str(alarm_count),
            "sample_fps": f"{sample_fps:.6f}",
            "conf": f"{conf:.6f}",
            "temporal_k": str(temporal_k),
            "temporal_window_s": f"{temporal_window_s:.6f}",
            "device": device,
            "processing_mode": PROCESSING_MODE,
            "notes": "public_mp4_emulator; CPU_ONNX_direction_for_paper",
        }
    ]

    metrics_path = run_dir / "metrics.csv"
    channel_metrics_path = run_dir / "channel_metrics.csv"
    events_path = run_dir / "events.csv"
    write_csv(
        metrics_path,
        [
            "config_path",
            "model_path",
            "channel_count",
            "elapsed_wall_clock_s",
            "frames_processed",
            "analyzed_fps",
            "predicted_frames",
            "total_pred_boxes",
            "alarm_count",
            "sample_fps",
            "conf",
            "temporal_k",
            "temporal_window_s",
            "device",
            "processing_mode",
            "notes",
        ],
        metrics_rows,
    )
    write_csv(
        channel_metrics_path,
        [
            "channel_id",
            "display_name",
            "expected_role",
            "zone_l1",
            "source_path",
            "frames_processed",
            "elapsed_wall_clock_s",
            "analyzed_fps",
            "predicted_frames",
            "total_pred_boxes",
            "alarm_count",
            "first_alarm_timestamp_s",
            "vid_stride",
            "processing_mode",
        ],
        channel_rows,
    )
    write_csv(
        events_path,
        [
            "event_id",
            "channel_id",
            "event_type",
            "frame_index",
            "timestamp_s",
            "confidence",
            "boxes_in_frame",
            "window_suspect_count",
            "snapshot_path",
            "notes",
        ],
        event_rows,
    )
    return metrics_path, channel_metrics_path, events_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "nvr_4stream_fast.json",
        help="Fast four-stream NVR config JSON path.",
    )
    parser.add_argument("--model", type=Path, required=True, help="YOLO .pt or .onnx model path.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "03_experiments" / "runtime_logs" / "fast_4stream_run",
        help="Output run directory for metrics, events, and snapshots.",
    )
    parser.add_argument("--sample-fps", type=positive_float, default=5.0, help="Nominal analysis sample FPS.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--temporal-k", type=positive_int, default=3, help="Suspect observations needed for alarm.")
    parser.add_argument(
        "--temporal-window-s",
        type=positive_float,
        default=10.0,
        help="Temporal verifier window length in seconds.",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device argument; default keeps CPU direction.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite known run output files.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        metrics_path, channel_metrics_path, events_path = run_fast_4stream_nvr(
            config=resolve_cli_path(args.config, ROOT),
            model_path=resolve_cli_path(args.model, ROOT),
            run_dir=resolve_cli_path(args.run_dir, ROOT),
            sample_fps=args.sample_fps,
            conf=args.conf,
            temporal_k=args.temporal_k,
            temporal_window_s=args.temporal_window_s,
            device=args.device,
            overwrite=args.overwrite,
            root=ROOT,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS metrics: {metrics_path}")
    print(f"PASS channel metrics: {channel_metrics_path}")
    print(f"PASS events: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

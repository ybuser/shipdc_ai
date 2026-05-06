"""Summarize a fast four-stream NVR emulator run for seminar tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE6_CSV = "table6_runtime_results.csv"
SUMMARY_TXT = "runtime_summary_for_paper.txt"


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        reader.fieldnames = [(fieldname or "").strip().lstrip("\ufeff").strip() for fieldname in reader.fieldnames or []]
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table_rows(metrics: dict[str, str], channel_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        {
            "scope": "aggregate",
            "channel_id": "all",
            "display_name": "All four MP4 emulator streams",
            "expected_role": "mixed_public_proxy",
            "frames_processed": metrics.get("frames_processed", ""),
            "elapsed_wall_clock_s": metrics.get("elapsed_wall_clock_s", ""),
            "analyzed_fps": metrics.get("analyzed_fps", ""),
            "predicted_frames": metrics.get("predicted_frames", ""),
            "total_pred_boxes": metrics.get("total_pred_boxes", ""),
            "alarm_count": metrics.get("alarm_count", ""),
            "first_alarm_timestamp_s": "",
            "processing_mode": metrics.get("processing_mode", ""),
            "notes": metrics.get("notes", ""),
        }
    ]
    for row in channel_rows:
        rows.append(
            {
                "scope": "channel",
                "channel_id": row.get("channel_id", ""),
                "display_name": row.get("display_name", ""),
                "expected_role": row.get("expected_role", ""),
                "frames_processed": row.get("frames_processed", ""),
                "elapsed_wall_clock_s": row.get("elapsed_wall_clock_s", ""),
                "analyzed_fps": row.get("analyzed_fps", ""),
                "predicted_frames": row.get("predicted_frames", ""),
                "total_pred_boxes": row.get("total_pred_boxes", ""),
                "alarm_count": row.get("alarm_count", ""),
                "first_alarm_timestamp_s": row.get("first_alarm_timestamp_s", ""),
                "processing_mode": row.get("processing_mode", ""),
                "notes": "public_mp4_channel",
            }
        )
    return rows


def write_summary_txt(path: Path, metrics: dict[str, str], channel_rows: list[dict[str, str]]) -> None:
    lines = [
        "Runtime summary for seminar paper",
        "",
        f"Processing mode: {metrics.get('processing_mode', 'unknown')}",
        f"Channels summarized: {metrics.get('channel_count', str(len(channel_rows)))}",
        f"Frames processed: {metrics.get('frames_processed', '')}",
        f"Elapsed wall-clock seconds: {metrics.get('elapsed_wall_clock_s', '')}",
        f"Analyzed FPS: {metrics.get('analyzed_fps', '')}",
        f"Predicted frames: {metrics.get('predicted_frames', '')}",
        f"Alarm count: {metrics.get('alarm_count', '')}",
        "",
        "Interpretation boundary:",
        "- Four local MP4 files emulate an NVR-style input set; this is not real CCTV, RTSP, or operational shipboard validation.",
        "- The runtime direction reported in the paper remains CPU-only ONNX.",
        "- These generated run outputs live under experiment/log paths and should not be committed.",
        "- Results are preliminary public-data-only feasibility evidence, not a detector benchmark claim.",
        "",
        "Channel details:",
    ]
    for row in channel_rows:
        lines.append(
            f"- {row.get('channel_id', '')}: role={row.get('expected_role', '')}, "
            f"frames={row.get('frames_processed', '')}, analyzed_fps={row.get('analyzed_fps', '')}, "
            f"alarms={row.get('alarm_count', '')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(*, run_dir: Path, out_dir: Path) -> tuple[Path, Path, int]:
    metrics_rows = read_csv_rows(run_dir / "metrics.csv")
    if not metrics_rows:
        raise ValueError(f"metrics.csv has no data rows: {run_dir / 'metrics.csv'}")
    channel_rows = read_csv_rows(run_dir / "channel_metrics.csv")
    metrics = metrics_rows[0]
    output_rows = table_rows(metrics, channel_rows)
    table_path = out_dir / TABLE6_CSV
    summary_path = out_dir / SUMMARY_TXT
    write_csv(
        table_path,
        [
            "scope",
            "channel_id",
            "display_name",
            "expected_role",
            "frames_processed",
            "elapsed_wall_clock_s",
            "analyzed_fps",
            "predicted_frames",
            "total_pred_boxes",
            "alarm_count",
            "first_alarm_timestamp_s",
            "processing_mode",
            "notes",
        ],
        output_rows,
    )
    write_summary_txt(summary_path, metrics, channel_rows)
    return table_path, summary_path, len(output_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing metrics.csv.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "04_papers" / "seminar" / "tables",
        help="Output directory for seminar runtime table files.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        table_path, summary_path, row_count = run_summary(
            run_dir=resolve_cli_path(args.run_dir, ROOT),
            out_dir=resolve_cli_path(args.out_dir, ROOT),
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS runtime table rows: {row_count}")
    print(f"PASS table CSV: {table_path}")
    print(f"PASS paper summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

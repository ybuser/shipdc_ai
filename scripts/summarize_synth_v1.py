"""Summarize synthetic v1 manifest counts for seminar-paper tables."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_synth_v1 import SYNTH_MANIFEST_HEADER, resolve_cli_path  # noqa: E402


SUMMARY_CSV = "table4_synth_v1_summary.csv"
SUMMARY_TXT = "synth_v1_summary_for_paper.txt"


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in SYNTH_MANIFEST_HEADER if column not in fieldnames]
        if missing:
            raise ValueError(f"manifest missing required columns: {', '.join(missing)}")
        return list(reader)


def count_column(rows: list[dict[str, str]], column: str) -> Counter[str]:
    return Counter(row.get(column, "").strip() or "unknown" for row in rows)


def summary_table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = [{"section": "total", "name": "synthetic_samples", "count": str(len(rows))}]
    for section, counter in [
        ("class_name", count_column(rows, "class_name")),
        ("background_zone_l1", count_column(rows, "background_zone_l1")),
        ("background_asset_id", count_column(rows, "background_asset_id")),
        ("donor_source_name", count_column(rows, "donor_source_name")),
    ]:
        for name, count in sorted(counter.items()):
            table_rows.append({"section": section, "name": name, "count": str(count)})
    return table_rows


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "name", "count"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_table_rows(rows))


def counter_lines(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"{title}:"]
    for name, count in sorted(counter.items()):
        lines.append(f"- {name}: {count}")
    return lines


def write_summary_txt(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "Synthetic v1 summary for seminar paper",
        "",
        f"Total synthetic samples: {len(rows)}",
        "",
    ]
    lines.extend(counter_lines("Class counts", count_column(rows, "class_name")))
    lines.append("")
    lines.extend(counter_lines("Background zone counts", count_column(rows, "background_zone_l1")))
    lines.append("")
    lines.extend(counter_lines("Background asset counts", count_column(rows, "background_asset_id")))
    lines.append("")
    lines.extend(counter_lines("Donor source counts", count_column(rows, "donor_source_name")))
    lines.extend(
        [
            "",
            "Public-data-only statement: synthetic v1 uses public ship-like background frames and public fire/smoke donor crops only.",
            "Claim boundary: these samples are synthetic positive proxies for preliminary data preparation, not real shipboard positive labels and not real naval CCTV validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(*, manifest: Path, out_dir: Path) -> tuple[Path, Path, int]:
    rows = load_manifest_rows(manifest)
    csv_path = out_dir / SUMMARY_CSV
    txt_path = out_dir / SUMMARY_TXT
    write_summary_csv(csv_path, rows)
    write_summary_txt(txt_path, rows)
    return csv_path, txt_path, len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "synth_v1_manifest.csv",
        help="Synthetic v1 manifest CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "04_papers" / "seminar" / "tables",
        help="Output directory for seminar summary table files.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        csv_path, txt_path, row_count = run_summary(
            manifest=resolve_cli_path(args.manifest, ROOT),
            out_dir=resolve_cli_path(args.out_dir, ROOT),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS synthetic samples summarized: {row_count}")
    print(f"PASS table CSV: {csv_path}")
    print(f"PASS paper summary: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect public donor YOLO label sources for donor crop bank readiness."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_donor_crop_bank import (  # noqa: E402
    UnknownClassMappingError,
    class_map_from_source_config,
    count_images,
    count_labels,
    load_config,
    manifest_rows_for_source,
    read_master_manifest,
    resolve_cli_path,
    source_root_from_config,
    source_splits,
)


@dataclass(frozen=True)
class SplitInspection:
    split: str
    images_dir: Path
    labels_dir: Path
    image_count: int
    label_count: int


@dataclass
class SourceInspection:
    source_name: str
    enabled: bool
    root: Path | None = None
    manifest_rows: int = 0
    rights: list[str] = field(default_factory=list)
    splits: list[SplitInspection] = field(default_factory=list)
    class_map: dict[str, str] | None = None
    class_map_status: str = "unknown"
    notes: list[str] = field(default_factory=list)


def inspect_sources(
    *,
    manifest_path: Path,
    config_path: Path,
    selected_sources: set[str] | None = None,
    root: Path = ROOT,
) -> list[SourceInspection]:
    config = load_config(config_path)
    master_rows = read_master_manifest(manifest_path)
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("config sources must be a list")

    reports: list[SourceInspection] = []
    for source_config in raw_sources:
        if not isinstance(source_config, dict):
            raise ValueError("each source config must be an object")
        source_name = str(source_config.get("source_name", "")).strip()
        if selected_sources and source_name not in selected_sources:
            continue

        report = SourceInspection(source_name=source_name, enabled=bool(source_config.get("enabled", False)))
        rows = manifest_rows_for_source(master_rows, source_name)
        report.manifest_rows = len(rows)
        report.rights = sorted({row.get("license_or_rights", "").strip() for row in rows if row.get("license_or_rights")})
        if not rows:
            report.notes.append("not found as source_group=donor in master manifest")

        try:
            report.root = source_root_from_config(source_config, root)
            if not report.root.is_dir():
                report.notes.append(f"root missing: {report.root}")
            for split_source in source_splits(source_config, root):
                report.splits.append(
                    SplitInspection(
                        split=split_source.split,
                        images_dir=split_source.images_dir,
                        labels_dir=split_source.labels_dir,
                        image_count=count_images(split_source.images_dir),
                        label_count=count_labels(split_source.labels_dir),
                    )
                )
            if not report.splits:
                report.notes.append("no split image/label directories found")

            class_map = class_map_from_source_config(source_config, report.root)
            report.class_map = class_map
            report.class_map_status = "known"
        except UnknownClassMappingError as error:
            report.class_map_status = "unknown"
            report.notes.append(str(error))
        reports.append(report)
    return reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "master_manifest.csv",
        help="Master manifest CSV path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "donor_crop_v1.json",
        help="Donor crop config JSON path.",
    )
    parser.add_argument("--source", action="append", help="Inspect only this source_name. Repeat for multiple sources.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        reports = inspect_sources(
            manifest_path=resolve_cli_path(args.manifest, ROOT),
            config_path=resolve_cli_path(args.config, ROOT),
            selected_sources=set(args.source) if args.source else None,
            root=ROOT,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS inspected sources: {len(reports)}")
    for report in reports:
        enabled = "enabled" if report.enabled else "disabled"
        print(f"SOURCE {report.source_name} ({enabled})")
        print(f"  manifest_rows: {report.manifest_rows}")
        print(f"  rights: {', '.join(report.rights) if report.rights else 'none'}")
        print(f"  root: {report.root if report.root else 'unknown'}")
        if report.class_map_status == "known" and report.class_map:
            class_text = ", ".join(f"{key}={value}" for key, value in sorted(report.class_map.items()))
            print(f"  class_map: PASS {class_text}")
        else:
            print("  class_map: WARN unknown")
        if report.splits:
            for split in report.splits:
                print(f"  split {split.split}: images={split.image_count} labels={split.label_count}")
        else:
            print("  splits: none")
        for note in report.notes:
            print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

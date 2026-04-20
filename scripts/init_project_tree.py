"""Create the shipdc_ai scaffold directories and empty .gitkeep files."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_DIRECTORIES = [
    "00_admin/seminar_docs",
    "00_admin/accounts/mivia",
    "00_admin/notes",
    "01_raw/01_donor/DFire/downloads",
    "01_raw/01_donor/DFire/unzipped",
    "01_raw/01_donor/FIRESENSE/downloads",
    "01_raw/01_donor/FIRESENSE/unzipped",
    "01_raw/01_donor/MIVIA/LFDN/downloads",
    "01_raw/01_donor/MIVIA/LFDN/unzipped",
    "01_raw/01_donor/MIVIA/FireDetection/downloads",
    "01_raw/01_donor/MIVIA/FireDetection/unzipped",
    "01_raw/01_donor/MIVIA/SmokeDetection/downloads",
    "01_raw/01_donor/MIVIA/SmokeDetection/unzipped",
    "01_raw/02_shiplike/DVIDS/downloads",
    "01_raw/02_shiplike/DVIDS/clips",
    "01_raw/02_shiplike/DVIDS/frames",
    "01_raw/02_shiplike/NOAA/downloads",
    "01_raw/02_shiplike/NOAA/clips",
    "01_raw/02_shiplike/NOAA/images",
    "01_raw/02_shiplike/NOAA/frames",
    "01_raw/02_shiplike/NARA/downloads",
    "01_raw/02_shiplike/NARA/images",
    "02_processed/manifests",
    "02_processed/frames",
    "02_processed/labels",
    "02_processed/synth_v1",
    "02_processed/hard_negative_v1",
    "03_experiments/training",
    "03_experiments/onnx",
    "03_experiments/runtime_logs",
    "03_experiments/eval_reports",
    "04_papers/seminar_2026",
    "04_papers/figures",
    "configs",
    "docs",
    "scripts",
    "src/shipdc_ai",
    "tests",
]

GITKEEP_DIRECTORIES = [
    "00_admin/seminar_docs",
    "00_admin/accounts/mivia",
    "00_admin/notes",
    "01_raw/01_donor/DFire/downloads",
    "01_raw/01_donor/DFire/unzipped",
    "01_raw/01_donor/FIRESENSE/downloads",
    "01_raw/01_donor/FIRESENSE/unzipped",
    "01_raw/01_donor/MIVIA/LFDN/downloads",
    "01_raw/01_donor/MIVIA/LFDN/unzipped",
    "01_raw/01_donor/MIVIA/FireDetection/downloads",
    "01_raw/01_donor/MIVIA/FireDetection/unzipped",
    "01_raw/01_donor/MIVIA/SmokeDetection/downloads",
    "01_raw/01_donor/MIVIA/SmokeDetection/unzipped",
    "01_raw/02_shiplike/DVIDS/downloads",
    "01_raw/02_shiplike/DVIDS/clips",
    "01_raw/02_shiplike/DVIDS/frames",
    "01_raw/02_shiplike/NOAA/downloads",
    "01_raw/02_shiplike/NOAA/clips",
    "01_raw/02_shiplike/NOAA/images",
    "01_raw/02_shiplike/NOAA/frames",
    "01_raw/02_shiplike/NARA/downloads",
    "01_raw/02_shiplike/NARA/images",
    "02_processed/frames",
    "02_processed/labels",
    "02_processed/synth_v1",
    "02_processed/hard_negative_v1",
    "03_experiments/training",
    "03_experiments/onnx",
    "03_experiments/runtime_logs",
    "03_experiments/eval_reports",
    "04_papers/seminar_2026",
    "04_papers/figures",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_tree(root: Path, dry_run: bool = False) -> list[Path]:
    created: list[Path] = []
    for relative in REQUIRED_DIRECTORIES:
        directory = root / relative
        if not directory.exists():
            created.append(directory)
            if not dry_run:
                directory.mkdir(parents=True, exist_ok=True)

    for relative in GITKEEP_DIRECTORIES:
        keep_path = root / relative / ".gitkeep"
        if not keep_path.exists():
            created.append(keep_path)
            if not dry_run:
                keep_path.parent.mkdir(parents=True, exist_ok=True)
                keep_path.touch()
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root(),
        help="Repository root. Defaults to the parent of the scripts directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List missing items without creating them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    created = create_tree(root, dry_run=args.dry_run)
    action = "Would create" if args.dry_run else "Created"
    for path in created:
        print(f"{action}: {path}")
    if not created:
        print("Scaffold directories and .gitkeep files already exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

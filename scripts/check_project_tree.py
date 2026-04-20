"""Check that the scaffold directories and files exist."""

from __future__ import annotations

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

REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "configs/paths.example.json",
    "configs/cameras.example.json",
    "configs/temporal_rules.example.json",
    "docs/PROJECT_CONTEXT.md",
    "docs/DATA_ACQUISITION_GUIDE.md",
    "docs/MANIFEST_SPEC.md",
    "docs/REPO_USAGE.md",
    "docs/CLAIM_BOUNDARY.md",
    "02_processed/manifests/master_manifest.csv",
    "02_processed/manifests/README.md",
    "scripts/init_project_tree.py",
    "scripts/check_project_tree.py",
    "scripts/add_manifest_row.py",
    "scripts/validate_manifest.py",
    "scripts/hash_files.py",
    "src/shipdc_ai/__init__.py",
    "src/shipdc_ai/manifest.py",
    "src/shipdc_ai/paths.py",
    "src/shipdc_ai/nvr_emulator.py",
    "src/shipdc_ai/cheap_filter.py",
    "src/shipdc_ai/detector_runtime.py",
    "src/shipdc_ai/temporal_verifier.py",
    "src/shipdc_ai/event_store.py",
    "tests/test_manifest_header.py",
    "tests/test_imports.py",
]

GITKEEP_FILES = [
    "00_admin/seminar_docs/.gitkeep",
    "00_admin/accounts/mivia/.gitkeep",
    "00_admin/notes/.gitkeep",
    "01_raw/01_donor/DFire/downloads/.gitkeep",
    "01_raw/01_donor/DFire/unzipped/.gitkeep",
    "01_raw/01_donor/FIRESENSE/downloads/.gitkeep",
    "01_raw/01_donor/FIRESENSE/unzipped/.gitkeep",
    "01_raw/01_donor/MIVIA/LFDN/downloads/.gitkeep",
    "01_raw/01_donor/MIVIA/LFDN/unzipped/.gitkeep",
    "01_raw/01_donor/MIVIA/FireDetection/downloads/.gitkeep",
    "01_raw/01_donor/MIVIA/FireDetection/unzipped/.gitkeep",
    "01_raw/01_donor/MIVIA/SmokeDetection/downloads/.gitkeep",
    "01_raw/01_donor/MIVIA/SmokeDetection/unzipped/.gitkeep",
    "01_raw/02_shiplike/DVIDS/downloads/.gitkeep",
    "01_raw/02_shiplike/DVIDS/clips/.gitkeep",
    "01_raw/02_shiplike/DVIDS/frames/.gitkeep",
    "01_raw/02_shiplike/NOAA/downloads/.gitkeep",
    "01_raw/02_shiplike/NOAA/clips/.gitkeep",
    "01_raw/02_shiplike/NOAA/images/.gitkeep",
    "01_raw/02_shiplike/NOAA/frames/.gitkeep",
    "01_raw/02_shiplike/NARA/downloads/.gitkeep",
    "01_raw/02_shiplike/NARA/images/.gitkeep",
    "02_processed/frames/.gitkeep",
    "02_processed/labels/.gitkeep",
    "02_processed/synth_v1/.gitkeep",
    "02_processed/hard_negative_v1/.gitkeep",
    "03_experiments/training/.gitkeep",
    "03_experiments/onnx/.gitkeep",
    "03_experiments/runtime_logs/.gitkeep",
    "03_experiments/eval_reports/.gitkeep",
    "04_papers/seminar_2026/.gitkeep",
    "04_papers/figures/.gitkeep",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def missing_directories(root: Path) -> list[str]:
    return [path for path in REQUIRED_DIRECTORIES if not (root / path).is_dir()]


def missing_files(root: Path) -> list[str]:
    return [path for path in REQUIRED_FILES if not (root / path).is_file()]


def missing_gitkeep(root: Path) -> list[str]:
    return [path for path in GITKEEP_FILES if not (root / path).is_file()]


def report_category(name: str, missing: list[str], expected_count: int) -> bool:
    if not missing:
        print(f"PASS {name}: {expected_count} expected items present")
        return True
    print(f"FAIL {name}: {len(missing)} missing of {expected_count}")
    for path in missing:
        print(f"  missing: {path}")
    return False


def main() -> int:
    root = project_root()
    results = [
        report_category("required directories", missing_directories(root), len(REQUIRED_DIRECTORIES)),
        report_category("required files", missing_files(root), len(REQUIRED_FILES)),
        report_category("gitkeep placeholders", missing_gitkeep(root), len(GITKEEP_FILES)),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

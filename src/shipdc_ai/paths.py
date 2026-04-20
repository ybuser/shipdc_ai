"""Repository path helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_root: Path
    processed_root: Path
    manifest_path: Path
    experiment_root: Path
    paper_root: Path

    @classmethod
    def default(cls, root: Path | None = None) -> "ProjectPaths":
        base = (root or project_root()).resolve()
        return cls(
            root=base,
            raw_root=base / "01_raw",
            processed_root=base / "02_processed",
            manifest_path=base / "02_processed" / "manifests" / "master_manifest.csv",
            experiment_root=base / "03_experiments",
            paper_root=base / "04_papers",
        )


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data

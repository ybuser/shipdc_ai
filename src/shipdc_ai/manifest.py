"""Manifest helpers for public-data asset tracking."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MANIFEST_HEADER = [
    "asset_id",
    "source_group",
    "source_name",
    "source_url",
    "download_date",
    "local_path",
    "asset_type",
    "license_or_rights",
    "zone_type",
    "confuser_tag",
    "use_purpose",
    "sha256",
    "notes",
]


@dataclass(frozen=True)
class ManifestRow:
    asset_id: str
    source_group: str = ""
    source_name: str = ""
    source_url: str = ""
    download_date: str = ""
    local_path: str = ""
    asset_type: str = ""
    license_or_rights: str = ""
    zone_type: str = ""
    confuser_tag: str = ""
    use_purpose: str = ""
    sha256: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in MANIFEST_HEADER}


def read_manifest_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return next(reader, [])


def validate_manifest_header(path: Path) -> list[str]:
    if not path.exists():
        return [f"manifest does not exist: {path}"]
    if not path.is_file():
        return [f"manifest path is not a file: {path}"]
    actual = read_manifest_header(path)
    if actual == MANIFEST_HEADER:
        return []
    return [f"header mismatch: found {actual!r}"]


def asset_id_exists(path: Path, asset_id: str) -> bool:
    errors = validate_manifest_header(path)
    if errors:
        raise ValueError("; ".join(errors))
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("asset_id") == asset_id:
                return True
    return False


def append_manifest_row(path: Path, values: Mapping[str, str]) -> None:
    errors = validate_manifest_header(path)
    if errors:
        raise ValueError("; ".join(errors))
    asset_id = values.get("asset_id", "").strip()
    if not asset_id:
        raise ValueError("asset_id is required")
    if asset_id_exists(path, asset_id):
        raise ValueError(f"duplicate asset_id: {asset_id}")

    row = {field: values.get(field, "") for field in MANIFEST_HEADER}
    row["asset_id"] = asset_id
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_HEADER)
        writer.writerow(row)

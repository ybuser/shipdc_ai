"""Compute SHA256 hashes for one file or all files under a directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


CHUNK_SIZE = 1024 * 1024


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    raise FileNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="File or directory to hash.")
    parser.add_argument("--no-header", action="store_true", help="Do not print the CSV header row.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    try:
        files = iter_files(args.path)
    except FileNotFoundError:
        print(f"FAIL path not found: {args.path}", file=sys.stderr)
        return 1

    if not args.no_header:
        writer.writerow(["path", "sha256", "size_bytes"])
    for path in files:
        writer.writerow([str(path), hash_file(path), path.stat().st_size])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

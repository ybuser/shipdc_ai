"""Validate the master manifest CSV header."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shipdc_ai.manifest import MANIFEST_HEADER, validate_manifest_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "master_manifest.csv",
        help="Manifest CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_manifest_header(args.manifest)
    if errors:
        print(f"FAIL manifest header: {args.manifest}")
        for error in errors:
            print(f"  {error}")
        print("Expected header:")
        print(",".join(MANIFEST_HEADER))
        return 1
    print(f"PASS manifest header: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

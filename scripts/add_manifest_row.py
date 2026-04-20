"""Append one row to the master manifest CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shipdc_ai.manifest import MANIFEST_HEADER, append_manifest_row, asset_id_exists  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "master_manifest.csv",
        help="Manifest CSV path.",
    )
    parser.add_argument("--asset-id", required=True, help="Unique asset identifier.")
    parser.add_argument("--source-group", default="")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--download-date", default="")
    parser.add_argument("--local-path", default="")
    parser.add_argument("--asset-type", default="")
    parser.add_argument("--license-or-rights", default="")
    parser.add_argument("--zone-type", default="")
    parser.add_argument("--confuser-tag", default="")
    parser.add_argument("--use-purpose", default="")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if asset_id_exists(args.manifest, args.asset_id):
        print(f"FAIL duplicate asset_id: {args.asset_id}")
        return 1

    values = {
        "asset_id": args.asset_id,
        "source_group": args.source_group,
        "source_name": args.source_name,
        "source_url": args.source_url,
        "download_date": args.download_date,
        "local_path": args.local_path,
        "asset_type": args.asset_type,
        "license_or_rights": args.license_or_rights,
        "zone_type": args.zone_type,
        "confuser_tag": args.confuser_tag,
        "use_purpose": args.use_purpose,
        "sha256": args.sha256,
        "notes": args.notes,
    }
    append_manifest_row(args.manifest, values)
    print(f"PASS appended manifest row: {args.asset_id}")
    print(",".join(MANIFEST_HEADER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

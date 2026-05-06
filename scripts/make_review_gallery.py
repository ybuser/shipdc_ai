"""Create a static HTML review gallery from a ship-like tagging sheet."""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_ship_like_tagging_sheet import TAGGING_SHEET_HEADER, normalize_frame_path  # noqa: E402


DISPLAY_COLUMNS = [
    "frame_id",
    "asset_id",
    "frame_path",
    "timestamp_s",
    "width",
    "height",
]


@dataclass(frozen=True)
class GallerySummary:
    input_sheet: Path
    output_html: Path
    total_rows: int
    displayed_rows: int
    groups: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def read_tagging_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"input tagging sheet does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input tagging sheet path is not a file: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in TAGGING_SHEET_HEADER if column not in fieldnames]
        if missing:
            raise ValueError(f"missing required tagging sheet columns: {', '.join(missing)}")
        return list(reader)


def select_gallery_groups(
    rows: list[dict[str, str]],
    *,
    limit_per_asset: int,
) -> OrderedDict[tuple[str, str], list[dict[str, str]]]:
    groups: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    counts_by_asset: defaultdict[str, int] = defaultdict(int)

    for row in rows:
        asset_id = row.get("asset_id", "").strip()
        if counts_by_asset[asset_id] >= limit_per_asset:
            continue

        source_name = row.get("source_name", "").strip()
        key = (asset_id, source_name)
        groups.setdefault(key, []).append(row)
        counts_by_asset[asset_id] += 1

    return groups


def relative_image_src(frame_path: str, *, output_html: Path, root: Path) -> str:
    normalized = normalize_frame_path(frame_path)
    image_path = (root / normalized).resolve()
    output_dir = output_html.parent.resolve()
    relative = os.path.relpath(image_path, output_dir)
    return Path(relative).as_posix()


def render_metadata(row: dict[str, str]) -> str:
    lines = []
    for column in DISPLAY_COLUMNS:
        value = row.get(column, "").strip()
        if column == "timestamp_s" and not value:
            value = "still"
        lines.append(
            "<div class=\"metadata-row\">"
            f"<dt>{html.escape(column)}</dt>"
            f"<dd>{html.escape(value)}</dd>"
            "</div>"
        )
    return "\n".join(lines)


def render_frame_card(row: dict[str, str], *, output_html: Path, root: Path) -> str:
    image_src = relative_image_src(row.get("frame_path", ""), output_html=output_html, root=root)
    frame_id = row.get("frame_id", "").strip()
    return (
        "<figure class=\"frame-card\">"
        f"<img loading=\"lazy\" src=\"{html.escape(image_src, quote=True)}\" "
        f"alt=\"{html.escape(frame_id, quote=True)}\">"
        "<figcaption>"
        "<dl>"
        f"{render_metadata(row)}"
        "</dl>"
        "</figcaption>"
        "</figure>"
    )


def render_gallery_html(
    groups: OrderedDict[tuple[str, str], list[dict[str, str]]],
    *,
    output_html: Path,
    root: Path,
    total_rows: int,
    displayed_rows: int,
    limit_per_asset: int,
) -> str:
    sections = []
    for (asset_id, source_name), rows in groups.items():
        cards = "\n".join(render_frame_card(row, output_html=output_html, root=root) for row in rows)
        sections.append(
            "<section class=\"asset-group\">"
            "<header class=\"group-header\">"
            f"<h2>{html.escape(asset_id or 'asset_unknown')}</h2>"
            f"<p>{html.escape(source_name or 'source_unknown')} | {len(rows)} displayed</p>"
            "</header>"
            f"<div class=\"frame-grid\">{cards}</div>"
            "</section>"
        )

    body = "\n".join(sections) if sections else "<p class=\"empty\">No rows to display.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ship-Like Review Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --text: #182126;
      --muted: #5d6b73;
      --line: #d8e0e4;
      --accent: #1d6f75;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.4;
    }}
    main {{
      width: min(1440px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .page-header {{
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      margin: 0;
      color: var(--muted);
    }}
    .asset-group {{
      margin-top: 22px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}
    .group-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .group-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .frame-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
    }}
    .frame-card {{
      margin: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    .frame-card img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 10;
      object-fit: contain;
      background: #12181b;
      border-bottom: 1px solid var(--line);
    }}
    figcaption {{
      padding: 10px;
    }}
    dl {{
      display: grid;
      gap: 6px;
      margin: 0;
    }}
    .metadata-row {{
      min-width: 0;
    }}
    dt {{
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    .empty {{
      color: var(--muted);
    }}
    @media (max-width: 640px) {{
      main {{
        width: min(100% - 20px, 1440px);
        padding-top: 16px;
      }}
      .group-header {{
        display: block;
      }}
      .group-header p {{
        margin-top: 4px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <h1>Ship-Like Review Gallery</h1>
      <p class="summary">Public-data proxy review only. Displaying {displayed_rows} of {total_rows} rows with a limit of {limit_per_asset} per asset.</p>
    </header>
    {body}
  </main>
</body>
</html>
"""


def write_gallery(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_gallery(
    *,
    input_sheet: Path,
    output_html: Path,
    limit_per_asset: int,
    root: Path = ROOT,
) -> GallerySummary:
    rows = read_tagging_rows(input_sheet)
    groups = select_gallery_groups(rows, limit_per_asset=limit_per_asset)
    displayed_rows = sum(len(group_rows) for group_rows in groups.values())
    content = render_gallery_html(
        groups,
        output_html=output_html,
        root=root,
        total_rows=len(rows),
        displayed_rows=displayed_rows,
        limit_per_asset=limit_per_asset,
    )
    write_gallery(output_html, content)
    return GallerySummary(
        input_sheet=input_sheet,
        output_html=output_html,
        total_rows=len(rows),
        displayed_rows=displayed_rows,
        groups=len(groups),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--manifest",
        dest="input",
        type=Path,
        default=ROOT / "02_processed" / "manifests" / "ship_like_tagging_sheet.csv",
        help="Input ship-like tagging sheet CSV path.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=ROOT / "02_processed" / "review" / "ship_like_gallery.html",
        help="Output static HTML gallery path.",
    )
    parser.add_argument(
        "--limit-per-asset",
        type=positive_int,
        default=80,
        help="Maximum frames to display per asset_id. Default: 80.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        summary = run_gallery(
            input_sheet=resolve_cli_path(args.input, ROOT),
            output_html=resolve_cli_path(args.output, ROOT),
            limit_per_asset=args.limit_per_asset,
            root=ROOT,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    print(f"PASS input sheet: {summary.input_sheet}")
    print(f"PASS output gallery: {summary.output_html}")
    print(f"PASS total rows: {summary.total_rows}")
    print(f"PASS displayed rows: {summary.displayed_rows}")
    print(f"PASS groups: {summary.groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

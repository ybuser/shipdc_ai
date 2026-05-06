# Keyframe Extraction

This repository uses Windows-native, public-data-only tooling to create local frame candidates from manifested ship-like assets. The scripts do not download datasets and do not use Torch, Ultralytics, CUDA, Docker, WSL, GPU runtimes, or third-party Python packages.

Generated frames are local processed data. They are written under `02_processed/frames/` and must stay out of Git history.

## Requirements

- Run from Windows with Python available.
- Use only assets already listed in `02_processed/manifests/master_manifest.csv`.
- For video extraction, provide `ffmpeg.exe` on `PATH` or with `--ffmpeg`.
- Still-image assets do not require `ffmpeg.exe`.

The tooling is for public ship-like context data only. Do not use it to reference or process real naval ship CCTV, classified layouts, RTSP camera feeds, or restricted operational data.

## Extract Frames

Extract DVIDS video frames and still-image candidates at the default 1 fps:

```powershell
python scripts/extract_keyframes.py --asset-prefix SHIP_DVIDS
```

Extract NOAA assets with an explicit `ffmpeg.exe`, 0.5 fps, and a cap of 25 frames per video:

```powershell
python scripts/extract_keyframes.py `
  --asset-prefix SHIP_NOAA `
  --fps 0.5 `
  --max-per-asset 25 `
  --ffmpeg C:\tools\ffmpeg\bin\ffmpeg.exe
```

Use `--overwrite` only when replacing existing extracted frames for selected assets is intentional. Without it, assets with existing target frames are skipped and existing manifest rows are preserved.

## Output Layout

Frames are written to:

```text
02_processed/frames/ship_like/<source>/<batch>/<asset_id>/
```

- `<source>` comes from `source_name` in the master manifest and is sanitized for filesystem safety.
- `<batch>` comes from `download_date` as `YYYYMMDD`, or `unbatched` when no usable date is available.
- Video frames use deterministic names such as `frame_000000.jpg`.
- Still-image candidates use names such as `still_000000.jpg` and are not treated as video keyframes.

## Keyframe Manifest

The extractor writes:

```text
02_processed/manifests/ship_like_keyframes.csv
```

Required columns:

```text
asset_id,source_name,source_asset_path,frame_path,timestamp_s,frame_index,width,height,extraction_fps,sha256,notes
```

Video rows include `timestamp_s`, `frame_index`, and `extraction_fps`. Still-image rows leave `timestamp_s` and `extraction_fps` blank, set `frame_index` to `0`, and include `still_image_candidate; not_video_keyframe` in `notes`.

The manifest stores repository-relative paths with forward slashes and SHA256 hashes for every copied or extracted frame.

## Validate

Validate the derived manifest after extraction:

```powershell
python scripts/validate_keyframe_manifest.py
```

The validator checks required columns, verifies that each local frame file exists, recomputes populated SHA256 values, and prints counts by `asset_id` and `source_name`.

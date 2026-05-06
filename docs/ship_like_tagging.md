# Ship-Like Tagging

This workflow prepares reviewer labels for public ship-like proxy imagery. It is not a real naval CCTV dataset and must not be described as one. Do not add references to classified ship layouts, operational camera systems, RTSP feeds, restricted data, or non-public naval surveillance.

The tagging tools are Windows-native and standard-library-only. They do not download data, extract new frames, require Torch, Ultralytics, CUDA, Docker, WSL, GPU runtimes, or third-party Python packages.

## Files

Create the reviewer CSV from the derived keyframe manifest:

```powershell
python scripts/make_ship_like_tagging_sheet.py
```

Default input:

```text
02_processed/manifests/ship_like_keyframes.csv
```

Default output:

```text
02_processed/manifests/ship_like_tagging_sheet.csv
```

Create a local static review gallery:

```powershell
python scripts/make_review_gallery.py `
  --input 02_processed/manifests/ship_like_tagging_sheet.csv `
  --output 02_processed/review/ship_like_gallery.html
```

Older workflow notes may use `--manifest` for the input sheet and `--out` for the output HTML path. Those aliases remain accepted, but prefer `--input` and `--output` in new commands.

Default output:

```text
02_processed/review/ship_like_gallery.html
```

The generated tagging sheet and gallery are local review artifacts. Do not commit generated manifests, frames, extracted media, logs, experiment outputs, or model files.

## Tagging Sheet Columns

Required columns:

```text
frame_id,asset_id,source_name,frame_path,timestamp_s,frame_index,width,height,zone_l1,confuser_tags,use_purpose,split,quality_flag,reviewer_notes,sha256
```

`frame_path` stays repository-relative so the CSV can move with the repository. `frame_id` is deterministic from the frame path and is intended to remain stable across repeated sheet generation as long as the frame path is unchanged.

## Allowed Labels

`zone_l1` describes the visible ship-like area:

- `engine_room`: Machinery spaces, engine-room-like equipment, piping, engines, pumps, or similar mechanical spaces.
- `bridge_control`: Bridge, helm, navigation, watch station, or vessel-control area.
- `control_room`: Monitoring room, control console area, operations center, or instrument-heavy control space.
- `passageway_ladder`: Corridor, hatch, ladder, stair, access trunk, or transit space.
- `service_maintenance_support`: Workshop, maintenance bay, storage, galley-adjacent support area, or general service area.
- `exterior_or_unknown`: Exterior, deck, non-interior view, unclear location, or insufficient evidence.

Initial value: `exterior_or_unknown`.

`confuser_tags` is a comma-separated reviewer field for visual conditions that may confuse fire or smoke detection. Use `none` when no confuser is present. Hard-negative examples include:

- `welding`
- `red_light`
- `reflection`
- `low_light`
- `control_panel_light`
- `steam_or_training_smoke`

Initial value: `none`.

`use_purpose` describes how the frame may be used:

- `background_train`: Background-only training candidate.
- `background_val`: Background-only validation candidate.
- `hard_negative_eval`: Hard-negative evaluation candidate with smoke/fire-like confusers but no real fire target.
- `qualitative_figure`: Candidate for paper or seminar figures.
- `synth_background`: Candidate background for later synthetic smoke/fire composition.
- `exclude`: Do not use.

Initial value: `exclude`.

`split` describes dataset partition:

- `train`
- `val`
- `test`
- `qualitative`
- `exclude`

Initial value: `exclude`.

`quality_flag` describes review quality:

- `good`: Usable and readable.
- `low_light`: Too dark or heavily underexposed.
- `blurred`: Motion blur, focus blur, or compression blur limits value.
- `duplicate`: Near-duplicate of another frame.
- `occluded`: Important scene content is blocked.
- `ambiguous`: Scene or use decision is unclear.
- `exclude`: Not suitable for use.

Initial value: `good`.

`reviewer_notes` is free text for short reviewer comments. Initial value: blank.

## Asset-Level Split Discipline

Keep splits at the `asset_id` level. All frames derived from one `asset_id` must stay in one split to reduce leakage from adjacent frames, repeated backgrounds, camera angle, lighting, or source-specific visual style.

Do not place frames from the same `asset_id` across `train`, `val`, and `test`. If an asset is used only for qualitative figures or is excluded, keep every frame from that asset aligned with `qualitative` or `exclude`.

When in doubt, leave `use_purpose` and `split` as `exclude` until an asset-level decision is made.

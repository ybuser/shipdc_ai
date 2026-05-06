# Synthetic V1 Positive Proxy Generation

Synthetic v1 is a small, auditable data-prep step for preliminary seminar experiments. It creates public-data-only ship-like synthetic positive proxy images by pasting public fire/smoke donor crops onto public ship-like train backgrounds.

This tooling does not claim realism, does not create real shipboard positive labels, and does not validate against real naval CCTV. It also does not change the later CPU-only ONNX Runtime deployment direction.

## Inputs

Default inputs are configured in `configs/synth_v1.json`:

```text
C:\_code\shipdc_ai\02_processed\manifests\ship_like_background_train.csv
C:\_code\shipdc_ai\02_processed\manifests\donor_crop_manifest_v1.csv
```

The background manifest is filtered to rows where:

```text
use_purpose == synth_background
split == train
quality_flag == good
```

Rows are excluded when either `reviewer_notes` or `review_note` contains `with_human`. This keeps person-containing ship-like frames out of synthetic background generation.

`ship_like_hard_negative_eval.csv` is explicitly excluded from synthetic generation. Validation and test background manifests are also excluded; synthetic v1 only uses train backgrounds.

The donor crop manifest is filtered to normalized `class_name` values of `fire` and `smoke`. Other donor classes, if present later, are ignored.

Class ids are fixed:

```text
0 = fire
1 = smoke
```

## Generation

`scripts/build_synth_v1.py` generates a small, deterministic v1 dataset from a base seed. The default cap is 170 samples with 50 QA previews.

The build prefers balanced class generation. When enough backgrounds are available, the schedule creates one fire sample and one smoke sample per background before repeating backgrounds.

The compositor is intentionally simple:

- resize the donor crop within configured scale bounds
- avoid placing the crop too close to image edges
- apply opacity
- apply feathered alpha blending
- optionally apply light blur
- write one YOLO-format label per generated image
- write preview images with visible bounding boxes for manual QA

## Outputs

Default outputs are:

```text
C:\_code\shipdc_ai\02_processed\synth\v1
C:\_code\shipdc_ai\02_processed\manifests\synth_v1_manifest.csv
```

The output directory contains `images`, `labels`, and `previews`. Generated synthetic images, labels, previews, generated manifests, logs, and experiment outputs are local artifacts and should not be committed.

The synthetic manifest columns are:

```text
synth_id,image_path,label_path,background_frame_id,background_asset_id,background_frame_path,background_zone_l1,donor_crop_id,donor_source_name,donor_crop_path,class_name,class_id,bbox_xywh_norm,generation_seed,image_width,image_height,sha256,quality_flag,notes
```

`bbox_xywh_norm` is the normalized YOLO bbox as a JSON list matching the label file values.

## Commands

Show build options:

```powershell
python scripts/build_synth_v1.py --help
```

Run the default build:

```powershell
python scripts/build_synth_v1.py --overwrite
```

Validate a generated manifest:

```powershell
python scripts/validate_synth_manifest.py --manifest "C:\_code\shipdc_ai\02_processed\manifests\synth_v1_manifest.csv"
```

Create seminar summary tables:

```powershell
python scripts/summarize_synth_v1.py `
  --manifest "C:\_code\shipdc_ai\02_processed\manifests\synth_v1_manifest.csv" `
  --out-dir "C:\_code\shipdc_ai\04_papers\seminar\tables"
```

## Paper Claim Boundary

Use language such as:

```text
We use public ship-like background frames and public fire/smoke donor crops to create a small synthetic positive proxy set for preliminary analysis.
```

Do not describe synthetic v1 as realistic shipboard fire/smoke data. Do not describe it as real shipboard positive labels. Do not claim validation on real naval CCTV or restricted operational data.

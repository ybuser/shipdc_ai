# Donor Crop Bank V1

This tooling builds a fire/smoke crop bank for synthetic v1 data preparation. It is not runtime inference code and does not change the CPU-only ONNX Runtime deployment direction.

The crop bank uses public donor datasets only:

- `DFire`: public D-Fire images and YOLO labels. The configured class map is explicit: `0=smoke`, `1=fire`.
- `CQU_FireSmokeYOLO`: CQUniversity open-access YOLO dataset. The configured class map follows the local `data.yaml`: `0=fire`, `1=smoke`.
- `IndoorFireSmoke`: Zenodo indoor fire/smoke dataset. Bbox labels are present locally, but the local `data.yaml` names are numeric (`['0', '1']`). It is disabled in `configs/donor_crop_v1.json` until the class id meaning is verified.

The build script intentionally fails if an enabled source has an unknown class mapping or a label references an unmapped class id. Do not guess class ids.

## Dependency Boundary

`scripts/build_donor_crop_bank.py` imports Pillow locally because it must crop image files. Pillow is a data-prep-only dependency and must not be added to `pyproject.toml` runtime dependencies.

Install it only in the data-prep environment when building crops:

```powershell
python -m pip install Pillow
```

No Torch, Ultralytics, CUDA, Docker, WSL, GPU runtime assumptions, or model training dependencies are used by this tooling.

## Outputs

Default outputs are configured in `configs/donor_crop_v1.json`:

```text
02_processed/crops/fire_smoke_donor_v1/images
02_processed/crops/fire_smoke_donor_v1/preview
02_processed/manifests/donor_crop_manifest_v1.csv
```

The generated crops, previews, and `donor_crop_manifest_v1.csv` are local generated artifacts and should not be committed.

Manifest columns:

```text
crop_id,source_name,source_image_path,source_label_path,crop_path,class_name,bbox_xyxy,crop_width,crop_height,sha256,notes
```

All paths are repository-relative and use forward slashes.

## Inspect Sources

Run inspection before building:

```powershell
python scripts/inspect_donor_label_sources.py --manifest "C:\_code\shipdc_ai\02_processed\manifests\master_manifest.csv"
```

Inspection reports master-manifest donor rows, rights notes, split image/label counts, and whether the class map is known.

## Build Crops

Full build:

```powershell
python scripts/build_donor_crop_bank.py --overwrite
```

Small smoke-test build:

```powershell
python scripts/build_donor_crop_bank.py --max-crops-per-source 10 --overwrite
```

Build one source:

```powershell
python scripts/build_donor_crop_bank.py --source DFire --overwrite
```

Selecting `IndoorFireSmoke` without first adding a verified `class_map` will fail clearly.

## Validate

After a build:

```powershell
python scripts/validate_donor_crop_manifest.py
```

Validation checks the exact header, class names, repository-relative paths, crop file existence, SHA256, bbox geometry, and crop dimensions.

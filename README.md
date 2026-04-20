# shipdc_ai

Public-data-only research scaffold for a ship-like multi-CCTV fire and smoke early warning and damage-control support AI prototype.

## Current Phase

This repository is in scaffold-only mode. It contains directory layout, manifest tooling, documentation, and lightweight Python stubs. It does not contain datasets, trained models, GPU code, camera integrations, or heavy ML dependencies.

## Non-Negotiable Constraints

- Target local environment: Windows 11.
- No WSL.
- No Docker.
- No GPU inference.
- Runtime must remain CPU-only, with ONNX Runtime intended for a later phase.
- Runtime must not depend on Torch, Ultralytics, CUDA, or OpenCV during this phase.
- NVR emulation is MP4-first. Do not assume RTSP cameras or live ship CCTV.
- Real naval ship CCTV, classified ship layouts, and restricted operational data are out of scope.

## Quickstart

From the repository root:

```powershell
python --version
python scripts/check_project_tree.py
python scripts/validate_manifest.py
```

Optional standard-library test run:

```powershell
python -m unittest discover tests
```

## Data Policy

Raw data stays local and outside Git history. Downloaded archives, extracted media, generated frames, synthetic data, ONNX artifacts, logs, and experiment outputs are ignored by default. Only the directory placeholders, documentation, source code, configs, and manifest template belong in Git at this phase.

Every future external asset should be recorded in `02_processed/manifests/master_manifest.csv` with its source, local path, purpose, rights note, and SHA256 hash when available.

## Next Steps

1. D-Fire download: place archives in `01_raw/01_donor/DFire/downloads/`, extract to `01_raw/01_donor/DFire/unzipped/`, then add manifest rows.
2. MIVIA account setup: keep account notes outside Git under `00_admin/accounts/mivia/`, then store approved downloads under the MIVIA raw folders.
3. FIRESENSE download: store archives and extracted data under `01_raw/01_donor/FIRESENSE/`.
4. DVIDS/NOAA starter pack: collect public ship-like clips, images, and derived frames under `01_raw/02_shiplike/` and document each asset in the manifest.

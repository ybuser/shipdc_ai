# Data Acquisition Guide

This guide explains where future public data should be placed. Do not download data during scaffold initialization.

## General Rules

1. Download only public or properly licensed data.
2. Keep archives in the matching `downloads/` directory.
3. Extract data into the matching `unzipped/`, `clips/`, `images/`, or `frames/` directory.
4. Do not commit downloaded or extracted data.
5. Add one manifest row per useful archive, clip, image set, frame set, label file, or derived asset.
6. Record license or rights information in plain language. If uncertain, write `review_needed`.
7. Compute SHA256 hashes for files that will be cited, processed, or reused.

## Donor Fire and Smoke Datasets

- D-Fire: store downloads in `01_raw/01_donor/DFire/downloads/` and extracted contents in `01_raw/01_donor/DFire/unzipped/`.
- FIRESENSE: store downloads in `01_raw/01_donor/FIRESENSE/downloads/` and extracted contents in `01_raw/01_donor/FIRESENSE/unzipped/`.
- MIVIA LFDN: store downloads in `01_raw/01_donor/MIVIA/LFDN/downloads/` and extracted contents in `01_raw/01_donor/MIVIA/LFDN/unzipped/`.
- MIVIA FireDetection: store downloads in `01_raw/01_donor/MIVIA/FireDetection/downloads/` and extracted contents in `01_raw/01_donor/MIVIA/FireDetection/unzipped/`.
- MIVIA SmokeDetection: store downloads in `01_raw/01_donor/MIVIA/SmokeDetection/downloads/` and extracted contents in `01_raw/01_donor/MIVIA/SmokeDetection/unzipped/`.

## Ship-Like Public Context Data

- DVIDS: store downloads in `01_raw/02_shiplike/DVIDS/downloads/`, clips in `clips/`, and extracted frames in `frames/`.
- NOAA: store downloads in `01_raw/02_shiplike/NOAA/downloads/`, clips in `clips/`, images in `images/`, and extracted frames in `frames/`.
- NARA: store downloads in `01_raw/02_shiplike/NARA/downloads/` and images in `images/`.

## Useful Commands

```powershell
python scripts/hash_files.py path\to\file_or_directory
python scripts/add_manifest_row.py --asset-id demo_001 --source-group donor --source-name DFire --local-path 01_raw/01_donor/DFire/downloads/example.zip
python scripts/validate_manifest.py
```

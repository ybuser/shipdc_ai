# Repository Usage

## Verify the Scaffold

```powershell
python scripts/check_project_tree.py
python scripts/validate_manifest.py
python -m unittest discover tests
```

## Add an Asset to the Manifest

```powershell
python scripts/add_manifest_row.py `
  --asset-id dfire_archive_001 `
  --source-group donor `
  --source-name DFire `
  --source-url https://example.invalid/source-page `
  --download-date 2026-04-20 `
  --local-path 01_raw/01_donor/DFire/downloads/example.zip `
  --asset-type archive `
  --license-or-rights review_needed `
  --zone-type donor_fire `
  --use-purpose train `
  --notes "Example only"
```

## Compute Hashes

```powershell
python scripts/hash_files.py 01_raw/01_donor/DFire/downloads/example.zip
python scripts/hash_files.py 01_raw/02_shiplike/DVIDS/clips
```

## Expected Local-Only Material

The following should remain outside Git history:

- Raw dataset downloads.
- Extracted videos, images, and frames.
- Synthetic image or video outputs.
- Training runs and evaluation artifacts.
- ONNX model files.
- Runtime logs and SQLite databases.
- Account notes, API keys, credentials, and access forms.

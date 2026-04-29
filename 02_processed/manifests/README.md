# Master Manifest

The master manifest CSV must start with the exact required header. It may contain real asset rows after the header. Do not place commented rows inside the CSV.

Example rows for reference are shown with leading `#` characters for readability only; do not include those markers in the CSV:

```text
# dfire_archive_001,donor,DFire,https://example.invalid/dfire,2026-04-20,01_raw/01_donor/DFire/downloads/dfire.zip,archive,review_needed,donor_fire,,train,,Downloaded archive placeholder example.
# mivia_lfdn_001,donor,MIVIA LFDN,https://example.invalid/mivia,2026-04-20,01_raw/01_donor/MIVIA/LFDN/downloads/lfdn.zip,archive,account_required_research,donor_fire,,validation,,Account-based download placeholder example.
# dvids_clip_001,shiplike,DVIDS,https://example.invalid/dvids,2026-04-20,01_raw/02_shiplike/DVIDS/clips/example.mp4,clip,public_distribution_review_needed,ship_like_internal,none,demo,,Public ship-like clip placeholder example.
```

Before adding or updating rows, confirm the source rights and compute hashes when practical:

```powershell
python scripts/hash_files.py path\to\asset
python scripts/add_manifest_row.py --asset-id unique_id --source-group donor --source-name DFire --local-path path\to\asset
python scripts/validate_manifest.py
```

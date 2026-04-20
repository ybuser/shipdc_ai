# Manifest Specification

The master manifest is stored at:

```text
02_processed/manifests/master_manifest.csv
```

The CSV header must be exactly:

```text
asset_id,source_group,source_name,source_url,download_date,local_path,asset_type,license_or_rights,zone_type,confuser_tag,use_purpose,sha256,notes
```

## Column Notes

- `asset_id`: unique stable identifier, such as `dfire_archive_001`.
- `source_group`: broad group, such as `donor`, `shiplike`, `synthetic`, or `processed`.
- `source_name`: dataset or source name.
- `source_url`: source page or download URL, if public and stable.
- `download_date`: ISO date, preferably `YYYY-MM-DD`.
- `local_path`: repository-relative path to the local asset.
- `asset_type`: archive, clip, image, frame_set, label, synthetic, report, or other clear type.
- `license_or_rights`: rights note such as public_domain, academic_research, attribution_required, or review_needed.
- `zone_type`: scene context, such as donor_fire, ship_like_internal, ship_like_external, engine_room_like, galley_like, or unknown.
- `confuser_tag`: optional negative or confusing condition, such as steam, fog, welding, glare, dust, or none.
- `use_purpose`: intended use, such as train, validation, test, hard_negative, calibration, demo, or citation.
- `sha256`: SHA256 hash for a file asset when available.
- `notes`: short free-text note.

Use `scripts/validate_manifest.py` after editing the manifest.

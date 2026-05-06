# Fast Detector And Runtime Experiment

This repository includes a minimal detector and runtime path for seminar-paper evidence only. The goal is preliminary feasibility under public-data-only ship-like proxy conditions, not a SOTA detector study or an operational safety claim.

## Detector Dataset

Use `scripts/build_fast_detector_dataset.py` to build a YOLO-format dataset under the canonical dataset root. For the fast seminar run, use:

`C:\_code\shipdc_ai\03_experiments\detector_datasets\fire_smoke_fast_v1`

The script accepts either `--output-dir 03_experiments/detector_datasets --dataset-name fire_smoke_fast_v1` or `--output-dir 03_experiments/detector_datasets/fire_smoke_fast_v1 --dataset-name fire_smoke_fast_v1`; both resolve to that same canonical root. The dataset combines:

- Public donor fire/smoke crops from `donor_crop_manifest_v1.csv`.
- Synthetic public-data-only positive proxy samples from `synth_v1_manifest.csv`.
- Public ship-like background train/validation frames as empty-label negatives.
- Public ship-like hard-negative frames in an evaluation-only folder.

Class ids are fixed:

- `0 = fire`
- `1 = smoke`

Donor crop positives are labeled with a full-image proxy bounding box, `0.5 0.5 0.98 0.98`. This is a deliberate fast baseline limitation. It is useful for quick detector evidence but does not provide real object localization quality.

Held-out synthetic validation samples are also copied to `eval/synth_val/` for synthetic proxy evaluation. These samples are synthetic overlays on public ship-like backgrounds, not real shipboard positive labels.

Hard-negative frames are copied to `eval/hard_negative/` with empty labels. They are excluded from training and used only to estimate preliminary false-alarm behavior on public ship-like confusers.

## Prediction Summary

Use `scripts/summarize_yolo_predictions.py` after saving YOLO prediction labels:

- `--task synthetic_proxy` reports hit rates using same-class IoU matching.
- `--task hard_negative` reports false-alarm frame rate.
- `--task combined` can merge the two table-5 CSV summaries.

These summaries are preliminary table artifacts for the seminar paper. They should be framed as public proxy evidence, not as validation on real ship CCTV.

## Four-Stream Runtime Emulator

Use `scripts/build_runtime_mp4_inputs.py` to normalize selected public images to 640x360 JPEG sequences and build four MP4 streams:

- `ch01_synth_positive.mp4`
- `ch02_engine_background.mp4`
- `ch03_hard_negative_confuser.mp4`
- `ch04_val_background.mp4`

The script also writes `configs/nvr_4stream_fast.json`. The config points to the four local MP4 files and records `channel_id`, `display_name`, `source_path`, `expected_role`, and `zone_l1`.

This is an MP4 NVR emulator. It is not real CCTV, not RTSP, and not evidence from a naval ship camera system. The scripts prefer non-human images and avoid rows marked `with_human` where an alternative is available.

## Runtime Run

Use `scripts/run_fast_4stream_nvr.py` in the experiment environment with Ultralytics available. The script accepts `.pt` or `.onnx` YOLO model paths and defaults to `--device cpu`.

The current implementation processes channels sequentially as an emulated four-stream batch. It records the processing mode in metrics so the paper can state the limitation directly.

Temporal verification is simple K-in-window logic:

- A frame with at least one prediction above `--conf` becomes a suspect observation.
- An alarm occurs when at least `--temporal-k` suspect observations occur inside `--temporal-window-s`.

The run writes `metrics.csv`, `channel_metrics.csv`, `events.csv`, and first suspect/alarm snapshots under the run directory.

Use `scripts/summarize_fast_nvr_run.py` to create:

- `04_papers/seminar/tables/table6_runtime_results.csv`
- `04_papers/seminar/tables/runtime_summary_for_paper.txt`

## Boundary

Generated detector datasets, model weights, ONNX files, generated frames, generated MP4s, prediction outputs, SQLite databases, logs, and experiment run outputs must not be committed. The runtime direction in the paper remains CPU-only ONNX Runtime. The public-data-only boundary must stay explicit in any figure, table, or narrative that uses these tools.

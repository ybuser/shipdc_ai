# Project Context

`shipdc_ai` supports a seminar paper and prototype for a public-data-only ship-like multi-CCTV fire and smoke early warning and damage-control support AI system.

The local repository is intentionally constrained. It is not a naval ship integration project, does not use real ship CCTV, and does not model classified compartments or layouts. The prototype will use public donor fire/smoke data, public ship-like imagery or clips, synthetic data, and CPU-only deployment assumptions.

## Local Development Direction

- Windows-native repository.
- Standard-library-only utilities during scaffold phase.
- MP4-first NVR emulator in a later phase.
- CPU-only ONNX Runtime deployment in a later phase.
- No Docker, WSL, GPU inference, Torch, Ultralytics, CUDA, or live RTSP assumptions.

Training may later occur on a separate Linux GPU server, but that is outside the local runtime scaffold.

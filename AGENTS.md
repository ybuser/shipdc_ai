# AGENTS.md

Instructions for future Codex sessions working in this repository:

- Do not download large datasets unless the user explicitly asks for that action.
- Do not commit data, extracted media, generated frames, ONNX files, logs, or experiment outputs.
- Do not introduce Torch, Ultralytics, CUDA, GPU runtime assumptions, Docker, or WSL requirements.
- Keep runtime code Windows-native and standard-library-only unless the user approves a later dependency phase.
- Prefer small, reviewable changes with clear file ownership.
- Update documentation and manifest-related code when changing the data layout, manifest columns, or acquisition workflow.
- Keep public-data-only boundaries explicit. Do not add references to real naval ship CCTV, classified layouts, RTSP cameras, or restricted operational data.
- If adding future inference code, preserve the CPU-only ONNX Runtime deployment direction.

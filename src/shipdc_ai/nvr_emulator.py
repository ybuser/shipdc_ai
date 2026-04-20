"""Configuration stubs for a future MP4-first NVR emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    display_name: str
    zone_type: str
    mp4_source_path: Path
    enabled: bool = True
    fps_hint: float | None = None


@dataclass(frozen=True)
class NvrConfig:
    name: str
    cameras: tuple[CameraConfig, ...] = field(default_factory=tuple)
    loop: bool = False

    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)

"""Placeholder cheap-filter types for future frame triage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChangeScore:
    camera_id: str
    timestamp: datetime
    motion_score: float = 0.0
    color_shift_score: float = 0.0
    notes: str = ""


def score_placeholder(*_args: object, **_kwargs: object) -> ChangeScore:
    """Placeholder until a dependency-free frame representation is chosen."""
    raise NotImplementedError("Cheap filter scoring is planned for a later phase.")


# TODO: Define a lightweight frame-summary input before adding real filtering logic.

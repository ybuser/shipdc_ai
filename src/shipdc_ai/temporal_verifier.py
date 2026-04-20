"""Dependency-free temporal verifier using K-in-window logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class AlarmState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    ALARM = "ALARM"


@dataclass(frozen=True)
class TemporalRule:
    window_seconds: float = 10.0
    min_confidence: float = 0.5
    suspect_count: int = 2
    alarm_count: int = 3

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.suspect_count <= 0:
            raise ValueError("suspect_count must be positive")
        if self.alarm_count < self.suspect_count:
            raise ValueError("alarm_count must be greater than or equal to suspect_count")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class DetectionVote:
    timestamp: datetime
    confidence: float
    camera_id: str = ""
    label: str = ""


@dataclass(frozen=True)
class VerificationSnapshot:
    state: AlarmState
    vote_count: int
    window_start: datetime
    window_end: datetime


@dataclass
class TemporalVerifier:
    rule: TemporalRule = field(default_factory=TemporalRule)
    _votes: list[DetectionVote] = field(default_factory=list)
    state: AlarmState = AlarmState.NORMAL

    def update(
        self,
        timestamp: datetime,
        confidence: float,
        *,
        camera_id: str = "",
        label: str = "",
    ) -> VerificationSnapshot:
        vote = DetectionVote(timestamp=timestamp, confidence=confidence, camera_id=camera_id, label=label)
        self._votes.append(vote)
        self._trim(timestamp)
        self.state = self._state_for_current_window(timestamp)
        return self.snapshot(timestamp)

    def snapshot(self, timestamp: datetime) -> VerificationSnapshot:
        window_start = timestamp - timedelta(seconds=self.rule.window_seconds)
        return VerificationSnapshot(
            state=self.state,
            vote_count=self._qualified_count(timestamp),
            window_start=window_start,
            window_end=timestamp,
        )

    def reset(self) -> None:
        self._votes.clear()
        self.state = AlarmState.NORMAL

    def _trim(self, timestamp: datetime) -> None:
        window_start = timestamp - timedelta(seconds=self.rule.window_seconds)
        self._votes = [vote for vote in self._votes if vote.timestamp >= window_start]

    def _qualified_count(self, timestamp: datetime) -> int:
        window_start = timestamp - timedelta(seconds=self.rule.window_seconds)
        return sum(
            1
            for vote in self._votes
            if window_start <= vote.timestamp <= timestamp and vote.confidence >= self.rule.min_confidence
        )

    def _state_for_current_window(self, timestamp: datetime) -> AlarmState:
        count = self._qualified_count(timestamp)
        if count >= self.rule.alarm_count:
            return AlarmState.ALARM
        if count >= self.rule.suspect_count:
            return AlarmState.SUSPECT
        return AlarmState.NORMAL

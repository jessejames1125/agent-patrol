"""Base interface for failure-mode detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from agent_patrol.trace import Trace


class FailureMode(Enum):
    LOOP = auto()
    STALL = auto()
    DRIFT = auto()


@dataclass
class Detection:
    """A detected failure mode with evidence."""

    mode: FailureMode
    confidence: float  # 0.0 to 1.0
    evidence: str
    suggested_intervention: str | None = None

    def __bool__(self) -> bool:
        return self.confidence > 0.0


class Detector(ABC):
    """Base class for all failure-mode detectors."""

    @abstractmethod
    def check(self, trace: Trace) -> Detection | None:
        """Analyze the trace and return a Detection if a failure mode is found."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state."""
        ...

"""Data structures for recording agent execution traces."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class StepKind(Enum):
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    ASSISTANT_MESSAGE = auto()
    USER_MESSAGE = auto()


@dataclass(frozen=True, slots=True)
class Step:
    """A single step in an agent's execution trace."""

    kind: StepKind
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def fingerprint(self) -> str:
        """Stable hash for comparing steps structurally."""
        raw = f"{self.kind.name}:{self.tool_name}:{_stable_hash(self.tool_args)}:{self.content[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def action_key(self) -> str:
        """Coarser key: same tool + same args = same action, ignoring content."""
        if self.tool_name:
            return f"{self.tool_name}:{_stable_hash(self.tool_args)}"
        return f"msg:{self.content[:80]}"


def _stable_hash(obj: object) -> str:
    if obj is None:
        return "none"
    return hashlib.sha256(repr(sorted(obj.items()) if isinstance(obj, dict) else repr(obj)).encode()).hexdigest()[:12]


@dataclass
class Trace:
    """Append-only record of an agent's execution."""

    task_description: str = ""
    steps: list[Step] = field(default_factory=list)

    def append(self, step: Step) -> None:
        self.steps.append(step)

    def tail(self, n: int) -> list[Step]:
        return self.steps[-n:]

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == StepKind.TOOL_CALL]

    def __len__(self) -> int:
        return len(self.steps)

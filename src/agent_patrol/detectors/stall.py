"""Detect when an agent stops making meaningful progress.

Stalls are identified by tracking "progress signals" — tool calls that
produce new information vs. repetitive or empty results — and by
monitoring time between meaningful actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent_patrol.detectors.base import Detection, Detector, FailureMode
from agent_patrol.trace import Step, StepKind, Trace

# Patterns in assistant messages that suggest the agent is spinning its wheels
_HEDGE_PATTERNS = [
    "let me try",
    "i'll attempt",
    "let me think",
    "i'm not sure",
    "let me reconsider",
    "apologies",
    "i apologize",
    "sorry about that",
    "let me try again",
    "let me retry",
    "hmm",
    "i need to rethink",
]


@dataclass
class StallDetector(Detector):
    """Detects when an agent stops making forward progress.

    Parameters
    ----------
    window : int
        Number of recent steps to evaluate.
    progress_threshold : float
        Fraction of recent steps that must show "progress" to avoid a
        stall detection. If fewer than this fraction are productive,
        the agent is considered stalled.
    hedge_limit : int
        Number of hedging/uncertain messages in the window that triggers
        a stall alert.
    empty_result_limit : int
        Number of empty or near-empty tool results in the window that
        triggers a stall alert.
    custom_progress_fn : callable or None
        Optional function (Step) -> bool that returns True if a step
        represents meaningful progress. Overrides default heuristics.
    """

    window: int = 10
    progress_threshold: float = 0.3
    hedge_limit: int = 4
    empty_result_limit: int = 4
    custom_progress_fn: object = None  # Callable[[Step], bool] | None
    _seen_fingerprints: set[str] = field(default_factory=set, repr=False)

    def check(self, trace: Trace) -> Detection | None:
        recent = trace.tail(self.window)
        if len(recent) < self.window // 2:
            return None

        signals = self._analyze(recent)

        # Check for hedge-heavy conversations
        if signals["hedge_count"] >= self.hedge_limit:
            return Detection(
                mode=FailureMode.STALL,
                confidence=min(1.0, signals["hedge_count"] / (self.hedge_limit + 2)),
                evidence=(
                    f"{signals['hedge_count']} hedging/uncertain messages in "
                    f"last {len(recent)} steps"
                ),
                suggested_intervention="reprompt",
            )

        # Check for empty tool results
        if signals["empty_results"] >= self.empty_result_limit:
            return Detection(
                mode=FailureMode.STALL,
                confidence=min(1.0, signals["empty_results"] / (self.empty_result_limit + 2)),
                evidence=(
                    f"{signals['empty_results']} empty/near-empty tool results in "
                    f"last {len(recent)} steps"
                ),
                suggested_intervention="reprompt",
            )

        # Check overall progress rate
        if signals["total_actions"] > 0:
            progress_rate = signals["novel_actions"] / signals["total_actions"]
            if progress_rate < self.progress_threshold:
                return Detection(
                    mode=FailureMode.STALL,
                    confidence=round(1.0 - progress_rate, 3),
                    evidence=(
                        f"Only {signals['novel_actions']}/{signals['total_actions']} "
                        f"recent actions are novel (progress rate: {progress_rate:.0%})"
                    ),
                    suggested_intervention="reprompt",
                )

        return None

    def _analyze(self, steps: list[Step]) -> dict:
        hedge_count = 0
        empty_results = 0
        novel_actions = 0
        total_actions = 0

        for step in steps:
            if step.kind == StepKind.ASSISTANT_MESSAGE:
                content_lower = step.content.lower()
                if any(p in content_lower for p in _HEDGE_PATTERNS):
                    hedge_count += 1

            elif step.kind == StepKind.TOOL_RESULT:
                if len(step.content.strip()) < 10:
                    empty_results += 1

            elif step.kind == StepKind.TOOL_CALL:
                total_actions += 1
                if self.custom_progress_fn:
                    if self.custom_progress_fn(step):
                        novel_actions += 1
                else:
                    fp = step.fingerprint
                    if fp not in self._seen_fingerprints:
                        self._seen_fingerprints.add(fp)
                        novel_actions += 1

        return {
            "hedge_count": hedge_count,
            "empty_results": empty_results,
            "novel_actions": novel_actions,
            "total_actions": total_actions,
        }

    def reset(self) -> None:
        self._seen_fingerprints.clear()

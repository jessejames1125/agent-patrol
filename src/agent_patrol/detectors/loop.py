"""Detect repeating action patterns in agent execution traces.

Uses cycle detection over action keys to find repeated subsequences,
not just exact consecutive duplicates. For example, detects A->B->A->B
as a loop of period 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_patrol.detectors.base import Detection, Detector, FailureMode
from agent_patrol.trace import Step, StepKind, Trace


@dataclass
class LoopDetector(Detector):
    """Detects repeating action-call cycles in an agent trace.

    Parameters
    ----------
    window : int
        Number of recent tool calls to consider. Larger windows can catch
        longer-period loops but cost more to scan.
    min_repetitions : int
        How many times a cycle must repeat before it counts as a loop.
    max_period : int
        Maximum cycle length to search for. A period-1 loop is the same
        action repeated; period-2 is A->B->A->B, etc.
    similarity_threshold : float
        Fraction of steps in a candidate cycle that must match for the
        cycle to be considered a loop. 1.0 means exact match only.
    """

    window: int = 20
    min_repetitions: int = 3
    max_period: int = 6
    similarity_threshold: float = 0.9
    _last_alert_len: int = field(default=0, repr=False)

    def check(self, trace: Trace) -> Detection | None:
        calls = trace.tool_calls()
        if len(calls) < self.min_repetitions * 2:
            return None

        recent = calls[-self.window :]
        keys = [s.action_key for s in recent]

        best = _find_best_cycle(
            keys,
            max_period=self.max_period,
            min_reps=self.min_repetitions,
            threshold=self.similarity_threshold,
        )

        if best is None:
            return None

        period, reps, match_rate = best

        # Don't re-alert on the same loop unless it's gotten worse
        if len(calls) == self._last_alert_len:
            return None
        self._last_alert_len = len(calls)

        cycle_actions = keys[-period:]
        confidence = min(1.0, match_rate * (reps / (self.min_repetitions + 2)))

        return Detection(
            mode=FailureMode.LOOP,
            confidence=round(confidence, 3),
            evidence=(
                f"Detected cycle of period {period} repeated {reps} times "
                f"(match rate {match_rate:.0%}): {' -> '.join(cycle_actions)}"
            ),
            suggested_intervention="reprompt",
        )

    def reset(self) -> None:
        self._last_alert_len = 0


def _find_best_cycle(
    keys: list[str],
    max_period: int,
    min_reps: int,
    threshold: float,
) -> tuple[int, int, float] | None:
    """Find the strongest repeating cycle in a sequence of action keys.

    Checks if the last `period` keys repeat when walking backwards in
    aligned chunks of size `period`.

    Returns (period, repetitions, match_rate) or None.
    """
    n = len(keys)
    best: tuple[int, int, float] | None = None

    for period in range(1, min(max_period + 1, n // min_reps + 1)):
        cycle = keys[n - period : n]

        # Walk backwards in aligned chunks: [n-2p : n-p], [n-3p : n-2p], ...
        reps = 1  # count the reference cycle itself
        total_matches = period  # reference cycle matches itself
        total_compared = period

        chunk_end = n - period
        while chunk_end - period >= 0:
            chunk = keys[chunk_end - period : chunk_end]
            chunk_matches = sum(1 for a, b in zip(chunk, cycle) if a == b)
            total_matches += chunk_matches
            total_compared += period

            if chunk_matches / period >= threshold:
                reps += 1
            else:
                break
            chunk_end -= period

        if reps < min_reps:
            continue

        match_rate = total_matches / total_compared
        if match_rate < threshold:
            continue

        if best is None or reps > best[1] or (reps == best[1] and period < best[0]):
            best = (period, reps, match_rate)

    return best

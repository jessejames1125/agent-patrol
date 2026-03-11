"""Core patrol monitor that ties detectors and interventions together."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_patrol.detectors.base import Detection, Detector, FailureMode
from agent_patrol.interventions import (
    Intervention,
    InterventionResult,
    Reprompt,
)
from agent_patrol.trace import Step, StepKind, Trace

logger = logging.getLogger("agent_patrol")


@dataclass
class PatrolConfig:
    """Configuration for the patrol monitor.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence for a detection to trigger an intervention.
    check_interval : int
        Run detectors every N steps (not every single step, for performance).
    on_detection : callable or None
        Optional hook called on every detection, even below threshold.
        Signature: (Detection) -> None.
    """

    confidence_threshold: float = 0.5
    check_interval: int = 1
    on_detection: Callable[[Detection], None] | None = None


class Patrol:
    """The main monitor that watches an agent's execution trace.

    Usage
    -----
    >>> from agent_patrol import Patrol, LoopDetector, StallDetector
    >>> patrol = Patrol(task="Write a sorting algorithm")
    >>> patrol.add_detector(LoopDetector())
    >>> patrol.add_detector(StallDetector())
    >>>
    >>> # In your agent loop:
    >>> result = patrol.record_tool_call("bash", {"cmd": "python sort.py"}, "output here")
    >>> if result and result.should_halt:
    ...     break
    """

    def __init__(
        self,
        task: str = "",
        detectors: list[Detector] | None = None,
        interventions: dict[FailureMode, Intervention] | None = None,
        config: PatrolConfig | None = None,
    ):
        self.config = config or PatrolConfig()
        self.trace = Trace(task_description=task)
        self._detectors: list[Detector] = detectors or []
        self._interventions: dict[FailureMode, Intervention] = interventions or {
            FailureMode.LOOP: Reprompt(),
            FailureMode.STALL: Reprompt(),
            FailureMode.DRIFT: Reprompt(),
        }
        self._step_count = 0
        self._detections: list[Detection] = []

    def add_detector(self, detector: Detector) -> Patrol:
        self._detectors.append(detector)
        return self

    def set_intervention(self, mode: FailureMode, intervention: Intervention) -> Patrol:
        self._interventions[mode] = intervention
        return self

    def record_step(self, step: Step) -> InterventionResult | None:
        """Record a step and run detectors if due."""
        self.trace.append(step)
        self._step_count += 1

        if self._step_count % self.config.check_interval != 0:
            return None

        return self._run_detectors()

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        result: str = "",
    ) -> InterventionResult | None:
        """Convenience: record a tool call + result pair and check for failures."""
        call_result = self.record_step(Step(
            kind=StepKind.TOOL_CALL,
            content=f"{tool_name}({tool_args})",
            tool_name=tool_name,
            tool_args=tool_args,
        ))
        result_result = self.record_step(Step(
            kind=StepKind.TOOL_RESULT,
            content=result,
            tool_name=tool_name,
        ))
        # Return the most urgent intervention (prefer halt over non-halt)
        if call_result and call_result.should_halt:
            return call_result
        if result_result and result_result.should_halt:
            return result_result
        return call_result or result_result

    def record_message(self, content: str, role: str = "assistant") -> InterventionResult | None:
        """Convenience: record an assistant or user message."""
        kind = StepKind.ASSISTANT_MESSAGE if role == "assistant" else StepKind.USER_MESSAGE
        return self.record_step(Step(kind=kind, content=content))

    def check_now(self) -> InterventionResult | None:
        """Force an immediate check regardless of check_interval."""
        return self._run_detectors()

    @property
    def detections(self) -> list[Detection]:
        """All detections so far."""
        return list(self._detections)

    def reset(self) -> None:
        """Reset all state."""
        self.trace = Trace(task_description=self.trace.task_description)
        self._step_count = 0
        self._detections.clear()
        for d in self._detectors:
            d.reset()

    def _run_detectors(self) -> InterventionResult | None:
        worst: Detection | None = None

        for detector in self._detectors:
            detection = detector.check(self.trace)
            if detection is None:
                continue

            self._detections.append(detection)

            if self.config.on_detection:
                self.config.on_detection(detection)

            logger.debug(
                "Detection: %s (confidence=%.2f) %s",
                detection.mode.name,
                detection.confidence,
                detection.evidence,
            )

            if detection.confidence >= self.config.confidence_threshold:
                if worst is None or detection.confidence > worst.confidence:
                    worst = detection

        if worst is None:
            return None

        intervention = self._interventions.get(worst.mode)
        if intervention is None:
            logger.warning("No intervention configured for %s", worst.mode.name)
            return None

        context = {"task": self.trace.task_description, "trace_length": len(self.trace)}
        result = intervention.act(worst, context)
        logger.info(
            "Intervention: %s -> %s",
            worst.mode.name,
            result.action,
        )
        return result

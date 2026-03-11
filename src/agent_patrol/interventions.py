"""Built-in intervention strategies for responding to detected failures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from agent_patrol.detectors.base import Detection, FailureMode


class Intervention(ABC):
    """Base class for intervention strategies."""

    @abstractmethod
    def act(self, detection: Detection, context: dict[str, Any]) -> InterventionResult:
        ...


@dataclass
class InterventionResult:
    """The outcome of an intervention."""

    action: str  # "reprompt", "abort", "escalate", "custom"
    message: str | None = None
    should_halt: bool = False
    metadata: dict[str, Any] | None = None


class Reprompt(Intervention):
    """Inject a corrective message back into the agent's context.

    Parameters
    ----------
    template : str or None
        Custom message template. Can use {evidence}, {mode}, {task} placeholders.
    """

    def __init__(self, template: str | None = None):
        self.template = template or (
            "You appear to be {mode}. {evidence} "
            "Please refocus on the original task: {task}"
        )

    def act(self, detection: Detection, context: dict[str, Any]) -> InterventionResult:
        msg = self.template.format(
            mode=detection.mode.name.lower() + "ing",
            evidence=detection.evidence,
            task=context.get("task", "the original task"),
        )
        return InterventionResult(action="reprompt", message=msg)


class Abort(Intervention):
    """Halt execution immediately."""

    def __init__(self, reason: str = "Agent failure detected"):
        self.reason = reason

    def act(self, detection: Detection, context: dict[str, Any]) -> InterventionResult:
        return InterventionResult(
            action="abort",
            message=f"Aborted: {self.reason} ({detection.evidence})",
            should_halt=True,
        )


class Escalate(Intervention):
    """Call a user-provided callback to handle the failure.

    Parameters
    ----------
    callback : callable
        Function (Detection, context) -> None. Could send a Slack message,
        log to an external service, page a human, etc.
    """

    def __init__(self, callback: Callable[[Detection, dict[str, Any]], None]):
        self.callback = callback

    def act(self, detection: Detection, context: dict[str, Any]) -> InterventionResult:
        self.callback(detection, context)
        return InterventionResult(
            action="escalate",
            message=f"Escalated: {detection.mode.name} detected",
            metadata={"detection": detection},
        )


class CompositeIntervention(Intervention):
    """Run multiple interventions in sequence."""

    def __init__(self, *interventions: Intervention):
        self.interventions = interventions

    def act(self, detection: Detection, context: dict[str, Any]) -> InterventionResult:
        last_result = None
        for intervention in self.interventions:
            last_result = intervention.act(detection, context)
            if last_result.should_halt:
                return last_result
        return last_result or InterventionResult(action="noop")

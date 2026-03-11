"""Detect when an agent drifts away from its original task.

Supports two modes:
  - Keyword-based (default, zero dependencies): extracts salient terms from
    the task description and checks what fraction appear in recent actions.
  - Embedding-based (optional, requires numpy): uses a user-provided embedding
    function for semantic similarity.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from agent_patrol.detectors.base import Detection, Detector, FailureMode
from agent_patrol.trace import Step, StepKind, Trace

# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could to of in for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all each every both few more most other some such no nor "
    "not only own same so than too very just don t s it its and but or "
    "if this that these those i me my we our you your he him his she "
    "her they them their what which who whom".split()
)


@dataclass
class DriftDetector(Detector):
    """Detects when recent agent activity diverges from the original task.

    Parameters
    ----------
    window : int
        Number of recent steps to compare against the task description.
    threshold : float
        Minimum relevance score (0-1) below which drift is flagged.
    min_steps : int
        Don't check for drift until the trace has at least this many steps.
    embed_fn : callable or None
        Optional function (str) -> list[float] for embedding-based similarity.
        If provided, uses cosine similarity instead of keyword overlap.
    """

    window: int = 15
    threshold: float = 0.25
    min_steps: int = 8
    embed_fn: object = None  # Callable[[str], list[float]] | None
    _task_keywords: set[str] = field(default_factory=set, repr=False)
    _task_embedding: list[float] | None = field(default=None, repr=False)

    def check(self, trace: Trace) -> Detection | None:
        if len(trace) < self.min_steps or not trace.task_description:
            return None

        if self.embed_fn is not None:
            return self._check_embedding(trace)
        return self._check_keywords(trace)

    def _check_keywords(self, trace: Trace) -> Detection | None:
        if not self._task_keywords:
            self._task_keywords = _extract_keywords(trace.task_description)

        if not self._task_keywords:
            return None

        recent = trace.tail(self.window)
        recent_text = " ".join(
            s.content for s in recent
            if s.kind in (StepKind.TOOL_CALL, StepKind.ASSISTANT_MESSAGE)
        )
        recent_keywords = _extract_keywords(recent_text)

        if not recent_keywords:
            return None

        overlap = self._task_keywords & recent_keywords
        relevance = len(overlap) / len(self._task_keywords)

        if relevance >= self.threshold:
            return None

        return Detection(
            mode=FailureMode.DRIFT,
            confidence=round(1.0 - relevance, 3),
            evidence=(
                f"Task relevance dropped to {relevance:.0%}. "
                f"Task keywords: {_top_n(self._task_keywords, 8)}. "
                f"Recent keywords: {_top_n(recent_keywords, 8)}."
            ),
            suggested_intervention="reprompt",
        )

    def _check_embedding(self, trace: Trace) -> Detection | None:
        try:
            import numpy as np
        except ImportError:
            return self._check_keywords(trace)

        if self._task_embedding is None:
            self._task_embedding = self.embed_fn(trace.task_description)

        recent = trace.tail(self.window)
        recent_text = " ".join(
            s.content for s in recent
            if s.kind in (StepKind.TOOL_CALL, StepKind.ASSISTANT_MESSAGE)
        )

        if not recent_text.strip():
            return None

        recent_emb = self.embed_fn(recent_text)
        similarity = _cosine_similarity(
            np.array(self._task_embedding),
            np.array(recent_emb),
        )

        if similarity >= self.threshold:
            return None

        return Detection(
            mode=FailureMode.DRIFT,
            confidence=round(1.0 - similarity, 3),
            evidence=f"Semantic similarity to task dropped to {similarity:.2f} (threshold: {self.threshold})",
            suggested_intervention="reprompt",
        )

    def reset(self) -> None:
        self._task_keywords.clear()
        self._task_embedding = None


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z_]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _top_n(keywords: set[str], n: int) -> str:
    return ", ".join(sorted(keywords)[:n])


def _cosine_similarity(a, b) -> float:
    dot = float(a @ b)
    norm = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
    if norm == 0:
        return 0.0
    return dot / norm

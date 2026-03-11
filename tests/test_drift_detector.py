"""Tests for drift detection."""

from agent_patrol import DriftDetector, Step, StepKind, Trace


def test_no_drift_on_short_trace():
    detector = DriftDetector(min_steps=8)
    trace = Trace(task_description="implement a sorting algorithm")
    for i in range(3):
        trace.append(Step(kind=StepKind.TOOL_CALL, content="edit()", tool_name="edit"))
    assert detector.check(trace) is None


def test_no_drift_when_on_task():
    detector = DriftDetector(min_steps=4, window=10, threshold=0.2)
    trace = Trace(task_description="implement a quicksort algorithm in python")
    for _ in range(5):
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content="Writing quicksort partition function in python",
            tool_name="edit",
            tool_args={"file": "sort.py"},
        ))
        trace.append(Step(
            kind=StepKind.ASSISTANT_MESSAGE,
            content="Now implementing the quicksort algorithm recursion",
        ))

    assert detector.check(trace) is None


def test_detects_drift():
    detector = DriftDetector(min_steps=4, window=10, threshold=0.3)
    trace = Trace(task_description="implement a quicksort algorithm in python")

    # Start on task
    trace.append(Step(kind=StepKind.ASSISTANT_MESSAGE, content="implementing quicksort"))

    # Then drift to something unrelated
    for _ in range(6):
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content="Setting up docker container for database migration",
            tool_name="bash",
            tool_args={"cmd": "docker compose up"},
        ))
        trace.append(Step(
            kind=StepKind.ASSISTANT_MESSAGE,
            content="Now configuring the nginx reverse proxy and SSL certificates",
        ))

    result = detector.check(trace)
    assert result is not None
    assert result.confidence > 0


def test_no_drift_without_task_description():
    detector = DriftDetector(min_steps=2)
    trace = Trace(task_description="")
    for _ in range(10):
        trace.append(Step(kind=StepKind.TOOL_CALL, content="random stuff", tool_name="bash"))
    assert detector.check(trace) is None


def test_reset_clears_cached_keywords():
    detector = DriftDetector()
    trace = Trace(task_description="implement sorting")
    # Force keyword extraction
    detector._task_keywords = {"sorting", "implement"}
    detector.reset()
    assert len(detector._task_keywords) == 0

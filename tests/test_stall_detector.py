"""Tests for stall detection."""

from agent_patrol import StallDetector, Step, StepKind, Trace


def test_no_stall_on_short_trace():
    detector = StallDetector(window=10)
    trace = Trace(task_description="test")
    trace.append(Step(kind=StepKind.TOOL_CALL, content="bash()", tool_name="bash"))
    assert detector.check(trace) is None


def test_detects_hedge_stall():
    """Many hedging messages should trigger a stall."""
    detector = StallDetector(window=10, hedge_limit=3)
    trace = Trace(task_description="test")
    for _ in range(6):
        trace.append(Step(
            kind=StepKind.ASSISTANT_MESSAGE,
            content="Let me try again, I apologize for the confusion.",
        ))
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content="bash()",
            tool_name="bash",
            tool_args={"cmd": "echo retry"},
        ))

    result = detector.check(trace)
    assert result is not None
    assert "hedging" in result.evidence


def test_detects_empty_result_stall():
    """Many empty tool results should trigger a stall."""
    detector = StallDetector(window=10, empty_result_limit=3)
    trace = Trace(task_description="test")
    for _ in range(6):
        trace.append(Step(kind=StepKind.TOOL_CALL, content="search()", tool_name="search"))
        trace.append(Step(kind=StepKind.TOOL_RESULT, content="", tool_name="search"))

    result = detector.check(trace)
    assert result is not None
    assert "empty" in result.evidence


def test_detects_low_novelty_stall():
    """Repeated identical tool calls with no new information."""
    detector = StallDetector(window=10, progress_threshold=0.3)
    trace = Trace(task_description="test")
    # First call is novel, rest are duplicates
    for _ in range(8):
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content="bash({'cmd': 'cat file.py'})",
            tool_name="bash",
            tool_args={"cmd": "cat file.py"},
        ))
        trace.append(Step(
            kind=StepKind.TOOL_RESULT,
            content="some file content here that is not empty",
            tool_name="bash",
        ))

    result = detector.check(trace)
    assert result is not None
    assert "novel" in result.evidence


def test_no_stall_on_productive_trace():
    """Varied, productive actions should not trigger."""
    detector = StallDetector(window=10, progress_threshold=0.3)
    trace = Trace(task_description="test")
    for i in range(6):
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content=f"edit({i})",
            tool_name="edit",
            tool_args={"file": f"file_{i}.py", "line": i},
        ))
        trace.append(Step(
            kind=StepKind.TOOL_RESULT,
            content=f"Successfully edited file_{i}.py",
            tool_name="edit",
        ))

    assert detector.check(trace) is None


def test_custom_progress_fn():
    """User-provided progress function should override default heuristics."""
    detector = StallDetector(
        window=10,
        progress_threshold=0.5,
        custom_progress_fn=lambda step: "write" in (step.tool_name or ""),
    )
    trace = Trace(task_description="test")
    for _ in range(6):
        trace.append(Step(
            kind=StepKind.TOOL_CALL,
            content="read()",
            tool_name="read",
            tool_args={"file": "foo.py"},
        ))

    result = detector.check(trace)
    assert result is not None

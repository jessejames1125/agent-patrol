"""Tests for loop detection."""

from agent_patrol import LoopDetector, Step, StepKind, Trace


def _tool_step(name: str, args: dict | None = None) -> Step:
    return Step(kind=StepKind.TOOL_CALL, content=f"{name}()", tool_name=name, tool_args=args)


def test_no_loop_on_empty_trace():
    detector = LoopDetector()
    trace = Trace(task_description="test")
    assert detector.check(trace) is None


def test_no_loop_with_varied_actions():
    detector = LoopDetector(min_repetitions=3)
    trace = Trace(task_description="test")
    for i in range(10):
        trace.append(_tool_step(f"tool_{i}", {"arg": i}))
    assert detector.check(trace) is None


def test_detects_period_1_loop():
    """Same tool called repeatedly = period-1 loop."""
    detector = LoopDetector(min_repetitions=3, window=20)
    trace = Trace(task_description="test")
    for _ in range(8):
        trace.append(_tool_step("bash", {"cmd": "python main.py"}))

    result = detector.check(trace)
    assert result is not None
    assert result.confidence > 0
    assert "period 1" in result.evidence


def test_detects_period_2_loop():
    """A->B->A->B->A->B pattern."""
    detector = LoopDetector(min_repetitions=3, window=20)
    trace = Trace(task_description="test")
    for _ in range(6):
        trace.append(_tool_step("read_file", {"path": "foo.py"}))
        trace.append(_tool_step("edit_file", {"path": "foo.py"}))

    result = detector.check(trace)
    assert result is not None
    assert result.confidence > 0


def test_no_false_positive_on_similar_but_different():
    """Tools with same name but different args should not trigger."""
    detector = LoopDetector(min_repetitions=3)
    trace = Trace(task_description="test")
    for i in range(10):
        trace.append(_tool_step("search", {"query": f"term_{i}"}))

    assert detector.check(trace) is None


def test_reset_clears_state():
    detector = LoopDetector()
    trace = Trace(task_description="test")
    for _ in range(8):
        trace.append(_tool_step("bash", {"cmd": "echo hi"}))
    detector.check(trace)
    detector.reset()
    assert detector._last_alert_len == 0

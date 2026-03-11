"""Tests for the Patrol monitor (integration)."""

from agent_patrol import (
    Abort,
    FailureMode,
    LoopDetector,
    Patrol,
    PatrolConfig,
    Reprompt,
    StallDetector,
)


def test_patrol_basic_usage():
    patrol = Patrol(
        task="Write a sorting algorithm",
        detectors=[LoopDetector(), StallDetector(window=20)],
    )
    # A few varied tool calls should not trigger anything
    for i in range(5):
        result = patrol.record_tool_call(f"tool_{i}", {"arg": i}, f"Successfully completed operation {i}")
        assert result is None


def test_patrol_detects_loop():
    patrol = Patrol(
        task="Write tests",
        detectors=[LoopDetector(min_repetitions=3)],
        config=PatrolConfig(confidence_threshold=0.1),
    )
    for _ in range(8):
        result = patrol.record_tool_call("bash", {"cmd": "pytest"}, "FAILED")

    # Should have at least one detection
    assert len(patrol.detections) > 0
    assert any(d.mode == FailureMode.LOOP for d in patrol.detections)


def test_patrol_abort_halts():
    patrol = Patrol(
        task="Test task",
        detectors=[LoopDetector(min_repetitions=3)],
        interventions={FailureMode.LOOP: Abort("loop detected")},
        config=PatrolConfig(confidence_threshold=0.1),
    )
    halted = False
    for _ in range(10):
        result = patrol.record_tool_call("bash", {"cmd": "echo hi"}, "hi")
        if result and result.should_halt:
            halted = True
            break

    assert halted


def test_patrol_reprompt_message():
    patrol = Patrol(
        task="Build a parser",
        detectors=[LoopDetector(min_repetitions=3)],
        interventions={FailureMode.LOOP: Reprompt("Hey, you're looping. Task: {task}")},
        config=PatrolConfig(confidence_threshold=0.1),
    )
    result = None
    for _ in range(10):
        result = patrol.record_tool_call("bash", {"cmd": "make"}, "error")

    # Find the last non-None result
    assert any(d.mode == FailureMode.LOOP for d in patrol.detections)


def test_patrol_on_detection_callback():
    detections_seen = []
    patrol = Patrol(
        task="Test",
        detectors=[LoopDetector(min_repetitions=3)],
        config=PatrolConfig(
            confidence_threshold=0.1,
            on_detection=lambda d: detections_seen.append(d),
        ),
    )
    for _ in range(8):
        patrol.record_tool_call("bash", {"cmd": "echo"}, "")

    assert len(detections_seen) > 0


def test_patrol_reset():
    patrol = Patrol(
        task="Test",
        detectors=[LoopDetector()],
    )
    for _ in range(5):
        patrol.record_tool_call("bash", {"cmd": "echo"}, "")

    patrol.reset()
    assert len(patrol.trace) == 0
    assert len(patrol.detections) == 0


def test_patrol_record_message():
    patrol = Patrol(
        task="Test",
        detectors=[StallDetector(window=6, hedge_limit=3)],
        config=PatrolConfig(confidence_threshold=0.1),
    )
    for _ in range(6):
        result = patrol.record_message("Let me try again, I apologize")

    assert len(patrol.detections) > 0


def test_check_interval():
    """Detectors should only run every check_interval steps."""
    call_count = [0]

    class CountingDetector:
        def check(self, trace):
            call_count[0] += 1
            return None
        def reset(self):
            pass

    patrol = Patrol(
        task="Test",
        detectors=[CountingDetector()],
        config=PatrolConfig(check_interval=3),
    )
    for i in range(9):
        patrol.record_step(
            __import__("agent_patrol").Step(
                kind=__import__("agent_patrol").StepKind.TOOL_CALL,
                content="x",
                tool_name="x",
            )
        )

    assert call_count[0] == 3  # 9 steps / interval of 3

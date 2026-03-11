"""Basic example: monitor an agent loop for common failure modes."""

from agent_patrol import (
    Abort,
    DriftDetector,
    Escalate,
    FailureMode,
    LoopDetector,
    Patrol,
    PatrolConfig,
    Reprompt,
    StallDetector,
)

# Set up patrol with all three detectors
patrol = Patrol(
    task="Refactor the authentication module to use JWT tokens",
    detectors=[
        LoopDetector(window=20, min_repetitions=3),
        StallDetector(window=10, hedge_limit=4),
        DriftDetector(window=15, threshold=0.25),
    ],
    interventions={
        FailureMode.LOOP: Reprompt(
            "You've been repeating the same actions. {evidence} "
            "Try a different approach to: {task}"
        ),
        FailureMode.STALL: Reprompt(
            "You don't seem to be making progress. {evidence} "
            "Focus on the next concrete step for: {task}"
        ),
        FailureMode.DRIFT: Reprompt(
            "You've drifted from the original task. {evidence} "
            "Get back to: {task}"
        ),
    },
    config=PatrolConfig(
        confidence_threshold=0.5,
        check_interval=2,
    ),
)


# Simulate an agent that gets stuck in a loop
def simulate_looping_agent():
    print("Simulating an agent stuck in a loop...\n")
    for i in range(12):
        result = patrol.record_tool_call(
            "bash",
            {"cmd": "python test_auth.py"},
            "FAILED: ImportError: jwt module not found",
        )
        if result:
            print(f"  Step {i}: INTERVENTION -> {result.action}")
            if result.message:
                print(f"    Message: {result.message}")
            if result.should_halt:
                print("    Agent halted.")
                return
        else:
            print(f"  Step {i}: ok")

    print(f"\nTotal detections: {len(patrol.detections)}")


if __name__ == "__main__":
    simulate_looping_agent()

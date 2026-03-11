# agent-patrol

Lightweight failure-mode detection for LLM agents. Catches loops, stalls, and task drift before they waste tokens and time.

## The problem

Long-running LLM agents fail in predictable ways:

- **Loops** — the agent repeats the same tool call (or cycle of calls) without realizing it's stuck
- **Stalls** — the agent hedges, retries, and apologizes without making forward progress
- **Drift** — the agent gradually wanders away from the original task into unrelated work

Most agent frameworks handle this with a blunt `max_iterations` counter. That stops the bleeding but doesn't diagnose the problem, distinguish productive iteration from unproductive looping, or allow targeted recovery.

## What this does

`agent-patrol` monitors your agent's execution trace in real-time and applies pattern detection to catch these failure modes early. When a failure is detected, it triggers a configurable intervention — re-prompt the agent, abort, escalate to a human, or run custom logic.

It's framework-agnostic. Works with any agent loop that makes tool calls.

## Install

```bash
pip install agent-patrol
```

For embedding-based drift detection (optional):

```bash
pip install agent-patrol[embeddings]
```

## Quick start

```python
from agent_patrol import (
    LoopDetector,
    StallDetector,
    DriftDetector,
    Patrol,
    PatrolConfig,
)

patrol = Patrol(
    task="Refactor auth module to use JWT",
    detectors=[
        LoopDetector(),
        StallDetector(),
        DriftDetector(),
    ],
)

# In your agent loop:
while not done:
    action = agent.step()
    result = patrol.record_tool_call(
        tool_name=action.tool,
        tool_args=action.args,
        result=action.output,
    )
    if result and result.should_halt:
        break
    if result and result.message:
        # Inject the corrective message back into the agent's context
        agent.add_message(result.message)
```

## Detectors

### LoopDetector

Finds repeating action cycles using sequence matching. Catches not just exact repeats (`A→A→A`) but multi-step cycles (`A→B→A→B→A→B`).

```python
LoopDetector(
    window=20,           # how many recent tool calls to scan
    min_repetitions=3,   # how many cycle repeats before alerting
    max_period=6,        # longest cycle length to search for
    similarity_threshold=0.9,  # how exact the match must be
)
```

### StallDetector

Tracks progress signals: novel vs. repeated actions, hedging language in messages, and empty tool results.

```python
StallDetector(
    window=10,
    progress_threshold=0.3,  # fraction of actions that must be novel
    hedge_limit=4,           # hedging messages before alerting
    empty_result_limit=4,    # empty tool results before alerting
    custom_progress_fn=None, # optional (Step) -> bool override
)
```

### DriftDetector

Compares recent agent activity against the original task description. Two modes:

- **Keyword-based** (default, zero deps): extracts salient terms and checks overlap
- **Embedding-based** (optional): uses a user-provided embedding function for semantic similarity

```python
# Keyword mode (default)
DriftDetector(window=15, threshold=0.25)

# Embedding mode
DriftDetector(
    embed_fn=my_embedding_function,  # (str) -> list[float]
    threshold=0.4,
)
```

## Interventions

Built-in strategies for responding to detections:

```python
from agent_patrol import Reprompt, Abort, Escalate, FailureMode

patrol.set_intervention(
    FailureMode.LOOP,
    Reprompt("You're looping. Try a different approach to: {task}")
)

patrol.set_intervention(
    FailureMode.STALL,
    Abort("Agent stalled, giving up")
)

patrol.set_intervention(
    FailureMode.DRIFT,
    Escalate(callback=lambda detection, ctx: notify_human(detection))
)
```

`Reprompt` templates support `{task}`, `{evidence}`, and `{mode}` placeholders.

## Recording steps

Three ways to feed data into the monitor:

```python
# 1. Tool call + result (most common)
patrol.record_tool_call("bash", {"cmd": "pytest"}, "3 passed, 1 failed")

# 2. Messages
patrol.record_message("Let me try a different approach", role="assistant")

# 3. Raw steps (full control)
from agent_patrol import Step, StepKind
patrol.record_step(Step(
    kind=StepKind.TOOL_CALL,
    content="searching for auth module",
    tool_name="grep",
    tool_args={"pattern": "class Auth"},
))
```

## Configuration

```python
from agent_patrol import PatrolConfig

config = PatrolConfig(
    confidence_threshold=0.5,  # minimum confidence to trigger intervention
    check_interval=2,          # run detectors every N steps (performance)
    on_detection=my_logger,    # callback for ALL detections, even below threshold
)
```

## Design decisions

- **No external dependencies** for core functionality. Numpy is optional, only for embedding-based drift detection.
- **Framework-agnostic.** No coupling to LangChain, OpenAI, Anthropic, or any specific agent framework. If your agent makes tool calls, this works.
- **Detection, not prevention.** This library tells you what's happening and suggests interventions. It doesn't hijack your agent loop — you decide what to do with the information.
- **Cheap to run.** Detectors operate on small windows of recent history, not the full trace. Cycle detection is O(window * max_period), not O(n^2).

## License

MIT

# Agent Stopping Conditions & Safety

Without stopping conditions, an agent can loop forever, consuming tokens, money, and time. These guards are essential.

---

## 1. `max_iterations`

The most important guard. Stops the agent after N tool-call cycles.

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,   # stop after 10 Thought→Action→Observation cycles
    verbose=True,
)
```

When `max_iterations` is hit:
- Returns the best answer found so far
- Or raises `AgentFinish` with a "stopped early" message

---

## 2. `max_execution_time`

Time-based limit in seconds. Useful when tools can be slow (web requests, database queries).

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=30,   # stop after 30 seconds total
    max_iterations=15,
)
```

---

## 3. `early_stopping_method`

Controls what happens when `max_iterations` or `max_execution_time` is reached.

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    early_stopping_method="generate",  # ask LLM for best answer with info gathered so far
    # OR
    early_stopping_method="force",     # just return "Agent stopped due to iteration limit."
)
```

---

## 4. `handle_parsing_errors`

When the LLM produces output that can't be parsed as a tool call (malformatted JSON, wrong tool name), this prevents a crash.

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # LLM gets to see the error and try again
    # OR provide a custom message:
    handle_parsing_errors="Please reformat your output and try again.",
)
```

---

## 5. Final Answer Detection

For ReAct agents, the loop ends when the LLM outputs `"Final Answer:"` in its response. This is a text-based signal.

For tool-calling agents, the loop ends when the LLM response contains **no `tool_calls`** — it's just text.

---

## 6. LangGraph — Interrupt Before Dangerous Actions

In LangGraph, you can pause the agent before executing specific nodes (e.g., a node that sends emails or deletes data) and require human approval.

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email"],  # pause before this node
    interrupt_after=["web_search"],   # pause after this node
)

# Run until interrupt
state = graph.invoke(input, config)
# Agent is paused — human can review state.values

# Resume after approval
graph.invoke(None, config)  # None = continue from checkpoint
```

---

## 7. Tool-Level Timeouts

Prevent a single slow tool from stalling the entire agent:

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Tool timed out after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@tool
def slow_web_search(query: str) -> str:
    """Search the web. May be slow."""
    try:
        with timeout(10):  # 10 second limit
            return actual_search(query)
    except TimeoutError:
        return "Search timed out. Please try a simpler query."
```

---

## 8. Token Budget Guard

Track total tokens used and stop before hitting limits or budget:

```python
class TokenBudgetCallbackHandler:
    def __init__(self, max_tokens=10000):
        self.total_tokens = 0
        self.max_tokens = max_tokens

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        if self.total_tokens > self.max_tokens:
            raise Exception(f"Token budget exceeded: {self.total_tokens}")
```

---

## Safety Checklist for Agents

```
□ Set max_iterations (always — no exception)
□ Set max_execution_time for long-running tools
□ Use handle_parsing_errors=True
□ Validate and sanitize tool inputs
□ Return errors as strings from tools (don't raise exceptions)
□ Use HITL (interrupt_before) for irreversible actions
□ Log all agent steps (LangSmith or custom)
□ Never give agents more tools than they need
□ Review tool descriptions — they should limit scope clearly
□ Test agent failure modes explicitly
```

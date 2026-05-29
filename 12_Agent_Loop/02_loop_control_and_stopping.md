# Loop Control — Stopping, Safety & Limits

Without proper controls, an agent loop can run forever, burn tokens, or get stuck. Robust agents need explicit stopping mechanisms.

---

## Stopping Conditions

An agent loop ends when **any** of these is true:

| Condition | Mechanism |
|---|---|
| LLM returns no tool calls | `tools_condition` routes to END |
| Max iterations reached | Counter check or `recursion_limit` |
| Max execution time exceeded | Timeout wrapper |
| Task explicitly marked complete | Special "done" tool or state flag |
| Human interrupts | `interrupt_before` in LangGraph |
| Error threshold exceeded | Custom error counting in state |

---

## 1. Max Iterations

### In a Manual Loop

```python
MAX_ITERATIONS = 15
iteration = 0

while True:
    iteration += 1
    if iteration > MAX_ITERATIONS:
        # Return best-effort answer instead of crashing
        messages.append(AIMessage(
            content=f"I reached the maximum of {MAX_ITERATIONS} steps. "
                    f"Here's what I found so far: ..."
        ))
        break
    
    response = llm.invoke(messages)
    messages.append(response)
    
    if not response.tool_calls:
        break   # Natural stopping condition
    
    # ... execute tools
```

### In LangGraph (recursion_limit)

```python
# LangGraph enforces a hard cap on graph steps
graph = builder.compile()

result = graph.invoke(
    {"messages": [HumanMessage("Research quantum computing")]},
    config={"recursion_limit": 25},   # default is 25
)

# If limit is hit: raises GraphRecursionError
from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke(input, config={"recursion_limit": 10})
except GraphRecursionError:
    print("Agent hit recursion limit — returning partial result")
```

---

## 2. Token Budget Guard

Track token usage and stop before hitting context window limits:

```python
from langchain_core.messages import trim_messages, BaseMessage

def agent_with_token_budget(state: State) -> State:
    messages = state["messages"]
    
    # Trim conversation to stay within 8K tokens
    trimmed = trim_messages(
        messages,
        max_tokens=8000,
        token_counter=ChatOpenAI(model="gpt-4o"),
        strategy="last",           # keep most recent messages
        include_system=True,       # always keep system message
        allow_partial=False,
    )
    
    response = llm_with_tools.invoke(trimmed)
    return {"messages": [response]}
```

---

## 3. Timeout

```python
import asyncio
from functools import wraps

async def run_agent_with_timeout(graph, input_data, config, timeout_seconds=60):
    """Run agent with a hard time limit."""
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(input_data, config=config),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"Agent timed out after {timeout_seconds}s", "partial": True}

# Usage
result = asyncio.run(
    run_agent_with_timeout(graph, {"messages": [HumanMessage("...")]}, config, timeout_seconds=30)
)
```

---

## 4. Task-Complete Signal (Explicit Done Tool)

Instead of relying on "no tool calls", give the agent an explicit `task_complete` tool:

```python
from langchain_core.tools import tool
from pydantic import BaseModel

class TaskCompleteInput(BaseModel):
    summary: str
    confidence: float  # 0.0 to 1.0

@tool(args_schema=TaskCompleteInput)
def task_complete(summary: str, confidence: float) -> str:
    """
    Call this tool when you have fully completed the user's request.
    Do NOT call this until you have all the information needed.
    
    Args:
        summary: A brief summary of what you accomplished
        confidence: Your confidence level (0.0-1.0) that the task is done
    """
    return f"TASK_COMPLETE: {summary} (confidence: {confidence})"

# In the loop: detect TASK_COMPLETE tool call and route to END
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "task_complete":
            return "end"
    return "continue"
```

---

## 5. Error Loop Detection

Detect when the agent is stuck in a repetitive error cycle:

```python
from collections import Counter
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    error_count: int          # track consecutive errors
    tool_call_history: list   # track which tools were called

def agent(state: State) -> State:
    # Detect repetitive tool calls (stuck in a loop)
    recent_calls = state.get("tool_call_history", [])[-6:]  # last 6 calls
    if len(recent_calls) >= 4:
        call_counts = Counter(recent_calls)
        if max(call_counts.values()) >= 3:
            # Same tool called 3+ times recently — agent is stuck
            return {
                "messages": [AIMessage(
                    "I seem to be stuck in a loop. Let me try a different approach or ask for clarification."
                )],
                "tool_call_history": [],
            }
    
    response = llm_with_tools.invoke(state["messages"])
    
    # Track calls made this iteration
    new_calls = [tc["name"] for tc in (response.tool_calls or [])]
    
    return {
        "messages": [response],
        "tool_call_history": state.get("tool_call_history", []) + new_calls,
    }
```

---

## 6. Human-in-the-Loop Interrupt

Stop the loop and wait for human approval before continuing:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause BEFORE executing any tool
)

config = {"configurable": {"thread_id": "safe_agent_1"}}

# Run until the interrupt
result = graph.invoke({"messages": [HumanMessage("Delete all temp files")]}, config)

# Show pending tool calls to human
current_state = graph.get_state(config)
pending = current_state.values["messages"][-1].tool_calls
print("Agent wants to call:")
for call in pending:
    print(f"  {call['name']}({call['args']})")

# Human approves → resume
approved = input("Approve? (y/n): ")
if approved == "y":
    graph.invoke(None, config)   # resume from checkpoint
else:
    # Inject a message refusing the action
    graph.update_state(config, {
        "messages": [HumanMessage("Do not proceed. Find an alternative approach.")]
    })
    graph.invoke(None, config)
```

---

## Stopping Condition Decision Tree

```
Agent response received
        │
        ▼
Has tool_calls? ──No──► Return final answer → END
        │
       Yes
        │
        ▼
Is "task_complete" tool? ──Yes──► END
        │
        No
        ▼
recursion_limit hit? ──Yes──► Raise error / return partial
        │
        No
        ▼
Token budget exceeded? ──Yes──► Trim + continue with warning
        │
        No
        ▼
Stuck in repetition? ──Yes──► Inject "try different approach" message
        │
        No
        ▼
Error count > threshold? ──Yes──► Return graceful failure
        │
        No
        ▼
Interrupt configured? ──Yes──► Pause for human review
        │
        No
        ▼
Execute tools → loop back
```

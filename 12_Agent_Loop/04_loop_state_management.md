# Loop State Management

State is the memory of the loop — everything the agent knows between iterations. Managing it correctly is critical for reliability.

---

## What Goes in Loop State?

```python
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # ── Core (required) ──────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]   # full conversation

    # ── Loop tracking ────────────────────────────────────────────────
    iteration: int              # how many times we've looped
    tool_call_count: int        # total tool calls made
    
    # ── Task context ─────────────────────────────────────────────────
    goal: str                   # original user objective
    sub_goals: List[str]        # decomposed tasks
    completed_sub_goals: List[str]  # what's been finished
    
    # ── Quality control ──────────────────────────────────────────────
    error_count: int            # consecutive errors
    last_error: Optional[str]   # most recent error message
    
    # ── Outputs ──────────────────────────────────────────────────────
    gathered_facts: List[str]   # information collected so far
    final_answer: Optional[str] # set when complete
```

---

## State Reducers

Reducers define how state fields are updated when nodes return partial state:

```python
import operator
from typing import Annotated

def keep_latest(old, new):
    """Replace old value with new value."""
    return new

def append_unique(old: list, new: list) -> list:
    """Append only items not already in the list."""
    return old + [item for item in new if item not in old]

class State(TypedDict):
    # add_messages: appends new messages, updates by ID, respects RemoveMessage
    messages: Annotated[list, add_messages]
    
    # operator.add: simple list concatenation
    facts: Annotated[list, operator.add]
    
    # Default (no annotation): last-write-wins (replaced each time)
    current_plan: str
    iteration: int
```

---

## Passing State Between Iterations

Each time around the loop, the node receives the **full current state** and returns a **partial update**:

```python
def agent_node(state: AgentState) -> dict:
    # Receive full state
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    errors = state.get("error_count", 0)
    
    response = llm_with_tools.invoke(messages)
    
    # Return ONLY the fields that changed
    return {
        "messages": [response],        # add_messages appends this
        "iteration": iteration + 1,    # increment counter
        "tool_call_count": state.get("tool_call_count", 0) + len(response.tool_calls or []),
    }
    # Fields not returned are UNCHANGED in state
```

---

## Context Window Management

As the loop runs, the message list grows. Long loops can exceed the LLM's context window:

```python
from langchain_core.messages import trim_messages, SystemMessage, RemoveMessage

def agent_with_memory_management(state: AgentState) -> dict:
    messages = state["messages"]
    
    # Strategy 1: Trim to fit token budget
    trimmed = trim_messages(
        messages,
        max_tokens=6000,
        token_counter=ChatOpenAI(model="gpt-4o"),
        strategy="last",          # keep most recent
        include_system=True,      # always keep system message
    )
    
    response = llm_with_tools.invoke(trimmed)
    return {"messages": [response]}


def agent_with_summarization(state: AgentState) -> dict:
    messages = state["messages"]
    
    # Strategy 2: Summarize old messages when list gets long
    if len(messages) > 20:
        # Summarize everything except the last 5 messages
        old_messages = messages[:-5]
        summary = ChatOpenAI(model="gpt-4o").invoke(
            f"Summarize this conversation in 3-4 sentences: {[m.content for m in old_messages]}"
        )
        
        # Replace old messages with a summary + keep recent messages
        keep = messages[-5:]
        summary_msg = SystemMessage(f"[Earlier conversation summary]: {summary.content}")
        
        # Delete old messages from state
        delete_ops = [RemoveMessage(id=m.id) for m in old_messages]
        return {
            "messages": delete_ops + [summary_msg] + keep,
        }
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

---

## Persisting State Across Sessions

Use checkpointers to save and restore loop state:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory (lost on restart)
graph = builder.compile(checkpointer=MemorySaver())

# SQLite (survives restarts)
with SqliteSaver.from_conn_string("./agent_state.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    # Same thread_id = continues where it left off
    config = {"configurable": {"thread_id": "research_session_42"}}
    
    result1 = graph.invoke({"messages": [HumanMessage("Start researching quantum computing")]}, config)
    # ... (agent loops, tools run, state saved) ...
    
    # Later, or in a new process:
    result2 = graph.invoke({"messages": [HumanMessage("What did you find so far?")]}, config)
    # Agent has full memory of previous loop iterations
```

---

## Inspecting & Modifying State Mid-Loop

```python
# Get current state (including all messages from all loop iterations)
state = graph.get_state(config)
print(f"Loop has run {state.values.get('iteration', 0)} times")
print(f"Messages so far: {len(state.values['messages'])}")
print(f"Next node: {state.next}")

# Manually inject state (e.g., fix a bad tool result)
graph.update_state(config, {
    "messages": [ToolMessage(
        content="Corrected result: The population is 13.96 million",
        tool_call_id="bad_call_id_here",
    )]
})

# Resume the loop with corrected state
graph.invoke(None, config)

# View full iteration history
for checkpoint in graph.get_state_history(config):
    step = checkpoint.metadata.get("step", 0)
    msg_count = len(checkpoint.values.get("messages", []))
    print(f"Step {step}: {msg_count} messages, next={checkpoint.next}")
```

---

## State Anti-patterns

```python
# ❌ BAD: Mutating state directly
def bad_node(state):
    state["messages"].append(new_msg)   # mutates in place
    return state

# ✅ GOOD: Return a dict with only the changed fields
def good_node(state):
    return {"messages": [new_msg]}      # add_messages reducer appends

# ❌ BAD: Storing large objects in state
class State(TypedDict):
    raw_documents: list[bytes]   # huge blobs slow down checkpointing

# ✅ GOOD: Store references, not data
class State(TypedDict):
    document_ids: list[str]      # look up from a store when needed

# ❌ BAD: Growing lists without bound
class State(TypedDict):
    all_search_results: Annotated[list, operator.add]  # grows forever

# ✅ GOOD: Keep only what's needed
class State(TypedDict):
    relevant_facts: list         # curated, deduplicated list
```

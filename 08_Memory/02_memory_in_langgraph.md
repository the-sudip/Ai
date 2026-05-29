# Memory in LangGraph

LangGraph has a first-class, built-in memory system based on **State** (short-term) and **Checkpointers** (persistent).

---

## How LangGraph Memory Works

```
Thread 1: user_123
  ┌──────────────────────────────────────────┐
  │  Checkpoint 1: {"messages": [msg1]}       │
  │  Checkpoint 2: {"messages": [msg1, msg2]} │
  │  Checkpoint 3: {"messages": [..., msg3]}  │  ← current
  └──────────────────────────────────────────┘

Thread 2: user_456
  ┌──────────────────────────────────────────┐
  │  Checkpoint 1: {"messages": [msg_a]}      │
  └──────────────────────────────────────────┘
```

Each **thread** has its own isolated state history. Memory is scoped by `thread_id`.

---

## `add_messages` Reducer — How Messages Accumulate

Without `add_messages`, returning `{"messages": [new_msg]}` would **replace** the list. With `add_messages`, it **appends**.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #                   ^^^^  ^^^^^^^^^^^
    #                   type  reducer function
```

`add_messages` is smart:
- Appends new messages
- Updates existing messages by ID (for streaming updates)
- Handles `RemoveMessage` to delete specific messages

---

## MemorySaver — In-Memory Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")
checkpointer = MemorySaver()  # stores checkpoints in RAM

# Build graph with checkpointer
graph = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
)

# Thread 1 — user Alice
config_alice = {"configurable": {"thread_id": "alice"}}

graph.invoke({"messages": [HumanMessage("My name is Alice and I'm a data scientist")]}, config_alice)
response = graph.invoke({"messages": [HumanMessage("What do I do for work?")]}, config_alice)
print(response["messages"][-1].content)  # "You are a data scientist."

# Thread 2 — user Bob (no memory of Alice)
config_bob = {"configurable": {"thread_id": "bob"}}
response = graph.invoke({"messages": [HumanMessage("What do I do for work?")]}, config_bob)
print(response["messages"][-1].content)  # "I don't know — could you tell me?"
```

**Note**: `MemorySaver` data is lost when the Python process restarts.

---

## SqliteSaver — Persistent File-Based Storage

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Creates/opens a SQLite database file
with SqliteSaver.from_conn_string("./conversation_memory.db") as checkpointer:
    graph = create_react_agent(llm, tools, checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user_42"}}
    graph.invoke({"messages": [HumanMessage("Remember: I prefer Python over Java")]}, config)

# ← Data survives restart ─────────────────────────────────────────────────────

# Next day, new Python process:
with SqliteSaver.from_conn_string("./conversation_memory.db") as checkpointer:
    graph = create_react_agent(llm, tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "user_42"}}
    result = graph.invoke({"messages": [HumanMessage("What language do I prefer?")]}, config)
    print(result["messages"][-1].content)  # "You prefer Python over Java."
```

---

## PostgresSaver — Production-Scale Persistence

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost:5432/mydb"
) as checkpointer:
    checkpointer.setup()  # create tables if they don't exist
    graph = builder.compile(checkpointer=checkpointer)
```

---

## Inspecting Memory State

```python
# Get current state
snapshot = graph.get_state(config)

print(snapshot.values)        # {"messages": [...]}
print(snapshot.next)          # nodes that will run next
print(snapshot.metadata)      # run id, step number, etc.
print(snapshot.created_at)    # when this checkpoint was created

# Get state history (all checkpoints for this thread)
for state in graph.get_state_history(config):
    print(state.metadata["step"], len(state.values["messages"]))
```

---

## Updating Memory Manually

```python
# Correct a mistake in the agent's state
graph.update_state(
    config,
    {"messages": [HumanMessage("Actually, my name is Bob, not Alice.")]},
)

# Delete specific messages
from langchain_core.messages import RemoveMessage

# Remove the last message
current = graph.get_state(config)
last_msg_id = current.values["messages"][-1].id
graph.update_state(config, {"messages": [RemoveMessage(id=last_msg_id)]})
```

---

## Thread-Scoped vs Cross-Thread Memory

| | Thread-Scoped (Checkpointer) | Cross-Thread (Store) |
|---|---|---|
| Scope | Single conversation thread | Across all threads for a user |
| Duration | Until thread is deleted | Long-term persistent |
| Use case | Conversation history | User preferences, facts about user |
| API | `graph.get_state(config)` | `store.get(namespace, key)` |

---

## LangGraph Store API — Cross-Thread Long-Term Memory

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Save information across threads
store.put(
    namespace=("user_42", "preferences"),
    key="language",
    value={"preferred_language": "Python", "reason": "works in data science"}
)

store.put(
    namespace=("user_42", "facts"),
    key="location",
    value={"city": "Berlin", "timezone": "CET"}
)

# Retrieve
pref = store.get(("user_42", "preferences"), "language")
print(pref.value)  # {"preferred_language": "Python", ...}

# Search by namespace
all_prefs = store.search(("user_42", "preferences"))
```

Using the store inside a LangGraph node:
```python
def agent_with_long_term_memory(state: State, *, store: BaseStore):
    # Recall past user preferences
    user_prefs = store.search(("user_42", "preferences"))
    context = "\n".join(str(p.value) for p in user_prefs)
    
    # Use in prompt
    response = llm.invoke([
        SystemMessage(f"User context: {context}"),
        *state["messages"],
    ])
    return {"messages": [response]}

builder.add_node("agent", agent_with_long_term_memory)
graph = builder.compile(checkpointer=checkpointer, store=store)
```

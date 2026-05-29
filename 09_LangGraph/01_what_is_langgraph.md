# What is LangGraph?

---

## Overview

**LangGraph** is a library for building **stateful, multi-actor applications** with LLMs. It models your application as a **directed graph** where:

- **Nodes** = Python functions that do work (LLM calls, tool execution, logic)
- **Edges** = connections between nodes (fixed or conditional)
- **State** = shared data that flows through the graph

It's built on top of LangChain and is designed specifically for **agentic workflows** — loops, branching, multi-agent coordination, and human-in-the-loop.

---

## Why LangGraph Over LCEL?

| | LCEL Chains | LangGraph |
|---|---|---|
| **Structure** | Linear / DAG (no cycles) | Cyclic graph (supports loops) |
| **State** | Passed through pipes | Persistent TypedDict |
| **Loops** | Not supported natively | Core feature |
| **HITL** | Not built-in | Native with checkpointers |
| **Multi-agent** | Complex to implement | First-class support |
| **Debugging** | LangSmith traces | LangSmith + state inspection |
| **Use case** | Simple RAG, single-turn chains | Agents, multi-step workflows |

---

## Core Concepts

### State
A `TypedDict` (or Pydantic model) that holds all shared data. Passed to every node. Nodes return partial state updates.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # accumulates messages
    task: str                                 # simple string field
    step_count: int                           # counter
    is_complete: bool                         # flag
```

### Nodes
Plain Python functions `(state) → dict` that return partial state updates:

```python
def my_node(state: MyState) -> dict:
    # Do work using state
    new_message = llm.invoke(state["messages"])
    # Return only the fields being updated
    return {
        "messages": [new_message],
        "step_count": state["step_count"] + 1,
    }
```

### Edges
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MyState)

# Static edge: always go from A to B
builder.add_edge("node_a", "node_b")

# START and END are special reserved nodes
builder.add_edge(START, "node_a")
builder.add_edge("node_b", END)
```

### Conditional Edges
```python
def route(state: MyState) -> str:
    if state["is_complete"]:
        return END
    return "continue_processing"

builder.add_conditional_edges(
    "check_node",   # from this node
    route,          # function that decides next node
    {               # map return values to node names
        END: END,
        "continue_processing": "process_node",
    }
)
```

---

## Minimal Working Graph

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile
graph = builder.compile()

# Run
result = graph.invoke({
    "messages": [HumanMessage("What is LangGraph?")]
})
print(result["messages"][-1].content)
```

---

## LangGraph vs Traditional Code

**Traditional (imperative):**
```python
def agent_loop(input):
    messages = [input]
    for i in range(10):
        response = llm.invoke(messages)
        if response.tool_calls:
            tool_result = execute_tools(response.tool_calls)
            messages.append(tool_result)
        else:
            return response.content
```

**LangGraph (declarative graph):**
```python
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()
```

Benefits of graph approach:
- **Persistent state** across iterations
- **Checkpointing** for free (memory, HITL)
- **Visualization** of the workflow
- **Streaming** of state updates
- **Easy debugging** — inspect state at any point

---

## Graph Compilation Options

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),        # enables memory & HITL
    interrupt_before=["risky_node"],   # pause before these nodes
    interrupt_after=["research_node"], # pause after these nodes
)
```

---

## Visualizing the Graph

```python
# ASCII visualization
graph.get_graph().print_ascii()

# PNG image (requires pygraphviz or Mermaid)
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

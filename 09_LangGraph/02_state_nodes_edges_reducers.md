# State, Nodes, Edges & Reducers

These are the four fundamental building blocks of every LangGraph application.

---

## 1. State

The **State** is a shared data container — a `TypedDict` (or Pydantic model) that every node reads from and writes to.

### Rules
- Every node receives the **full current state**
- Each node returns a **dict with only the fields it updates**
- Fields not returned are left unchanged
- Reducers define how returned values are merged into existing state

```python
from typing import Annotated, TypedDict, Optional, Literal
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    # Accumulated list — add_messages appends instead of replacing
    messages: Annotated[list, add_messages]
    
    # Accumulated integer — operator.add adds to existing value
    token_count: Annotated[int, operator.add]
    
    # Last-write wins — no reducer = overwrite
    task: str
    status: Literal["pending", "in_progress", "complete"]
    result: Optional[str]
    
    # Boolean flag
    needs_human_review: bool
```

---

## 2. Reducers

A **reducer** is a function that controls how a field update is merged into the existing state.

```python
# No reducer — overwrites the field
class State(TypedDict):
    status: str   # each node replaces the value

# add_messages reducer — appends messages to the list
messages: Annotated[list, add_messages]

# operator.add — adds numbers
token_count: Annotated[int, operator.add]

# Custom reducer — take the maximum value
def take_max(existing: int, new: int) -> int:
    return max(existing, new)

priority: Annotated[int, take_max]
```

### `add_messages` Details

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

# Appending
state = {"messages": [HumanMessage("Hello")]}
update = {"messages": [AIMessage("Hi!")]}
# After reducer: {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}

# Updating by ID (for streaming)
existing_msg = AIMessage("I am", id="msg_1")
streaming_update = AIMessage("I am thinking...", id="msg_1")  # same ID
# After reducer: replaces the message with the same ID

# Removing
state = {"messages": [HumanMessage("Hello", id="h1"), AIMessage("Hi!", id="a1")]}
remove = {"messages": [RemoveMessage(id="h1")]}
# After reducer: {"messages": [AIMessage("Hi!", id="a1")]}
```

---

## 3. Nodes

A node is any Python callable `(state) → dict`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Simple node
def greet(state: AgentState) -> dict:
    return {"messages": [AIMessage("Hello! How can I help?")]}

# LLM node
def chat(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "token_count": response.usage_metadata.get("total_tokens", 0),
    }

# Logic node
def check_completion(state: AgentState) -> dict:
    last = state["messages"][-1]
    is_done = "DONE" in last.content.upper()
    return {
        "status": "complete" if is_done else "in_progress",
    }

# Async node
async def async_node(state: AgentState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

### Adding Nodes to Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

builder.add_node("greet", greet)
builder.add_node("chat", chat)
builder.add_node("check", check_completion)

# Node names must be unique strings
```

---

## 4. Edges

### Static Edge — Always go from A to B

```python
builder.add_edge(START, "greet")   # entry point
builder.add_edge("greet", "chat")
builder.add_edge("chat", "check")
builder.add_edge("check", END)     # exit point
```

### Conditional Edge — Dynamic routing

```python
def router(state: AgentState) -> str:
    """Return the name of the next node."""
    if state["status"] == "complete":
        return "finalize"
    elif state["needs_human_review"]:
        return "review"
    else:
        return "continue"

# Must list all possible return values in the mapping
builder.add_conditional_edges(
    "check",   # from this node
    router,    # function that returns next node name
    {
        "finalize": "finalize",
        "review": "human_review",
        "continue": "chat",
    }
)
```

Simplified when return values match node names exactly:
```python
# If router returns "tools", "agent", or END
builder.add_conditional_edges("agent", tools_condition)
# tools_condition is prebuilt: returns "tools" if tool_calls exist, else END
```

---

## Full Example — All Concepts Together

```python
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import operator

class State(TypedDict):
    messages: Annotated[list, add_messages]
    call_count: Annotated[int, operator.add]
    final_answer: str

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def agent(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "call_count": 1}

def finalize(state: State) -> dict:
    return {"final_answer": state["messages"][-1].content}

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("finalize", finalize)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "finalize"})
builder.add_edge("tools", "agent")
builder.add_edge("finalize", END)

graph = builder.compile()

result = graph.invoke({
    "messages": [HumanMessage("What is 7 * 8?")],
    "call_count": 0,
    "final_answer": "",
})
print(result["final_answer"])   # "7 × 8 = 56"
print(result["call_count"])     # 2 (agent called twice: once for tool call, once for final answer)
```

# ToolNode in LangGraph

`ToolNode` is a prebuilt LangGraph node that **automatically executes tool calls** found in the last `AIMessage` and returns `ToolMessage` results — no manual routing code needed.

---

## What ToolNode Does

1. Reads the last `AIMessage` in state
2. Finds all `tool_calls` in that message
3. Executes each tool with the specified arguments
4. Wraps results in `ToolMessage` objects (with correct `tool_call_id`)
5. Returns them as new messages to add to state

---

## Basic ToolNode Usage

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"{city}: 22°C, sunny"

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime("%H:%M:%S %Z")

tools = [get_weather, get_time]
tool_node = ToolNode(tools)

# Simulate what happens when an LLM returns tool_calls:
from langchain_core.messages import AIMessage

fake_ai_message = AIMessage(
    content="",  # content is empty when calling tools
    tool_calls=[
        {"name": "get_weather", "args": {"city": "London"}, "id": "call_001", "type": "tool_call"},
        {"name": "get_time", "args": {"timezone": "Europe/London"}, "id": "call_002", "type": "tool_call"},
    ]
)

# ToolNode processes both calls
result = tool_node.invoke({"messages": [fake_ai_message]})
print(result)
# {"messages": [
#   ToolMessage(content="London: 22°C, sunny", tool_call_id="call_001"),
#   ToolMessage(content="14:30:00 BST", tool_call_id="call_002"),
# ]}
```

---

## ToolNode in a Full Agent Graph

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# LLM with tools bound
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# Agent node
def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Graph
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))  # ← just pass the list of tools

builder.add_edge(START, "agent")

# tools_condition: built-in — routes to "tools" if tool_calls exist, else END
builder.add_conditional_edges("agent", tools_condition)

builder.add_edge("tools", "agent")  # after tool execution, go back to agent

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("What's the weather in London right now?")]})
print(result["messages"][-1].content)
```

---

## `tools_condition` — Built-in Router

`tools_condition` is a prebuilt conditional edge function:

```python
from langgraph.prebuilt import tools_condition

# Equivalent to writing this yourself:
def tools_condition(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

---

## Parallel Tool Execution

`ToolNode` automatically executes all tool calls **in parallel** if the LLM requests multiple tools at once:

```python
# LLM requests both tools simultaneously:
ai_message = AIMessage(
    content="",
    tool_calls=[
        {"name": "get_weather", "args": {"city": "London"}, "id": "c1"},
        {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "c2"},
        {"name": "get_time", "args": {"timezone": "Asia/Tokyo"}, "id": "c3"},
    ]
)

# ToolNode runs all 3 concurrently
result = tool_node.invoke({"messages": [ai_message]})
# Returns 3 ToolMessages simultaneously
```

---

## Custom Error Handling in ToolNode

```python
# By default, ToolNode catches errors and returns them as ToolMessages
# so the agent can see the error and retry

tool_node = ToolNode(
    tools,
    handle_tool_errors=True,  # default: True
)

# With custom error message
tool_node = ToolNode(
    tools,
    handle_tool_errors="Tool execution failed. Please try a different approach.",
)
```

---

## InjectedToolArg — Inject State into Tools

Sometimes tools need access to the graph state (e.g., user ID, auth token) that should not be passed by the LLM:

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def get_user_orders(
    customer_id: str,  # LLM provides this
    db_connection: Annotated[object, InjectedToolArg],  # injected from state, not from LLM
) -> str:
    """Get orders for a specific customer."""
    return db_connection.query(f"SELECT * FROM orders WHERE customer_id = '{customer_id}'")

# When creating the ToolNode, inject the db_connection
tool_node = ToolNode(tools).with_config({
    "configurable": {"db_connection": my_db_connection}
})
```

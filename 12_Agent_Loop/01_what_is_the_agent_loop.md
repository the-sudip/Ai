# What is the Agent Loop?

The **agent loop** (also called the agentic loop or reasoning loop) is the core execution cycle every AI agent runs through repeatedly until the task is complete.

---

## The Core Cycle

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                            │
│                                                          │
│   ┌──────────┐                                           │
│   │  START   │                                           │
│   └────┬─────┘                                           │
│        │ user input / goal                               │
│        ▼                                                 │
│   ┌──────────┐     no tool needed     ┌──────────┐       │
│   │  THINK   │ ─────────────────────► │   END    │       │
│   │  (LLM)   │                        └──────────┘       │
│   └────┬─────┘                                           │
│        │ decides to call a tool                          │
│        ▼                                                 │
│   ┌──────────┐                                           │
│   │   ACT    │                                           │
│   │ (Tool    │                                           │
│   │  call)   │                                           │
│   └────┬─────┘                                           │
│        │ tool result                                     │
│        ▼                                                 │
│   ┌──────────┐                                           │
│   │ OBSERVE  │                                           │
│   │(ToolMsg) │                                           │
│   └────┬─────┘                                           │
│        │ loop back                                       │
│        └──────────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**Phases:**
1. **Think** — LLM sees current state (messages + observations) and decides what to do
2. **Act** — Calls a tool (or answers directly if done)
3. **Observe** — Tool result is added to state as a `ToolMessage`
4. **Repeat** — Loop runs again with updated state

---

## Why a Loop, Not a Chain?

| | Chain | Agent Loop |
|---|---|---|
| **Steps** | Fixed, predetermined | Dynamic, decided at runtime |
| **Iterations** | Always 1 pass | N iterations until task done |
| **Tool use** | Optional, hardcoded | Core mechanism |
| **Goal** | Transform data | Achieve a goal |
| **Control** | Full (deterministic) | Partial (LLM decides) |

A chain is like a recipe. An agent loop is like a chef improvising until the dish is right.

---

## Minimal Agent Loop in Python

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 22°C, partly cloudy"

@tool
def get_population(city: str) -> str:
    """Get population of a city."""
    populations = {"Tokyo": "13.96M", "London": "9.5M", "Paris": "2.1M"}
    return populations.get(city, "Unknown")

llm = ChatOpenAI(model="gpt-4o").bind_tools([get_weather, get_population])
tools_map = {"get_weather": get_weather, "get_population": get_population}

# Manual agent loop
messages = [HumanMessage("What's the weather and population of Tokyo?")]

MAX_ITERATIONS = 10
for i in range(MAX_ITERATIONS):
    response = llm.invoke(messages)
    messages.append(response)
    
    # No tool calls → agent is done
    if not response.tool_calls:
        print(f"Final answer (after {i+1} iterations):")
        print(response.content)
        break
    
    # Execute all tool calls
    for tool_call in response.tool_calls:
        tool_fn = tools_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
else:
    print(f"Reached max iterations ({MAX_ITERATIONS})")
```

---

## The Loop in LangGraph

LangGraph makes the agent loop explicit as a graph:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([search])

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent(state: State) -> State:
    """THINK phase — LLM decides next action."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the loop graph
builder = StateGraph(State)
builder.add_node("agent", agent)         # THINK
builder.add_node("tools", ToolNode([search]))  # ACT + OBSERVE
builder.add_edge(START, "agent")

# The key conditional: loop back or end
builder.add_conditional_edges(
    "agent",
    tools_condition,    # has tool calls? → "tools", else → END
)
builder.add_edge("tools", "agent")  # always loop back after tool use

graph = builder.compile()

# The graph loops: agent → tools → agent → tools → ... → END
result = graph.invoke({"messages": [HumanMessage("Search for LangGraph tutorials")]})
print(result["messages"][-1].content)
```

---

## Loop Execution Trace

For the question *"What's the capital of Australia and its population?"*:

```
Iteration 1:
  THINK: I need to find the capital of Australia.
  ACT:   get_capital(country="Australia")
  OBS:   "Canberra"

Iteration 2:
  THINK: Now I need the population of Canberra.
  ACT:   get_population(city="Canberra")
  OBS:   "467,000"

Iteration 3:
  THINK: I have all the information. No more tools needed.
  OUTPUT: "The capital of Australia is Canberra with a population of ~467,000."
  → STOP
```

Each iteration, the LLM sees the **full accumulated history** — all previous actions and observations — before deciding what to do next.

---

## Loop State Accumulation

```
Start:    [HumanMessage("Capital and population of Australia?")]

After 1:  [Human, AI(tool_call: get_capital), ToolMessage("Canberra")]

After 2:  [Human, AI(tool_call: get_capital), ToolMessage("Canberra"),
           AI(tool_call: get_population), ToolMessage("467,000")]

After 3:  [Human, AI(tool_call: get_capital), ToolMessage("Canberra"),
           AI(tool_call: get_population), ToolMessage("467,000"),
           AI("The capital is Canberra with population ~467,000.")]
           → END
```

The state grows with each iteration — this is why **context window management** matters in long loops.

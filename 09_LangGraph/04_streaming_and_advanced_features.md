# Streaming & Advanced LangGraph Features

---

## 1. Streaming Graph Output

LangGraph supports multiple streaming modes to see output as it's produced:

### `graph.stream()` — Stream State Updates

Yields state updates after each **node** completes:

```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "stream_test"}}

for event in graph.stream(
    {"messages": [HumanMessage("Research the latest AI breakthroughs")]},
    config=config,
    stream_mode="updates",   # "updates" (default) or "values"
):
    for node_name, state_update in event.items():
        print(f"\n{'='*40}")
        print(f"Node: {node_name}")
        if "messages" in state_update:
            for msg in state_update["messages"]:
                print(f"  {type(msg).__name__}: {str(msg.content)[:200]}")

# stream_mode="values" — emits the FULL state after each node (not just the diff)
for state in graph.stream(input, config, stream_mode="values"):
    print(state["messages"][-1].content)
```

### `graph.astream_events()` — Token-Level Streaming

Streams individual tokens from the LLM as they are generated:

```python
import asyncio

async def stream_tokens():
    async for event in graph.astream_events(
        {"messages": [HumanMessage("Tell me about neural networks")]},
        config=config,
        version="v2",
    ):
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream":
            # Individual LLM tokens
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        
        elif event_type == "on_tool_start":
            print(f"\n[Calling tool: {event['name']}]")
        
        elif event_type == "on_tool_end":
            print(f"[Tool result: {event['data']['output'][:100]}]")
        
        elif event_type == "on_chain_end":
            print(f"\n[Node complete: {event['name']}]")

asyncio.run(stream_tokens())
```

---

## 2. `Command` — Combined State Update + Routing

`Command` lets a node both update state AND specify the next node — no separate conditional edge needed:

```python
from langgraph.types import Command

def smart_router(state: State) -> Command:
    """This node both updates state AND decides routing."""
    
    # Analyze the last message
    last_msg = state["messages"][-1].content
    
    if "calculate" in last_msg.lower():
        return Command(
            update={"messages": [AIMessage("Routing to calculator...")], "mode": "math"},
            goto="calculator_node",
        )
    elif "search" in last_msg.lower():
        return Command(
            update={"mode": "search"},
            goto="search_node",
        )
    else:
        return Command(
            update={"mode": "general"},
            goto="chat_node",
        )

# No add_conditional_edges needed! The node handles routing itself.
builder.add_node("router", smart_router)
builder.add_edge(START, "router")
# "calculator_node", "search_node", "chat_node" must exist in the graph
```

---

## 3. `Send` — Dynamic Parallel Execution

`Send` creates dynamic edges at runtime — fan-out to process items in parallel:

```python
from langgraph.types import Send

def distribute_tasks(state: State) -> list[Send]:
    """Fan out to process multiple items."""
    tasks = state["pending_tasks"]
    # Each Send creates an independent parallel execution
    return [
        Send("process_task", {"task": task, "task_id": i})
        for i, task in enumerate(tasks)
    ]

builder.add_conditional_edges("distribute", distribute_tasks)
```

---

## 4. State History & Time Travel

LangGraph stores every checkpoint. You can go back to any point in time:

```python
# Get full history for a thread
history = list(graph.get_state_history(config))

for h in history:
    print(f"Step {h.metadata['step']}: {len(h.values['messages'])} messages, next={h.next}")

# Go back to step 3
target_checkpoint = history[-3]  # 3rd from most recent

# Resume from that checkpoint
graph.invoke(None, target_checkpoint.config)
```

This is "time travel" — rewind and replay from any point.

---

## 5. Subgraph Communication

Subgraphs (nested graphs) communicate with parent via shared state keys:

```python
class ParentState(TypedDict):
    task: str
    result: str
    messages: Annotated[list, add_messages]

class SubgraphState(TypedDict):
    task: str    # ← shared with parent (same name)
    result: str  # ← shared with parent (same name)
    internal_steps: int  # ← private to subgraph

# The subgraph reads "task" from parent state
# and writes "result" back to parent state
# "internal_steps" is private and not visible to parent
```

---

## 6. `create_react_agent` — Prebuilt Full Agent

The fastest way to build a production-ready tool-calling agent in LangGraph:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 22°C, sunny"

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather],
    state_modifier="""You are a helpful weather assistant.
    Always provide temperatures in both Celsius and Fahrenheit.""",
    checkpointer=MemorySaver(),   # adds memory
)

config = {"configurable": {"thread_id": "weather_1"}}
result = agent.invoke(
    {"messages": [HumanMessage("What's the weather in Tokyo?")]},
    config=config,
)
print(result["messages"][-1].content)
```

What `create_react_agent` creates internally:
```
START → [agent node] ← → [ToolNode] → END
              ↓ (no tool calls)
             END
```

---

## 7. Graph Visualization

```python
# Text visualization
graph.get_graph().print_ascii()
# Output:
#          +-----------+
#          | __start__ |
#          +-----------+
#                *
#           +-------+
#           | agent |
#           +-------+
#          /         \
#    +-------+      +-----+
#    | tools |      | END |
#    +-------+      +-----+

# Mermaid diagram (for markdown/docs)
print(graph.get_graph().draw_mermaid())

# PNG (requires graphviz)
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

# Multi-Agent Systems in LangGraph

LangGraph is purpose-built for multi-agent architectures — systems where multiple specialized agents collaborate to accomplish complex tasks.

---

## Why Multi-Agent?

A single agent with many tools becomes unfocused and hard to debug. Multi-agent systems:
- Give each agent a **focused responsibility**
- Allow **parallel execution** of independent subtasks
- Enable **specialization** (researcher, coder, reviewer)
- Make the system **easier to test and maintain**

---

## Pattern 1: Sequential Pipeline

Agents run one after another, each enriching the shared state:

```
Input → Research Agent → Analysis Agent → Writing Agent → Output
```

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage

class PipelineState(TypedDict):
    task: str
    research: str
    analysis: str
    final_report: str
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o")

# Specialized agents
research_agent = create_react_agent(
    llm, [DuckDuckGoSearchRun()],
    state_modifier="You are a research specialist. Search and gather facts."
)

def research_node(state: PipelineState) -> dict:
    result = research_agent.invoke({
        "messages": [HumanMessage(f"Research: {state['task']}")]
    })
    return {"research": result["messages"][-1].content}

def analysis_node(state: PipelineState) -> dict:
    prompt = f"Analyze these research findings:\n{state['research']}\n\nProvide key insights:"
    result = llm.invoke([HumanMessage(prompt)])
    return {"analysis": result.content}

def writing_node(state: PipelineState) -> dict:
    prompt = f"""Write a professional report based on:
Task: {state['task']}
Research: {state['research']}
Analysis: {state['analysis']}"""
    result = llm.invoke([HumanMessage(prompt)])
    return {"final_report": result.content}

builder = StateGraph(PipelineState)
builder.add_node("research", research_node)
builder.add_node("analyze", analysis_node)
builder.add_node("write", writing_node)
builder.add_edge(START, "research")
builder.add_edge("research", "analyze")
builder.add_edge("analyze", "write")
builder.add_edge("write", END)

pipeline = builder.compile()
result = pipeline.invoke({"task": "Impact of AI on software development jobs"})
print(result["final_report"])
```

---

## Pattern 2: Supervisor (Dynamic Routing)

A supervisor LLM decides which agent to call next based on current state:

```python
from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

AGENTS = ["researcher", "coder", "reviewer"]

class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "reviewer", "FINISH"]
    reason: str

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a project supervisor managing these agents:
- researcher: Searches for information and gathers facts
- coder: Writes and tests Python code  
- reviewer: Reviews code for bugs and quality

Based on the current conversation, decide who should act next.
Return FINISH when the task is fully complete."""),
    MessagesPlaceholder("messages"),
])

supervisor_chain = supervisor_prompt | llm.with_structured_output(SupervisorDecision)

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

def supervisor_node(state: SupervisorState) -> dict:
    decision = supervisor_chain.invoke({"messages": state["messages"]})
    return {"next_agent": decision.next}

def route_to_agent(state: SupervisorState) -> str:
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return END
    return next_agent

builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_agent)

# After each agent, return to supervisor
for agent in ["researcher", "coder", "reviewer"]:
    builder.add_edge(agent, "supervisor")

graph = builder.compile()
```

---

## Pattern 3: Subgraphs (Hierarchical Agents)

A compiled graph can be used as a **node inside another graph**:

```python
# Build specialized sub-graphs
def build_research_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("search", search_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()

def build_coding_graph():
    builder = StateGraph(CodingState)
    builder.add_node("write", write_code_node)
    builder.add_node("test", test_code_node)
    builder.add_edge(START, "write")
    builder.add_edge("write", "test")
    builder.add_edge("test", END)
    return builder.compile()

research_graph = build_research_graph()
coding_graph = build_coding_graph()

# Parent graph uses subgraphs as nodes
parent_builder = StateGraph(ParentState)
parent_builder.add_node("research", research_graph)  # ← subgraph as node
parent_builder.add_node("coding", coding_graph)
parent_builder.add_edge(START, "research")
parent_builder.add_edge("research", "coding")
parent_builder.add_edge("coding", END)

parent_graph = parent_builder.compile()
```

---

## Pattern 4: Fan-Out / Map-Reduce

Process multiple items in parallel, then aggregate results:

```python
from langgraph.types import Send

def fan_out_node(state: MapReduceState) -> list[Send]:
    """Send each item to be processed in parallel."""
    return [
        Send("process_item", {"item": item, "item_id": i})
        for i, item in enumerate(state["items"])
    ]

def process_item_node(state: ItemState) -> dict:
    """Process a single item."""
    result = llm.invoke([HumanMessage(f"Analyze: {state['item']}")])
    return {"results": [{"id": state["item_id"], "analysis": result.content}]}

def aggregate_node(state: MapReduceState) -> dict:
    """Combine all results."""
    combined = "\n".join(f"Item {r['id']}: {r['analysis']}" for r in state["results"])
    summary = llm.invoke([HumanMessage(f"Summarize these analyses:\n{combined}")])
    return {"final_summary": summary.content}

builder = StateGraph(MapReduceState)
builder.add_node("distribute", fan_out_node)   # returns list[Send]
builder.add_node("process_item", process_item_node)
builder.add_node("aggregate", aggregate_node)

builder.add_conditional_edges("distribute", lambda x: x)  # pass Send objects directly
builder.add_edge("process_item", "aggregate")
builder.add_edge("aggregate", END)
```

---

## Key Design Principles for Multi-Agent Systems

1. **Keep agents focused** — each agent should do one thing well
2. **Share state minimally** — only pass what each agent needs
3. **Route deterministically when possible** — supervisor adds latency/cost
4. **Use subgraphs for encapsulation** — hide complexity
5. **Add checkpoints** — long multi-agent runs need recovery points
6. **Monitor all agents** — use LangSmith to trace the full pipeline

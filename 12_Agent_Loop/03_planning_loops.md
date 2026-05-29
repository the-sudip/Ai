# Planning Loops — Think Before You Act

Simple reactive agents (ReAct) act one step at a time. **Planning loops** add a separate reasoning phase where the agent creates a full plan before executing any actions.

---

## Reactive vs Planning Loop

```
REACTIVE (ReAct):
  Input → Think → Act → Observe → Think → Act → ... → Answer

PLANNING:
  Input → [Plan] → [Execute Step 1] → [Execute Step 2] → ... → [Synthesize] → Answer
              ↑                                                        │
              └──────────── Re-plan if needed ◄──────────────────────┘
```

---

## Plan-and-Execute Pattern

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── TOOLS ────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Web results for '{query}': ...]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    import ast, operator
    # safe eval
    return str(eval(expression))  # simplified; use safe_eval in production

tools = [search_web, calculate]
tools_map = {t.name: t for t in tools}

# ── STATE ─────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    plan: List[str]          # ordered list of steps
    current_step: int        # which step we're on
    step_results: List[str]  # results from each completed step

# ── PLANNER ───────────────────────────────────────────────────────────

class Plan(BaseModel):
    steps: List[str]

planner_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Plan)

PLANNER_PROMPT = """You are a planning agent. 
Break the user's task into a sequence of clear, actionable steps.
Each step should be achievable with the available tools: {tools}.
Return 2-5 steps maximum."""

def planner(state: State) -> State:
    """Create a step-by-step plan for the task."""
    user_query = state["messages"][0].content
    tool_names = ", ".join(tools_map.keys())
    
    prompt = PLANNER_PROMPT.format(tools=tool_names) + f"\n\nTask: {user_query}"
    plan_result = planner_llm.invoke(prompt)
    
    print(f"\n📋 PLAN:")
    for i, step in enumerate(plan_result.steps, 1):
        print(f"  {i}. {step}")
    
    return {
        "plan": plan_result.steps,
        "current_step": 0,
        "step_results": [],
    }

# ── EXECUTOR ──────────────────────────────────────────────────────────

executor_llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def executor(state: State) -> State:
    """Execute the current step of the plan."""
    current_step = state["current_step"]
    step_description = state["plan"][current_step]
    
    # Build context from previous steps
    context = ""
    if state["step_results"]:
        context = "\n\nPrevious results:\n" + "\n".join(
            f"Step {i+1}: {r}" for i, r in enumerate(state["step_results"])
        )
    
    prompt = f"""Execute this step: {step_description}
{context}

Use tools if needed."""
    
    print(f"\n⚙️ Executing step {current_step + 1}: {step_description}")
    response = executor_llm.invoke([HumanMessage(prompt)])
    
    # Handle tool calls
    step_output = response.content
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_result = tools_map[tc["name"]].invoke(tc["args"])
            step_output = str(tool_result)
            print(f"   Tool: {tc['name']}({tc['args']}) → {step_output[:100]}")
    
    return {
        "step_results": state["step_results"] + [step_output],
        "current_step": current_step + 1,
    }

# ── SYNTHESIZER ───────────────────────────────────────────────────────

synthesizer_llm = ChatOpenAI(model="gpt-4o")

def synthesizer(state: State) -> State:
    """Combine all step results into a final answer."""
    user_query = state["messages"][0].content
    
    steps_summary = "\n".join(
        f"Step {i+1} ({state['plan'][i]}): {result}"
        for i, result in enumerate(state["step_results"])
    )
    
    prompt = f"""Original question: {user_query}

Steps completed and their results:
{steps_summary}

Please synthesize a clear, comprehensive final answer."""
    
    final_answer = synthesizer_llm.invoke(prompt)
    print(f"\n✅ Final Answer:\n{final_answer.content}")
    
    return {"messages": [final_answer]}

# ── ROUTING ───────────────────────────────────────────────────────────

def should_continue(state: State) -> str:
    """Continue executing steps or synthesize?"""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# ── GRAPH ─────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_conditional_edges("executor", should_continue, {
    "execute": "executor",      # loop: run next step
    "synthesize": "synthesizer",  # done: combine results
})
builder.add_edge("synthesizer", END)

graph = builder.compile()

# Run
result = graph.invoke({
    "messages": [HumanMessage("What is the GDP of Japan and how does it compare to Germany?")],
    "plan": [],
    "current_step": 0,
    "step_results": [],
})
```

---

## Reflection Loop

The agent critiques its own output and revises until quality is acceptable:

```python
from pydantic import BaseModel, Field

class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="Quality score 1-10")
    feedback: str = Field(description="Specific improvements needed")
    is_acceptable: bool = Field(description="True if score >= 7")

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    critique: str
    revision_count: int

generator_llm = ChatOpenAI(model="gpt-4o")
critic_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Critique)

def generate(state: ReflectionState) -> ReflectionState:
    """Generate or revise a draft."""
    query = state["messages"][0].content
    context = f"\n\nFeedback to address: {state['critique']}" if state.get("critique") else ""
    
    prompt = f"Write a high-quality response to: {query}{context}"
    draft = generator_llm.invoke(prompt).content
    
    return {"draft": draft, "revision_count": state.get("revision_count", 0) + 1}

def reflect(state: ReflectionState) -> ReflectionState:
    """Critique the draft."""
    critique = critic_llm.invoke(
        f"Rate this response (1-10) and provide feedback:\n\n{state['draft']}"
    )
    return {"critique": critique.feedback, "messages": [AIMessage(f"Score: {critique.score}")]}

def should_revise(state: ReflectionState) -> str:
    """Revise if quality is low, finalize if good enough."""
    max_revisions = 3
    if state["revision_count"] >= max_revisions:
        return "finalize"
    
    critique = critic_llm.invoke(f"Rate 1-10:\n{state['draft']}")
    return "revise" if not critique.is_acceptable else "finalize"

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_edge("generate", "reflect")
builder.add_conditional_edges("reflect", should_revise, {
    "revise": "generate",     # loop: regenerate with feedback
    "finalize": END,          # done: acceptable quality
})

reflection_graph = builder.compile()
```

---

## Loop Patterns Summary

| Pattern | Loop Structure | Best For |
|---|---|---|
| **ReAct** | agent ↔ tools (reactive) | Open-ended tasks, tool use |
| **Plan-Execute** | plan → [execute]×N → synthesize | Multi-step, predictable tasks |
| **Reflection** | generate → critique → revise | Writing, code generation, quality output |
| **CRAG** | retrieve → grade → [re-query] → generate | RAG with quality control |
| **Self-Ask** | break → answer sub-questions → combine | Complex multi-part questions |

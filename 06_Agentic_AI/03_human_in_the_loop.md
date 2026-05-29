# Human-in-the-Loop (HITL)

**Human-in-the-Loop** is the practice of pausing an autonomous agent workflow at critical points to get **human review, approval, or correction** before proceeding.

---

## Why HITL?

Fully autonomous agents can:
- Take irreversible actions (delete files, send emails, make purchases)
- Make mistakes that cascade through downstream steps
- Take actions the user didn't intend

HITL provides a safety net for high-stakes operations.

---

## HITL in LangGraph

LangGraph has first-class support for HITL via **interrupt_before** and **interrupt_after** on graph nodes.

### Basic Interrupt Before a Node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    action_approved: bool

def plan_action(state: State):
    # Agent decides what to do
    return {"messages": [AIMessage("I will send an email to the client about the delay.")]}

def execute_action(state: State):
    # Executes the irreversible action
    return {"messages": [AIMessage("Email sent successfully.")]}

builder = StateGraph(State)
builder.add_node("plan", plan_action)
builder.add_node("execute", execute_action)
builder.add_edge(START, "plan")
builder.add_edge("plan", "execute")
builder.add_edge("execute", END)

# ← This is the key: pause before "execute"
checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"],
)

config = {"configurable": {"thread_id": "approval_flow_1"}}

# Run until the interrupt point
state = graph.invoke({"messages": [HumanMessage("Send an update to the client")]}, config)
print("Agent plans to:", state["messages"][-1].content)

# ── HUMAN REVIEW HAPPENS HERE ──
# Human sees the plan, decides to approve or reject
human_approves = True

if human_approves:
    # Resume from checkpoint
    final_state = graph.invoke(None, config)
    print("Result:", final_state["messages"][-1].content)
else:
    # Update state with rejection
    graph.update_state(config, {
        "messages": [HumanMessage("Do not send the email. Escalate to manager instead.")]
    })
    final_state = graph.invoke(None, config)
```

---

## Interrupt After a Node

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["web_search"],  # pause after search, before LLM processes results
)
```

Useful when you want to review what was retrieved before the LLM uses it.

---

## `graph.get_state()` — Inspect What's Happening

```python
# After interrupt, inspect current state
snapshot = graph.get_state(config)
print(snapshot.values)    # current state dict
print(snapshot.next)      # list of nodes that will run next
print(snapshot.metadata)  # checkpoint metadata
```

---

## `graph.update_state()` — Human Correction

```python
# Human corrects the agent's plan before resuming
graph.update_state(
    config,
    {"messages": [HumanMessage("Actually, don't email — just add a note to the CRM.")]},
    as_node="plan",  # apply this update as if it came from the "plan" node
)

# Then resume
graph.invoke(None, config)
```

---

## Approval Workflow — Full Example

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    status: Literal["pending", "approved", "rejected"]

def draft_email(state: WorkflowState):
    """Agent drafts an email."""
    draft = "Dear Client, We regret to inform you of a 2-day delay..."
    return {"draft": draft, "status": "pending"}

def send_email(state: WorkflowState):
    """Sends the email after approval."""
    if state["status"] == "approved":
        # send_email_api(state["draft"])
        return {"messages": [AIMessage("Email sent!")], "status": "sent"}
    return {"messages": [AIMessage("Email sending cancelled.")], "status": "cancelled"}

builder = StateGraph(WorkflowState)
builder.add_node("draft", draft_email)
builder.add_node("send", send_email)
builder.add_edge(START, "draft")
builder.add_edge("draft", "send")
builder.add_edge("send", END)

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send"],  # pause before sending
)

config = {"configurable": {"thread_id": "email_1"}}

# Run until pause
state = graph.invoke({"messages": [], "draft": "", "status": "pending"}, config)
print("Draft email:", state["draft"])

# Human approves
graph.update_state(config, {"status": "approved"})

# Resume and send
final = graph.invoke(None, config)
print(final["messages"][-1].content)  # "Email sent!"
```

---

## HITL Patterns

| Pattern | When to Use |
|---|---|
| **Approve before action** | Irreversible operations (send, delete, pay) |
| **Review before synthesis** | Check retrieved docs before LLM uses them |
| **Correction loop** | Human edits agent's draft/plan |
| **Escalation** | Agent asks for help when uncertain |
| **Audit trail** | Log all agent decisions for compliance |

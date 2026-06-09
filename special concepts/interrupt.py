from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    user: str
    report: str


# -----------------------------
# Nodes
# -----------------------------
def generate_report(state: State):
    print("Generating report...")

    return {
        "report": f"Financial report for {state['user']}"
    }


def dangerous_action(state: State):
    print("Executing dangerous action...")
    print(f"Sending report: {state['report']}")

    return {}


# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(State)

builder.add_node("generate_report", generate_report)
builder.add_node("dangerous_action", dangerous_action)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", "dangerous_action")
builder.add_edge("dangerous_action", END)


# -----------------------------
# Checkpointer
# -----------------------------
memory = MemorySaver()


# -----------------------------
# Compile with breakpoint
# -----------------------------
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_action"]
)


# -----------------------------
# Thread ID
# -----------------------------
config = {
    "configurable": {
        "thread_id": "report-thread"
    }
}


# -----------------------------
# First Run
# -----------------------------
print("\nRUN #1\n")

result = graph.invoke(
    {
        "user": "Sudip"
    },
    config=config
)

print("\nGraph paused\n")


# -----------------------------
# Inspect State
# -----------------------------
state = graph.get_state(config)

print("Current State:")
print(state.values)


# -----------------------------
# Human Approval
# -----------------------------
input("\nPress Enter to approve...")


# -----------------------------
# Resume
# -----------------------------
print("\nRUN #2 (RESUME)\n")

result = graph.invoke(
    None,
    config=config
)

print("\nFinished")
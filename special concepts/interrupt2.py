from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =====================================================
# State
# =====================================================

class State(TypedDict):
    user: str
    report: str
    approved: bool
    comment: str


# =====================================================
# Node 1
# =====================================================

def generate_report(state: State):

    print("\nGenerating report...")

    report = f"""
    Monthly Sales Report

    User: {state["user"]}
    Revenue: $125,000
    Profit: $35,000
    """

    return {
        "report": report
    }


# =====================================================
# Node 2 (Human Approval)
# =====================================================

def human_review(state: State):

    response = interrupt(
        {
            "message": "Please review the report",
            "report": state["report"]
        }
    )

    return {
        "approved": response["approved"],
        "comment": response["comment"]
    }


# =====================================================
# Node 3
# =====================================================

def send_report(state: State):

    if not state["approved"]:

        print("\nReport Rejected")
        print("Comment:", state["comment"])

        return {}

    print("\nSending Report...")
    print(state["report"])

    print("\nReviewer Comment:")
    print(state["comment"])

    return {}


# =====================================================
# Graph
# =====================================================

builder = StateGraph(State)

builder.add_node("generate_report", generate_report)
builder.add_node("human_review", human_review)
builder.add_node("send_report", send_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", "human_review")
builder.add_edge("human_review", "send_report")
builder.add_edge("send_report", END)

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# =====================================================
# Thread Config
# =====================================================

config = {
    "configurable": {
        "thread_id": "report-thread-1"
    }
}


# =====================================================
# First Run
# =====================================================

result = graph.invoke(
    {
        "user": "Sudip"
    },
    config=config
)

print("\nGRAPH PAUSED\n")
print(result)


# =====================================================
# Inspect Current State
# =====================================================

snapshot = graph.get_state(config)

print("\nCurrent State")
print(snapshot.values)


# =====================================================
# Human Input
# =====================================================

approval = input("\nApprove? (yes/no): ")

comment = input("Comment: ")

approved = approval.lower() == "yes"


# =====================================================
# Resume Graph
# =====================================================

graph.invoke(
    Command(
        resume={
            "approved": approved,
            "comment": comment
        }
    ),
    config=config
)
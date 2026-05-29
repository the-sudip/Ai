# Agentic Patterns — Planning, Reflection, Multi-Agent

---

## 1. Planning Pattern

The agent first **creates a detailed plan** for the task, then executes each step in order. Separates high-level reasoning from low-level execution.

### Why Plan First?
Without planning, agents can get lost in details and lose sight of the overall goal. A plan provides a roadmap.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

llm = ChatOpenAI(model="gpt-4o")

# Step 1: Generate a plan
class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered list of steps to accomplish the task")

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a detailed step-by-step plan to accomplish the given task. Be specific."),
    ("human", "Task: {task}"),
])

planner = planner_prompt | llm.with_structured_output(Plan)

plan = planner.invoke({"task": "Research electric vehicles and write a summary report"})
print(plan.steps)
# ["1. Search for latest EV market statistics", "2. Find top EV models and specs", ...]

# Step 2: Execute each step
results = []
executor_prompt = ChatPromptTemplate.from_template("""
Complete this step of the task:
Step: {step}

Previous results:
{previous_results}

Result:""")

executor_chain = executor_prompt | llm | StrOutputParser()

for step in plan.steps:
    result = executor_chain.invoke({
        "step": step,
        "previous_results": "\n".join(results[-3:]),  # last 3 results as context
    })
    results.append(result)

# Step 3: Synthesize
final_prompt = ChatPromptTemplate.from_template(
    "Synthesize these research results into a coherent report:\n\n{results}"
)
final_report = (final_prompt | llm | StrOutputParser()).invoke({"results": "\n\n".join(results)})
```

---

## 2. Reflection Pattern

After generating output, the agent **critiques its own work** and iterates to improve it. Mimics human self-review.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

# Initial generation
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer. Write clean, well-documented code."),
    ("human", "Write a Python function to: {task}"),
])
generate_chain = generate_prompt | llm | StrOutputParser()

# Reflection / critique
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a critical code reviewer. Analyze the code for:
    - Bugs and logic errors
    - Edge cases not handled
    - Performance issues
    - Missing error handling
    - Code style and readability
    Be specific and actionable."""),
    ("human", "Review this code:\n\n{code}"),
])
reflect_chain = reflect_prompt | llm | StrOutputParser()

# Revision
revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Revise the code based on the critique. Fix all identified issues."),
    ("human", "Original code:\n{code}\n\nCritique:\n{critique}\n\nRevised code:"),
])
revise_chain = revise_prompt | llm | StrOutputParser()

# Execute reflection loop
task = "implement binary search on a sorted list"

code = generate_chain.invoke({"task": task})
print("Draft:\n", code)

for i in range(2):  # 2 rounds of reflection
    critique = reflect_chain.invoke({"code": code})
    print(f"\nCritique {i+1}:\n", critique)
    code = revise_chain.invoke({"code": code, "critique": critique})
    print(f"\nRevision {i+1}:\n", code)
```

---

## 3. Multi-Agent Pattern

Different **specialized agents** handle different parts of a complex task. An **orchestrator** coordinates them.

```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    task: str
    research: str
    code: str
    review: str
    messages: Annotated[list, add_messages]

# Specialized agents
research_agent = create_react_agent(llm, [web_search_tool], "You are a research specialist.")
coding_agent = create_react_agent(llm, [python_repl_tool], "You are a Python coding specialist.")
review_agent = create_react_agent(llm, [], "You are a code reviewer. Find bugs and improvements.")

def research_node(state: MultiAgentState):
    result = research_agent.invoke({"messages": [HumanMessage(state["task"])]})
    return {"research": result["messages"][-1].content}

def coding_node(state: MultiAgentState):
    prompt = f"Task: {state['task']}\nResearch findings: {state['research']}\nWrite the code."
    result = coding_agent.invoke({"messages": [HumanMessage(prompt)]})
    return {"code": result["messages"][-1].content}

def review_node(state: MultiAgentState):
    result = review_agent.invoke({"messages": [HumanMessage(f"Review:\n{state['code']}")]})
    return {"review": result["messages"][-1].content}

# Build the pipeline
builder = StateGraph(MultiAgentState)
builder.add_node("research", research_node)
builder.add_node("code", coding_node)
builder.add_node("review", review_node)
builder.add_edge(START, "research")
builder.add_edge("research", "code")
builder.add_edge("code", "review")
builder.add_edge("review", END)

pipeline = builder.compile()
result = pipeline.invoke({"task": "Create a web scraper for news headlines", "messages": []})
```

---

## 4. Supervisor Pattern

A **supervisor LLM** dynamically routes tasks between specialized sub-agents.

```python
from pydantic import BaseModel

class RouterDecision(BaseModel):
    next: str  # "researcher" | "coder" | "FINISH"
    reason: str

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a supervisor managing these agents:
    - researcher: searches for information
    - coder: writes and runs Python code
    - FINISH: when the task is complete

    Based on the conversation, who should act next?"""),
    MessagesPlaceholder("messages"),
])

supervisor = supervisor_prompt | llm.with_structured_output(RouterDecision)

def supervisor_node(state):
    decision = supervisor.invoke({"messages": state["messages"]})
    if decision.next == "FINISH":
        return END
    return decision.next

builder.add_conditional_edges("supervisor", supervisor_node, {
    "researcher": "researcher",
    "coder": "coder",
    END: END,
})
```

---

## 5. Tool-Use Pattern (Augmented Agent)

The most common pattern — a single agent with several tools.

```python
@tool
def search_web(query: str) -> str:
    """Search the internet for current information."""
    return web_search_api.search(query)

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    # Use a sandbox!
    return sandbox.run(code)

@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    with open(path) as f:
        return f.read()

agent = create_react_agent(llm, [search_web, run_python, read_file])
```

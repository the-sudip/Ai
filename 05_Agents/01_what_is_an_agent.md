# What is an AI Agent?

---

## Definition

An **AI Agent** is a system where an LLM acts as the **reasoning engine** to autonomously decide what actions to take, execute those actions (via tools), observe the results, and repeat until a goal is achieved.

```
Agent = LLM (brain) + Tools (hands) + Loop (persistence)
```

Unlike a chain (fixed sequence of steps), an agent **determines its own execution path at runtime**.

---

## The Agent Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│  User Goal                                  │
│       ↓                                     │
│  LLM Reasons: "What should I do next?"      │
│       ↓                                     │
│  LLM Decides: Call tool X with args Y       │
│       ↓                                     │
│  Tool Executes → Returns result             │
│       ↓                                     │
│  LLM Observes result                        │
│       ↓                                     │
│  LLM Reasons again...                       │
│       ↓                                     │
│  (repeat until goal met or max_iterations)  │
│       ↓                                     │
│  Final Answer                               │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Chain vs Agent

| | Chain | Agent |
|---|---|---|
| **Execution path** | Fixed, developer-defined | Dynamic, LLM-decided |
| **Tool use** | None (or fixed) | Autonomous tool selection |
| **Loops** | No native looping | Core feature |
| **Flexibility** | Predictable, reliable | Flexible, powerful |
| **Risk** | Low (deterministic) | Higher (can go wrong) |
| **Use case** | Simple pipelines, RAG | Open-ended tasks |

---

## When to Use an Agent

Use an agent when:
- The task requires **multiple steps** that can't be predetermined
- The path depends on **intermediate results**
- You need to **call external tools** dynamically
- The task is **open-ended** ("research this topic and write a report")

Use a chain when:
- Steps are **always the same**
- Predictability and reliability are critical
- Latency must be minimal

---

## Minimal Agent Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Define a tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# Create agent
llm = ChatOpenAI(model="gpt-4o")
tools = [add_numbers, multiply_numbers]
agent = create_react_agent(llm, tools)

# Run
result = agent.invoke({
    "messages": [HumanMessage("What is (15 + 7) × 3?")]
})
print(result["messages"][-1].content)
# Agent calls add_numbers(15, 7) → 22, then multiply_numbers(22, 3) → 66
# "The answer is 66"
```

---

## What Makes a Good Agent?

1. **Clear, descriptive tool names and docstrings** — the LLM uses these to decide
2. **Appropriate tools** — don't give the agent tools it doesn't need
3. **Good system prompt** — constrains behavior, defines the agent's role
4. **Stopping conditions** — `max_iterations`, `max_execution_time`
5. **Error handling** — tools should return errors as strings, not raise exceptions

---

## Agent Failure Modes

| Failure | Cause | Fix |
|---|---|---|
| Infinite loop | LLM keeps calling tools without progress | Set `max_iterations` |
| Wrong tool selected | Poor tool description | Write clearer docstrings |
| Tool argument errors | LLM passes wrong arg types | Use Pydantic schemas on tools |
| Over-thinking | Too many reasoning steps | Reduce temperature |
| Prompt injection | Malicious tool output hijacks agent | Sanitize tool outputs |

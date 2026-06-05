# The Runnable Interface

`Runnable` is the **core protocol** that every component in LangChain implements. It provides a standard interface so any two components can be composed together.

---

## Why Runnable Exists

Before LCEL, LangChain had incompatible chain types (`LLMChain`, `SequentialChain`, etc.) with different APIs. The Runnable interface unifies everything:

```
PromptTemplate  ─┐
ChatOpenAI      ─┤  All implement Runnable → all composable with |
StrOutputParser ─┘
BaseTool
BaseRetriever
```

---

## The Runnable Protocol

Every Runnable must implement:

| Method | Description | Returns |
|---|---|---|
| `invoke(input, config?)` | Run once, return result | Single output |
| `batch(inputs, config?)` | Run on list of inputs | List of outputs |
| `stream(input, config?)` | Run and yield chunks | Iterator |
| `ainvoke(input, config?)` | Async invoke | Awaitable |
| `abatch(inputs, config?)` | Async batch | Awaitable list |
| `astream(input, config?)` | Async stream | AsyncIterator |

---

## `invoke` — Basic Execution

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

# Each component is a Runnable
result = llm.invoke("What is 2+2?")           # LLM invoke
print(result.content)  # "4"

prompt = ChatPromptTemplate.from_template("Say hello to {name}")
formatted = prompt.invoke({"name": "Alice"})   # Prompt invoke
print(formatted.messages[0].content)           # "Say hello to Alice"

parser = StrOutputParser()
text = parser.invoke(result)                   # Parser invoke
print(text)  # "4"
```

---

## `batch` — Parallel Execution

Runs multiple inputs efficiently, optionally in parallel:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

questions = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Brazil?",
]

# Runs all 3 concurrently
answers = llm.batch(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a.content}\n")

# Control concurrency
answers = llm.batch(
    questions,
    config={"max_concurrency": 2},  # max 2 at a time
)
```

---

## `stream` — Streaming Tokens

Yields output chunks as they arrive (token by token for LLMs):

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write a poem about {topic}")
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

chain = prompt | llm | parser

# Stream token by token
for chunk in chain.stream({"topic": "the ocean"}):
    print(chunk, end="", flush=True)
print()  # newline at end
```

---

## `astream_events` — Detailed Async Streaming

For observing every event in a chain (tools, LLM tokens, chain starts/ends):

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Explain {topic} briefly")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

async def stream_with_events():
    async for event in chain.astream_events({"topic": "RAG"}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_chain_start":
            print(f"\n[Start: {event['name']}]")
        elif kind == "on_chain_end":
            print(f"\n[End: {event['name']}]")

asyncio.run(stream_with_events())
```

---

## Runnable Methods for Introspection

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Answer: {question}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# What does the chain expect as input?
print(chain.input_schema.schema())
# {"properties": {"question": {"type": "string"}}, "required": ["question"]}

# What does it output?
print(chain.output_schema.schema())
# {"type": "string"}

# Visualize the chain structure
chain.get_graph().print_ascii()
# PromptTemplate → ChatOpenAI → StrOutputParser
```

---

## `with_config` — Runtime Configuration

```python
chain = prompt | llm | parser

# Override config per-call
result = chain.invoke(
    {"topic": "Python"},
    config={
        "run_name": "my_poem_chain",      # name for tracing
        "tags": ["poetry", "creative"],   # for filtering in LangSmith
        "metadata": {"user_id": "u_123"}, # custom metadata
        "max_concurrency": 5,             # for batch
        "callbacks": [my_callback],       # attach callbacks
        "configurable": {"model": "gpt-4o-mini"},  # for configurable chains
    }
)

# Or bind config permanently
fast_chain = chain.with_config(
    run_name="fast_poem",
    tags=["production"],
)
```

---

## `with_retry` — Automatic Retries

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Retry on specific errors, up to 3 times with exponential backoff
resilient_llm = llm.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

result = resilient_llm.invoke("Hello")
```

---

## `with_fallbacks` — Graceful Degradation

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o")
fallback = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# If primary fails, automatically try fallback
resilient_llm = primary.with_fallbacks([fallback])

result = resilient_llm.invoke("What is LangChain?")
# Uses GPT-4o; if that fails (rate limit, outage), uses Claude
```

---

## `bind` — Pre-filling Arguments

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Create specialized versions with pre-bound kwargs
creative_llm = llm.bind(temperature=1.2, max_tokens=500)
precise_llm  = llm.bind(temperature=0.0, max_tokens=200)

# bind_tools — attach tools to an LLM
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm_with_tools = llm.bind_tools([calculator])
```

---

## Implementing a Custom Runnable

```python
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from typing import Optional

class UpperCaseRunnable(RunnableSerializable[str, str]):
    """A custom Runnable that uppercases its input."""
    
    prefix: str = ""
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        return self.prefix + input.upper()

upper = UpperCaseRunnable(prefix=">> ")
print(upper.invoke("hello world"))  # ">> HELLO WORLD"

# Works with all Runnable methods
print(upper.batch(["hello", "world"]))    # [">> HELLO", ">> WORLD"]
for chunk in upper.stream("hello"):       # streams character by character
    print(chunk, end="")
```

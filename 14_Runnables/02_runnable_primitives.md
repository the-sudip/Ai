# Runnable Primitives

LangChain provides a set of built-in Runnables that cover the most common data manipulation patterns when building chains.

---

## `RunnablePassthrough` — Pass Input Through

Passes input unchanged. Essential for keeping the original input available further down the chain.

```python
from langchain_core.runnables import RunnablePassthrough

# Simply passes its input through unchanged
passthrough = RunnablePassthrough()
print(passthrough.invoke({"question": "What is RAG?"}))
# → {"question": "What is RAG?"}

# ── Common use: keep original question in a RAG chain ────────────────
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

setup = RunnableParallel({
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),   # passes the original query string through
})
# Input: "What is RAG?"
# Output: {"context": "...retrieved docs...", "question": "What is RAG?"}
```

### `RunnablePassthrough.assign` — Add Keys to a Dict

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

chain = RunnablePassthrough.assign(
    answer=lambda x: llm.invoke(x["question"]).content,
    question_length=lambda x: len(x["question"]),
)

result = chain.invoke({"question": "What is the capital of France?"})
print(result)
# {
#   "question": "What is the capital of France?",  ← original preserved
#   "answer": "Paris",                              ← added
#   "question_length": 36,                          ← added
# }
```

---

## `RunnableParallel` — Run Multiple Runnables in Parallel

Runs several Runnables on the same input simultaneously and merges results into a dict:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o")

parallel = RunnableParallel({
    "original": RunnablePassthrough(),              # keep original input
    "uppercase": lambda x: x.upper(),              # transform
    "word_count": lambda x: len(x.split()),         # compute
    "summary": llm,                                 # LLM call
})

result = parallel.invoke("LangChain makes building LLM apps easier.")
# All four run on the same input concurrently
# {
#   "original": "LangChain makes building LLM apps easier.",
#   "uppercase": "LANGCHAIN MAKES BUILDING LLM APPS EASIER.",
#   "word_count": 7,
#   "summary": AIMessage(content="..."),
# }
```

### Dict Shorthand (most common)

```python
# The pipe operator accepts a plain dict as a RunnableParallel shorthand
chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
} | prompt | llm | parser
```

---

## `RunnableLambda` — Wrap Any Function

Turns a plain Python function into a Runnable:

```python
from langchain_core.runnables import RunnableLambda

# Wrap a regular function
def double(x: int) -> int:
    return x * 2

runnable_double = RunnableLambda(double)
print(runnable_double.invoke(5))    # 10
print(runnable_double.batch([1, 2, 3]))  # [2, 4, 6]

# Inline with lambda
chain = prompt | llm | RunnableLambda(lambda msg: msg.content.strip())

# Async support
import asyncio

async def async_fetch(url: str) -> str:
    # ... async http call
    return "fetched content"

async_runnable = RunnableLambda(async_fetch)
result = asyncio.run(async_runnable.ainvoke("https://example.com"))
```

---

## `RunnableBranch` — Conditional Routing

Routes input to different Runnables based on conditions:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Define specialized chains
technical_chain = (
    ChatPromptTemplate.from_template("Answer this technical question precisely: {question}")
    | llm
)
casual_chain = (
    ChatPromptTemplate.from_template("Answer this conversationally: {question}")
    | llm
)
fallback_chain = (
    ChatPromptTemplate.from_template("Answer: {question}")
    | llm
)

branch = RunnableBranch(
    # (condition, runnable_if_true)
    (lambda x: "code" in x["question"].lower() or "function" in x["question"].lower(),
     technical_chain),
    (lambda x: x["question"].endswith("?") and len(x["question"]) < 50,
     casual_chain),
    # Default fallback (no condition)
    fallback_chain,
)

result = branch.invoke({"question": "How do I write a Python function?"})
# Routes to technical_chain because "function" is in the question
```

---

## `RunnableWithFallbacks` — Error Recovery

Tries primary Runnable; on failure, tries fallbacks in order:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o")
backup1 = ChatOpenAI(model="gpt-4o-mini")
backup2 = ChatAnthropic(model="claude-3-5-haiku-20241022")

# Method 1: .with_fallbacks()
resilient = primary.with_fallbacks([backup1, backup2])

# Method 2: RunnableWithFallbacks directly
from langchain_core.runnables import RunnableWithFallbacks

resilient = RunnableWithFallbacks(
    runnable=primary,
    fallbacks=[backup1, backup2],
    exception_key="error",         # store exception in output if all fail
)

result = resilient.invoke("Hello!")
# Tries primary → if fails, tries backup1 → if fails, tries backup2
```

---

## `itemgetter` — Extract Dict Values

The standard library `operator.itemgetter` works as a Runnable in chains:

```python
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
llm = ChatOpenAI(model="gpt-4o")

chain = (
    {
        "language": itemgetter("target_lang"),   # extract from input dict
        "text": itemgetter("content"),           # extract from input dict
    }
    | prompt
    | llm
)

result = chain.invoke({
    "target_lang": "Spanish",
    "content": "Hello, how are you?",
    "extra_field": "ignored",         # not used
})
print(result.content)  # "Hola, ¿cómo estás?"
```

---

## `RunnableGenerator` — Streaming Custom Logic

For custom streaming logic that yields chunks:

```python
from langchain_core.runnables import RunnableGenerator
from typing import Iterator

def sentence_by_sentence(input: Iterator[str]) -> Iterator[str]:
    """Re-chunk stream output sentence by sentence."""
    buffer = ""
    for chunk in input:
        buffer += chunk
        while ". " in buffer:
            sentence, buffer = buffer.split(". ", 1)
            yield sentence + ". "
    if buffer:
        yield buffer

chain = prompt | llm | StrOutputParser() | RunnableGenerator(sentence_by_sentence)

for sentence in chain.stream({"topic": "LangChain"}):
    print(f"[Sentence]: {sentence}")
```

---

## Quick Reference

| Primitive | Purpose | Key Use Case |
|---|---|---|
| `RunnablePassthrough` | Pass input unchanged | Keep original query in RAG chain |
| `RunnablePassthrough.assign` | Add new keys to dict | Enrich state without losing original |
| `RunnableParallel` | Run multiple in parallel | Fetch context + pass query simultaneously |
| `RunnableLambda` | Wrap any function | Custom transforms in a chain |
| `RunnableBranch` | Conditional routing | Route to specialized chains |
| `RunnableWithFallbacks` | Error recovery | Primary LLM → backup LLM |
| `itemgetter` | Extract dict keys | Map input dict fields to prompt variables |
| `RunnableGenerator` | Custom streaming | Re-chunk or filter streamed output |

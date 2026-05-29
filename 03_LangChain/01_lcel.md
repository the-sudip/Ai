# LangChain Expression Language (LCEL)

LCEL is LangChain's **declarative composition syntax** using the `|` pipe operator. It lets you chain components together into a pipeline, where the output of each step feeds into the next.

---

## The Pipe Operator `|`

```python
chain = component_1 | component_2 | component_3
result = chain.invoke(input)
```

This is equivalent to:
```python
result = component_3.invoke(component_2.invoke(component_1.invoke(input)))
```

But LCEL gives you **free streaming, async, batching, and tracing** automatically.

---

## The Runnable Interface

Every LangChain component (LLM, prompt, parser, retriever, tool) implements the `Runnable` interface:

| Method | Description |
|---|---|
| `.invoke(input)` | Single synchronous call |
| `.stream(input)` | Streaming (yields chunks) |
| `.batch(inputs)` | Parallel batch of inputs |
| `.ainvoke(input)` | Async single call |
| `.astream(input)` | Async streaming |
| `.abatch(inputs)` | Async batch |

---

## Basic LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "{question}"),
])

parser = StrOutputParser()

# Compose the chain
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"question": "What is LCEL?"})
print(result)  # "LCEL stands for LangChain Expression Language..."
```

---

## Streaming

Because every Runnable supports streaming, LCEL chains stream automatically:

```python
# Stream tokens as they arrive
for chunk in chain.stream({"question": "Tell me about Python."}):
    print(chunk, end="", flush=True)
```

---

## Async Invocation

```python
import asyncio

async def main():
    result = await chain.ainvoke({"question": "What is async programming?"})
    print(result)

asyncio.run(main())
```

---

## Batch Processing

Run the chain for multiple inputs in parallel:

```python
questions = [
    {"question": "What is Python?"},
    {"question": "What is JavaScript?"},
    {"question": "What is Rust?"},
]

results = chain.batch(questions)
# All 3 LLM calls run concurrently
for r in results:
    print(r)
```

---

## `RunnableParallel` — Run Branches in Parallel

```python
from langchain_core.runnables import RunnableParallel

pros_chain = pros_prompt | llm | StrOutputParser()
cons_chain = cons_prompt | llm | StrOutputParser()

parallel = RunnableParallel({
    "pros": pros_chain,
    "cons": cons_chain,
})

result = parallel.invoke({"topic": "working from home"})
# {"pros": "...", "cons": "..."}
# Both chains run at the same time
```

Shorthand using a plain dict at the start of a chain:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## `RunnablePassthrough` — Forward Input Unchanged

```python
from langchain_core.runnables import RunnablePassthrough

# In a RAG chain, pass the original question AND retrieved docs
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),  # passes the query string as-is
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## `RunnableLambda` — Wrap Any Python Function

```python
from langchain_core.runnables import RunnableLambda

def validate_input(text: str) -> str:
    if len(text) < 5:
        raise ValueError("Query too short")
    return text.strip().lower()

chain = RunnableLambda(validate_input) | prompt | llm | StrOutputParser()
```

---

## `RunnableWithFallbacks` — Fallback on Error

```python
primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatOpenAI(model="gpt-4o-mini")

robust_chain = (prompt | primary_llm | parser).with_fallbacks(
    [prompt | fallback_llm | parser]
)
# If primary_llm raises an error, automatically tries fallback_llm
```

---

## Inspecting a Chain

```python
chain.input_schema.schema()   # what input the chain expects
chain.output_schema.schema()  # what output the chain produces
chain.get_graph().print_ascii()  # visualize the DAG
```

---

## LCEL vs Traditional Chains

| | LCEL (`|` operator) | Old `LLMChain` |
|---|---|---|
| Composition | Explicit pipes | Configuration-based |
| Streaming | Built-in | Manual |
| Async | Built-in | Manual |
| Batch | Built-in | Manual |
| Debugging | LangSmith traces every step | Limited |
| Flexibility | Compose any Runnable | Fixed chain types |

> **LCEL is the current standard**. The old `LLMChain`, `ConversationalChain`, etc. are deprecated.

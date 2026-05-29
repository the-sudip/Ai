# LangSmith — Observability & Tracing

LangSmith is LangChain's platform for **debugging, monitoring, and evaluating** LLM applications. It traces every call in your chain/agent, showing inputs, outputs, token counts, latency, and errors.

---

## Why LangSmith?

Without tracing, debugging a multi-step chain or agent is like flying blind. LangSmith shows you:
- What prompt was sent to the LLM (after template formatting)
- What the LLM responded
- Which tool was called with what arguments
- How long each step took
- How many tokens were consumed

---

## Setup

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "my-interview-prep"  # group traces by project

# That's it! All subsequent chain/agent calls are automatically traced.
```

Or use a `.env` file:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=my_project
```

---

## What Gets Traced Automatically

Once the environment variables are set, **every Runnable, chain, agent, and tool call** is traced — no code changes needed.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "transformers"})
# → Automatically creates a trace in LangSmith with all intermediate steps
```

---

## Manual Tracing with `@traceable`

For custom functions not built with LangChain components:

```python
from langsmith import traceable

@traceable(name="my_custom_function")
def preprocess_query(query: str) -> str:
    """Custom preprocessing step."""
    return query.strip().lower()

@traceable
def full_pipeline(user_input: str) -> str:
    cleaned = preprocess_query(user_input)
    return chain.invoke({"topic": cleaned})
```

---

## Adding Metadata to Traces

```python
chain.invoke(
    {"question": "What is AI?"},
    config={
        "metadata": {
            "user_id": "user_123",
            "session_id": "abc",
            "experiment": "gpt4o_v2",
        },
        "tags": ["production", "search"],
        "run_name": "user_query_v2",
    }
)
```

---

## Datasets & Evaluation

LangSmith lets you create test datasets and run evaluations:

```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset("QA Test Set")
client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "RAG stands for Retrieval-Augmented Generation..."}],
    dataset_id=dataset.id,
)

# Run evaluation
from langsmith.evaluation import evaluate

results = evaluate(
    lambda inputs: chain.invoke(inputs),
    data="QA Test Set",
    evaluators=["correctness"],
)
```

---

## Key LangSmith Features

| Feature | Description |
|---|---|
| **Traces** | Full tree of every step in a chain/agent run |
| **Playground** | Test prompts interactively in the UI |
| **Datasets** | Store test inputs and expected outputs |
| **Evaluations** | Automated scoring against datasets |
| **Monitoring** | Track latency, errors, costs in production |
| **Annotation Queue** | Human review and labeling of traces |

---

## Viewing a Trace

In the LangSmith UI (smith.langchain.com), a trace for a RAG chain looks like:

```
▼ RAG Chain (1.2s, 847 tokens)
  ▼ Retriever (0.3s)
    Input: "What is attention?"
    Output: [3 documents...]
  ▼ ChatOpenAI (0.9s, 734 tokens)
    Input: [formatted prompt with context]
    Output: "Attention is a mechanism..."
  ▼ StrOutputParser (0ms)
    Output: "Attention is a mechanism..."
```

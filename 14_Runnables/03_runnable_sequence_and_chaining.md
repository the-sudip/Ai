# Building Chains with the Pipe Operator

The `|` operator is the heart of LCEL — it composes Runnables into a `RunnableSequence`.

---

## The Pipe Operator `|`

```python
chain = A | B | C

# Equivalent to:
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(first=A, middle=[B], last=C)

# Data flows left to right:
# input → A → (A's output) → B → (B's output) → C → final output
```

### Rules
- `A`'s **output type** must match `B`'s **input type**
- Any dict `{"key": runnable}` is automatically a `RunnableParallel`
- Plain functions and lambdas become `RunnableLambda` automatically

---

## Classic LLM Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
llm   = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Three Runnables piped together
chain = prompt | llm | parser
#       dict → ChatPromptValue → AIMessage → str

result = chain.invoke({"text": "LangChain is a framework for building LLM-powered applications..."})
print(result)  # "LangChain is a framework for building LLM-powered applications."
```

---

## RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings()).as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""Answer using only the context below.
Context: {context}
Question: {question}""")

rag_chain = (
    {
        "context": retriever | format_docs,   # retriever + formatter
        "question": RunnablePassthrough(),    # original query passes through
    }
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LCEL?")
```

---

## Chain with Source Documents

Return both the answer and the source documents used:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

retrieval_chain = RunnableParallel({
    "docs": retriever,
    "question": RunnablePassthrough(),
})

answer_chain = (
    {
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

full_chain = retrieval_chain | RunnablePassthrough.assign(answer=answer_chain)

result = full_chain.invoke("What is RAG?")
print(result["answer"])   # LLM answer
print(result["docs"])     # source documents
```

---

## Multi-Step Transformation Chain

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import List

class KeyPoints(BaseModel):
    points: List[str]
    sentiment: str

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(KeyPoints)

# Step 1: Preprocess input
def clean_text(text: str) -> str:
    return text.strip().replace("\n", " ")

# Step 2: Extract key points
extract_chain = (
    RunnableLambda(clean_text)
    | RunnableLambda(lambda t: f"Extract key points from: {t}")
    | structured_llm
)

# Step 3: Format output
def format_key_points(kp: KeyPoints) -> str:
    points = "\n".join(f"• {p}" for p in kp.points)
    return f"Sentiment: {kp.sentiment}\n\nKey Points:\n{points}"

full_chain = extract_chain | RunnableLambda(format_key_points)

result = full_chain.invoke("  LangChain is great for RAG. It supports many vector stores.  ")
print(result)
```

---

## Branching and Merging

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# Fan-out: one input, multiple outputs
fan_out = RunnableParallel({
    "formal":   prompt_formal   | llm | parser,
    "casual":   prompt_casual   | llm | parser,
    "technical": prompt_technical | llm | parser,
})

# Fan-in: merge multiple outputs into one
def merge_responses(x: dict) -> str:
    return f"""FORMAL: {x['formal']}

CASUAL: {x['casual']}

TECHNICAL: {x['technical']}"""

chain = fan_out | RunnableLambda(merge_responses)

result = chain.invoke({"topic": "quantum computing"})
```

---

## Inspecting Chain Structure

```python
chain = prompt | llm | parser

# Schema
print(chain.input_schema.schema())
print(chain.output_schema.schema())

# ASCII graph
chain.get_graph().print_ascii()
# PromptTemplate
#       |
# ChatOpenAI
#       |
# StrOutputParser

# Steps in the sequence
print(chain.steps)
# [ChatPromptTemplate, ChatOpenAI, StrOutputParser]

# Number of steps
print(len(chain.steps))  # 3
```

---

## Chaining vs Old-Style Chains

```python
# ❌ Old (legacy): LLMChain
from langchain.chains import LLMChain
old_chain = LLMChain(llm=llm, prompt=prompt)
result = old_chain.run("input")   # inconsistent API

# ✅ New (LCEL): RunnableSequence via |
new_chain = prompt | llm | StrOutputParser()
result = new_chain.invoke({"question": "input"})  # consistent everywhere

# LCEL benefits:
# - Streaming built-in
# - Async built-in
# - Batch built-in
# - Better tracing in LangSmith
# - Composable with any Runnable
```

---

## `RunnableSequence` Directly

```python
from langchain_core.runnables import RunnableSequence

# Explicit sequence (equivalent to | chaining)
chain = RunnableSequence(
    first=prompt,
    middle=[llm],
    last=parser,
)

# Or from a list
chain = RunnableSequence.from_list([prompt, llm, parser])
```

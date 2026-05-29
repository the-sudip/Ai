# Deep Lake as a LangChain Vector Store

Full integration guide for using Deep Lake as the vector store in LangChain RAG pipelines.

---

## Setup

```bash
pip install deeplake langchain-community langchain-openai
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["ACTIVELOOP_TOKEN"] = "your_activeloop_token"  # for Hub storage
```

---

## Creating a Vector Store

### From Documents (most common)

```python
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content="LangGraph uses a StateGraph to define agent workflows.", metadata={"topic": "langgraph"}),
    Document(page_content="RAG combines retrieval with LLM generation.", metadata={"topic": "rag"}),
    Document(page_content="Embeddings encode semantic meaning as vectors.", metadata={"topic": "embeddings"}),
]

# Local storage
vectorstore = DeepLake.from_documents(
    documents=docs,
    embedding=embedder,
    dataset_path="./langchain_deep_lake",
    overwrite=True,
)

# Activeloop Hub (persistent, shareable)
vectorstore = DeepLake.from_documents(
    documents=docs,
    embedding=embedder,
    dataset_path="hub://your_username/langchain_docs",
    overwrite=True,
)
```

### Load Existing Dataset

```python
# Load without re-embedding (fast)
vectorstore = DeepLake(
    dataset_path="./langchain_deep_lake",
    embedding=embedder,
    read_only=True,    # prevents accidental writes
)
```

---

## Search Methods

### Similarity Search

```python
results = vectorstore.similarity_search(
    query="how do agents work in LangGraph?",
    k=3,
)

for doc in results:
    print(f"[{doc.metadata['topic']}] {doc.page_content}")
```

### Similarity Search with Scores

```python
results = vectorstore.similarity_search_with_score(
    query="vector embeddings",
    k=3,
)

for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}")
```

### Maximum Marginal Relevance (MMR)

Returns diverse results — avoids returning near-duplicate documents:

```python
results = vectorstore.max_marginal_relevance_search(
    query="agent workflows",
    k=4,
    fetch_k=20,      # fetch 20 candidates, return 4 most diverse
    lambda_mult=0.5, # 0 = max diversity, 1 = max relevance
)
```

### Metadata Filtering

```python
# Filter using TQL (Tensor Query Language)
results = vectorstore.similarity_search(
    query="graph-based agents",
    k=3,
    filter={"metadata": {"topic": "langgraph"}},
)

# Or with Deep Lake's TQL directly
results = vectorstore.similarity_search(
    query="agent workflows",
    k=3,
    tql="SELECT * WHERE metadata['topic'] == 'langgraph'",
)
```

---

## As a LangChain Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",     # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5},
)

# In a RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("""Answer using only the context below.
Context: {context}
Question: {question}""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = chain.invoke("How does LangGraph handle agent state?")
print(answer)
```

---

## Adding & Managing Documents

```python
# Add new documents to existing store
new_docs = [
    Document(page_content="Checkpointers persist LangGraph state to a database.", metadata={"topic": "langgraph"}),
]
vectorstore.add_documents(new_docs)

# Add with custom IDs
ids = vectorstore.add_documents(new_docs, ids=["doc_checkpointer_001"])

# Delete by ID
vectorstore.delete(ids=["doc_checkpointer_001"])

# Check dataset size
import deeplake
ds = deeplake.load("./langchain_deep_lake")
print(f"Total documents: {len(ds)}")
```

---

## Ingestion Pipeline — Documents to Deep Lake

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings

# 1. Load
loader = DirectoryLoader("./documents/", glob="**/*.pdf", loader_cls=PyPDFLoader)
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} pages")

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunks")

# 3. Embed and store
embedder = OpenAIEmbeddings()
vectorstore = DeepLake.from_documents(
    documents=chunks,
    embedding=embedder,
    dataset_path="hub://your_username/company_knowledge_base",
)
print(f"Stored {len(chunks)} chunks in Deep Lake")

# 4. Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What is our refund policy?")
```

---

## TQL — Tensor Query Language

Deep Lake has its own SQL-like query language for filtering:

```python
import deeplake

ds = deeplake.load("./langchain_deep_lake")

# Filter by metadata value
view = ds.query("SELECT * WHERE metadata['topic'] == 'langgraph'")

# Combine vector search with filter
results = vectorstore.similarity_search(
    query="state management",
    k=5,
    tql="SELECT * WHERE metadata['topic'] IN ('langgraph', 'agents')"
)

# Range filter
results = vectorstore.similarity_search(
    query="embeddings",
    k=5,
    tql="SELECT * WHERE metadata['year'] >= 2024",
)
```

---

## Multi-modal Storage (Beyond Text)

Deep Lake can store images alongside text embeddings:

```python
import deeplake
import numpy as np
from PIL import Image

ds = deeplake.Dataset("./image_dataset")
ds.create_tensor("images", htype="image", sample_compression="jpeg")
ds.create_tensor("captions", htype="text")
ds.create_tensor("embedding", htype="embedding", dims=512)

with ds:
    ds.images.append(deeplake.read("photo.jpg"))
    ds.captions.append("A photo of a mountain landscape")
    ds.embedding.append(np.random.rand(512).astype("float32"))  # your CLIP embedding

# Retrieve by image embedding similarity
query_embedding = clip_model.encode(query_image)
view = ds.query(f"SELECT * ORDER BY l2_norm(embedding - ARRAY{query_embedding.tolist()}) LIMIT 5")
```

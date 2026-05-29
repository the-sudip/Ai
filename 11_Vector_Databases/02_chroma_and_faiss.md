# Chroma & FAISS — Local Vector Stores

The two most common **in-process** vector stores for development and smaller deployments.

---

## Chroma

**Chroma** is an open-source, batteries-included vector database designed for LLM applications. It runs in-process (no server needed) or as a persistent store.

```bash
pip install langchain-chroma chromadb
```

### In-Memory (Development)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content="The Eiffel Tower is in Paris.", metadata={"country": "France"}),
    Document(page_content="Mount Fuji is in Japan.", metadata={"country": "Japan"}),
    Document(page_content="The Colosseum is in Rome.", metadata={"country": "Italy"}),
]

# In-memory (lost when process exits)
db = Chroma.from_documents(docs, embedder)

results = db.similarity_search("famous landmark in Europe", k=2)
for doc in results:
    print(doc.page_content, doc.metadata)
```

### Persistent Storage

```python
# Save to disk — survives restarts
db = Chroma.from_documents(
    documents=docs,
    embedding=embedder,
    persist_directory="./chroma_db",
    collection_name="landmarks",
)

# Load from disk later
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder,
    collection_name="landmarks",
)
```

### Adding Documents Incrementally

```python
# Add more documents after creation
new_docs = [
    Document(page_content="Big Ben is in London.", metadata={"country": "UK"}),
]
db.add_documents(new_docs)

# Add with explicit IDs (for deduplication/updates)
db.add_documents(new_docs, ids=["doc_bigben"])

# Delete by ID
db.delete(ids=["doc_bigben"])
```

### Search Methods

```python
# Basic similarity search
results = db.similarity_search("European landmarks", k=3)

# Similarity search with relevance scores (0.0–1.0)
results_with_scores = db.similarity_search_with_relevance_scores("European landmarks", k=3)
for doc, score in results_with_scores:
    print(f"Score: {score:.3f} | {doc.page_content}")

# Maximum Marginal Relevance — diverse results (less repetitive)
results_mmr = db.max_marginal_relevance_search(
    query="famous landmarks",
    k=3,
    fetch_k=10,   # fetch 10 candidates, pick 3 most diverse
    lambda_mult=0.5,  # 0 = max diversity, 1 = max relevance
)

# With metadata filter
results_filtered = db.similarity_search(
    query="famous place",
    k=2,
    filter={"country": "France"}
)
```

### Chroma Collection Management

```python
import chromadb

# Direct Chroma client (lower-level)
client = chromadb.PersistentClient(path="./chroma_db")

# List all collections
print(client.list_collections())

# Get or create a collection
collection = client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},  # explicitly set distance metric
)

# Peek at stored data
print(collection.peek(5))
print(f"Total documents: {collection.count()}")
```

---

## FAISS

**FAISS** (Facebook AI Similarity Search) is a battle-tested C++ library with Python bindings, optimized for billion-scale vector search.

```bash
pip install langchain-community faiss-cpu
# GPU version: pip install faiss-gpu
```

### Basic FAISS Usage

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embedder = OpenAIEmbeddings()

docs = [
    Document(page_content="Transformers revolutionized NLP.", metadata={"field": "AI"}),
    Document(page_content="BERT uses bidirectional attention.", metadata={"field": "AI"}),
    Document(page_content="Python is the most popular language for ML.", metadata={"field": "Programming"}),
]

# Build index
db = FAISS.from_documents(docs, embedder)

results = db.similarity_search("language models", k=2)
```

### Saving and Loading

```python
# Save to disk
db.save_local("./faiss_index")

# Load from disk
db = FAISS.load_local(
    "./faiss_index",
    embeddings=embedder,
    allow_dangerous_deserialization=True,  # required flag
)
```

### Merging Two FAISS Indexes

```python
# Useful for distributed indexing
db1 = FAISS.from_documents(docs_batch1, embedder)
db2 = FAISS.from_documents(docs_batch2, embedder)

db1.merge_from(db2)  # db1 now contains all documents
```

### FAISS Index Types

```python
import faiss

# Flat (exact, brute force) — small datasets
index = faiss.IndexFlatL2(1536)   # L2 distance, 1536 dims
index = faiss.IndexFlatIP(1536)   # Inner product (cosine if normalized)

# IVF (Inverted File) — large datasets, approximate
quantizer = faiss.IndexFlatL2(1536)
index = faiss.IndexIVFFlat(quantizer, 1536, nlist=100)  # 100 clusters
index.train(training_vectors)  # must train first

# HNSW — graph-based, fast and accurate
index = faiss.IndexHNSWFlat(1536, 32)  # 32 = graph connectivity
```

---

## Chroma vs FAISS Comparison

| Feature | Chroma | FAISS |
|---|---|---|
| **Persistence** | Built-in (SQLite) | Manual save/load |
| **Metadata filtering** | Yes, flexible | Limited |
| **Scalability** | Millions of docs | Billions of vectors |
| **Index types** | HNSW (automatic) | Many (flat, IVF, HNSW, PQ...) |
| **Ease of use** | Very easy | Moderate |
| **Server mode** | Yes (chromadb server) | No |
| **Best for** | Development, RAG apps | High-performance production |

---

## As LangChain Retriever

Both integrate seamlessly as LangChain retrievers:

```python
# Chroma retriever
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
)

# FAISS retriever
retriever = faiss_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5},
)

# Use in a RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("""Answer based on context only.
Context: {context}
Question: {question}""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

print(chain.invoke("What is BERT?"))
```

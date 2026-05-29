# Pinecone, Weaviate & Qdrant — Cloud/Production Vector Stores

For production workloads that need scalability, high availability, and managed infrastructure.

---

## Pinecone

**Pinecone** is a fully managed, serverless vector database — no infrastructure to run.

```bash
pip install langchain-pinecone pinecone-client
```

### Setup & Indexing

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone
pc = Pinecone(api_key="your_pinecone_api_key")

# Create index (one-time setup)
if "my-index" not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name="my-index",
        dimension=1536,           # must match your embedding model
        metric="cosine",          # "cosine", "euclidean", "dotproduct"
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Create LangChain vector store
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embedder,
    pinecone_api_key="your_pinecone_api_key",
)

# Add documents
from langchain_core.documents import Document
docs = [
    Document(page_content="LangChain connects LLMs with tools.", metadata={"source": "docs"}),
    Document(page_content="RAG improves LLM accuracy with retrieval.", metadata={"source": "paper"}),
]
vectorstore.add_documents(docs)
```

### Querying

```python
# Similarity search
results = vectorstore.similarity_search("how to improve LLM accuracy?", k=3)

# With metadata filter
results = vectorstore.similarity_search(
    "LLM tools",
    k=3,
    filter={"source": {"$eq": "docs"}},
)

# With score
results = vectorstore.similarity_search_with_score("LLM accuracy", k=3)
for doc, score in results:
    print(f"{score:.4f}: {doc.page_content[:80]}")
```

### Namespaces (Multi-tenancy)

```python
# Isolate data per user/tenant using namespaces
user_store = PineconeVectorStore(
    index_name="my-index",
    embedding=embedder,
    namespace="user_123",  # data is isolated per namespace
)
```

---

## Weaviate

**Weaviate** is an open-source vector database with a GraphQL API, strong schema support, and hybrid search built in.

```bash
pip install langchain-weaviate weaviate-client
```

### Connecting

```python
import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings

# Connect to Weaviate Cloud (WCS)
client = weaviate.connect_to_wcs(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("your_weaviate_api_key"),
    headers={"X-OpenAI-Api-Key": "your_openai_key"},  # for auto-vectorization
)

# Or connect to local Weaviate instance
client = weaviate.connect_to_local()

embedder = OpenAIEmbeddings()

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="Document",       # Weaviate class name (must start with uppercase)
    text_key="content",
    embedding=embedder,
)
```

### Adding Documents & Searching

```python
from langchain_core.documents import Document

docs = [Document(page_content="Weaviate supports hybrid search.", metadata={"type": "feature"})]
vectorstore.add_documents(docs)

# Semantic search
results = vectorstore.similarity_search("vector database features", k=3)

# Hybrid search (BM25 + semantic)
results = vectorstore.similarity_search(
    query="hybrid search",
    k=3,
    search_kwargs={"alpha": 0.5},  # 0 = pure BM25, 1 = pure semantic, 0.5 = balanced
)
```

### Weaviate Schema (Classes)

```python
# Weaviate organizes data in "classes" (like tables with vector index)
class_obj = {
    "class": "Article",
    "vectorizer": "text2vec-openai",  # auto-vectorize with OpenAI
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "author", "dataType": ["text"]},
        {"name": "publishedAt", "dataType": ["date"]},
    ],
}
client.schema.create_class(class_obj)
```

---

## Qdrant

**Qdrant** is a high-performance vector database written in Rust. Available as open-source (self-hosted) or cloud.

```bash
pip install langchain-qdrant qdrant-client
```

### In-Memory (Development)

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# In-memory client
client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

embedder = OpenAIEmbeddings()

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_docs",
    embedding=embedder,
)

docs = [
    Document(page_content="Qdrant is written in Rust for high performance.", metadata={"tag": "qdrant"}),
    Document(page_content="Vector databases enable semantic search.", metadata={"tag": "general"}),
]
vectorstore.add_documents(docs)
```

### Persistent / Cloud

```python
# Self-hosted
client = QdrantClient(host="localhost", port=6333)

# Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your_qdrant_api_key",
)
```

### Advanced Qdrant Features

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Metadata filtering with Qdrant filter model
results = vectorstore.similarity_search(
    query="high performance database",
    k=3,
    filter=Filter(
        must=[
            FieldCondition(key="metadata.tag", match=MatchValue(value="qdrant"))
        ]
    ),
)

# Sparse + Dense hybrid search (Qdrant 1.7+)
from langchain_qdrant import FastEmbedSparse, RetrievalMode

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embedding=embedder,
    sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
    collection_name="hybrid_collection",
    retrieval_mode=RetrievalMode.HYBRID,
)
```

---

## Production Comparison

| Feature | Pinecone | Weaviate | Qdrant |
|---|---|---|---|
| **Type** | Managed SaaS only | Open-source + Cloud | Open-source + Cloud |
| **Language** | — | Go | Rust |
| **Hybrid Search** | No (semantic only) | Yes (built-in BM25) | Yes (sparse + dense) |
| **Schema** | Schemaless | Typed schema | Flexible |
| **Multi-tenancy** | Namespaces | Multi-tenancy API | Collections / payload |
| **GraphQL API** | No | Yes | No |
| **Self-hosting** | No | Yes | Yes |
| **Filtering** | Metadata filters | GraphQL where | Qdrant filter DSL |
| **Best for** | Simplicity, SaaS | Schema-rich data, hybrid | Performance, open-source |

---

## Choosing the Right Vector Store

```
Dev / prototype:
  → Chroma (simplest) or FAISS (fastest locally)

Small-medium production, managed:
  → Pinecone (easiest ops) 

Need hybrid search (keyword + semantic):
  → Weaviate or Qdrant

Max performance, self-hosted:
  → Qdrant

Already using PostgreSQL:
  → pgvector (via langchain-postgres)
```

---

## pgvector — Vector Search in PostgreSQL

```python
# Use your existing Postgres as a vector store
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

CONNECTION_STRING = "postgresql+psycopg://user:pass@localhost:5432/mydb"

vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
    connection=CONNECTION_STRING,
)

vectorstore.add_documents(docs)
results = vectorstore.similarity_search("query text", k=5)
```

# Hybrid Search — Combining Keyword & Semantic

Pure semantic search misses exact keyword matches. Hybrid search combines both for better results.

---

## The Problem with Pure Semantic Search

```
Query: "GPT-4o pricing"

Semantic search retrieves: documents about LLMs, pricing models, AI costs
→ Might miss the exact doc that says "GPT-4o costs $5/1M input tokens"

Keyword search retrieves: only docs containing exact tokens "GPT-4o" and "pricing"
→ Misses semantically relevant docs about "OpenAI API costs"

Hybrid: gets both
```

---

## BM25 — Keyword (Sparse) Search

**BM25** (Best Match 25) is the gold standard for keyword search. It's a TF-IDF variant that accounts for document length.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $f(t,d)$ = term frequency in document
- $|d|$ = document length
- $k_1$ = term saturation (default 1.5)
- $b$ = length normalization (default 0.75)

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

docs = [
    Document(page_content="GPT-4o costs $5 per million input tokens."),
    Document(page_content="OpenAI offers various pricing tiers for API access."),
    Document(page_content="Large language models are transforming software development."),
]

# BM25 — pure keyword
bm25_retriever = BM25Retriever.from_documents(docs, k=2)
results = bm25_retriever.invoke("GPT-4o pricing")
print(results[0].page_content)  # → "GPT-4o costs $5 per million input tokens."
```

---

## Ensemble Retriever (Hybrid Search)

LangChain's `EnsembleRetriever` merges results from multiple retrievers using **Reciprocal Rank Fusion (RRF)**:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="GPT-4o costs $5 per million input tokens.", metadata={"source": "pricing"}),
    Document(page_content="OpenAI API pricing varies by model and usage.", metadata={"source": "docs"}),
    Document(page_content="Large language models enable many AI applications.", metadata={"source": "overview"}),
    Document(page_content="Semantic search finds conceptually similar content.", metadata={"source": "ml"}),
]

embedder = OpenAIEmbeddings()

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(docs, k=3)

# Semantic retriever
chroma_db = Chroma.from_documents(docs, embedder)
semantic_retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

# Hybrid: 40% keyword + 60% semantic
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],
)

results = hybrid_retriever.invoke("GPT-4o API costs")
for doc in results:
    print(doc.page_content)
```

### Reciprocal Rank Fusion (RRF)

RRF merges ranked lists without needing to normalize scores from different systems:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where $k=60$ (constant), $R$ = set of rankers, $\text{rank}_r(d)$ = rank of doc $d$ in ranker $r$.

---

## Weaviate Native Hybrid Search

```python
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore(client=client, index_name="Document", 
                                   text_key="content", embedding=embedder)

# alpha=0 → pure BM25, alpha=1 → pure semantic, alpha=0.5 → balanced
results = vectorstore.similarity_search(
    "GPT-4o pricing",
    k=5,
    alpha=0.5,
)
```

---

## Qdrant Sparse + Dense Hybrid

```python
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embedding=OpenAIEmbeddings(),           # dense vectors
    sparse_embedding=FastEmbedSparse(       # sparse (BM25-like) vectors
        model_name="Qdrant/bm25"
    ),
    collection_name="hybrid_docs",
    retrieval_mode=RetrievalMode.HYBRID,    # use both
)

results = vectorstore.similarity_search("GPT-4o pricing", k=3)
```

---

## Contextual Compression (Re-ranking)

After retrieval, **re-rank** or **compress** results to keep only the most relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder re-ranker (more accurate than bi-encoder)
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Wrap any retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,
)

results = compression_retriever.invoke("GPT-4o pricing")
# Returns top 3 most relevant documents, re-scored by cross-encoder
```

### Bi-encoder vs Cross-encoder

| | Bi-encoder | Cross-encoder |
|---|---|---|
| **How** | Encode query & doc separately | Encode (query, doc) pair jointly |
| **Speed** | Fast (pre-compute doc embeddings) | Slow (must run at query time) |
| **Accuracy** | Good | Better (full attention on pair) |
| **Use** | First-stage retrieval | Re-ranking top-k results |

---

## Full Hybrid RAG Pipeline

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embedder = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")

# Build retrievers
bm25 = BM25Retriever.from_documents(docs, k=10)
semantic = Chroma.from_documents(docs, embedder).as_retriever(search_kwargs={"k": 10})

# Hybrid fusion
ensemble = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.3, 0.7],
)

# Re-rank
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=5,
)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble,
)

# RAG chain
prompt = ChatPromptTemplate.from_template("""Answer using the context below.
Context: {context}
Question: {question}""")

chain = (
    {"context": final_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What does GPT-4o cost?")
print(answer)
```

---

## When to Use Each Approach

| Scenario | Best Retrieval |
|---|---|
| General semantic Q&A | Semantic only |
| Exact product names, codes, IDs | BM25 or Hybrid |
| Legal/medical exact term matching | BM25 or Hybrid |
| High-precision requirement | Hybrid + re-ranking |
| Large corpus (10M+ docs) | Dense retrieval + re-ranking |
| Multilingual | Dense (semantic) retrieval |

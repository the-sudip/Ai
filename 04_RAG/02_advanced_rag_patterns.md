# Advanced RAG Patterns

Beyond basic RAG, these techniques significantly improve retrieval quality and answer accuracy.

---

## 1. HyDE — Hypothetical Document Embeddings

**Problem**: A user's question ("What is the return policy?") and the relevant document ("Items can be returned within 30 days...") may have very different vocabulary, causing poor retrieval.

**Solution**: Generate a **hypothetical answer** using the LLM, embed that, and search with it. A hypothetical answer will better match the style/vocabulary of stored documents.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")

# Step 1: Generate a hypothetical answer
hyde_prompt = ChatPromptTemplate.from_template(
    "Write a passage that would directly answer this question:\n\n{question}"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

hypothetical_doc = hyde_chain.invoke({"question": "What is the return policy?"})

# Step 2: Embed the hypothetical answer and search
results = vectorstore.similarity_search(hypothetical_doc, k=4)
```

Full chain:
```python
from langchain_core.runnables import RunnablePassthrough

full_hyde_chain = (
    {
        "context": hyde_chain | vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 2. Multi-Query Retrieval

**Problem**: A single query may not surface all relevant documents due to vocabulary mismatch or the query being phrased one specific way.

**Solution**: Generate 3–5 different phrasings of the question, retrieve for each, and take the union.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

# Generates multiple queries and deduplicated results
docs = multi_retriever.invoke("What are the benefits of transformer models?")
# Internally generates:
# 1. "What are the advantages of transformer architecture?"
# 2. "Why are transformers better than RNNs?"
# 3. "What makes transformers useful in NLP?"
```

---

## 3. Contextual Compression

**Problem**: Retrieved chunks contain a lot of text that's not relevant to the question. This wastes context window space and dilutes the LLM's focus.

**Solution**: Use an LLM to extract only the relevant sentences/paragraphs from each retrieved chunk.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Extractor: pulls only relevant sentences from each chunk
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

compressed_docs = compression_retriever.invoke("What is the refund window?")
# Returns shorter, focused snippets instead of full chunks
```

---

## 4. Re-ranking with Cross-Encoder

**Problem**: Embedding-based similarity is approximate. Two-stage retrieval: retrieve many with embeddings (fast), then re-rank with a more accurate cross-encoder (slower).

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder scores (query, document) pairs directly
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
    top_n=3,  # keep top 3 after reranking
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),  # fetch 20
)
```

---

## 5. Hybrid Search (BM25 + Semantic)

Combines **keyword search** (BM25 — good for exact terms) with **semantic search** (good for meaning). Often outperforms either alone.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 (keyword-based)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Semantic
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],  # 40% BM25, 60% semantic
)
docs = hybrid_retriever.invoke("transformer attention mechanism")
```

---

## 6. Self-RAG

The model decides **when to retrieve** and **validates the retrieved content** before using it.

```python
# Simplified Self-RAG logic
def self_rag(question: str) -> str:
    # Step 1: Does this question need retrieval?
    needs_retrieval = retrieval_decision_chain.invoke({"question": question})

    if needs_retrieval == "YES":
        docs = retriever.invoke(question)
        # Step 2: Are the retrieved docs relevant?
        relevant_docs = [
            doc for doc in docs
            if relevance_check_chain.invoke({"question": question, "doc": doc.page_content}) == "RELEVANT"
        ]
        context = format_docs(relevant_docs)
    else:
        context = ""

    # Step 3: Generate answer
    answer = generation_chain.invoke({"question": question, "context": context})

    # Step 4: Is the answer faithful to context?
    faithfulness = faithfulness_check_chain.invoke({"answer": answer, "context": context})
    return answer
```

---

## 7. Corrective RAG (CRAG)

If retrieved documents are not relevant enough, fall back to a web search.

```python
def corrective_rag(question: str) -> str:
    docs = retriever.invoke(question)

    # Check relevance of retrieved docs
    relevance_scores = [score_relevance(doc, question) for doc in docs]

    if max(relevance_scores) < 0.6:
        # Docs are not relevant — search the web
        web_docs = web_search_tool.invoke(question)
        return generate_answer(question, web_docs)
    else:
        return generate_answer(question, docs)
```

---

## Comparison of Advanced RAG Techniques

| Technique | Problem Solved | Complexity | Cost |
|---|---|---|---|
| HyDE | Query-document vocabulary mismatch | Low | Extra LLM call |
| Multi-query | Single phrasing missing results | Low | Extra LLM calls |
| Contextual compression | Noisy retrieved content | Medium | Extra LLM calls |
| Re-ranking | Coarse embedding similarity | Medium | ML model inference |
| Hybrid search | Keyword vs semantic trade-off | Medium | BM25 index |
| Self-RAG | Unnecessary retrieval, irrelevant docs | High | Multiple LLM calls |
| CRAG | Vector store knowledge gaps | High | Web search API |

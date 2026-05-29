# Document Loaders & Text Splitters

Before you can use documents in RAG or other LangChain pipelines, you need to **load** them and **split** them into manageable chunks.

---

## Document Object

All loaders return `Document` objects:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the text of the document.",
    metadata={
        "source": "my_file.pdf",
        "page": 1,
        "author": "Alice",
    }
)

print(doc.page_content)  # the text
print(doc.metadata)      # arbitrary metadata dict
```

---

## Document Loaders

### PyPDFLoader
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("annual_report.pdf")
docs = loader.load()

print(len(docs))          # one Document per page
print(docs[0].metadata)   # {"source": "annual_report.pdf", "page": 0}
```

### WebBaseLoader
```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/tutorial/"],
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("section",))},
)
docs = loader.load()
```

### TextLoader
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt", encoding="utf-8")
docs = loader.load()
```

### CSVLoader
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="products.csv",
    source_column="product_name",  # which column becomes source in metadata
)
docs = loader.load()
# Each row becomes a Document
```

### DirectoryLoader — Load All Files in a Folder
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./documents/",
    glob="**/*.pdf",   # only PDFs
    loader_cls=PyPDFLoader,
)
docs = loader.load()
```

### Lazy Loading — Memory Efficient
```python
# Don't load all at once; yield one doc at a time
for doc in loader.lazy_load():
    process(doc)
```

---

## Text Splitters

LLMs have context window limits. Splitting ensures chunks fit in the window and retrieval is precise.

### `RecursiveCharacterTextSplitter` — Recommended Default

Tries to split on semantic boundaries in order: `\n\n` → `\n` → `.` → ` ` → characters.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max characters per chunk
    chunk_overlap=200,    # characters shared between adjacent chunks
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

chunks = splitter.split_documents(docs)
print(len(chunks))         # number of chunks
print(chunks[0].page_content)
print(chunks[0].metadata)  # inherits from parent Document
```

**Why `chunk_overlap`?** Context at chunk boundaries is duplicated so that an answer spanning two chunks is captured by at least one chunk fully.

### `CharacterTextSplitter` — Split on a Single Character
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=100,
)
chunks = splitter.split_text(raw_text)  # or split_documents(docs)
```

### `TokenTextSplitter` — Split by Token Count (More Accurate)
```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,    # tokens, not characters
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
```

### `MarkdownHeaderTextSplitter` — Preserve Markdown Structure
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_splits = splitter.split_text(markdown_document)
# Each chunk's metadata includes the header hierarchy it belongs to
```

---

## Choosing Chunk Size

| Use Case | chunk_size | chunk_overlap |
|---|---|---|
| Dense technical docs (code) | 500–800 | 50–100 |
| General documents | 1000–1500 | 150–200 |
| Long-form narrative | 2000+ | 200–400 |

**Rule of thumb**: Chunk size should be small enough for precise retrieval but large enough to contain complete ideas. Experiment and evaluate!

---

## Full Pipeline: Load → Split → Embed → Store

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load
loader = PyPDFLoader("research_paper.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
)

print(f"Indexed {len(chunks)} chunks")
```

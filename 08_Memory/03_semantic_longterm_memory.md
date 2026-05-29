# Semantic (Long-term) Memory with Vector Stores

While checkpointers store conversation history, **semantic memory** stores arbitrary facts and allows **similarity-based retrieval** — finding relevant memories based on meaning, not just recency.

---

## Why Semantic Memory?

Conversation history grows large and most of it isn't relevant to the current question. Semantic memory lets you ask:

> "What does the user prefer for breakfast?"

And retrieve only the relevant memory — even if it was mentioned 50 conversations ago.

---

## Building a Semantic Memory Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Persistent semantic memory
memory_store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./semantic_memory",
)

def save_memory(content: str, user_id: str, memory_type: str = "general"):
    """Save a memory to the vector store."""
    doc = Document(
        page_content=content,
        metadata={
            "user_id": user_id,
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
        }
    )
    memory_store.add_documents([doc])
    print(f"Memory saved: '{content}'")

def recall_memory(query: str, user_id: str, k: int = 3) -> str:
    """Retrieve relevant memories for a user."""
    results = memory_store.similarity_search(
        query,
        k=k,
        filter={"user_id": user_id},  # only this user's memories
    )
    if not results:
        return "No relevant memories found."
    return "\n".join(f"- {doc.page_content}" for doc in results)

# Save memories
save_memory("User's name is Alice", "user_42", memory_type="identity")
save_memory("Alice prefers Python over Java for data science work", "user_42", memory_type="preference")
save_memory("Alice's favorite color is blue", "user_42", memory_type="preference")
save_memory("Alice lives in Berlin, Germany", "user_42", memory_type="location")
save_memory("Alice is a data scientist at TechCorp", "user_42", memory_type="work")

# Recall relevant memories
memories = recall_memory("What programming language does Alice like?", "user_42")
print(memories)
# - Alice prefers Python over Java for data science work
```

---

## Integrating Semantic Memory into an Agent

```python
from langchain_core.tools import tool

@tool
def save_user_memory(content: str, category: str = "general") -> str:
    """
    Save important information about the user to long-term memory.
    Use this when the user shares personal preferences, facts about themselves,
    or any information that should be remembered for future conversations.
    
    Args:
        content: The information to remember (e.g., "User's name is Alice")
        category: Category of memory (preference, identity, work, location)
    """
    save_memory(content, user_id=current_user_id, memory_type=category)
    return f"Memory saved: {content}"

@tool
def recall_user_memory(query: str) -> str:
    """
    Search long-term memory for information about the user.
    Use this to recall past preferences, facts, or context shared by the user.
    
    Args:
        query: What you're trying to remember about the user
    """
    return recall_memory(query, user_id=current_user_id)

# Agent with memory tools
agent = create_react_agent(
    llm,
    [save_user_memory, recall_user_memory, web_search],
)
```

---

## Memory-Aware System Prompt

Before each conversation, inject relevant memories into the system prompt:

```python
def build_system_prompt(user_id: str, current_question: str) -> str:
    # Retrieve memories relevant to the current question
    relevant_memories = recall_memory(current_question, user_id, k=5)
    
    return f"""You are a helpful personal assistant for the user.

IMPORTANT — What you remember about this user:
{relevant_memories}

Use these memories to personalize your responses. If a memory contradicts
what the user says, ask for clarification — their memory may have updated information.
"""

# Build the prompt before each session
system_prompt = build_system_prompt("user_42", "What coding language should I use for my ML project?")
# System prompt will include: "Alice prefers Python for data science work"
```

---

## Automatic Memory Extraction

Use an LLM to automatically extract memorable facts from conversations:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Optional

class MemoryExtraction(BaseModel):
    should_save: bool
    memory: Optional[str]
    category: Optional[str]  # identity, preference, work, location, other

memory_extractor = ChatPromptTemplate.from_template("""
Analyze this user message and determine if it contains information worth remembering long-term.

User message: {message}

Extract information that is:
- Personal facts (name, location, age, job)
- Explicit preferences (likes/dislikes)
- Important context for future conversations

Do NOT save: questions, greetings, temporary state

Respond with JSON: {{"should_save": bool, "memory": "fact to remember", "category": "category"}}
""") | llm | JsonOutputParser()

def process_and_maybe_save(user_message: str, user_id: str):
    extraction = memory_extractor.invoke({"message": user_message})
    if extraction.get("should_save") and extraction.get("memory"):
        save_memory(extraction["memory"], user_id, extraction.get("category", "general"))
```

---

## Memory Scoring & Relevance Threshold

```python
def recall_with_threshold(query: str, user_id: str, threshold: float = 0.7, k: int = 5) -> list:
    """Only return memories above a similarity threshold."""
    results = memory_store.similarity_search_with_score(
        query,
        k=k,
        filter={"user_id": user_id},
    )
    # Score is cosine distance (lower = more similar for some implementations)
    return [
        doc for doc, score in results
        if score >= threshold  # adjust based on your vector store's scoring
    ]
```

---

## Memory Architecture Summary

```
Incoming Message
      ↓
[Extract facts?] → YES → [Save to Semantic Memory (Chroma/FAISS)]
      ↓
[Recall relevant memories] ← query: current user message
      ↓
[Build context-rich prompt: recalled memories + conversation history]
      ↓
[LLM Response]
      ↓
[Save response facts?] → YES → [Update Semantic Memory]
```

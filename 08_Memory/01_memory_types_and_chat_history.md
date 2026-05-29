# Memory in LLM Applications

---

## Why Memory?

LLMs are **stateless** — every API call is completely independent. Without memory, the model has no idea what was said in the previous message.

```python
# Problem: No memory
llm.invoke("My name is Alice")    # "Nice to meet you, Alice!"
llm.invoke("What is my name?")    # "I don't know your name." ← forgot!
```

**Memory** adds continuity by storing and retrieving past interactions.

---

## Types of Memory

### 1. Short-term Memory (In-Context)
The full conversation history is kept in the active context window. Simple but limited by context size.

### 2. Summary Memory (Compressed Short-term)
An LLM periodically summarizes old conversation turns to compress the history. Stays within context limits.

### 3. Entity Memory (Structured Facts)
Extracts and stores facts about entities (people, places, products) mentioned in conversation.

### 4. Long-term Memory (External Store)
Persists important information beyond the current session in a database or vector store. Survives session restarts.

---

## Short-term Memory — Chat History

### Manual Approach
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o")

# Build and maintain history manually
history = [SystemMessage("You are a helpful assistant.")]

def chat(user_input: str) -> str:
    history.append(HumanMessage(user_input))
    response = llm.invoke(history)
    history.append(response)  # save AIMessage to history
    return response.content

print(chat("My name is Alice."))   # "Nice to meet you, Alice!"
print(chat("What is my name?"))    # "Your name is Alice."
```

---

## `ChatMessageHistory` — In-Memory Store

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("My name is Bob")
history.add_ai_message("Nice to meet you, Bob!")
history.add_user_message("What's my name?")

print(history.messages)
# [HumanMessage, AIMessage, HumanMessage]
```

---

## `RunnableWithMessageHistory` — Automatic History Management

Wraps any LCEL chain to automatically:
1. Load history at the start of each call
2. Inject it into the prompt
3. Save the new messages back to the store

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),   # ← history injected here
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Session store (in-memory — use Redis/DB for production)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with history
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session config identifies whose history to use
config = {"configurable": {"session_id": "alice_session"}}

# Turn 1
r1 = chain_with_memory.invoke({"input": "My name is Alice"}, config=config)
print(r1)  # "Nice to meet you, Alice!"

# Turn 2 (history automatically loaded and injected)
r2 = chain_with_memory.invoke({"input": "What's my name?"}, config=config)
print(r2)  # "Your name is Alice."

# Different session — no memory of Alice
config2 = {"configurable": {"session_id": "bob_session"}}
r3 = chain_with_memory.invoke({"input": "What's my name?"}, config=config2)
print(r3)  # "I don't know your name yet — could you tell me?"
```

---

## Persistent Storage — Redis / SQL

For production, replace in-memory store with persistent storage:

```python
# Redis (for scalable, fast production use)
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=3600,  # expire after 1 hour of inactivity
    )

# PostgreSQL
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string="postgresql://user:pass@localhost/mydb",
    )
```

---

## Conversation Trimming — Managing Context Window

As conversations grow long, trim older messages to stay within context limits:

```python
from langchain_core.messages import trim_messages

def trim_and_invoke(messages, question):
    trimmed = trim_messages(
        messages,
        max_tokens=4000,        # keep within 4000 tokens
        strategy="last",        # keep most recent messages
        token_counter=llm,      # use model's tokenizer
        include_system=True,    # always keep system message
        allow_partial=False,    # don't cut mid-message
    )
    return llm.invoke(trimmed + [HumanMessage(question)])
```

---

## Summary Memory

```python
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Progressively summarize the conversation. Add new information to the existing summary."),
    ("human", "Current summary:\n{summary}\n\nNew messages:\n{new_messages}\n\nNew summary:"),
])

summary_chain = summary_prompt | llm | StrOutputParser()

def update_summary(current_summary: str, new_messages: list) -> str:
    return summary_chain.invoke({
        "summary": current_summary,
        "new_messages": "\n".join(str(m) for m in new_messages),
    })

# In your chat loop:
# When history grows too long, replace old messages with a summary
if len(history) > 20:
    summary = update_summary(current_summary, history[-10:])
    history = [SystemMessage(f"Conversation summary: {summary}")] + history[-5:]
```

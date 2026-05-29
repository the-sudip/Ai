# Chat vs Completion Models

---

## The Two API Styles

### Completion (Legacy)
Takes a **single string** and returns a **string continuation**. The model simply predicts what comes next.

```python
# Legacy style (text-davinci-003, now deprecated)
response = openai.completions.create(
    model="text-davinci-003",
    prompt="The capital of France is",
    max_tokens=5,
)
print(response.choices[0].text)  # " Paris."
```

**Problem**: No clean separation between system instructions and user input. Developers had to use hacky prompt formatting.

---

### Chat (Modern — Use This)
Takes a **structured list of messages** with explicit roles. Returns a message object.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a friendly Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print(type(response))      # AIMessage
print(response.content)    # "A list comprehension is..."
```

---

## Message Roles

| Role | Class | Purpose |
|---|---|---|
| `system` | `SystemMessage` | Developer instructions, persona, rules. Persists across the conversation. |
| `user` / `human` | `HumanMessage` | What the user says or sends. |
| `assistant` / `ai` | `AIMessage` | Previous model responses included for context. |
| `tool` | `ToolMessage` | Result of a tool/function call, matched by `tool_call_id`. |

---

## Multi-Turn Conversation

To maintain conversation history, manually append messages:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o")

# Build conversation history
history = [
    SystemMessage(content="You are a helpful assistant. Be concise."),
]

# Turn 1
user_msg = HumanMessage(content="My name is Alice.")
history.append(user_msg)
response = llm.invoke(history)
history.append(response)  # append AIMessage

# Turn 2
user_msg2 = HumanMessage(content="What is my name?")
history.append(user_msg2)
response2 = llm.invoke(history)

print(response2.content)  # "Your name is Alice."
```

---

## AIMessage Structure

```python
response = llm.invoke([HumanMessage("What is 2+2?")])

print(response.content)          # "2 + 2 = 4."
print(response.response_metadata) # model info, finish reason
print(response.usage_metadata)    # token usage
print(response.id)                # unique message ID

# If the model called a tool:
print(response.tool_calls)
# [{"name": "calculator", "args": {"expression": "2+2"}, "id": "call_abc"}]
```

---

## Async Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

async def main():
    response = await llm.ainvoke([HumanMessage("Hello!")])
    print(response.content)

asyncio.run(main())
```

---

## Streaming

```python
for chunk in llm.stream([HumanMessage("Tell me a joke about Python.")]):
    print(chunk.content, end="", flush=True)
# Prints tokens as they are generated, not all at once
```

---

## Key Difference Summary

| | Completion | Chat |
|---|---|---|
| Input | Single string | List of role-tagged messages |
| Output | String | `AIMessage` object |
| History | Manual prompt engineering | Structured message list |
| Tool calls | Not supported | Native support via `tool_calls` |
| Status | Deprecated/legacy | Current standard |

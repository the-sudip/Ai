# AI Interview Study Notes
> Covers: LLMs · Prompt Engineering · LangChain · LangGraph · Agents · Agentic AI · Tools · Memory · RAG · MCP

---

## Table of Contents
1. [Large Language Models (LLMs)](#1-large-language-models-llms)
2. [Prompt Engineering](#2-prompt-engineering)
3. [LangChain](#3-langchain)
4. [Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
5. [Agents](#5-agents)
6. [Agentic AI](#6-agentic-ai)
7. [Tools](#7-tools)
8. [Memory](#8-memory)
9. [LangGraph](#9-langgraph)
10. [MCP – Model Context Protocol](#10-mcp--model-context-protocol)

---

## 1. Large Language Models (LLMs)

### 1.1 What is an LLM?
A **Large Language Model** is a deep neural network (usually Transformer-based) trained on massive text corpora to predict the next token. It learns statistical patterns of language and can generate, summarize, translate, and reason about text.

Key families: GPT-4/4o (OpenAI), Claude (Anthropic), Gemini (Google), LLaMA (Meta), Mistral.

### 1.2 How LLMs Work — Core Concepts

| Concept | Explanation |
|---|---|
| **Token** | Smallest unit of text (~0.75 words on average). "Hello world" ≈ 2 tokens. |
| **Context Window** | Max tokens the model can see at once (e.g., GPT-4o: 128k). |
| **Temperature** | Controls randomness. 0 = deterministic, 1+ = creative/random. |
| **Top-p (nucleus sampling)** | Only sample from top-p probability mass. |
| **Max tokens** | Limit on response length. |
| **System prompt** | Instructions set by the developer before the conversation begins. |

### 1.3 Transformer Architecture (Brief)
- **Input Embedding** → **Positional Encoding** → **Multi-head Self-Attention** → **Feed-Forward** layers → **Output logits**
- Self-attention lets every token attend to every other token in the context.
- LLMs are auto-regressive: they generate one token at a time, appending it to the context.

### 1.4 Inference Parameters

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,      # low = focused answers
    max_tokens=1024,      # cap response length
    top_p=0.9,
)

response = llm.invoke("Explain transformers in 2 sentences.")
print(response.content)
```

### 1.5 Chat vs Completion Models
- **Completion**: takes a raw string, returns a string (legacy).
- **Chat**: takes a list of messages (`system`, `human`, `ai`), returns a message object. All modern models use chat format.

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="What is backpropagation?"),
]
response = llm.invoke(messages)
```

### 1.6 Fine-tuning vs RAG vs Prompting
| Approach | When to Use |
|---|---|
| **Prompting** | Fast, no training. Best for general tasks. |
| **RAG** | Need up-to-date or private knowledge without retraining. |
| **Fine-tuning** | Need specific style/tone/domain expertise baked in permanently. |

---

## 2. Prompt Engineering

### 2.1 Prompt Components
```
[System Prompt]  → Role, rules, constraints
[Few-shot Examples] → (optional) input-output demonstrations
[Context] → Retrieved docs, user data
[User Query] → The actual question
```

### 2.2 Techniques

#### Zero-shot
```python
prompt = "Translate 'Good morning' to French."
```

#### Few-shot
```python
prompt = """
Translate to French:
English: Hello → French: Bonjour
English: Thank you → French: Merci
English: Good night → French: ?
"""
```

#### Chain-of-Thought (CoT)
```python
prompt = """
Solve step by step:
A store has 50 apples. It sells 20 on Monday and 15 on Tuesday.
How many remain?

Step 1: Start with 50.
Step 2: Subtract Monday sales: 50 - 20 = 30.
Step 3: Subtract Tuesday sales: 30 - 15 = 15.
Answer: 15 apples remain.

Now solve: A jar has 100 cookies...
"""
```

#### ReAct (Reason + Act) — used in agents
```
Thought: I need to find the current weather in London.
Action: search("current weather London")
Observation: It is 18°C and cloudy.
Thought: I now have the answer.
Final Answer: It is 18°C and cloudy in London.
```

### 2.3 PromptTemplate in LangChain

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
])

prompt = template.invoke({"domain": "machine learning", "question": "What is overfitting?"})
```

### 2.4 Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# String output
chain = template | llm | StrOutputParser()

# Structured JSON output
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary")

parser = JsonOutputParser(pydantic_object=MovieReview)
chain = template | llm | parser
```

---

## 3. LangChain

### 3.1 What is LangChain?
LangChain is a framework for building applications powered by LLMs. It provides:
- Abstractions for LLMs, Chat Models, Embeddings
- Prompt templates
- Chains (sequence of steps)
- Agents and Tools
- Memory integrations
- Retrieval components

### 3.2 LCEL — LangChain Expression Language
LCEL uses the `|` pipe operator to compose components into chains. Every component is a **Runnable**.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain: prompt → llm → parser
chain = prompt | llm | parser

result = chain.invoke({"topic": "Python programming"})
print(result)
```

### 3.3 Runnable Interface
Every LangChain component implements:
- `.invoke(input)` — single call
- `.stream(input)` — streaming generator
- `.batch([inputs])` — parallel calls
- `.ainvoke(input)` — async single call

```python
# Streaming
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([{"topic": "Python"}, {"topic": "Rust"}])
```

### 3.4 RunnableParallel — Run branches in parallel

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "pros": pros_chain,
    "cons": cons_chain,
})

result = parallel.invoke({"topic": "electric vehicles"})
# result = {"pros": "...", "cons": "..."}
```

### 3.5 RunnablePassthrough — Pass input unchanged

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough() | llm | StrOutputParser()
# Useful in RAG pipelines to pass the original question through
```

### 3.6 Document Loaders

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Web
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
```

### 3.7 Text Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # characters per chunk
    chunk_overlap=200,    # overlap between chunks
)
chunks = splitter.split_documents(docs)
```

---

## 4. Retrieval-Augmented Generation (RAG)

### 4.1 What is RAG?
RAG combines **retrieval** (fetching relevant documents from a store) with **generation** (LLM producing an answer). Solves the knowledge cutoff and hallucination problems.

**Pipeline:**
```
User Query → Embed Query → Vector Search → Top-K Docs → LLM + Docs → Answer
```

### 4.2 Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("What is RAG?")
# Returns a list of ~1536 floats
```

### 4.3 Vector Store

```python
from langchain_chroma import Chroma

# Build vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# Similarity search
results = vectorstore.similarity_search("What is RAG?", k=3)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 4.4 Basic RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return policy?")
```

### 4.5 Advanced RAG Patterns

| Pattern | Description |
|---|---|
| **Hybrid Search** | Combine keyword (BM25) + semantic (vector) search |
| **Re-ranking** | Use a cross-encoder to rerank retrieved docs |
| **HyDE** | Generate a hypothetical answer, embed it, then search |
| **Multi-query** | Generate multiple query variations, retrieve for each |
| **Self-RAG** | Model decides when to retrieve and validates retrieved content |
| **Contextual Compression** | Strip irrelevant content from retrieved docs before passing to LLM |

---

## 5. Agents

### 5.1 What is an Agent?
An **agent** is an LLM that can:
1. **Reason** about a goal
2. **Decide** which tool to use
3. **Execute** the tool
4. **Observe** the result
5. **Repeat** until the goal is achieved

### 5.2 Agent Components
```
Agent = LLM + Tools + Agent Loop (Reasoning Engine)
```

- **LLM**: The brain (decides what to do next)
- **Tools**: External capabilities (search, calculator, database, API)
- **Agent Executor**: Runs the loop until done

### 5.3 ReAct Agent Pattern

```
Thought → Action → Observation → Thought → Action → Observation → Final Answer
```

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# Get the ReAct prompt
prompt = hub.pull("hwchase17/react")

tools = [DuckDuckGoSearchRun()]
agent = create_react_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

result = executor.invoke({"input": "Who won the 2024 US presidential election?"})
```

### 5.4 Tool-Calling Agent (Modern Approach)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Modern agents use native tool-calling (function calling)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "Search for latest AI news"})
```

### 5.5 Agent Loop Flow

```
User Input
    ↓
LLM decides: "I need to call search()"
    ↓
Tool is executed → result returned
    ↓
LLM sees result, decides: "I now have the answer"
    ↓
LLM returns Final Answer
```

### 5.6 Agent Stopping Conditions
- `max_iterations` — prevents infinite loops
- `max_execution_time` — time-based limit
- LLM outputs "Final Answer:" — signals completion
- Early stopping on error

---

## 6. Agentic AI

### 6.1 What is Agentic AI?
Agentic AI refers to systems where AI acts **autonomously** over extended tasks with minimal human intervention. Goes beyond single Q&A — the agent **plans**, **executes multi-step tasks**, **uses tools**, **manages state**, and **adapts**.

### 6.2 Properties of Agentic Systems
| Property | Description |
|---|---|
| **Autonomy** | Operates without constant human guidance |
| **Goal-directedness** | Works toward a defined objective |
| **Reactivity** | Responds to environment changes |
| **Tool use** | Extends capabilities via external tools |
| **Memory** | Maintains context across steps |
| **Multi-step planning** | Breaks complex tasks into subtasks |

### 6.3 Agentic Patterns

#### Planning Pattern
```python
# Agent first creates a plan, then executes each step
plan_prompt = "Break down this task into steps: {task}"
execute_prompt = "Execute this step: {step}. Previous results: {history}"
```

#### Reflection Pattern
```python
# Agent reviews its own output and improves it
draft = agent.invoke({"task": "Write a report"})
reflection = critic_llm.invoke(f"Critique this: {draft}")
final = agent.invoke({"task": "Improve based on critique", "critique": reflection})
```

#### Multi-Agent Pattern
```
Orchestrator Agent
    ├── Research Agent (searches the web)
    ├── Analysis Agent (processes data)
    └── Writer Agent (produces final output)
```

### 6.4 Human-in-the-Loop (HITL)
Some agentic workflows pause and ask for human approval before proceeding with irreversible actions.

```python
# In LangGraph, use interrupt_before to pause at a node
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"],  # pause before this node
)
```

---

## 7. Tools

### 7.1 What are Tools?
Tools are functions the LLM can call to interact with the external world — APIs, databases, code interpreters, file systems, etc.

### 7.2 Defining a Tool with @tool decorator

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # In real code, call a weather API
    return f"The weather in {city} is 22°C and sunny."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

### 7.3 Tool with Pydantic Schema (Recommended)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum number of results")

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    # API call here
    return f"Results for '{query}': ..."

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the internet for current information",
    args_schema=SearchInput,
)
```

### 7.4 Binding Tools to an LLM

```python
from langchain_openai import ChatOpenAI

tools = [get_weather, calculator, search_tool]

# Bind tools to LLM (enables function/tool calling)
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("What is 25 * 4 + 10?")
# LLM returns a tool_call, not a text answer
print(response.tool_calls)
# [{"name": "calculator", "args": {"expression": "25 * 4 + 10"}, "id": "..."}]
```

### 7.5 Tool Execution

```python
from langchain_core.messages import ToolMessage

# Execute tool calls from LLM response
tool_map = {t.name: t for t in tools}

for tool_call in response.tool_calls:
    tool_fn = tool_map[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )
```

### 7.6 Built-in Tools

```python
from langchain_community.tools import (
    DuckDuckGoSearchRun,    # Web search
    WikipediaQueryRun,      # Wikipedia
    PythonREPLTool,         # Run Python code
    ShellTool,              # Run shell commands
)
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()
python_repl = PythonREPLTool()
```

---

## 8. Memory

### 8.1 Why Memory?
LLMs are **stateless** — each call is independent. Memory gives agents the ability to remember past interactions, facts, and context.

### 8.2 Types of Memory

| Type | What it stores | Scope |
|---|---|---|
| **Conversation Buffer** | Full chat history | Short-term |
| **Conversation Summary** | LLM-summarized history | Short-term (compressed) |
| **Entity Memory** | Facts about entities (people, places) | Short-term |
| **Vector Store Memory** | Semantically searchable past messages | Long-term |
| **External DB** | Structured data (SQL, Redis) | Long-term |

### 8.3 In-Memory Chat History (LangChain)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-memory store
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session 1
chain_with_memory.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "session_1"}},
)

# Session 1 continued (remembers Alice)
chain_with_memory.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "session_1"}},
)
```

### 8.4 Memory in LangGraph
LangGraph uses a **State** object that persists across nodes. A **Checkpointer** enables persistence across sessions.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # In-memory (use SqliteSaver for persistent)

graph = graph_builder.compile(checkpointer=checkpointer)

# Thread ID scopes the memory
config = {"configurable": {"thread_id": "user_123"}}

graph.invoke({"messages": [HumanMessage("Hi, I'm Bob")]}, config=config)
graph.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)
# Returns: "Your name is Bob."
```

### 8.5 Persistent Memory (SQLite)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
    # All states are persisted to SQLite
```

### 8.6 Semantic (Long-term) Memory with Vector Store

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

memory_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./agent_memory",
)

# Save a memory
memory_store.add_texts(["User prefers Python over Java", "User lives in Berlin"])

# Recall relevant memory
relevant = memory_store.similarity_search("What language does the user like?", k=1)
# Returns: "User prefers Python over Java"
```

---

## 9. LangGraph

### 9.1 What is LangGraph?
LangGraph is a library built on top of LangChain for building **stateful, multi-actor applications** using graphs. It models agent workflows as a directed graph where:
- **Nodes** = functions / agents / LLM calls
- **Edges** = transitions between nodes
- **State** = shared data passed between nodes

### 9.2 Core Concepts

| Concept | Description |
|---|---|
| **StateGraph** | The graph object you build |
| **State** | TypedDict or Pydantic model holding shared data |
| **Node** | A Python function `(state) → state` |
| **Edge** | Static connection from node A → node B |
| **Conditional Edge** | Dynamic routing based on state |
| **START / END** | Special graph entry/exit points |
| **Checkpointer** | Saves/restores state for persistence & HITL |

### 9.3 Minimal LangGraph Agent

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages = append, not overwrite

# 2. Create LLM
llm = ChatOpenAI(model="gpt-4o")

# 3. Define Node
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 5. Run
result = graph.invoke({"messages": [HumanMessage("Hello!")]})
print(result["messages"][-1].content)
```

### 9.4 Conditional Edges (Routing)

```python
def route(state: State) -> str:
    """Return the next node name based on state."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # go to tool node
    return END           # done

builder.add_conditional_edges(
    "chatbot",           # from this node
    route,               # function that returns next node name
    {
        "tools": "tool_node",   # mapping: return value → node name
        END: END,
    }
)
```

### 9.5 Tool Node in LangGraph

```python
from langgraph.prebuilt import ToolNode

tools = [get_weather, calculator]
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)  # auto-executes tool calls and returns ToolMessages

builder.add_node("tools", tool_node)
builder.add_edge("tools", "chatbot")  # after tools, go back to chatbot
```

### 9.6 Full ReAct Agent in LangGraph

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

tools = [search]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # tools_condition is built-in
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### 9.7 LangGraph Subgraphs (Multi-Agent)

```python
# Each agent is its own graph, composed into a parent graph
research_graph = build_research_graph()
writer_graph = build_writer_graph()

parent = StateGraph(ParentState)
parent.add_node("research", research_graph)
parent.add_node("write", writer_graph)
parent.add_edge(START, "research")
parent.add_edge("research", "write")
parent.add_edge("write", END)
```

### 9.8 LangGraph vs LangChain Chains

| | LangChain LCEL | LangGraph |
|---|---|---|
| Structure | Linear/parallel DAG | Cyclic graph |
| State | Passed through pipes | Persistent TypedDict |
| Loops | Not supported | Native |
| Human-in-loop | Not built-in | Built-in with checkpointers |
| Use case | Simple chains, RAG | Agents, multi-step workflows |

---

## 10. MCP — Model Context Protocol

### 10.1 What is MCP?
**Model Context Protocol (MCP)** is an open standard (by Anthropic) that defines how AI models communicate with external **tools, resources, and prompts** in a standardized way. Think of it as USB-C for AI integrations — one protocol, many connectors.

### 10.2 Architecture

```
┌─────────────────────────────────────────┐
│              MCP Host                    │
│  (Claude Desktop, VS Code, your app)     │
│                                          │
│  ┌──────────────┐                        │
│  │  MCP Client  │◄──── JSON-RPC 2.0 ────►│ MCP Server
│  └──────────────┘                        │  (exposes tools/resources)
└─────────────────────────────────────────┘
```

- **MCP Host**: The application using AI (e.g., Claude Desktop, Cursor, your app)
- **MCP Client**: Embedded in the host; manages server connections
- **MCP Server**: A lightweight process that exposes capabilities

### 10.3 MCP Primitives

| Primitive | Direction | Description |
|---|---|---|
| **Tools** | Server → Client | Functions the LLM can call (like function calling) |
| **Resources** | Server → Client | Data/files the LLM can read (URIs) |
| **Prompts** | Server → Client | Reusable prompt templates |
| **Sampling** | Client → Server | Server requests LLM completions |
| **Roots** | Client → Server | File system roots the server can access |

### 10.4 Creating an MCP Server (Python)

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Tool Server")

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # In real code: call financial API
    prices = {"AAPL": 189.50, "GOOGL": 175.30}
    return f"{ticker}: ${prices.get(ticker, 'Unknown')}"

@mcp.resource("file:///{path}")
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    with open(path) as f:
        return f.read()

@mcp.prompt()
def summarize_prompt(text: str) -> str:
    """Generate a summarization prompt."""
    return f"Please summarize the following text concisely:\n\n{text}"

if __name__ == "__main__":
    mcp.run()  # starts stdio server by default
```

### 10.5 MCP Transport Protocols

| Transport | Description | Use Case |
|---|---|---|
| **stdio** | stdin/stdout | Local tools, CLI integrations |
| **SSE (HTTP)** | Server-Sent Events over HTTP | Remote servers, web apps |
| **WebSocket** | Bidirectional streaming | Real-time apps |

### 10.6 Connecting MCP to LangChain

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def main():
    async with MultiServerMCPClient(
        {
            "my_server": {
                "command": "python",
                "args": ["server.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = await client.get_tools()
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke({"messages": [HumanMessage("Get AAPL stock price")]})
```

### 10.7 MCP vs Traditional Tool Calling

| | Traditional Tool Calling | MCP |
|---|---|---|
| **Standard** | Model-specific (OpenAI format) | Universal open standard |
| **Discovery** | Hardcoded in app | Dynamic at runtime |
| **Reuse** | One integration per app | One server, many clients |
| **Resources** | Not supported | Native (URIs, files) |

---

## Quick Reference Cheat Sheet

### LangChain Core Imports
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### LangGraph Core Imports
```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
```

### Key Patterns
```python
# LCEL Chain
chain = prompt | llm | parser

# RAG Chain
rag = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser

# Agent with tools
llm_with_tools = llm.bind_tools(tools)

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

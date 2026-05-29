# MCP Quick Reference & Interview Cheat Sheet

---

## Core Concepts at a Glance

| Concept | One-Line Definition |
|---|---|
| **MCP** | Open standard (Anthropic, 2024) for AI↔tool connectivity — "USB-C for AI" |
| **Host** | The app using AI (Claude Desktop, VS Code, your LangChain app) |
| **Client** | Embedded in the host; manages connections to MCP servers |
| **Server** | Process exposing tools/resources/prompts via JSON-RPC 2.0 |
| **Tool** | Function the LLM can call (with side effects) |
| **Resource** | Read-only data with a URI (like a GET endpoint) |
| **Prompt** | Reusable parameterized prompt template |
| **Sampling** | Server requests an LLM completion from the host |
| **Transport** | How client↔server communicate: stdio / HTTP+SSE / WebSocket |

---

## The Three Primitives

```
TOOLS      — do things     → @mcp.tool()
RESOURCES  — read things   → @mcp.resource("uri://{param}")
PROMPTS    — template things → @mcp.prompt()
```

---

## Minimal MCP Server

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.resource("data://greeting")
def greeting() -> str:
    """A friendly greeting."""
    return "Hello from MCP!"

@mcp.prompt()
def explain(topic: str) -> str:
    """Generate an explanation prompt."""
    return f"Explain {topic} in simple terms for a beginner."

if __name__ == "__main__":
    mcp.run()  # stdio by default
```

---

## Minimal LangChain MCP Client

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio

async def main():
    async with MultiServerMCPClient({
        "myserver": {
            "command": "python", "args": ["server.py"], "transport": "stdio"
        }
    }) as client:
        tools = await client.get_tools()
        agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)
        result = await agent.ainvoke({"messages": [HumanMessage("What is 3 + 5?")]})
        print(result["messages"][-1].content)

asyncio.run(main())
```

---

## Transport Quick Reference

| Transport | Use Case | Security |
|---|---|---|
| `stdio` | Local subprocesses | OS-level (safe) |
| `sse` (HTTP) | Remote web servers | HTTPS + API key |
| `websocket` | Real-time bidirectional | WSS + token auth |

---

## MCP vs Traditional Tools

| | `@tool` (LangChain) | MCP |
|---|---|---|
| Standard | LangChain-specific | Open standard |
| Reuse | One app | Any MCP client |
| Discovery | Hardcoded | Dynamic |
| Transport | In-process | Network / subprocess |
| Resources | No | Yes (URI) |

---

## Top 10 MCP Interview Questions

**Q1: What is MCP and what problem does it solve?**
MCP (Model Context Protocol) is an open standard by Anthropic for connecting AI models to external tools and data sources. It solves the M×N integration problem — without MCP, each AI app needs custom integrations for each tool. With MCP, you build once and any MCP-compatible client can use it.

**Q2: What are the three MCP primitives?**
1. **Tools** — functions the LLM can call (with side effects, like POST)
2. **Resources** — read-only data with URIs (like GET)
3. **Prompts** — reusable parameterized prompt templates

**Q3: What protocol does MCP use?**
JSON-RPC 2.0 — a lightweight remote procedure call protocol over various transports (stdio, HTTP, WebSocket).

**Q4: How do you create an MCP server in Python?**
Using FastMCP: `mcp = FastMCP("name")`, decorate functions with `@mcp.tool()`, `@mcp.resource("uri://{param}")`, `@mcp.prompt()`, then call `mcp.run()`.

**Q5: How does MCP integrate with LangChain?**
Via `langchain-mcp-adapters` package. Use `MultiServerMCPClient` as an async context manager, call `await client.get_tools()` to get standard `BaseTool` instances, then use them with any LangGraph/LangChain agent.

**Q6: What is the difference between Tools and Resources in MCP?**
Tools can have side effects (send email, run code, write to DB) and take structured arguments. Resources are read-only, addressed by URI, and return data without side effects.

**Q7: What is stdio transport and when do you use it?**
stdio transport runs the MCP server as a subprocess and communicates via stdin/stdout. Use it for local tools, CLI integrations, and development. Most secure (no network exposure).

**Q8: How does the MCP trust model work?**
The user has ultimate trust → host app enforces policies → MCP servers have minimal trust. The LLM cannot directly execute MCP calls — all calls go through the host application which can enforce access controls.

**Q9: What is "sampling" in MCP?**
Sampling is a reverse capability where the MCP **server** requests an LLM completion from the **client/host**. This lets MCP servers use AI without having their own LLM access or API keys.

**Q10: What is `MultiServerMCPClient` and how do you use it?**
It's the main class from `langchain-mcp-adapters` for connecting to multiple MCP servers simultaneously. Used as `async with MultiServerMCPClient({server_configs}) as client:`. Provides `get_tools()` which returns standard LangChain `BaseTool` instances compatible with any LangGraph agent.

---

## Common Pitfalls

1. **Not using async** — MCP client APIs are all `async`. Always `await` them.
2. **Forgetting `async with`** — The client context manager starts/stops server processes.
3. **Hardcoding credentials** — Always use environment variables for API keys.
4. **No input validation** — Validate all tool inputs to prevent injection attacks.
5. **Wrong transport** — Use `stdio` for local, `sse`/`websocket` for remote with HTTPS.

---

## Ecosystem Shortcuts

```bash
# Test your server without code
npx @modelcontextprotocol/inspector python server.py

# Popular pre-built servers
npx -y @modelcontextprotocol/server-filesystem /path/to/dir
npx -y @modelcontextprotocol/server-github
npx -y @modelcontextprotocol/server-postgres postgresql://...
npx -y @modelcontextprotocol/server-brave-search
```

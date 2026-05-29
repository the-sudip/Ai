# MCP vs Traditional Tool Calling

Understanding the difference helps you decide when to use MCP and when to use native LangChain tools.

---

## Side-by-Side Comparison

| Dimension | Traditional LangChain Tools | MCP |
|---|---|---|
| **Standard** | LangChain-specific (`@tool`, `StructuredTool`) | Open standard (any client, any server) |
| **Discovery** | Hardcoded in application code | Dynamic at runtime via `tools/list` |
| **Reusability** | One integration per app | Build once, use in any MCP client |
| **Language** | Python only | Language-agnostic (Python, TypeScript, Go, Rust, etc.) |
| **Resources** | Not a concept | First-class (URIs, file access) |
| **Transport** | In-process function call | stdio / HTTP / WebSocket |
| **Ecosystem** | LangChain tools library | Growing MCP server ecosystem |
| **Complexity** | Simple function + decorator | Client-server protocol setup |
| **Deployment** | Same process as app | Separate server process |

---

## When to Use Traditional LangChain Tools

Use `@tool` / `StructuredTool` when:
- You're building **within a Python application**
- The tool logic is **simple and internal**
- You **don't need to share** the tools across multiple apps
- You want **minimal overhead** (no network, no subprocess)

```python
# Great for internal, simple tools
@tool
def format_currency(amount: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

@tool
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax on an amount."""
    return round(amount * rate / 100, 2)
```

---

## When to Use MCP

Use MCP when:
- You want tools **reusable across multiple AI apps** (Claude Desktop, VS Code, your app)
- Your tools need to be **maintained independently** from the AI application
- You're building a **shared tool service** for a team
- The tool logic involves **non-Python languages or existing services**
- You need **resource access** (files, databases with URI addressing)

```python
# Great as an MCP server: shared company-wide
@mcp.tool()
def get_employee_info(employee_id: str) -> str:
    """Get employee information from HR system."""
    return hr_api.get_employee(employee_id)

@mcp.resource("hr://org-chart")
def get_org_chart() -> str:
    """Get the current organizational chart."""
    return hr_api.get_org_chart()
```

---

## Practical Example — Same Tool, Both Ways

### As LangChain Tool
```python
from langchain_core.tools import tool
import httpx

@tool
def search_products(query: str, max_results: int = 10) -> str:
    """Search the product catalog."""
    response = httpx.get(f"https://api.mystore.com/products?q={query}&limit={max_results}")
    return response.json()

# Available only in this Python app
agent = create_react_agent(llm, [search_products])
```

### As MCP Server
```python
# product_server.py — runs as a separate process
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("Product Catalog")

@mcp.tool()
def search_products(query: str, max_results: int = 10) -> str:
    """Search the product catalog."""
    response = httpx.get(f"https://api.mystore.com/products?q={query}&limit={max_results}")
    return str(response.json())

mcp.run()

# Now available to:
# - Your LangChain app (via MultiServerMCPClient)
# - Claude Desktop
# - VS Code Copilot
# - Any other MCP client
```

---

## The Full Modern Stack

In a sophisticated production system, all three layers work together:

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
│                                                          │
│  LangGraph Agent (orchestration)                         │
│    ├── LangChain @tools (internal, Python logic)         │
│    └── MCP Client → MCP Servers (external services)     │
│          ├── GitHub MCP Server                           │
│          ├── Database MCP Server                         │
│          └── Custom Business Logic MCP Server            │
│                                                          │
│  LLM: GPT-4o / Claude 3.5 (function calling)            │
└─────────────────────────────────────────────────────────┘
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Internal Python tools
@tool
def format_output(data: dict) -> str:
    """Format data for display."""
    return json.dumps(data, indent=2)

async def build_agent():
    async with MultiServerMCPClient({
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
        "database": {"command": "python", "args": ["db_server.py"]},
    }) as mcp_client:
        
        mcp_tools = await mcp_client.get_tools()
        all_tools = mcp_tools + [format_output]  # mix MCP + native tools
        
        agent = create_react_agent(llm, all_tools)
        return agent
```

---

## Summary

| | Use This |
|---|---|
| Simple internal function | `@tool` decorator |
| Complex schema validation | `StructuredTool` with Pydantic |
| Share tools across apps | MCP Server |
| Connect to external services | MCP (pre-built servers) |
| Mix both | `MultiServerMCPClient` + `@tool` together |

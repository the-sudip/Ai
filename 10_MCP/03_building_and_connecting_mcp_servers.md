# Building & Connecting MCP Servers

---

## Creating an MCP Server with FastMCP

**FastMCP** is the recommended high-level Python SDK for building MCP servers quickly.

```bash
pip install mcp fastmcp
```

### Complete MCP Server Example

```python
# server.py
from mcp.server.fastmcp import FastMCP
from typing import Optional
import json
import datetime

# Create the server
mcp = FastMCP(
    name="Business Intelligence Server",
    version="1.0.0",
)

# ── TOOLS ────────────────────────────────────────────────────────────

@mcp.tool()
def get_sales_report(
    start_date: str,
    end_date: str,
    region: Optional[str] = None,
) -> str:
    """
    Generate a sales report for a given date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        region: Optional region filter (e.g., 'APAC', 'EMEA', 'AMER')
    
    Returns:
        Sales summary as formatted string
    """
    # Mock data — real implementation queries a database
    data = {
        "total_sales": 1_250_000,
        "orders": 3_420,
        "region": region or "Global",
        "period": f"{start_date} to {end_date}",
    }
    return json.dumps(data, indent=2)

@mcp.tool()
def create_ticket(
    title: str,
    description: str,
    priority: str = "medium",
) -> str:
    """
    Create a support ticket in the ticketing system.
    
    Args:
        title: Brief title of the issue
        description: Detailed description of the problem
        priority: Ticket priority - 'low', 'medium', 'high', 'critical'
    """
    if priority not in ("low", "medium", "high", "critical"):
        return f"Error: Invalid priority '{priority}'. Must be one of: low, medium, high, critical"
    
    ticket_id = f"TKT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Ticket created: {ticket_id}\nTitle: {title}\nPriority: {priority}"

# ── RESOURCES ────────────────────────────────────────────────────────

@mcp.resource("report://{report_type}/latest")
def get_latest_report(report_type: str) -> str:
    """Get the most recent report of a given type."""
    reports = {
        "sales": "Q1 2025 Sales: $5.2M (+12% YoY)...",
        "inventory": "Current inventory: 45,230 units...",
        "customers": "Active customers: 12,450...",
    }
    return reports.get(report_type, f"No report found for type: {report_type}")

@mcp.resource("config://business-rules")
def get_business_rules() -> str:
    """Get the current business rules configuration."""
    return """
Business Rules:
- Discounts > 20% require manager approval
- Orders > $10,000 require credit check
- International orders add 8% customs fee
- Return window: 30 days from delivery
"""

# ── PROMPTS ──────────────────────────────────────────────────────────

@mcp.prompt()
def analyze_metrics(
    metric_name: str,
    current_value: str,
    target_value: str,
    time_period: str,
) -> str:
    """Generate a prompt for analyzing business metrics."""
    return f"""Analyze the following business metric:

Metric: {metric_name}
Current Value: {current_value}
Target Value: {target_value}
Time Period: {time_period}

Please provide:
1. Current performance vs target (gap analysis)
2. Likely root causes if below target
3. 3 specific, actionable recommendations
4. Risk assessment if the trend continues
"""

# ── ENTRY POINT ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run with stdio transport (default for local tools)
    mcp.run()
    
    # For HTTP/SSE transport:
    # mcp.run(transport="sse", host="0.0.0.0", port=8080)
```

---

## Running the Server

```bash
# Run directly (stdio transport)
python server.py

# Test with MCP inspector (development tool)
npx @modelcontextprotocol/inspector python server.py

# Install as a Claude Desktop tool (add to claude_desktop_config.json)
```

---

## Configuring in Claude Desktop

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "business-intelligence": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
    }
  }
}
```

---

## Connecting MCP to LangChain/LangGraph

```python
# Using the official LangChain MCP adapter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def run_agent_with_mcp():
    async with MultiServerMCPClient({
        "business": {
            "command": "python",
            "args": ["server.py"],
            "transport": "stdio",
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "transport": "stdio",
        },
    }) as client:
        # Get all tools from all connected servers
        tools = await client.get_tools()
        print(f"Available tools: {[t.name for t in tools]}")
        # ["get_sales_report", "create_ticket", "read_file", "list_directory", ...]
        
        # Create LangGraph agent with MCP tools
        agent = create_react_agent(llm, tools)
        
        result = await agent.ainvoke({
            "messages": [
                HumanMessage("Get the latest sales report and create a ticket to review the Q2 targets")
            ]
        })
        
        print(result["messages"][-1].content)

asyncio.run(run_agent_with_mcp())
```

---

## Remote MCP Server (HTTP/SSE)

```python
# Server (SSE transport)
mcp.run(transport="sse", host="0.0.0.0", port=8080)

# Client connects to remote server
async with MultiServerMCPClient({
    "remote_tools": {
        "url": "http://my-mcp-server.com:8080/sse",
        "transport": "sse",
        "headers": {"Authorization": "Bearer my_api_key"},
    }
}) as client:
    tools = await client.get_tools()
```

---

## Pre-built MCP Servers

The MCP ecosystem has growing library of ready-to-use servers:

| Server | What it provides |
|---|---|
| `@modelcontextprotocol/server-filesystem` | Read/write local files |
| `@modelcontextprotocol/server-github` | GitHub repos, issues, PRs |
| `@modelcontextprotocol/server-postgres` | PostgreSQL query access |
| `@modelcontextprotocol/server-slack` | Slack messages and channels |
| `@modelcontextprotocol/server-brave-search` | Brave web search |
| `@modelcontextprotocol/server-memory` | Key-value memory store |
| `mcp-server-fetch` | HTTP/web fetching |

```bash
# Use a pre-built server
npx -y @modelcontextprotocol/server-github
```

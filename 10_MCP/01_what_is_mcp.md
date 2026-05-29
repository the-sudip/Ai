# MCP — Model Context Protocol

---

## What is MCP?

**Model Context Protocol (MCP)** is an **open standard** created by Anthropic (November 2024) that defines a universal way for AI models to connect to **external tools, data sources, and services**.

Think of it as **USB-C for AI integrations** — one standard protocol, many connectors:

```
Before MCP:
  Claude → custom integration for each tool
  GPT-4 → different custom integration for same tools
  Your app → build integrations from scratch every time

After MCP:
  Any AI ←→ MCP Protocol ←→ Any tool/data source
```

---

## Why MCP Exists

Without a standard:
- Every AI app builds custom integrations (Slack, GitHub, DB, etc.)
- Same tools need to be re-implemented for different AI models
- No standard way to discover what a server can do
- No composable ecosystem

With MCP:
- Build a tool server once, use it with Claude, GPT, LangChain, anything
- Standardized discovery — clients learn capabilities automatically
- Growing ecosystem of pre-built MCP servers

---

## MCP Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MCP HOST                         │
│  (Claude Desktop, VS Code, your LangChain app)      │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              MCP CLIENT                      │   │
│  │  (manages connections to MCP servers)        │   │
│  └──────────────┬───────────────────────────────┘   │
└─────────────────┼───────────────────────────────────┘
                  │ JSON-RPC 2.0
          ┌───────┼───────┐
          │       │       │
    ┌─────┴──┐ ┌──┴───┐ ┌─┴──────┐
    │File    │ │GitHub│ │Database│  ← MCP Servers
    │Server  │ │Server│ │Server  │
    └────────┘ └──────┘ └────────┘
```

- **MCP Host**: The application that uses AI (your app, Claude Desktop, VS Code)
- **MCP Client**: Embedded in the host, manages server connections
- **MCP Server**: Lightweight process that exposes capabilities

---

## MCP Communication Protocol

MCP uses **JSON-RPC 2.0** — a lightweight remote procedure call protocol.

```json
// Client → Server: List available tools
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}

// Server → Client: Response with tool definitions
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather for a city",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    ]
  }
}

// Client → Server: Call a tool
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "London"}
  }
}
```

---

## MCP Transport Types

### 1. stdio (Local)
Server runs as a subprocess, communication via stdin/stdout.

```bash
# Server is launched by the host and communicates via pipes
python my_server.py  # reads from stdin, writes to stdout
```

Best for: local tools, CLI integrations, development.

### 2. HTTP + SSE (Remote)
Server runs as an HTTP service. Client sends HTTP POST, server streams responses via Server-Sent Events.

```
Client → POST /messages → Server
Client ← GET /sse ← Server (streams events)
```

Best for: remote servers, cloud-hosted tools, shared team tools.

### 3. Streamable HTTP (New in 2025)
Simplified HTTP transport replacing SSE in newer MCP versions.

---

## MCP Lifecycle

```
1. INITIALIZATION
   Client → initialize (protocol version, client info)
   Server → initialized (server info, capabilities)

2. DISCOVERY
   Client → tools/list
   Server → [list of tool definitions]
   
   Client → resources/list
   Server → [list of resources]

3. OPERATION (repeated)
   LLM decides to call a tool
   Client → tools/call (name, arguments)
   Server → [tool result]

4. SHUTDOWN
   Client closes connection
```

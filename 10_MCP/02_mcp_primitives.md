# MCP Primitives — Tools, Resources, Prompts

MCP defines three types of capabilities that a server can expose to clients.

---

## 1. Tools

**Tools** are functions the LLM can call to take actions or retrieve computed information. They may have side effects.

Think of them like POST endpoints — they **do something**.

### Defining Tools in FastMCP

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Tool Server")

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather conditions for a city.
    
    Args:
        city: The name of the city (e.g., 'London', 'Tokyo')
    
    Returns:
        Current weather as a human-readable string
    """
    # Real implementation would call a weather API
    weather = {"London": "18°C, cloudy", "Tokyo": "25°C, sunny"}
    return weather.get(city, f"No weather data for {city}")

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to a recipient.
    
    WARNING: This action is irreversible. Confirm before calling.
    """
    # email_api.send(to, subject, body)
    return f"Email sent to {to} with subject '{subject}'"

@mcp.tool()
def run_sql_query(query: str, database: str = "main") -> str:
    """
    Execute a read-only SQL SELECT query on the database.
    Only SELECT statements are allowed.
    """
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are permitted."
    # result = db.execute(query)
    return "Query result: [...]"
```

### Tool Schema

Tools expose a JSON Schema for their input arguments, which the LLM uses to structure its call:

```json
{
  "name": "get_weather",
  "description": "Get the current weather conditions for a city.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The name of the city"
      }
    },
    "required": ["city"]
  }
}
```

---

## 2. Resources

**Resources** are data sources the LLM can **read**. They have URIs (like file paths or URLs). No side effects.

Think of them like GET endpoints — they **return data**.

```python
@mcp.resource("file:///logs/{date}")
def get_log_file(date: str) -> str:
    """Read the application log file for a given date."""
    with open(f"/var/logs/app_{date}.log") as f:
        return f.read()

@mcp.resource("db://users/{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Get a user's profile from the database."""
    user = db.get_user(user_id)
    return f"Name: {user.name}\nEmail: {user.email}\nJoined: {user.created_at}"

@mcp.resource("config://app-settings")
def get_app_settings() -> str:
    """Get the current application configuration."""
    with open("config.yaml") as f:
        return f.read()
```

### Resource URI Templates

Resources use URI templates (RFC 6570):
```
file:///logs/{date}         → file:///logs/2025-05-29
db://users/{user_id}/profile → db://users/42/profile
config://app-settings       → config://app-settings (static)
```

---

## 3. Prompts

**Prompts** are reusable, parameterized prompt templates that the server exposes. The client can request them by name.

```python
@mcp.prompt()
def summarize(text: str, style: str = "concise") -> str:
    """
    Generate a summarization prompt.
    
    Args:
        text: The text to summarize
        style: Summary style - 'concise', 'detailed', or 'bullet_points'
    """
    style_instructions = {
        "concise": "in 2-3 sentences",
        "detailed": "with main points and supporting details",
        "bullet_points": "as a bulleted list of key points",
    }
    instruction = style_instructions.get(style, "concisely")
    return f"Please summarize the following text {instruction}:\n\n{text}"

@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """Generate a code review prompt for the given code."""
    return f"""Review the following {language} code for:
1. Bugs and logic errors
2. Security vulnerabilities (OWASP Top 10)
3. Performance issues
4. Code style and readability
5. Missing error handling

Code:
```{language}
{code}
```

Provide specific, actionable feedback."""
```

---

## 4. Sampling (Client → Server)

**Sampling** is a reverse capability — the **server** can ask the **client** to make an LLM call. This allows servers to use AI without having their own LLM access.

```python
# Server requests LLM completion from the host
async def server_side_llm_call(mcp_context):
    result = await mcp_context.sample(
        messages=[{"role": "user", "content": "Summarize: ..."}],
        max_tokens=500,
    )
    return result.content
```

---

## Primitive Comparison

| Primitive | Direction | Side Effects | URI | Use For |
|---|---|---|---|---|
| **Tool** | Client calls → Server executes | Yes | No | Actions, computations |
| **Resource** | Client reads ← Server provides | No | Yes (URI) | Data access, file reads |
| **Prompt** | Client requests → Server provides template | No | No | Reusable prompt templates |
| **Sampling** | Server requests → Client LLM call | Depends | No | Server-side AI processing |

---

## Listing Capabilities

```python
# Client discovery:
tools = await client.list_tools()
resources = await client.list_resources()
prompts = await client.list_prompts()

# Client usage:
weather = await client.call_tool("get_weather", {"city": "London"})
log = await client.read_resource("file:///logs/2025-05-29")
prompt = await client.get_prompt("summarize", {"text": "...", "style": "concise"})
```

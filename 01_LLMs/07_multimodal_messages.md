# Multimodal Messages

Modern LLMs like GPT-4o, Claude 3.5, and Gemini can process more than just text — they accept **images, audio, video, and documents** alongside text in a single message.

---

## What is Multimodal?

| Modality | Description | Supported By |
|---|---|---|
| **Text** | Plain text | All LLMs |
| **Image** | JPEG, PNG, GIF, WebP | GPT-4o, Claude 3.5, Gemini |
| **Audio** | WAV, MP3, etc. | GPT-4o Audio, Gemini |
| **Video** | MP4, frames | Gemini 1.5+ |
| **Document/PDF** | PDF, DOCX | Claude 3.5, Gemini |

---

## Image Messages in LangChain

LangChain represents multimodal content using a **list of content blocks** instead of a plain string.

### Image from URL

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

message = HumanMessage(content=[
    {
        "type": "text",
        "text": "What's in this image? Describe it in detail.",
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "detail": "high",   # "low", "high", or "auto"
        },
    },
])

response = llm.invoke([message])
print(response.content)
```

### Image from Base64 (local file)

```python
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def encode_image(path: str) -> str:
    """Encode a local image file to base64."""
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")

llm = ChatOpenAI(model="gpt-4o")

image_data = encode_image("chart.png")

message = HumanMessage(content=[
    {
        "type": "text",
        "text": "Analyze this chart and summarize the key trends.",
    },
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_data}",
        },
    },
])

response = llm.invoke([message])
print(response.content)
```

---

## Multiple Images in One Message

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

message = HumanMessage(content=[
    {"type": "text", "text": "Compare these two charts. What are the main differences?"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/chart_q1.png"},
    },
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/chart_q2.png"},
    },
])

response = llm.invoke([message])
print(response.content)
```

---

## Multimodal with Claude (Anthropic)

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64
from pathlib import Path

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

image_data = base64.standard_b64encode(Path("diagram.png").read_bytes()).decode()

message = HumanMessage(content=[
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",   # Claude uses "image" block with "source"
            "data": image_data,
        },
    },
    {
        "type": "text",
        "text": "Explain what this architecture diagram shows.",
    },
])

response = llm.invoke([message])
print(response.content)
```

> Note: Claude uses `"type": "image"` with a `"source"` block, while OpenAI uses `"type": "image_url"` with a `"url"` field. LangChain normalizes this via model-specific adapters.

---

## Multimodal with Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

message = HumanMessage(content=[
    {"type": "text", "text": "Describe what you see in this image."},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/photo.jpg"},
    },
])

response = llm.invoke([message])
print(response.content)
```

Gemini also supports **video** and **audio** via the same pattern:

```python
# Video input (Gemini 1.5+)
from langchain_core.messages import HumanMessage

message = HumanMessage(content=[
    {"type": "text", "text": "Summarize what happens in this video."},
    {
        "type": "media",
        "data": base64_video_data,
        "mime_type": "video/mp4",
    },
])
```

---

## Multimodal in Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Using a callable to build the content list dynamically
def build_vision_prompt(image_url: str, question: str) -> list:
    return [
        ("system", "You are an expert image analyst."),
        ("human", [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]),
    ]

llm = ChatOpenAI(model="gpt-4o")

messages = build_vision_prompt(
    image_url="https://example.com/invoice.png",
    question="Extract all line items, quantities, and prices from this invoice."
)

response = llm.invoke(messages)
print(response.content)
```

---

## Structured Output from Images

Combine multimodal input with `.with_structured_output()` to extract structured data:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import List

class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    items: List[InvoiceItem]
    subtotal: float
    tax: float
    total: float

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Invoice)

message = HumanMessage(content=[
    {"type": "text", "text": "Extract all invoice data from this image."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
])

invoice: Invoice = structured_llm.invoke([message])
print(f"Vendor: {invoice.vendor}")
print(f"Total: ${invoice.total:.2f}")
for item in invoice.items:
    print(f"  {item.description}: {item.quantity} × ${item.unit_price:.2f}")
```

---

## Multimodal in LCEL Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from pathlib import Path

llm = ChatOpenAI(model="gpt-4o")

def image_analysis_chain(image_path: str, task: str) -> str:
    """Analyze an image using an LCEL chain."""
    image_data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ])
    ]
    
    chain = llm | StrOutputParser()
    return chain.invoke(messages)

# Usage
result = image_analysis_chain(
    image_path="product_photo.jpg",
    task="Write a compelling product description for an e-commerce listing."
)
print(result)
```

---

## Detail Levels (OpenAI)

OpenAI's vision API supports a `detail` parameter that controls token usage and accuracy:

```python
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg",
        "detail": "low"    # ~85 tokens — fast, good for thumbnails
        # "detail": "high"  # up to 1105 tokens — fine details, charts, text in images
        # "detail": "auto"  # (default) model decides based on image size
    }
}
```

| Setting | Tokens Used | Best For |
|---|---|---|
| `"low"` | ~85 | Quick classification, simple scenes |
| `"high"` | 170–1105 | Charts, documents, OCR, fine detail |
| `"auto"` | Varies | General use, let model decide |

---

## Common Use Cases

```python
# 1. OCR — Extract text from images
task = "Extract all text from this image exactly as written."

# 2. Chart/graph analysis
task = "What does this chart show? List the key data points and trends."

# 3. UI/screenshot analysis
task = "Describe every UI element in this screenshot and identify any usability issues."

# 4. Document understanding
task = "Summarize this document and extract any dates, names, and key figures."

# 5. Product identification
task = "Identify the product in this image. List the brand, model, and key features."

# 6. Code from screenshot
task = "Convert this screenshot of code into runnable Python. Preserve exact logic."
```

---

## Key Points for Interviews

1. Multimodal content uses a **list of content blocks** instead of a plain string in `HumanMessage`
2. OpenAI uses `"image_url"` blocks; Claude uses `"image"` blocks with `"source"`; LangChain adapts each
3. Images can be passed as **URLs** or **base64-encoded data URIs**
4. Use `"detail": "high"` for documents/charts with small text; `"low"` for cost savings
5. `.with_structured_output()` works with multimodal input — great for data extraction
6. Gemini 1.5+ supports video and audio in addition to images

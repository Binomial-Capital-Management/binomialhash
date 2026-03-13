# BinomialHash Examples

Production-grade examples showing BinomialHash integrated into real AI systems.

## Agent Loops

| Example | What it shows |
|---------|---------------|
| `01_openai_agent_loop.py` | Full chat-completions tool-use loop with GPT-4o. BH ingests large API responses, model queries compressed data across turns. |
| `02_anthropic_tool_use.py` | Claude Messages API loop. BH tools registered as Anthropic tool definitions, tool_use blocks dispatched through the adapter. |
| `03_gemini_function_calling.py` | Google Gemini multi-turn function calling with BH tools as native function declarations. |

## SDK & Framework Integration

| Example | What it shows |
|---------|---------------|
| `04_openai_agents_sdk.py` | Bridge from `ToolSpec` to OpenAI Agents SDK `@function_tool`. Dynamic tool generation for the `agents` framework used in production. |
| `05_fastapi_streaming.py` | FastAPI SSE endpoint where BH compresses tool outputs mid-stream, returning summaries instead of raw JSON to the LLM. |
| `06_langchain_tools.py` | Converting BH `ToolSpec` objects into LangChain `StructuredTool` instances for use in LangGraph agents. |
| `07_multi_agent_handoff.py` | Two-agent system: data-fetcher agent ingests raw data into BH, analyst agent queries the stored data via BH tools. Shared state via contextvars. |

## Infrastructure Patterns

| Example | What it shows |
|---------|---------------|
| `08_mcp_server_middleware.py` | Decorating MCP server tool functions with `@bh_intercept` so every tool output is auto-compacted. Zero changes to existing tool code. |
| `09_multi_tenant_isolation.py` | Per-request BH instances via `asyncio.TaskGroup` and `contextvars.copy_context()`. Demonstrates that user A's data never leaks to user B. |

## Domain Pipelines

| Example | What it shows |
|---------|---------------|
| `10_financial_analysis.py` | Market data ingestion, regression analysis, manifold navigation, and insight extraction — a realistic quant workflow. |
| `11_rag_retrieval.py` | Retrieval-Augmented Generation: BH compresses large document chunks returned by a vector store, agent queries the compressed data. |

## Live API Tests

| Example | What it shows |
|---------|---------------|
| `15_openai_responses_api.py` | Full OpenAI Responses API loop (not Chat Completions). Multi-turn with `function_call` / `function_call_output` flow, namespace support, and strict mode. |
| `16_live_multi_provider.py` | Same BH analysis task run against OpenAI, Anthropic, Gemini, and xAI. Compares tool call count, latency, and answer quality side-by-side. |

## Output & Observability

| Example | What it shows |
|---------|---------------|
| `12_export_artifacts.py` | Generate downloadable CSV, Markdown tables, and structured artifacts from BH data for chat frontend consumption. |
| `13_context_budget.py` | Monitor token usage, compression ratios, and memory across a multi-tool conversation. Budget-aware tool selection. |
| `14_manifold_deep_dive.py` | Full manifold exploration: build surface, navigate, geodesic pathfinding, frontier detection, basin analysis, ridge tracing. |

## Running

```bash
pip install binomialhash[all]

# Most examples use mock data and don't need API keys:
python examples/10_financial_analysis.py

# Agent loop examples need real API keys:
OPENAI_API_KEY=sk-... python examples/01_openai_agent_loop.py
```

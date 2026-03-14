# Provider Adapters

BinomialHash defines tools in a provider-neutral format and uses adapters to translate them into each LLM provider's wire format.

## ToolSpec

Every tool is defined as a `ToolSpec` dataclass:

```python
@dataclass
class ToolSpec:
    name: str                      # e.g. "bh_retrieve"
    description: str               # LLM-visible description
    input_schema: Dict[str, Any]   # JSON Schema for parameters
    handler: Callable[..., Any]    # Function to call when invoked
    group: str = ""                # "retrieval", "stats", "manifold", "export"
```

ToolSpecs are created by binding a `BinomialHash` instance:

```python
from binomialhash import BinomialHash
from binomialhash.tools import get_all_tools, get_tools_by_group

bh = BinomialHash()
all_specs = get_all_tools(bh)           # 68 tools
retrieval = get_tools_by_group(bh, "retrieval")  # data access subset
stats     = get_tools_by_group(bh, "stats")      # 39 analysis tools
manifold  = get_tools_by_group(bh, "manifold")   # navigation & spatial
export    = get_tools_by_group(bh, "export")      # CSV, Markdown, etc.
```

## Tool Groups

| Group | Count | Examples |
|-------|-------|---------|
| `retrieval` | 7 | `bh_retrieve`, `bh_aggregate`, `bh_query`, `bh_group_by`, `bh_schema`, `bh_keys`, `bh_context_stats` |
| `stats` | 39 | `bh_regress`, `bh_distribution`, `bh_cluster`, `bh_causal_graph`, `bh_entropy_spectrum`, ... |
| `manifold` | 16 | `bh_manifold_state`, `bh_navigate`, `bh_geodesic`, `bh_orbit`, `bh_heat_kernel`, `bh_reeb_graph`, ... |
| `export` | 6 | `bh_export_csv`, `bh_export_markdown`, `bh_export_excel`, `bh_export_rows`, `bh_export_chunks`, `bh_export_artifact` |

## Provider Adapters

### OpenAI

```python
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call

# Responses API format (default)
tools = get_openai_tools(specs)

# Chat Completions format (legacy)
tools = get_openai_tools(specs, format="chat_completions")

# Handle a function call
result = handle_openai_tool_call(specs, function_name, arguments_json)
```

OpenAI's Structured Outputs requires `additionalProperties: false` and all properties marked as required. The adapter adds these automatically.

### Anthropic

```python
from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use

tools = get_anthropic_tools(specs)
result = handle_anthropic_tool_use(specs, tool_name, tool_input)
```

Anthropic validates tool names against `^[a-zA-Z0-9_-]{1,64}$`. The adapter enforces this regex. Tool examples (if any) are appended to the description.

### Google Gemini

```python
from google.genai import types
from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call

decls = get_gemini_tools(specs)
gemini_tools = types.Tool(function_declarations=decls)

result = handle_gemini_tool_call(specs, function_name, function_args)
```

Gemini names must match `^[a-zA-Z_][a-zA-Z0-9_]*$`. The adapter normalises names and handles Protobuf `MapComposite` arguments.

### xAI / Grok

```python
from binomialhash.adapters.xai import get_xai_tools, handle_xai_tool_call

tools = get_xai_tools(specs)
result = handle_xai_tool_call(specs, function_name, arguments_json)
```

xAI uses an OpenAI-compatible wire format (Responses API style).

### Provider Router

For dynamic provider selection:

```python
from binomialhash.adapters import get_tools_for_provider

tools = get_tools_for_provider(specs, provider="openai")
tools = get_tools_for_provider(specs, provider="anthropic")
tools = get_tools_for_provider(specs, provider="gemini")
tools = get_tools_for_provider(specs, provider="xai")
```

## Selecting a Subset of Tools

You don't have to register all 68 tools. Pick what your agent needs:

```python
# Only retrieval + export (13 tools)
specs = get_tools_by_group(bh, "retrieval") + get_tools_by_group(bh, "export")
tools = get_openai_tools(specs)

# Or cherry-pick individual tools
selected = [s for s in get_all_tools(bh) if s.name in {"bh_retrieve", "bh_aggregate", "bh_export_csv"}]
tools = get_openai_tools(selected)
```

## Handling Tool Calls

All `handle_*` functions follow the same pattern:

1. Look up the `ToolSpec` by name (O(1) via an internal dict)
2. Parse the arguments (JSON string or dict, depending on provider)
3. Call `spec.handler(**parsed_args)`
4. Return the result as a JSON-serializable dict

"""Live multi-provider benchmark — same BH workflow across 4 providers.

Runs the exact same analysis task against OpenAI, Anthropic, Gemini, and xAI
to compare how each handles BH tool calling. Each provider:
  1. Gets the same tools and data
  2. Ingests the same dataset via BH
  3. Calls bh_aggregate and bh_group_by to answer the question
  4. Results compared side-by-side

Requires API keys for whichever providers you want to test:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY
"""

import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools, get_tools_by_group
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call
from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use
from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call


# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------

def generate_dataset() -> str:
    random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
               "JPM", "GS", "JNJ", "XOM", "CVX"]
    sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
               "META": "Tech", "NVDA": "Tech", "TSLA": "Tech", "JPM": "Finance",
               "GS": "Finance", "JNJ": "Health", "XOM": "Energy", "CVX": "Energy"}
    rows = []
    for t in tickers:
        for d in range(1, 21):
            rows.append({
                "ticker": t, "sector": sectors[t],
                "date": f"2025-03-{d:02d}",
                "price": round(random.uniform(80, 900), 2),
                "volume": random.randint(3_000_000, 80_000_000),
                "pe_ratio": round(random.uniform(10, 70), 2),
                "return_1d": round(random.gauss(0.001, 0.02), 6),
            })
    return json.dumps(rows)


SYSTEM_PROMPT = """You are a financial analyst. You have BinomialHash tools to query
stored datasets. A dataset of stock data has been loaded (key provided in the summary).
Answer the user's question by calling the appropriate BH tools.
Be concise — just the key numbers and a one-sentence insight."""

USER_QUERY = "Which sector has the highest average price? And what's the overall mean return_1d?"


# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    provider: str
    model: str
    success: bool = False
    answer: str = ""
    tool_calls: int = 0
    latency_ms: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# OpenAI (Responses API)
# ---------------------------------------------------------------------------

def test_openai(bh_summary: str, key: str) -> ProviderResult:
    from openai import OpenAI

    bh = get_binomial_hash()
    specs = get_tools_by_group(bh, "retrieval")
    tools = get_openai_tools(specs, format="responses")
    model = "gpt-5.4"

    result = ProviderResult(provider="OpenAI", model=model)
    client = OpenAI()

    input_list = [
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{USER_QUERY}"},
    ]

    t0 = time.time()
    try:
        for _ in range(5):
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                tools=tools,
                input=input_list,
            )
            input_list += response.output

            calls_this_round = False
            for item in response.output:
                if item.type == "function_call":
                    calls_this_round = True
                    result.tool_calls += 1
                    tool_result = handle_openai_tool_call(specs, item.name, item.arguments)
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result, default=str),
                    })

            if not calls_this_round:
                result.answer = response.output_text
                result.success = True
                break

    except Exception as e:
        result.error = str(e)

    result.latency_ms = int((time.time() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# Anthropic (Messages API)
# ---------------------------------------------------------------------------

def test_anthropic(bh_summary: str, key: str) -> ProviderResult:
    from anthropic import Anthropic

    bh = get_binomial_hash()
    specs = get_tools_by_group(bh, "retrieval")
    tools = get_anthropic_tools(specs)
    model = "claude-sonnet-4-6"

    result = ProviderResult(provider="Anthropic", model=model)
    client = Anthropic()

    messages = [
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{USER_QUERY}"},
    ]

    t0 = time.time()
    try:
        for _ in range(5):
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result.tool_calls += 1
                    tool_result = handle_anthropic_tool_use(specs, block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result, default=str),
                    })
                elif block.type == "text" and block.text.strip():
                    result.answer += block.text

            if not tool_results:
                result.success = True
                break
            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        result.error = str(e)

    result.latency_ms = int((time.time() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

def test_gemini(bh_summary: str, key: str) -> ProviderResult:
    from google import genai
    from google.genai import types

    bh = get_binomial_hash()
    specs = get_tools_by_group(bh, "retrieval")
    decls = get_gemini_tools(specs)
    model = "gemini-3.1-flash-lite-preview"

    result = ProviderResult(provider="Gemini", model=model)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    tools_obj = types.Tool(function_declarations=decls)
    config = types.GenerateContentConfig(
        tools=[tools_obj],
        system_instruction=SYSTEM_PROMPT,
    )

    contents = [f"Data loaded: {bh_summary}\n\n{USER_QUERY}"]

    t0 = time.time()
    try:
        for _ in range(5):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            has_fc = False

            for part in candidate.content.parts:
                if part.function_call:
                    has_fc = True
                    result.tool_calls += 1
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_result = handle_gemini_tool_call(specs, fc.name, args)

                    contents.append(candidate.content)
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name=fc.name,
                                response={"result": tool_result},
                            )],
                        )
                    )
                elif part.text and part.text.strip():
                    result.answer += part.text

            if not has_fc:
                result.success = True
                break

    except Exception as e:
        result.error = str(e)

    result.latency_ms = int((time.time() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# xAI (OpenAI-compatible)
# ---------------------------------------------------------------------------

def test_xai(bh_summary: str, key: str) -> ProviderResult:
    from openai import OpenAI

    bh = get_binomial_hash()
    specs = get_tools_by_group(bh, "retrieval")
    tools = get_openai_tools(specs, format="chat_completions")
    model = "grok-4-1-fast-reasoning"

    result = ProviderResult(provider="xAI", model=model)
    client = OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{USER_QUERY}"},
    ]

    t0 = time.time()
    try:
        for _ in range(5):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    result.tool_calls += 1
                    tool_result = handle_openai_tool_call(specs, tc.function.name, tc.function.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, default=str),
                    })
            else:
                result.answer = choice.message.content or ""
                result.success = True
                break

    except Exception as e:
        result.error = str(e)

    result.latency_ms = int((time.time() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": ("OPENAI_API_KEY", test_openai),
    "anthropic": ("ANTHROPIC_API_KEY", test_anthropic),
    "gemini": ("GEMINI_API_KEY", test_gemini),
    "xai": ("XAI_API_KEY", test_xai),
}


def run_benchmark():
    print("=== Live Multi-Provider BinomialHash Benchmark ===\n")

    available = []
    for name, (env_key, _) in PROVIDERS.items():
        has_key = bool(os.environ.get(env_key))
        status = "ready" if has_key else "skipped (no key)"
        print(f"  {name:12s} {status}")
        if has_key:
            available.append(name)
    print()

    if not available:
        print("No API keys found. Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY")
        return

    # Ingest shared dataset
    bh = init_binomial_hash()
    raw = generate_dataset()
    summary = bh.ingest(raw, "benchmark_data")
    key = bh.keys()[0]["key"]
    print(f"Dataset: {len(raw):,} chars → {len(summary)} char summary ({len(raw) // max(len(summary), 1)}x compression)")
    print(f"Question: {USER_QUERY}\n")

    # Run each provider
    results: list[ProviderResult] = []
    for name in available:
        env_key, test_fn = PROVIDERS[name]
        print(f"--- {name} ---")
        try:
            r = test_fn(summary, key)
        except Exception as e:
            r = ProviderResult(provider=name, model="?", error=str(e))
            traceback.print_exc()
        results.append(r)

        status = "OK" if r.success else f"FAIL: {r.error[:80]}"
        print(f"  Status: {status}")
        print(f"  Model: {r.model}")
        print(f"  Tool calls: {r.tool_calls}")
        print(f"  Latency: {r.latency_ms}ms")
        if r.answer:
            print(f"  Answer: {r.answer[:300]}{'...' if len(r.answer) > 300 else ''}")
        print()

    # Comparison table
    print("=== Comparison ===")
    print(f"{'Provider':12s} {'Model':30s} {'OK':4s} {'Tools':6s} {'Latency':>8s}  Answer preview")
    print("-" * 100)
    for r in results:
        ok = "yes" if r.success else "NO"
        preview = r.answer.replace("\n", " ")[:40] if r.answer else r.error[:40]
        print(f"{r.provider:12s} {r.model:30s} {ok:4s} {r.tool_calls:6d} {r.latency_ms:>7d}ms  {preview}")

    stats = bh.context_stats()
    print(f"\n=== BH budget across all providers: {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")


if __name__ == "__main__":
    run_benchmark()

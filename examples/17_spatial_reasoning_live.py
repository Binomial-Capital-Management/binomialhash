"""Live spatial reasoning benchmark — latent pattern discovery.

Exercises the 6 new manifold spatial reasoning tools across 4 providers
with prompts designed to surface hidden geometric structure that basic
stats and navigation cannot detect:

  1. Bottleneck hunting (heat kernel)
  2. Regime skeleton mapping (Reeb graph)
  3. Flow topology (vector field)
  4. Manifold-native segmentation (Laplacian spectrum)
  5. Structural anomaly detection (scalar harmonics)
  6. Connectivity robustness (diffusion distance)

The dataset is synthetic but encodes deliberate latent structure:
  - Two hidden regimes with a narrow bottleneck between them
  - A cyclical feedback loop (returns -> vol -> spread -> returns)
  - Non-linear interactions invisible to pairwise correlation
  - Structural anomalies that deviate from the manifold trend

Requires at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY, or XAI_API_KEY.
"""

import json
import math
import os
import random
import time
import traceback
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call
from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use
from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call


# ---------------------------------------------------------------------------
# Dataset with deliberate latent structure
# ---------------------------------------------------------------------------

def generate_structured_dataset() -> str:
    """Build a dataset with hidden geometric features.

    Structure planted:
    - Regime A (high liquidity): tight spreads, moderate returns, low vol
    - Regime B (stress): wide spreads, extreme returns, high vol
    - Bottleneck: only a few ticker-date combos bridge the two regimes
    - Cyclical coupling: vol -> spread -> cost -> returns -> vol
    - Structural outliers: 5 points that violate the manifold trend
    """
    random.seed(2026)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "JPM", "GS", "BAC", "JNJ", "PFE", "XOM", "CVX", "COP",
    ]
    sectors = {
        "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
        "META": "Tech", "NVDA": "Tech", "TSLA": "Tech",
        "JPM": "Finance", "GS": "Finance", "BAC": "Finance",
        "JNJ": "Health", "PFE": "Health",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    }
    market_caps = {
        "AAPL": 3.2e12, "MSFT": 3.1e12, "GOOGL": 2.0e12, "AMZN": 1.8e12,
        "META": 1.3e12, "NVDA": 2.8e12, "TSLA": 0.8e12,
        "JPM": 0.6e12, "GS": 0.15e12, "BAC": 0.3e12,
        "JNJ": 0.4e12, "PFE": 0.15e12,
        "XOM": 0.5e12, "CVX": 0.3e12, "COP": 0.15e12,
    }

    rows = []
    for day in range(1, 31):
        is_stress = day in (5, 6, 7, 15, 16, 22, 23, 24)
        transition = day in (4, 8, 14, 17, 21, 25)

        for t in tickers:
            s = sectors[t]
            mc = market_caps[t]

            if is_stress:
                base_vol = 0.04 + random.gauss(0, 0.008)
                base_spread = 0.08 + random.gauss(0, 0.015)
                base_return = random.gauss(-0.005, 0.035)
            elif transition:
                base_vol = 0.025 + random.gauss(0, 0.005)
                base_spread = 0.04 + random.gauss(0, 0.01)
                base_return = random.gauss(0, 0.02)
            else:
                base_vol = 0.015 + random.gauss(0, 0.003)
                base_spread = 0.02 + random.gauss(0, 0.005)
                base_return = random.gauss(0.002, 0.012)

            sector_mult = {"Tech": 1.2, "Finance": 1.0, "Health": 0.7, "Energy": 1.4}
            vol = max(0.005, base_vol * sector_mult.get(s, 1.0))

            spread = max(0.005, base_spread * (1.0 + 0.3 * vol / 0.02))

            cost = spread * vol * 1000
            ret = base_return - 0.15 * cost + random.gauss(0, 0.003)

            pe = 20 + 30 * math.log10(mc / 1e11) + random.gauss(0, 5)
            momentum = ret * (1 + random.gauss(0, 0.3))

            volume = int(mc / 50 * (1 + 2 * vol) * random.uniform(0.5, 1.5))

            rows.append({
                "ticker": t,
                "sector": s,
                "date": f"2026-03-{day:02d}",
                "market_cap_bn": round(mc / 1e9, 1),
                "price": round(random.uniform(50, 900), 2),
                "volume": volume,
                "return_1d": round(ret, 6),
                "volatility": round(vol, 6),
                "bid_ask_spread": round(spread, 6),
                "pe_ratio": round(max(5, pe), 2),
                "momentum_1w": round(momentum, 6),
                "trading_cost": round(cost, 6),
            })

    # Plant 5 structural outliers that violate the manifold trend
    anomaly_indices = [42, 117, 203, 287, 391]
    for ai in anomaly_indices:
        if ai < len(rows):
            rows[ai]["return_1d"] = round(random.choice([-1, 1]) * 0.12, 6)
            rows[ai]["volatility"] = round(0.08 + random.uniform(0, 0.03), 6)
            rows[ai]["bid_ask_spread"] = round(0.005, 6)

    return json.dumps(rows)


# ---------------------------------------------------------------------------
# Complex prompts for latent pattern discovery
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative researcher with access to BinomialHash — a 
content-addressed data structure that builds manifold surfaces from tabular data 
and provides geometric analysis tools.

Your dataset contains 30 days of multi-stock market data. Use the full range of 
BH tools — especially the spatial reasoning tools (heat_kernel, reeb_graph, 
vector_field, laplacian_spectrum, scalar_harmonics, diffusion_distance) — to 
discover latent patterns the data hides.

Key approach:
1. First ingest and build manifold state (bh_manifold_state)
2. Use spatial tools to probe geometric structure
3. Cross-reference with statistical tools for validation
4. Report discoveries with evidence — coordinates, scores, magnitudes

Be precise and quantitative. Don't just describe what the tools returned — 
interpret what the patterns mean for an investor."""

PROMPTS = [
    # Prompt 1: Bottleneck + regime detection
    {
        "name": "Regime Bottleneck Discovery",
        "query": (
            "I suspect this market data contains hidden regimes connected by narrow "
            "transition corridors. Use the heat kernel to find geometric bottlenecks "
            "on the manifold, then use the Reeb graph on return_1d to map the "
            "topological skeleton of return regimes. Are there distinct connected "
            "components at certain return levels? What coordinates sit at the "
            "bottleneck between regimes?"
        ),
        "expected_tools": ["bh_manifold_state", "bh_heat_kernel", "bh_reeb_graph"],
    },
    # Prompt 2: Cyclical flow structure
    {
        "name": "Cyclical Flow Topology",
        "query": (
            "Analyze the gradient flow of volatility on the manifold. I want to "
            "know: where are the sources (volatility being created) and sinks "
            "(volatility being absorbed)? Is there any rotational structure — "
            "a cyclical pattern where volatility feeds into spreads which feed "
            "into costs which feed back into returns? Use the vector field tool "
            "on volatility and trading_cost, and compare their flow topologies."
        ),
        "expected_tools": ["bh_manifold_state", "bh_vector_field"],
    },
    # Prompt 3: Manifold-native segmentation vs feature clustering
    {
        "name": "Spectral Segmentation",
        "query": (
            "Use the Laplacian spectrum to find the natural segmentation of this "
            "data on the manifold. How many natural clusters does the spectral gap "
            "suggest? Compare the manifold-native partition (from the Fiedler "
            "vector) with a simple sector-based grouping. Are there stocks that "
            "the manifold groups together despite being in different sectors? "
            "That would indicate hidden structural similarity invisible to labels."
        ),
        "expected_tools": ["bh_manifold_state", "bh_laplacian_spectrum"],
    },
    # Prompt 4: Structural anomaly detection via harmonics
    {
        "name": "Harmonic Anomaly Detection",
        "query": (
            "Decompose the return_1d field into the manifold's natural harmonic "
            "modes using scalar harmonics. How many modes capture 90%% of the "
            "variance? Which points have the largest residuals — meaning they "
            "structurally deviate from the manifold's natural trend? These aren't "
            "just statistical outliers; they're points where the local geometry "
            "contradicts the global pattern. List the top anomalies with their "
            "coordinates and residual magnitudes."
        ),
        "expected_tools": ["bh_manifold_state", "bh_scalar_harmonics"],
    },
    # Prompt 5: Full spatial analysis pipeline
    {
        "name": "Full Geometric Intelligence",
        "query": (
            "Run a complete geometric analysis pipeline on this dataset:\n"
            "1. Build manifold state and check coverage\n"
            "2. Heat kernel: find bottlenecks between data regions\n"
            "3. Reeb graph on volatility: map the regime skeleton\n"
            "4. Vector field on return_1d: classify sources/sinks/vortices\n"
            "5. Laplacian spectrum: find natural segments\n"
            "6. Scalar harmonics on bid_ask_spread: find structural deviations\n"
            "7. Diffusion distance: which points are truly far apart on the manifold "
            "despite looking close in raw features?\n\n"
            "Synthesize these into a unified view: what is the hidden geometric "
            "structure of this market, and what would it mean for portfolio construction?"
        ),
        "expected_tools": [
            "bh_manifold_state", "bh_heat_kernel", "bh_reeb_graph",
            "bh_vector_field", "bh_laplacian_spectrum", "bh_scalar_harmonics",
            "bh_diffusion_distance",
        ],
    },
]


# ---------------------------------------------------------------------------
# Provider test runners
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    provider: str
    model: str
    prompt_name: str
    success: bool = False
    answer: str = ""
    tool_calls: int = 0
    tools_used: list = None
    latency_ms: int = 0
    error: str = ""

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


def _run_openai(bh_summary: str, prompt: dict, max_turns: int = 12) -> ProviderResult:
    from openai import OpenAI

    bh = get_binomial_hash()
    specs = get_all_tools(bh)
    tools = get_openai_tools(specs, format="responses")
    model = "gpt-5.4"
    result = ProviderResult(provider="OpenAI", model=model, prompt_name=prompt["name"])
    client = OpenAI()

    input_list = [
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{prompt['query']}"},
    ]

    t0 = time.time()
    try:
        for _ in range(max_turns):
            response = client.responses.create(
                model=model, instructions=SYSTEM_PROMPT,
                tools=tools, input=input_list,
            )
            input_list += response.output

            calls_this_round = False
            for item in response.output:
                if item.type == "function_call":
                    calls_this_round = True
                    result.tool_calls += 1
                    result.tools_used.append(item.name)
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


def _run_anthropic(bh_summary: str, prompt: dict, max_turns: int = 12) -> ProviderResult:
    from anthropic import Anthropic

    bh = get_binomial_hash()
    specs = get_all_tools(bh)
    tools = get_anthropic_tools(specs)
    model = "claude-sonnet-4-6"
    result = ProviderResult(provider="Anthropic", model=model, prompt_name=prompt["name"])
    client = Anthropic()

    messages = [
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{prompt['query']}"},
    ]

    t0 = time.time()
    try:
        for _ in range(max_turns):
            response = client.messages.create(
                model=model, max_tokens=4096,
                system=SYSTEM_PROMPT, tools=tools, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result.tool_calls += 1
                    result.tools_used.append(block.name)
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


def _run_gemini(bh_summary: str, prompt: dict, max_turns: int = 12) -> ProviderResult:
    from google import genai
    from google.genai import types

    bh = get_binomial_hash()
    specs = get_all_tools(bh)
    decls = get_gemini_tools(specs)
    model = "gemini-3.1-flash-lite-preview"
    result = ProviderResult(provider="Gemini", model=model, prompt_name=prompt["name"])
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    tools_obj = types.Tool(function_declarations=decls)
    config = types.GenerateContentConfig(
        tools=[tools_obj], system_instruction=SYSTEM_PROMPT,
    )
    contents = [f"Data loaded: {bh_summary}\n\n{prompt['query']}"]

    t0 = time.time()
    try:
        for _ in range(max_turns):
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            candidate = response.candidates[0]
            has_fc = False

            for part in candidate.content.parts:
                if part.function_call:
                    has_fc = True
                    result.tool_calls += 1
                    fc = part.function_call
                    result.tools_used.append(fc.name)
                    args = dict(fc.args) if fc.args else {}
                    tool_result = handle_gemini_tool_call(specs, fc.name, args)
                    contents.append(candidate.content)
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=fc.name, response={"result": tool_result},
                        )],
                    ))
                elif part.text and part.text.strip():
                    result.answer += part.text

            if not has_fc:
                result.success = True
                break

    except Exception as e:
        result.error = str(e)

    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _run_xai(bh_summary: str, prompt: dict, max_turns: int = 12) -> ProviderResult:
    from openai import OpenAI

    bh = get_binomial_hash()
    specs = get_all_tools(bh)
    tools = get_openai_tools(specs, format="chat_completions")
    model = "grok-4-1-fast-reasoning"
    result = ProviderResult(provider="xAI", model=model, prompt_name=prompt["name"])
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Data loaded: {bh_summary}\n\n{prompt['query']}"},
    ]

    t0 = time.time()
    try:
        for _ in range(max_turns):
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    result.tool_calls += 1
                    result.tools_used.append(tc.function.name)
                    tool_result = handle_openai_tool_call(
                        specs, tc.function.name, tc.function.arguments,
                    )
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
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
    "openai": ("OPENAI_API_KEY", _run_openai),
    "anthropic": ("ANTHROPIC_API_KEY", _run_anthropic),
    "gemini": ("GEMINI_API_KEY", _run_gemini),
    "xai": ("XAI_API_KEY", _run_xai),
}


def run_benchmark(prompt_index: int = None, provider_name: str = None):
    print("=== Spatial Reasoning Live Benchmark ===\n")

    available = []
    for name, (env_key, _) in PROVIDERS.items():
        if provider_name and name != provider_name:
            continue
        has_key = bool(os.environ.get(env_key))
        status = "ready" if has_key else "skipped"
        print(f"  {name:12s} {status}")
        if has_key:
            available.append(name)
    print()

    if not available:
        print("No API keys found. Set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY")
        return

    bh = init_binomial_hash()
    raw = generate_structured_dataset()
    summary = bh.ingest(raw, "market_structured")
    key = bh.keys()[0]["key"]
    print(f"Dataset: {len(json.loads(raw))} rows, {len(raw):,} chars")
    print(f"  -> {len(summary)} char summary ({len(raw) // max(len(summary), 1)}x compression)")
    print(f"  Key: {key}\n")

    prompts_to_run = PROMPTS if prompt_index is None else [PROMPTS[prompt_index]]

    all_results: list[ProviderResult] = []

    for pi, prompt in enumerate(prompts_to_run):
        print(f"{'='*70}")
        print(f"Prompt {pi+1}: {prompt['name']}")
        print(f"  Q: {prompt['query'][:120]}...")
        print(f"  Expected tools: {prompt['expected_tools']}")
        print(f"{'='*70}\n")

        for name in available:
            env_key, test_fn = PROVIDERS[name]
            print(f"--- {name} ---")
            try:
                r = test_fn(summary, prompt)
            except Exception as e:
                r = ProviderResult(
                    provider=name, model="?", prompt_name=prompt["name"],
                    error=str(e),
                )
                traceback.print_exc()

            all_results.append(r)
            status = "OK" if r.success else f"FAIL: {r.error[:100]}"
            print(f"  Status: {status}")
            print(f"  Model: {r.model}")
            print(f"  Tool calls: {r.tool_calls}")

            spatial_used = [t for t in r.tools_used if t.startswith("bh_heat") or
                           t.startswith("bh_reeb") or t.startswith("bh_vector") or
                           t.startswith("bh_laplacian") or t.startswith("bh_scalar") or
                           t.startswith("bh_diffusion")]
            expected_used = [t for t in prompt["expected_tools"] if t in r.tools_used]
            print(f"  Spatial tools used: {spatial_used}")
            print(f"  Expected hit rate: {len(expected_used)}/{len(prompt['expected_tools'])}")
            print(f"  Latency: {r.latency_ms}ms")

            if r.answer:
                preview = r.answer[:500].replace("\n", "\n    ")
                print(f"  Answer:\n    {preview}")
                if len(r.answer) > 500:
                    print(f"    ... ({len(r.answer)} chars total)")
            print()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Provider':12s} {'Prompt':30s} {'OK':4s} {'Tools':6s} {'Spatial':8s} {'Latency':>8s}")
    print("-" * 80)
    for r in all_results:
        ok = "yes" if r.success else "NO"
        spatial = len([t for t in r.tools_used if t.startswith("bh_heat") or
                      t.startswith("bh_reeb") or t.startswith("bh_vector") or
                      t.startswith("bh_laplacian") or t.startswith("bh_scalar") or
                      t.startswith("bh_diffusion")])
        print(f"{r.provider:12s} {r.prompt_name:30s} {ok:4s} {r.tool_calls:6d} {spatial:8d} {r.latency_ms:>7d}ms")

    stats = bh.context_stats()
    print(f"\nBH budget: {stats['chars_in_raw']:,} raw -> {stats['chars_out_to_llm']:,} to LLM "
          f"({stats['compression_ratio']:.1f}x) | {stats['tool_calls']} tool calls")


if __name__ == "__main__":
    import sys
    pi = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    prov = sys.argv[2] if len(sys.argv) > 2 else None
    run_benchmark(prompt_index=pi, provider_name=prov)

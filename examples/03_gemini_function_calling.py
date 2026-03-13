"""Google Gemini function-calling loop with BinomialHash.

Shows:
  1. BH ToolSpecs converted to Gemini function declarations (plain dicts)
  2. Gemini returns functionCall parts
  3. Dispatch through the Gemini adapter (handles protobuf MapComposite args)
  4. Return functionResponse parts

Requires: pip install google-genai binomialhash
Set GOOGLE_API_KEY or pass --mock.
"""

import argparse
import json
import os

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools, get_tools_by_group
from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call


# ---------------------------------------------------------------------------
# Simulated genomics data source
# ---------------------------------------------------------------------------

def fetch_gene_expression(organism: str) -> str:
    """Simulate a gene expression dataset — realistic BH use case outside finance."""
    import random
    random.seed(42)
    genes = [f"GENE_{i:04d}" for i in range(300)]
    tissues = ["brain", "liver", "kidney", "heart", "lung"]
    rows = []
    for gene in genes:
        for tissue in tissues:
            rows.append({
                "gene_id": gene,
                "tissue": tissue,
                "expression_level": round(random.uniform(0, 1000), 2),
                "methylation_score": round(random.uniform(0, 1), 4),
                "gc_content": round(random.uniform(0.3, 0.7), 4),
                "chromosome": f"chr{random.randint(1, 22)}",
                "p_value": round(random.uniform(0.0001, 0.5), 6),
                "fold_change": round(random.uniform(-5, 5), 3),
            })
    return json.dumps(rows)


def build_tools(bh: BinomialHash):
    retrieval = get_tools_by_group(bh, "retrieval")
    stats = get_tools_by_group(bh, "stats")
    specs = retrieval + stats

    gemini_decls = get_gemini_tools(specs)

    gemini_decls.insert(0, {
        "name": "fetch_gene_expression",
        "description": "Fetch gene expression data for an organism.",
        "parameters": {
            "type": "object",
            "properties": {"organism": {"type": "string", "description": "Species name"}},
            "required": ["organism"],
        },
    })
    return gemini_decls, specs


def handle_tool(specs, name, args):
    if name == "fetch_gene_expression":
        raw = fetch_gene_expression(args.get("organism", "human"))
        return get_binomial_hash().ingest(raw, "gene_expression")
    return json.dumps(handle_gemini_tool_call(specs, name, args), default=str)


def run_mock():
    bh = init_binomial_hash()
    decls, specs = build_tools(bh)

    print("=== Gemini mock loop ===\n")
    print(f"Registered {len(decls)} function declarations\n")

    # Simulate Gemini calling fetch_gene_expression
    print("[Gemini] functionCall: fetch_gene_expression({organism: 'human'})")
    result = handle_tool(specs, "fetch_gene_expression", {"organism": "human"})
    print(f"[BH] Ingested → {len(result)} chars (raw was ~150KB)\n")

    key = bh.keys()[0]["key"]

    # Simulate Gemini calling bh_partial_corr
    print("[Gemini] functionCall: bh_partial_corr(expression vs methylation, controlling gc_content)")
    pcorr = handle_tool(specs, "bh_partial_corr", {
        "key": key,
        "field_a": "expression_level",
        "field_b": "methylation_score",
        "controls_json": '["gc_content"]',
    })
    parsed = json.loads(pcorr)
    print(f"[Result] Raw corr: {parsed.get('raw_correlation', 'N/A'):.4f}")
    print(f"         Partial:  {parsed.get('partial_correlation', 'N/A'):.4f}")
    print(f"         → {'Spurious!' if abs(parsed.get('partial_correlation', 0)) < abs(parsed.get('raw_correlation', 0)) * 0.5 else 'Real relationship'}\n")

    # Simulate dependency screen
    print("[Gemini] functionCall: bh_dependency_screen(target=fold_change)")
    screen = handle_tool(specs, "bh_dependency_screen", {
        "key": key,
        "target": "fold_change",
        "candidates_json": '["expression_level", "methylation_score", "gc_content", "p_value"]',
        "top_k": 4,
    })
    parsed = json.loads(screen)
    print("[Result] Driver ranking:")
    for rank in parsed.get("ranked_candidates", parsed.get("rankings", [])):
        name = rank.get("candidate", rank.get("field", "?"))
        print(f"  {name:25s} corr={rank.get('raw_correlation', 0):+.4f}  partial={rank.get('partial_correlation', 0):+.4f}")

    stats = bh.context_stats()
    print(f"\n=== {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")

    print("\n--- Sample Gemini function declaration ---")
    print(json.dumps(decls[1], indent=2)[:400])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or not os.environ.get("GOOGLE_API_KEY"):
        run_mock()
    else:
        print("Live Gemini loop requires google-genai SDK — see README for setup")
        run_mock()

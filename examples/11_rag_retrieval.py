"""RAG (Retrieval-Augmented Generation) pipeline with BinomialHash.

Problem: vector store returns 20 document chunks (each 500-1000 tokens).
Passing all of them raw = 10K-20K tokens in context. Most of it is redundant
or low-relevance. The LLM can't efficiently process all of it.

Solution: BH ingests the retrieval results as structured data. The LLM gets
a compact schema summary, then uses bh_query to pull only the relevant chunks.

Pattern:
  1. Vector search returns chunks with metadata (score, source, section, etc.)
  2. BH ingests the result set → compact summary
  3. LLM sees: "20 chunks, scores 0.72-0.95, from 3 documents, 5 sections"
  4. LLM calls bh_query(where: score > 0.85) → gets top 5 chunks
  5. LLM calls bh_query(where: section = 'Risk Factors') → focused retrieval
"""

import json
import random
import hashlib

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_tools_by_group
from binomialhash.adapters.common import handle_tool_call


# ---------------------------------------------------------------------------
# Simulated vector store retrieval
# ---------------------------------------------------------------------------

def simulate_vector_search(query: str, top_k: int = 30) -> list[dict]:
    """Simulate a vector store returning ranked document chunks."""
    random.seed(hash(query) % 2**31)

    documents = [
        {"doc_id": "10K_2024", "doc_type": "10-K Filing", "company": "TechCorp"},
        {"doc_id": "10K_2023", "doc_type": "10-K Filing", "company": "TechCorp"},
        {"doc_id": "earnings_Q4", "doc_type": "Earnings Transcript", "company": "TechCorp"},
        {"doc_id": "analyst_report", "doc_type": "Analyst Report", "company": "TechCorp"},
    ]

    sections = [
        "Risk Factors", "Business Description", "MD&A",
        "Financial Statements", "Executive Summary", "Revenue Breakdown",
        "Competitive Landscape", "Regulatory Environment",
    ]

    chunks = []
    for i in range(top_k):
        doc = random.choice(documents)
        section = random.choice(sections)
        score = round(random.uniform(0.55, 0.98), 4)
        text = _generate_chunk_text(section, doc["company"])

        chunks.append({
            "chunk_id": hashlib.md5(f"{doc['doc_id']}_{section}_{i}".encode()).hexdigest()[:12],
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "company": doc["company"],
            "section": section,
            "relevance_score": score,
            "chunk_index": i,
            "text": text,
            "token_count": len(text.split()),
            "page_number": random.randint(1, 200),
            "fiscal_year": random.choice([2023, 2024]),
        })

    chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
    return chunks


def _generate_chunk_text(section: str, company: str) -> str:
    """Generate realistic-ish document text."""
    templates = {
        "Risk Factors": f"{company} faces significant risks including market volatility, "
            "regulatory changes, competitive pressures, and macroeconomic uncertainty. "
            "Our revenue concentration in cloud services exposes us to infrastructure costs "
            "and potential service disruptions. International operations face currency risk "
            "and geopolitical uncertainty across our major markets. " * 3,
        "MD&A": f"Management's discussion: {company} reported revenue growth of 15% YoY "
            "driven by cloud services and AI product adoption. Operating margins expanded "
            "200bps to 32.5% as we achieved scale economies in our data center operations. "
            "Capital expenditures increased 40% to support AI infrastructure buildout. " * 3,
        "Revenue Breakdown": f"{company} revenue by segment: Cloud Services $45.2B (+22%), "
            "Enterprise Software $28.1B (+8%), Consumer Products $12.3B (-3%), "
            "Professional Services $6.8B (+15%). Geographic mix: Americas 58%, EMEA 26%, "
            "APAC 16%. Recurring revenue represents 72% of total. " * 3,
    }
    return templates.get(section, f"[{section}] {company} information regarding {section.lower()}. " * 8)


# ---------------------------------------------------------------------------
# RAG pipeline with BH
# ---------------------------------------------------------------------------

def run_rag_pipeline():
    bh = init_binomial_hash()
    specs = get_tools_by_group(bh, "retrieval")

    print("=== RAG Pipeline with BinomialHash ===\n")

    # Step 1: Vector search
    query = "What are TechCorp's main revenue drivers and growth risks?"
    chunks = simulate_vector_search(query, top_k=30)
    raw = json.dumps(chunks)
    print(f"[1] Vector search: {len(chunks)} chunks, {len(raw):,} chars total")
    print(f"    Score range: {chunks[-1]['relevance_score']:.4f} - {chunks[0]['relevance_score']:.4f}")
    print(f"    Avg tokens per chunk: {sum(c['token_count'] for c in chunks) // len(chunks)}")
    print(f"    Total tokens (raw): ~{sum(c['token_count'] for c in chunks):,}\n")

    # Step 2: BH ingest — compress the entire result set
    summary = bh.ingest(raw, "rag_results")
    key = bh.keys()[0]["key"]
    print(f"[2] BH ingest: {len(raw):,} chars → {len(summary)} chars ({len(raw) // max(len(summary), 1)}x)")
    print(f"    LLM now sees a schema summary instead of {len(chunks)} full chunks\n")

    # Step 3: LLM inspects the schema
    schema = bh.schema(key)
    print(f"[3] Schema inspection:")
    print(f"    Columns: {schema['columns'][:8]}")
    score_stats = schema["col_stats"].get("relevance_score", {})
    print(f"    Score stats: mean={score_stats.get('mean', '?')}, max={score_stats.get('max', '?')}")
    section_stats = schema["col_stats"].get("section", {})
    print(f"    Sections: {section_stats.get('top_values', [])}\n")

    # Step 4: LLM queries high-relevance chunks only
    print(f"[4] Query: relevance_score > 0.85")
    high_rel = handle_tool_call(specs, "bh_query", {
        "key": key,
        "where_json": json.dumps({"column": "relevance_score", "op": ">", "value": 0.85}),
        "sort_by": "relevance_score",
        "sort_desc": True,
        "limit": 5,
        "columns": json.dumps(["section", "relevance_score", "text", "doc_type"]),
    })
    print(f"    Matched: {high_rel.get('matched', '?')} chunks, returned top {high_rel.get('returned', '?')}")
    for row in high_rel.get("rows", [])[:3]:
        print(f"    [{row.get('relevance_score', 0):.4f}] {row.get('section', '?')}: {str(row.get('text', ''))[:80]}...")

    # Step 5: LLM queries specific section
    print(f"\n[5] Query: section = 'Risk Factors'")
    risk_chunks = handle_tool_call(specs, "bh_query", {
        "key": key,
        "where_json": json.dumps({"column": "section", "op": "=", "value": "Risk Factors"}),
        "sort_by": "relevance_score",
        "sort_desc": True,
        "limit": 5,
    })
    print(f"    Matched: {risk_chunks.get('matched', '?')} risk factor chunks")

    # Step 6: Group by section to understand coverage
    print(f"\n[6] Group by section:")
    groups = bh.group_by(
        key, ["section"],
        json.dumps([
            {"column": "relevance_score", "func": "mean", "alias": "avg_score"},
            {"column": "token_count", "func": "sum", "alias": "total_tokens"},
            {"column": "chunk_id", "func": "count", "alias": "chunk_count"},
        ]),
        "avg_score", True, 10,
    )
    for g in groups.get("rows", []):
        print(f"    {g.get('section', '?'):25s} chunks={g.get('chunk_count', 0):>2}  "
              f"avg_score={g.get('avg_score', 0):.4f}  tokens={g.get('total_tokens', 0):>5}")

    # Budget comparison
    stats = bh.context_stats()
    naive_tokens = sum(c["token_count"] for c in chunks)
    print(f"\n=== Context budget comparison ===")
    print(f"  Naive (all chunks):    ~{naive_tokens:,} tokens")
    print(f"  BH summary + queries:   ~{stats['est_tokens_out']:,} tokens")
    print(f"  Savings:                ~{naive_tokens - stats['est_tokens_out']:,} tokens ({(1 - stats['est_tokens_out'] / max(naive_tokens, 1)) * 100:.0f}%)")
    print(f"  And the LLM got targeted data instead of scanning everything.")


if __name__ == "__main__":
    run_rag_pipeline()

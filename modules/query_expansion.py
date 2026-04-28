# modules/query_expansion.py
#
# Generates query variants for RAG-Fusion via a tightly constrained LLM prompt.
# The variants are used to run N parallel vector retrievals whose results are
# merged via Reciprocal Rank Fusion (RRF) before returning the top-k chunks.

import ollama


def generate_query_variants(query: str, n: int = 3, model: str = "llama3.2:3b") -> list[str]:
    """
    Ask the LLM to rephrase the clinical query N different ways.
    Returns a list of variant strings (always includes the original as the first entry).
    Falls back to [query] if the LLM call fails or returns too few variants.
    """
    prompt = (
        f"Rephrase the following clinical question {n} different ways. "
        "Each rephrasing must ask for the same information but use different wording. "
        f"Output exactly {n} lines, one rephrasing per line, no numbering, no explanation.\n\n"
        f"Question: {query}"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()][:n]
    except Exception:
        variants = []

    if len(variants) < 1:
        return [query]

    # Always lead with the original so it gets a rank signal in RRF
    return [query] + variants


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    n_results: int = 10,
) -> list[dict]:
    """
    Merge N ranked chunk lists using Reciprocal Rank Fusion.

    score(chunk) = sum over lists of 1 / (rank_in_list + k)

    k=60 is the standard RRF constant (Cormack et al. 2009) — dampens the
    outsized influence of rank-1 results.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk["chunk_id"]
            scores[cid]    = scores.get(cid, 0.0) + 1.0 / (rank + k)
            chunk_map[cid] = chunk

    fused = sorted(chunk_map.values(), key=lambda c: scores[c["chunk_id"]], reverse=True)
    for chunk in fused:
        chunk["score"] = scores[chunk["chunk_id"]]

    return fused[:n_results]

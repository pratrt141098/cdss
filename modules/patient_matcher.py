# modules/patient_matcher.py
#
# Cross-patient retrieval via query entity extraction.
#
# The problem RAG-Fusion failed to solve: given a natural-language query that
# describes a patient by clinical characteristics (diagnoses, medications,
# symptoms, demographics) rather than by ID, find the right patient from the
# full corpus and scope retrieval to them.
#
# Approach:
#   1. Use the LLM to extract clinical entities from the query (one short call)
#   2. Score every patient by entity overlap against their stored NER metadata
#   3. Return the top-N candidate hadm_ids for the caller to use as filters

import ollama


def extract_query_entities(query: str, model: str = "llama3.2:3b") -> list[str]:
    """
    Ask the LLM to pull clinical terms from the query that could identify a
    specific patient. Returns a flat list of lowercase entity strings.
    Falls back to empty list on failure so the caller degrades gracefully.
    """
    prompt = (
        "Extract every clinical term from the query below that could help identify "
        "a specific patient. Include: diagnoses, medications, procedures, anatomy, "
        "symptoms, and demographic descriptors (e.g. 'elderly', 'female').\n"
        "Output one term per line, lowercase, no numbering, no explanation.\n\n"
        f"Query: {query}"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        return [line.strip().lower() for line in raw.splitlines() if line.strip()]
    except Exception:
        return []


def score_patients(
    query_entities: list[str],
    patient_entity_map: dict[str, set[str]],
    top_n: int = 3,
) -> list[tuple[str, int]]:
    """
    Score each patient by raw entity overlap count between the query entity
    list and the patient's stored NER entity set.

    Returns list of (hadm_id, overlap_count) sorted descending, up to top_n.
    Only patients with at least one overlap are included.
    """
    query_set = {e.strip().lower() for e in query_entities if e.strip()}
    if not query_set:
        return []

    scored = []
    for hadm_id, entity_set in patient_entity_map.items():
        overlap = len(query_set & entity_set)
        if overlap > 0:
            scored.append((hadm_id, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def find_candidate_patients(
    query: str,
    patient_entity_map: dict[str, set[str]],
    top_n: int = 3,
    model: str = "llama3.2:3b",
) -> list[tuple[str, int]]:
    """
    End-to-end: extract entities from query, score patients, return top-N candidates.
    Each entry is (hadm_id, overlap_count).
    Returns empty list if no entities extracted or no matches found.
    """
    entities = extract_query_entities(query, model=model)
    if not entities:
        return []
    return score_patients(entities, patient_entity_map, top_n=top_n)

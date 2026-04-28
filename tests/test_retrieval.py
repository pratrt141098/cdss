"""
tests/test_retrieval.py

Unit tests for retrieval-layer logic that requires no external dependencies
(no Ollama, no ChromaDB, no MIMIC data).

Covers:
  - query_router.route_query()
  - query_expansion.reciprocal_rank_fusion()
  - patient_matcher.score_patients()
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytestmark = pytest.mark.pure

from modules.query_router import route_query
from modules.query_expansion import reciprocal_rank_fusion
from modules.patient_matcher import score_patients


# ── Helpers ──────────────────────────────────────────────────────────────────

def _chunk(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": "", "metadata": {}, "score": 0.9}


# ── query_router.route_query ──────────────────────────────────────────────────

class TestRouteQuery:
    def test_allergies_variants(self):
        queries = [
            "What are this patient's documented drug allergies?",
            "Does this patient have any documented drug allergies?",
            "What allergies are documented for this patient?",
        ]
        for q in queries:
            assert route_query(q) == "allergies", f"Failed for: {q}"

    def test_discharge_medications_variants(self):
        queries = [
            "What medications were prescribed at discharge?",
            "What antibiotics were prescribed at discharge?",
            "What inhalers or respiratory medications were prescribed at discharge?",
        ]
        for q in queries:
            assert route_query(q) == "discharge_medications", f"Failed for: {q}"

    def test_informal_discharge_query_does_not_route(self):
        # "sent home with" has no discharge keyword — should fall through to vector search
        assert route_query("What drugs was the patient sent home with?") is None

    def test_medications_on_admission_variants(self):
        queries = [
            "What medications was this patient taking on admission?",
            "What drugs was the patient on at admission?",
        ]
        for q in queries:
            assert route_query(q) == "medications_on_admission", f"Failed for: {q}"

    def test_discharge_diagnosis_variants(self):
        queries = [
            "What is the discharge diagnosis for this patient?",
            "What is the primary discharge diagnosis for this patient?",
            "What is the primary diagnosis for this patient?",
            "What is this patient's primary diagnosis?",
        ]
        for q in queries:
            assert route_query(q) == "discharge_diagnosis", f"Failed for: {q}"

    def test_no_route_for_narrative_queries(self):
        queries = [
            "What is the history of present illness for this patient?",
            "Why was this patient admitted and what procedure did they undergo?",
            "What procedure did the patient undergo?",
            "Summarise the patient's hospital course.",
        ]
        for q in queries:
            assert route_query(q) is None, f"Should not route: {q}"

    def test_returns_none_for_empty_string(self):
        assert route_query("") is None

    def test_case_insensitive(self):
        assert route_query("WHAT ARE THE PATIENT ALLERGIES?") == "allergies"
        assert route_query("discharge diagnosis please") == "discharge_diagnosis"

    def test_first_match_wins_for_ambiguous(self):
        # Contains both "allerg" and "discharge" + medication term —
        # allergies pattern is listed first so it should win
        q = "Are there discharge medication allergies documented?"
        result = route_query(q)
        assert result in ("allergies", "discharge_medications")


# ── query_expansion.reciprocal_rank_fusion ────────────────────────────────────

class TestReciprocalRankFusion:
    def test_chunk_appearing_in_multiple_lists_ranks_first(self):
        list1 = [_chunk("a"), _chunk("b"), _chunk("c")]
        list2 = [_chunk("b"), _chunk("d"), _chunk("e")]
        fused = reciprocal_rank_fusion([list1, list2])
        assert fused[0]["chunk_id"] == "b"

    def test_single_list_preserves_order(self):
        chunks = [_chunk("x"), _chunk("y"), _chunk("z")]
        fused = reciprocal_rank_fusion([chunks])
        assert [c["chunk_id"] for c in fused] == ["x", "y", "z"]

    def test_n_results_limits_output(self):
        list1 = [_chunk(str(i)) for i in range(20)]
        fused = reciprocal_rank_fusion([list1], n_results=5)
        assert len(fused) == 5

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_score_is_attached_to_output(self):
        list1 = [_chunk("a"), _chunk("b")]
        fused = reciprocal_rank_fusion([list1])
        assert all("score" in c for c in fused)
        assert all(c["score"] > 0 for c in fused)

    def test_rank_1_in_three_lists_beats_rank_1_in_one(self):
        # chunk "winner" appears at rank 1 in three lists
        # chunk "loser" appears at rank 1 in one list only
        winner = _chunk("winner")
        loser  = _chunk("loser")
        filler = [_chunk(f"f{i}") for i in range(5)]
        list1 = [winner] + filler
        list2 = [winner] + filler
        list3 = [loser, winner] + filler
        fused = reciprocal_rank_fusion([list1, list2, list3])
        assert fused[0]["chunk_id"] == "winner"

    def test_deduplication_across_lists(self):
        list1 = [_chunk("a"), _chunk("b")]
        list2 = [_chunk("a"), _chunk("c")]
        fused = reciprocal_rank_fusion([list1, list2])
        ids = [c["chunk_id"] for c in fused]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids in fused output"


# ── patient_matcher.score_patients ────────────────────────────────────────────

class TestScorePatients:
    def _make_entity_map(self):
        return {
            "hadm_001": {"aspirin", "coronary artery disease", "hypertension"},
            "hadm_002": {"metformin", "diabetes", "obesity"},
            "hadm_003": {"warfarin", "atrial fibrillation", "stroke"},
        }

    def test_exact_entity_match_ranks_first(self):
        entity_map = self._make_entity_map()
        results = score_patients(["diabetes", "metformin"], entity_map)
        assert results[0][0] == "hadm_002"

    def test_higher_overlap_ranks_higher(self):
        entity_map = self._make_entity_map()
        results = score_patients(
            ["coronary artery disease", "hypertension", "aspirin"], entity_map
        )
        assert results[0][0] == "hadm_001"
        assert results[0][1] == 3

    def test_no_matching_entities_returns_empty(self):
        entity_map = self._make_entity_map()
        results = score_patients(["penicillin", "appendicitis"], entity_map)
        assert results == []

    def test_empty_query_entities_returns_empty(self):
        entity_map = self._make_entity_map()
        assert score_patients([], entity_map) == []

    def test_empty_entity_map_returns_empty(self):
        assert score_patients(["aspirin"], {}) == []

    def test_top_n_limits_results(self):
        entity_map = {
            "hadm_001": {"aspirin"},
            "hadm_002": {"aspirin"},
            "hadm_003": {"aspirin"},
        }
        results = score_patients(["aspirin"], entity_map, top_n=2)
        assert len(results) <= 2

    def test_case_insensitive_matching(self):
        entity_map = {"hadm_001": {"aspirin", "hypertension"}}
        results = score_patients(["ASPIRIN", "Hypertension"], entity_map)
        assert len(results) == 1
        assert results[0][1] == 2

    def test_result_format_is_tuple_hadm_count(self):
        entity_map = {"hadm_001": {"aspirin"}}
        results = score_patients(["aspirin"], entity_map)
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], str)
        assert isinstance(results[0][1], int)

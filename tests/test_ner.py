"""
tests/test_ner.py

Unit tests for NERModule and EmbeddingModule helper methods.
No spaCy model loaded, no Ollama calls — tests only the pure-logic helpers.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytestmark = pytest.mark.heavy

from dataclasses import dataclass, field
from modules.ner import NERModule, AnnotatedChunk, EMPTY_ENTITIES
from modules.embedding import EmbeddingModule


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_annotated_chunk(
    section: str,
    text: str,
    medications: list[str] | None = None,
    diseases: list[str] | None = None,
) -> AnnotatedChunk:
    return AnnotatedChunk(
        chunk_id=f"{section}_0",
        hadm_id=12345,
        patient_id=99,
        section=section,
        text=text,
        entities={
            "medications": medications or [],
            "diseases":    diseases    or [],
            "anatomy":     [],
            "procedures":  [],
        },
    )


# ── NERModule._extract_entities (section routing logic) ──────────────────────

class TestNERSectionRouting:
    """
    NERModule._extract_entities routes flat ENTITY labels by section name.
    We test the routing logic by mocking a spaCy doc with preset entities.
    """

    @pytest.fixture
    def ner(self):
        """Instantiate NERModule without loading spaCy (model not needed for these tests)."""
        import unittest.mock as mock
        with mock.patch("spacy.load", return_value=mock.MagicMock()):
            return NERModule()

    def _make_doc_with_entities(self, labels_texts: list[tuple[str, str]]):
        """Return a mock spaCy doc whose .ents list has the given (label, text) pairs."""
        import unittest.mock as mock
        doc = mock.MagicMock()
        ents = []
        for label, text in labels_texts:
            ent = mock.MagicMock()
            ent.label_ = label
            ent.text = text
            ents.append(ent)
        doc.ents = ents
        return doc

    def test_flat_entity_in_medication_section_goes_to_medications(self, ner):
        doc = self._make_doc_with_entities([("ENTITY", "Metformin 500mg")])
        result = ner._extract_entities(doc, "discharge_medications")
        assert "Metformin 500mg" in result["medications"]
        assert result["diseases"] == []

    def test_flat_entity_in_disease_section_goes_to_diseases(self, ner):
        doc = self._make_doc_with_entities([("ENTITY", "Type 2 diabetes")])
        result = ner._extract_entities(doc, "discharge_diagnosis")
        assert "Type 2 diabetes" in result["diseases"]
        assert result["medications"] == []

    def test_typed_chemical_label_always_goes_to_medications(self, ner):
        doc = self._make_doc_with_entities([("CHEMICAL", "Aspirin")])
        result = ner._extract_entities(doc, "history_of_present_illness")
        assert "Aspirin" in result["medications"]

    def test_typed_disease_label_always_goes_to_diseases(self, ner):
        doc = self._make_doc_with_entities([("DISEASE", "Heart failure")])
        result = ner._extract_entities(doc, "discharge_medications")
        assert "Heart failure" in result["diseases"]

    def test_short_entities_are_filtered(self, ner):
        doc = self._make_doc_with_entities([("ENTITY", "IV")])  # len < 4
        result = ner._extract_entities(doc, "discharge_medications")
        assert result["medications"] == []

    def test_noise_entities_are_filtered(self, ner):
        doc = self._make_doc_with_entities([("ENTITY", "no known")])
        result = ner._extract_entities(doc, "allergies")
        assert result["medications"] == []

    def test_duplicate_entities_deduplicated(self, ner):
        doc = self._make_doc_with_entities([
            ("ENTITY", "Aspirin"),
            ("ENTITY", "Aspirin"),
        ])
        result = ner._extract_entities(doc, "discharge_medications")
        assert result["medications"].count("Aspirin") == 1


# ── EmbeddingModule._build_embed_text ────────────────────────────────────────

class TestBuildEmbedText:
    @pytest.fixture
    def embedder(self):
        import unittest.mock as mock
        with mock.patch("ollama.embed"):
            return EmbeddingModule()

    def test_output_starts_with_search_document_prefix(self, embedder):
        chunk = _make_annotated_chunk("allergies", "Penicillin allergy documented.")
        text = embedder._build_embed_text(chunk)
        assert text.startswith("search_document:")

    def test_section_name_included(self, embedder):
        chunk = _make_annotated_chunk("discharge_diagnosis", "Coronary artery disease.")
        text = embedder._build_embed_text(chunk)
        assert "discharge_diagnosis" in text

    def test_entity_string_included_when_entities_present(self, embedder):
        chunk = _make_annotated_chunk(
            "discharge_medications", "Aspirin 81mg.", medications=["Aspirin"]
        )
        text = embedder._build_embed_text(chunk)
        assert "Aspirin" in text
        assert "Entities:" in text

    def test_no_entity_prefix_when_no_entities(self, embedder):
        chunk = _make_annotated_chunk("allergies", "No known drug allergies.")
        text = embedder._build_embed_text(chunk)
        assert "Entities:" not in text

    def test_chunk_text_always_included(self, embedder):
        chunk = _make_annotated_chunk("allergies", "Penicillin — hives.")
        text = embedder._build_embed_text(chunk)
        assert "Penicillin — hives." in text

    def test_entities_capped_at_ten(self, embedder):
        many_meds = [f"Drug{i}" for i in range(20)]
        chunk = _make_annotated_chunk("discharge_medications", "text", medications=many_meds)
        text = embedder._build_embed_text(chunk)
        # count how many Drug{i} appear
        included = [m for m in many_meds if m in text]
        assert len(included) <= 10

    def test_query_embedding_uses_search_query_prefix(self, embedder):
        """embed_query wraps the query with search_query: prefix."""
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        with pytest.raises(Exception):
            # Without a live Ollama call this will error — just check the prefix
            # is built correctly by inspecting the call argument
            pass

        import unittest.mock as mock
        with mock.patch("ollama.embed", return_value=mock_response) as mock_embed:
            embedder.embed_query("what medications?")
        called_input = mock_embed.call_args[1].get("input") or mock_embed.call_args[0][1]
        assert called_input.startswith("search_query:")

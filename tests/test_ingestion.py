"""
tests/test_ingestion.py

Unit tests for PreprocessingModule — section splitting and chunking logic.
No MIMIC data required: all tests use inline fixture strings that mimic
the structure of real MIMIC-IV discharge notes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytestmark = pytest.mark.heavy

from modules.preprocessing import PreprocessingModule
from modules.ingestion import PatientRecord


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_NOTE = """
Chief Complaint: shortness of breath

History of Present Illness:
Patient is a year old female with known heart failure presenting with
worsening dyspnea and orthopnea over the past 3 days. She denies
chest pain or palpitations.

Allergies:
Penicillin — hives
Sulfa drugs — rash

Medications on Admission:
Furosemide 40mg daily
Lisinopril 10mg daily
Metoprolol succinate 25mg daily

Discharge Medications:
Furosemide 80mg daily (increased)
Lisinopril 10mg daily
Metoprolol succinate 25mg daily
Spironolactone 25mg daily (new)

Discharge Diagnosis:
Acute decompensated heart failure
"""

NOTE_NO_SECTIONS = "Patient presented with chest pain. ECG showed ST changes. Admitted for observation."


def _make_record(note: str, hadm_id: str = "99999", patient_id: str = "11111") -> PatientRecord:
    return PatientRecord(
        hadm_id=hadm_id,
        patient_id=patient_id,
        admission_type="EMERGENCY",
        diagnoses=[],
        medications=[],
        discharge_notes=[note],
    )


@pytest.fixture
def preprocessor():
    return PreprocessingModule(config={"chunk_size": 50, "chunk_overlap": 10})


@pytest.fixture
def small_preprocessor():
    """Tiny chunk size to force multi-chunk splitting."""
    return PreprocessingModule(config={"chunk_size": 10, "chunk_overlap": 2})


# ── Section splitting ─────────────────────────────────────────────────────────

class TestSectionSplitting:
    def test_known_sections_are_extracted(self, preprocessor):
        sections = preprocessor._split_into_sections(SAMPLE_NOTE)
        assert "allergies" in sections
        assert "discharge_medications" in sections
        assert "discharge_diagnosis" in sections

    def test_section_content_is_correct(self, preprocessor):
        sections = preprocessor._split_into_sections(SAMPLE_NOTE)
        assert "Penicillin" in sections["allergies"]
        assert "Furosemide 80mg" in sections["discharge_medications"]

    def test_fallback_to_general_when_no_headers(self, preprocessor):
        sections = preprocessor._split_into_sections(NOTE_NO_SECTIONS)
        assert "general" in sections
        assert "chest pain" in sections["general"]

    def test_empty_sections_are_not_included(self, preprocessor):
        sections = preprocessor._split_into_sections(SAMPLE_NOTE)
        for name, text in sections.items():
            assert text.strip() != "", f"Section '{name}' is empty"

    def test_section_names_use_underscores(self, preprocessor):
        sections = preprocessor._split_into_sections(SAMPLE_NOTE)
        for name in sections:
            assert " " not in name, f"Section name has spaces: {name}"

    def test_section_names_are_lowercase(self, preprocessor):
        sections = preprocessor._split_into_sections(SAMPLE_NOTE)
        for name in sections:
            assert name == name.lower(), f"Section name not lowercase: {name}"


# ── Text chunking ─────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_text_produces_single_chunk(self, preprocessor):
        chunks = preprocessor._chunk_text("Aspirin 81mg daily.")
        assert len(chunks) == 1

    def test_empty_text_produces_no_chunks(self, preprocessor):
        assert preprocessor._chunk_text("") == []
        assert preprocessor._chunk_text("   ") == []

    def test_long_text_produces_multiple_chunks(self, small_preprocessor):
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = small_preprocessor._chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_overlap(self, small_preprocessor):
        words = [f"word{i}" for i in range(20)]
        text = " ".join(words)
        chunks = small_preprocessor._chunk_text(text)
        if len(chunks) > 1:
            # Last word of chunk N should appear in chunk N+1 due to overlap
            last_word_of_first = chunks[0].split()[-1]
            assert last_word_of_first in chunks[1]

    def test_chunk_size_is_respected(self, small_preprocessor):
        words = [f"word{i}" for i in range(50)]
        chunks = small_preprocessor._chunk_text(" ".join(words))
        for chunk in chunks:
            assert len(chunk.split()) <= small_preprocessor.chunk_size


# ── Text cleaning ─────────────────────────────────────────────────────────────

class TestCleanText:
    def test_multiple_newlines_collapsed(self, preprocessor):
        text = "line one\n\n\nline two"
        cleaned = preprocessor._clean_text(text)
        assert "\n\n" not in cleaned

    def test_multiple_spaces_collapsed(self, preprocessor):
        text = "word1    word2     word3"
        cleaned = preprocessor._clean_text(text)
        assert "  " not in cleaned

    def test_mimic_underline_separators_removed(self, preprocessor):
        text = "Header\n______________________\nContent"
        cleaned = preprocessor._clean_text(text)
        assert "__" not in cleaned


# ── Full process() integration ────────────────────────────────────────────────

class TestProcessIntegration:
    def test_produces_text_chunks(self, preprocessor):
        record = _make_record(SAMPLE_NOTE)
        chunks = preprocessor.process({"99999": record})
        assert len(chunks) > 0

    def test_all_chunks_have_correct_hadm_id(self, preprocessor):
        record = _make_record(SAMPLE_NOTE, hadm_id="77777")
        chunks = preprocessor.process({"77777": record})
        for chunk in chunks:
            assert chunk.hadm_id == "77777"

    def test_chunk_ids_are_unique(self, preprocessor):
        record = _make_record(SAMPLE_NOTE)
        chunks = preprocessor.process({"99999": record})
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_section_slugs_assigned_to_chunks(self, preprocessor):
        record = _make_record(SAMPLE_NOTE)
        chunks = preprocessor.process({"99999": record})
        sections_seen = {c.section for c in chunks}
        assert "allergies" in sections_seen
        assert "discharge_medications" in sections_seen

    def test_multiple_records_produce_chunks_for_each(self, preprocessor):
        records = {
            "hadm_001": _make_record(SAMPLE_NOTE, hadm_id="hadm_001"),
            "hadm_002": _make_record(SAMPLE_NOTE, hadm_id="hadm_002"),
        }
        chunks = preprocessor.process(records)
        hadm_ids_in_chunks = {c.hadm_id for c in chunks}
        assert "hadm_001" in hadm_ids_in_chunks
        assert "hadm_002" in hadm_ids_in_chunks

"""
tests/test_generation.py

Unit tests for GenerationModule helper methods.
No Ollama calls — tests only the prompt/context construction logic.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytestmark = pytest.mark.heavy

from unittest.mock import patch, MagicMock
from modules.generation import GenerationModule


@pytest.fixture
def generator():
    return GenerationModule()


def _make_chunk(section: str, text: str, score: float = 0.85) -> dict:
    return {
        "chunk_id": f"{section}_0",
        "text":     text,
        "metadata": {"section": section, "hadm_id": "12345"},
        "score":    score,
    }


# ── _build_context ────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_section_name_formatted_correctly(self, generator):
        chunk = _make_chunk("discharge_medications", "Aspirin 81mg daily.")
        ctx = generator._build_context([chunk])
        assert "Discharge Medications" in ctx

    def test_score_included_in_context(self, generator):
        chunk = _make_chunk("allergies", "Penicillin — hives.", score=0.92)
        ctx = generator._build_context([chunk])
        assert "0.92" in ctx

    def test_chunk_text_included(self, generator):
        chunk = _make_chunk("discharge_diagnosis", "Coronary artery disease.")
        ctx = generator._build_context([chunk])
        assert "Coronary artery disease." in ctx

    def test_multiple_chunks_numbered(self, generator):
        chunks = [
            _make_chunk("allergies",            "Penicillin."),
            _make_chunk("discharge_medications", "Aspirin 81mg."),
        ]
        ctx = generator._build_context(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_empty_chunks_returns_empty_string(self, generator):
        assert generator._build_context([]) == ""


# ── _build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_query_included_in_prompt(self, generator):
        prompt = generator._build_prompt("What medications is the patient on?", "context here")
        assert "What medications is the patient on?" in prompt

    def test_context_included_in_prompt(self, generator):
        prompt = generator._build_prompt("Any question?", "Aspirin 81mg daily.")
        assert "Aspirin 81mg daily." in prompt

    def test_prompt_instructs_grounded_answer(self, generator):
        prompt = generator._build_prompt("q", "ctx")
        assert "ONLY" in prompt or "only" in prompt

    def test_prompt_mentions_deidentification(self, generator):
        prompt = generator._build_prompt("q", "ctx")
        assert "de-identif" in prompt.lower()

    def test_prompt_has_answer_marker(self, generator):
        prompt = generator._build_prompt("q", "ctx")
        assert "ANSWER:" in prompt


# ── GenerationModule.process (mocked) ────────────────────────────────────────

class TestProcessMocked:
    def test_process_returns_string(self, generator):
        mock_response = {"message": {"content": "Patient is on aspirin 81mg daily."}}
        with patch("modules.generation.ollama.chat", return_value=mock_response):
            result = generator.process({
                "query":          "What medications?",
                "context_chunks": [_make_chunk("discharge_medications", "Aspirin 81mg.")],
            })
        assert isinstance(result, str)
        assert result == "Patient is on aspirin 81mg daily."

    def test_process_passes_correct_model(self, generator):
        mock_response = {"message": {"content": "answer"}}
        with patch("modules.generation.ollama.chat", return_value=mock_response) as mock_chat:
            generator.process({
                "query":          "q",
                "context_chunks": [],
            })
        call_kwargs = mock_chat.call_args
        assert call_kwargs[1]["model"] == "llama3.2:3b" or call_kwargs[0][0] == "llama3.2:3b" or \
               call_kwargs.kwargs.get("model") == "llama3.2:3b"

    def test_stream_yields_tokens(self, generator):
        mock_chunks = [
            {"message": {"content": "The "}},
            {"message": {"content": "patient "}},
            {"message": {"content": "has aspirin."}},
        ]
        with patch("modules.generation.ollama.chat", return_value=iter(mock_chunks)):
            tokens = list(generator.stream({
                "query":          "What medications?",
                "context_chunks": [],
            }))
        assert tokens == ["The ", "patient ", "has aspirin."]

    def test_stream_skips_empty_tokens(self, generator):
        mock_chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": ""}},
            {"message": {"content": " world"}},
        ]
        with patch("modules.generation.ollama.chat", return_value=iter(mock_chunks)):
            tokens = list(generator.stream({
                "query":          "q",
                "context_chunks": [],
            }))
        assert "" not in tokens
        assert tokens == ["Hello", " world"]

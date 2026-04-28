# tests/conftest.py
#
# pytest configuration for the CDSS test suite.
#
# Markers:
#   pure   — no external deps (no Ollama, spaCy, pandas, ChromaDB). Runs anywhere.
#   heavy  — imports spaCy or pandas; requires the full cdss conda environment.
#            May segfault on macOS ARM with mismatched numpy builds.
#
# Run only pure tests:
#   pytest tests/ -m pure
#
# Run everything:
#   pytest tests/

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "pure: no heavy ML/data dependencies")
    config.addinivalue_line("markers", "heavy: requires spaCy / pandas / full env")

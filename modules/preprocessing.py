# modules/preprocessing.py

import re
from dataclasses import dataclass
from typing import Optional
from modules.base import BaseModule
from modules.ingestion import PatientRecord
from config.settings import settings


@dataclass
class TextChunk:
    hadm_id: str
    patient_id: str
    chunk_id: str
    text: str
    section: str
    chunk_index: int


# MIMIC-IV discharge notes have consistent section headers
SECTION_PATTERNS = [
    "chief complaint",
    "history of present illness",
    "past medical history",
    "medications on admission",
    "discharge medications",
    "allergies",
    "physical exam",
    "pertinent results",
    "assessment and plan",
    "discharge diagnosis",
    "discharge condition",
    "discharge instructions",
    "followup instructions",
    "social history",
    "family history",
    "review of systems",
]


class PreprocessingModule(BaseModule):
    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", settings.chunk_size)
        self.chunk_overlap = self.config.get("chunk_overlap", settings.chunk_overlap)

    def process(self, input_data: dict[str, PatientRecord]) -> list[TextChunk]:
        """
        Takes a dict of hadm_id -> PatientRecord, returns a flat list of TextChunks.
        """
        all_chunks = []

        for hadm_id, record in input_data.items():
            for note_text in record.discharge_notes:
                sections = self._split_into_sections(note_text)
                for section_name, section_text in sections.items():
                    chunks = self._chunk_text(section_text)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(TextChunk(
                            hadm_id=hadm_id,
                            patient_id=record.patient_id,
                            chunk_id=f"{hadm_id}_{section_name}_{i}",
                            text=chunk,
                            section=section_name,
                            chunk_index=i,
                        ))

        print(f"[{self.name}] Produced {len(all_chunks)} chunks from {len(input_data)} admissions")
        return all_chunks

    def _split_into_sections(self, text: str) -> dict[str, str]:
        """
        Splits a clinical note into sections based on known MIMIC-IV headers.
        Falls back to 'general' if no headers are found.
        """
        pattern = r'(' + '|'.join(
            re.escape(s) for s in SECTION_PATTERNS
        ) + r')[\s]*:'

        splits = re.split(pattern, text, flags=re.IGNORECASE)

        if len(splits) <= 1:
            return {"general": self._clean_text(text)}

        sections = {}
        # splits alternates: [pre-text, header, content, header, content ...]
        i = 1
        while i < len(splits) - 1:
            section_name = splits[i].strip().lower().replace(" ", "_")
            section_text = self._clean_text(splits[i + 1])
            if section_text:
                sections[section_name] = section_text
            i += 2

        return sections if sections else {"general": self._clean_text(text)}

    def _chunk_text(self, text: str) -> list[str]:
        """
        Splits text into overlapping word-level chunks.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'_{2,}', '', text)   # remove underline separators common in MIMIC
        return text.strip()
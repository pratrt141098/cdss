# eval/generation_eval.py
#
# Evaluates generation quality for:
#   1. RAG pipeline  — retrieve top-k chunks scoped to query.hadm_id, then generate
#   2. No-RAG baseline — same query sent to llama3.2:3b with no context
#
# Metrics:
#   • ROUGE-1, ROUGE-2, ROUGE-L  (rouge_score)
#   • BERTScore F1                (bert_score)
#   • Faithfulness score          — fraction of response sentences that contain
#                                   at least one *content* token from the retrieved
#                                   chunks (stopwords excluded)
#
# Reference text for ROUGE / BERTScore:
#   The actual text of the ground-truth chunk (hadm_id + expected_section) looked
#   up directly from ChromaDB. Falls back to keywords if no chunk is found.

from __future__ import annotations

import re
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ollama
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

import chromadb
from chromadb.config import Settings

from modules.embedding import EmbeddingModule
from eval.query_set import QUERY_SET, EvalQuery
from config.settings import settings

TOP_K = 5
COLLECTION_NAME = "cdss_147"
GEN_MODEL = "llama3.2:3b"
FAITHFULNESS_THRESHOLD = 0.5

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@dataclass
class GenerationResult:
    query_id: str
    query: str
    hadm_id: str
    reference: str          # ground-truth chunk text (or keyword fallback)
    reference_source: str   # "chunk" | "keywords_fallback"

    rag_response: str = ""
    rag_rouge1: float = 0.0
    rag_rouge2: float = 0.0
    rag_rougeL: float = 0.0
    rag_bertscore: float = 0.0
    rag_faithfulness: float = 0.0
    rag_faithfulness_flag: bool = False   # True if score < threshold

    norag_response: str = ""
    norag_rouge1: float = 0.0
    norag_rouge2: float = 0.0
    norag_rougeL: float = 0.0
    norag_bertscore: float = 0.0

    retrieved_chunks: list[dict] = field(default_factory=list)


class GenerationEvaluator:
    def __init__(self, collection_name: str = COLLECTION_NAME, top_k: int = TOP_K):
        self.top_k = top_k
        self.embedder = EmbeddingModule()

        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)
        print(f"[GenerationEvaluator] Connected to '{collection_name}' "
              f"({self.collection.count()} docs)")

        # Index all chunks by (hadm_id, section) for ground-truth lookup
        raw = self.collection.get(
            include=["documents", "metadatas"],
            limit=self.collection.count(),
        )
        self._chunk_index: dict[tuple[str, str], list[str]] = {}
        for doc, meta in zip(raw["documents"], raw["metadatas"]):
            key = (str(meta["hadm_id"]), meta["section"])
            self._chunk_index.setdefault(key, []).append(doc)
        print(f"[GenerationEvaluator] Indexed {len(self._chunk_index)} "
              f"(hadm_id, section) pairs for reference lookup")

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, queries: list[EvalQuery] | None = None) -> list[GenerationResult]:
        queries = queries or QUERY_SET
        results: list[GenerationResult] = []

        print(f"[GenerationEvaluator] Running {len(queries)} queries "
              f"(RAG + no-RAG)...")

        for q in queries:
            print(f"  {q.query_id}: {q.query[:60]}...")
            reference, ref_source = self._lookup_reference(q)

            # Retrieve
            chunks = self._retrieve(q)

            # Generate
            rag_resp   = self._generate_rag(q.query, chunks)
            norag_resp = self._generate_norag(q.query)

            # Score
            rag_rouge   = self._rouge(rag_resp, reference)
            norag_rouge = self._rouge(norag_resp, reference)

            faith = self._faithfulness(rag_resp, chunks)

            result = GenerationResult(
                query_id=q.query_id,
                query=q.query,
                hadm_id=q.hadm_id,
                reference=reference,
                reference_source=ref_source,
                rag_response=rag_resp,
                rag_rouge1=rag_rouge["rouge1"],
                rag_rouge2=rag_rouge["rouge2"],
                rag_rougeL=rag_rouge["rougeL"],
                rag_faithfulness=faith,
                rag_faithfulness_flag=faith < FAITHFULNESS_THRESHOLD,
                norag_response=norag_resp,
                norag_rouge1=norag_rouge["rouge1"],
                norag_rouge2=norag_rouge["rouge2"],
                norag_rougeL=norag_rouge["rougeL"],
                retrieved_chunks=chunks,
            )
            results.append(result)

        # BERTScore is expensive — batch all at once
        print("[GenerationEvaluator] Computing BERTScore (batched)...")
        results = self._attach_bertscores(results)

        return results

    @staticmethod
    def aggregate(results: list[GenerationResult]) -> dict:
        n = len(results)
        if n == 0:
            return {}

        def mean(vals):
            return sum(vals) / n

        flagged = [r.query_id for r in results if r.rag_faithfulness_flag]

        return {
            "n_queries": n,
            "rag_mean_rouge1":       mean(r.rag_rouge1       for r in results),
            "rag_mean_rouge2":       mean(r.rag_rouge2       for r in results),
            "rag_mean_rougeL":       mean(r.rag_rougeL       for r in results),
            "rag_mean_bertscore":    mean(r.rag_bertscore    for r in results),
            "rag_mean_faithfulness": mean(r.rag_faithfulness for r in results),
            "norag_mean_rouge1":     mean(r.norag_rouge1     for r in results),
            "norag_mean_rouge2":     mean(r.norag_rouge2     for r in results),
            "norag_mean_rougeL":     mean(r.norag_rougeL     for r in results),
            "norag_mean_bertscore":  mean(r.norag_bertscore  for r in results),
            "low_faithfulness_queries": flagged,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _retrieve(self, q: EvalQuery) -> list[dict]:
        vec = self.embedder.embed_query(q.query)
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=self.top_k,
            where={"hadm_id": q.hadm_id},
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": res["ids"][0][i],
                "text":     res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score":    1 - res["distances"][0][i],
            }
            for i in range(len(res["ids"][0]))
        ]

    def _lookup_reference(self, q: EvalQuery) -> tuple[str, str]:
        """
        Return (reference_text, source) where source is 'chunk' or
        'keywords_fallback'. Joins all chunks for the target section so
        multi-chunk sections are fully represented.
        """
        key = (str(q.hadm_id), q.expected_section)
        texts = self._chunk_index.get(key)
        if texts:
            return " ".join(texts), "chunk"
        return " ".join(q.expected_answer_keywords), "keywords_fallback"

    def _generate_rag(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            section = c["metadata"].get("section", "unknown").replace("_", " ").title()
            context_parts.append(
                f"[{i}] Section: {section} (score: {c['score']:.2f})\n{c['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            "You are a clinical decision support assistant. "
            "Use ONLY the patient records below to answer. "
            "Important: these records are from a de-identified dataset. "
            "Patient ages, names, and dates have been removed and may appear as blank spaces "
            "or truncated phrases (e.g. 'year old female'). "
            "Treat such fragments as valid clinical content and answer from whatever information is present. "
            "Only say 'This information is not available in the provided records.' "
            "if the topic is genuinely absent — not because a sentence looks incomplete.\n\n"
            f"PATIENT RECORDS:\n{context}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]

    def _generate_norag(self, query: str) -> str:
        prompt = (
            "You are a clinical decision support assistant with broad medical knowledge. "
            "You have NOT been given any patient records for this query. "
            "Answer the question using only your general medical training knowledge. "
            "Do NOT refuse to answer — provide a clinically plausible response based on "
            "typical presentations, standard medications, or common diagnoses as applicable. "
            "Acknowledge at the end that this is a general answer not derived from patient records.\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]

    @staticmethod
    def _rouge(hypothesis: str, reference: str) -> dict[str, float]:
        scores = _ROUGE.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    # Generic clinical/English words that appear in nearly every note and
    # would trivially inflate faithfulness if counted as "content tokens".
    _STOPWORDS: frozenset[str] = frozenset({
        "the", "and", "for", "with", "this", "that", "from", "were", "have",
        "been", "will", "was", "are", "has", "not", "but", "also", "which",
        "patient", "patients", "history", "discharge", "admission", "hospital",
        "medical", "clinical", "note", "notes", "please", "follow", "given",
        "started", "continued", "taken", "per", "daily", "dose", "tablet",
        "capsule", "oral", "intravenous", "pain", "blood", "pressure", "heart",
        "rate", "normal", "significant", "relevant", "following", "assessed",
        "section", "information", "records", "provided", "available",
    })

    @classmethod
    def _faithfulness(cls, response: str, chunks: list[dict]) -> float:
        """
        Fraction of response sentences that contain at least one *content*
        token from the retrieved chunks, where content tokens are words that:
          - appear in the retrieved chunk text
          - are 4+ characters
          - are NOT in the clinical/generic stopword list
        This prevents common clinical prose from inflating the score.
        """
        if not chunks:
            return 0.0

        chunk_text = " ".join(c["text"] for c in chunks).lower()
        chunk_tokens = {
            tok for tok in re.findall(r"\b[a-z]{4,}\b", chunk_text)
            if tok not in cls._STOPWORDS
        }

        if not chunk_tokens:
            return 0.0

        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        if not sentences:
            return 0.0

        grounded = sum(
            1
            for sent in sentences
            if any(
                tok in chunk_tokens
                for tok in re.findall(r"\b[a-z]{4,}\b", sent.lower())
                if tok not in cls._STOPWORDS
            )
        )
        return grounded / len(sentences)

    @staticmethod
    def _attach_bertscores(results: list[GenerationResult]) -> list[GenerationResult]:
        references  = [r.reference      for r in results]
        rag_hyps    = [r.rag_response    for r in results]
        norag_hyps  = [r.norag_response  for r in results]

        _, _, rag_f1 = bert_score_fn(
            rag_hyps, references, lang="en", verbose=False
        )
        _, _, norag_f1 = bert_score_fn(
            norag_hyps, references, lang="en", verbose=False
        )

        for r, rf, nf in zip(results, rag_f1.tolist(), norag_f1.tolist()):
            r.rag_bertscore   = rf
            r.norag_bertscore = nf

        return results

# eval/retrieval_eval.py
#
# Evaluates retrieval quality for three strategies:
#   1. RAG global   — vector search across all chunks (no hadm_id filter)
#   2. RAG scoped   — vector search filtered to query.hadm_id (mirrors app "Single patient" mode)
#   3. Keyword      — keyword hit-count ranking across all chunks (global)
#
# True positive definition:
#   retrieved chunk.hadm_id == query.hadm_id
#   AND chunk.section      == query.expected_section
#
# Metrics computed per query and aggregated:
#   precision = tp / retrieved_k
#   recall    = tp / total_relevant_in_collection

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from chromadb.config import Settings

from modules.embedding import EmbeddingModule
from modules.query_router import route_query
from modules.vector_store import VectorStoreModule
from eval.query_set import QUERY_SET, EvalQuery
from config.settings import settings

TOP_K = 10
COLLECTION_NAME = "cdss_147"
HYBRID_ALPHAS = [0.3, 0.5, 0.7]


@dataclass
class RetrievalResult:
    query_id: str
    query: str
    hadm_id: str
    expected_section: str

    rag_retrieved: list[dict] = field(default_factory=list)
    rag_tp: int = 0
    rag_precision: float = 0.0
    rag_recall: float = 0.0

    scoped_retrieved: list[dict] = field(default_factory=list)
    scoped_tp: int = 0
    scoped_precision: float = 0.0
    scoped_recall: float = 0.0

    kw_retrieved: list[dict] = field(default_factory=list)
    kw_tp: int = 0
    kw_precision: float = 0.0
    kw_recall: float = 0.0

    routed_retrieved: list[dict] = field(default_factory=list)
    routed_tp: int = 0
    routed_precision: float = 0.0
    routed_recall: float = 0.0
    routed_section: str | None = None

    # hybrid results keyed by alpha value e.g. {0.3: [...], 0.5: [...], 0.7: [...]}
    hybrid_by_alpha: dict = field(default_factory=dict)

    fusion_retrieved: list[dict] = field(default_factory=list)
    fusion_tp: int = 0
    fusion_precision: float = 0.0
    fusion_recall: float = 0.0

    total_relevant: int = 0


class RetrievalEvaluator:
    def __init__(self, collection_name: str = COLLECTION_NAME, top_k: int = TOP_K):
        self.top_k = top_k
        self.embedder = EmbeddingModule()

        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)
        self.store = VectorStoreModule(config={
            "persist_dir":       settings.chroma_persist_dir,
            "collection_name":   collection_name,
        })
        print(f"[RetrievalEvaluator] Connected to '{collection_name}' "
              f"({self.collection.count()} docs)")

        # Cache all chunks for the keyword baseline (avoid repeated fetches)
        print("[RetrievalEvaluator] Fetching all chunks for keyword baseline...")
        raw = self.collection.get(
            include=["documents", "metadatas"],
            limit=self.collection.count(),
        )
        self._all_chunks: list[dict] = [
            {
                "chunk_id": cid,
                "text":     doc,
                "metadata": meta,
            }
            for cid, doc, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        ]
        print(f"[RetrievalEvaluator] Cached {len(self._all_chunks)} chunks")

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, queries: list[EvalQuery] | None = None) -> list[RetrievalResult]:
        queries = queries or QUERY_SET
        results: list[RetrievalResult] = []

        for q in queries:
            total_relevant = self._count_relevant(q)
            query_vec      = self.embedder.embed_query(q.query)  # embed once, reuse everywhere
            rag_chunks     = self._rag_retrieve(q, query_vec, hadm_filter=None)
            scoped_chunks  = self._rag_retrieve(q, query_vec, hadm_filter=q.hadm_id)
            kw_chunks      = self._keyword_retrieve(q)
            routed_section, routed_chunks = self._routed_retrieve(q)

            fusion_chunks  = self._fusion_retrieve(q, query_vec)

            rag_tp    = sum(1 for c in rag_chunks     if self._is_tp(c, q))
            scoped_tp = sum(1 for c in scoped_chunks  if self._is_tp(c, q))
            kw_tp     = sum(1 for c in kw_chunks      if self._is_tp(c, q))
            routed_tp = sum(1 for c in routed_chunks  if self._is_tp(c, q))
            fusion_tp = sum(1 for c in fusion_chunks  if self._is_tp(c, q))
            hybrid_by_alpha: dict = {}
            for alpha in HYBRID_ALPHAS:
                h_chunks = self.store.hybrid_query(
                    query_embedding=query_vec,
                    keywords=q.expected_answer_keywords,
                    n_results=self.top_k,
                    hadm_id=q.hadm_id,
                    alpha=alpha,
                )
                h_tp = sum(1 for c in h_chunks if self._is_tp(c, q))
                hybrid_by_alpha[alpha] = {
                    "chunks":    h_chunks,
                    "tp":        h_tp,
                    "precision": h_tp / self.top_k if self.top_k else 0.0,
                    "recall":    h_tp / total_relevant if total_relevant else 0.0,
                }

            n_routed = len(routed_chunks) if routed_chunks else self.top_k

            results.append(RetrievalResult(
                query_id=q.query_id,
                query=q.query,
                hadm_id=q.hadm_id,
                expected_section=q.expected_section,

                rag_retrieved=rag_chunks,
                rag_tp=rag_tp,
                rag_precision=rag_tp / self.top_k if self.top_k else 0.0,
                rag_recall=rag_tp / total_relevant if total_relevant else 0.0,

                scoped_retrieved=scoped_chunks,
                scoped_tp=scoped_tp,
                scoped_precision=scoped_tp / self.top_k if self.top_k else 0.0,
                scoped_recall=scoped_tp / total_relevant if total_relevant else 0.0,

                kw_retrieved=kw_chunks,
                kw_tp=kw_tp,
                kw_precision=kw_tp / self.top_k if self.top_k else 0.0,
                kw_recall=kw_tp / total_relevant if total_relevant else 0.0,

                routed_retrieved=routed_chunks,
                routed_tp=routed_tp,
                routed_precision=routed_tp / n_routed if n_routed else 0.0,
                routed_recall=routed_tp / total_relevant if total_relevant else 0.0,
                routed_section=routed_section,
                hybrid_by_alpha=hybrid_by_alpha,

                fusion_retrieved=fusion_chunks,
                fusion_tp=fusion_tp,
                fusion_precision=fusion_tp / self.top_k if self.top_k else 0.0,
                fusion_recall=fusion_tp / total_relevant if total_relevant else 0.0,

                total_relevant=total_relevant,
            ))

            hybrid_str = "  ".join(
                f"α={a} R={hybrid_by_alpha[a]['recall']:.2f}"
                for a in HYBRID_ALPHAS
            )
            print(
                f"  {q.query_id} | "
                f"RAG P={results[-1].rag_precision:.2f} R={results[-1].rag_recall:.2f} | "
                f"Scoped P={results[-1].scoped_precision:.2f} R={results[-1].scoped_recall:.2f} | "
                f"KW P={results[-1].kw_precision:.2f} R={results[-1].kw_recall:.2f} | "
                f"Routed({'→' + routed_section if routed_section else 'n/a'}) "
                f"P={results[-1].routed_precision:.2f} R={results[-1].routed_recall:.2f} | "
                f"Hybrid [{hybrid_str}] | "
                f"Fusion P={results[-1].fusion_precision:.2f} R={results[-1].fusion_recall:.2f} | "
                f"relevant={total_relevant}"
            )

        return results

    @staticmethod
    def aggregate(results: list[RetrievalResult]) -> dict:
        n = len(results)
        if n == 0:
            return {}
        return {
            "n_queries": n,
            "rag_mean_precision":       sum(r.rag_precision     for r in results) / n,
            "rag_mean_recall":          sum(r.rag_recall        for r in results) / n,
            "scoped_mean_precision":    sum(r.scoped_precision  for r in results) / n,
            "scoped_mean_recall":       sum(r.scoped_recall     for r in results) / n,
            "kw_mean_precision":        sum(r.kw_precision      for r in results) / n,
            "kw_mean_recall":           sum(r.kw_recall         for r in results) / n,
            "routed_mean_precision":    sum(r.routed_precision  for r in results) / n,
            "routed_mean_recall":       sum(r.routed_recall     for r in results) / n,
            "routed_coverage":          sum(1 for r in results if r.routed_section) / n,
            **{
                f"hybrid_alpha{a}_mean_precision": sum(
                    r.hybrid_by_alpha[a]["precision"] for r in results if a in r.hybrid_by_alpha
                ) / n
                for a in HYBRID_ALPHAS
            },
            **{
                f"hybrid_alpha{a}_mean_recall": sum(
                    r.hybrid_by_alpha[a]["recall"] for r in results if a in r.hybrid_by_alpha
                ) / n
                for a in HYBRID_ALPHAS
            },
            "fusion_mean_precision":    sum(r.fusion_precision for r in results) / n,
            "fusion_mean_recall":       sum(r.fusion_recall    for r in results) / n,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _rag_retrieve(self, q: EvalQuery, query_vec: list[float], hadm_filter: str | None = None) -> list[dict]:
        where = {"hadm_id": hadm_filter} if hadm_filter else None
        res   = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
            where=where,
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

    def _fusion_retrieve(self, q: EvalQuery, query_vec: list[float]) -> list[dict]:
        """
        RAG-Fusion (unscoped): retrieve using the original query vector plus
        LLM-generated variants, merge with RRF. No hadm_id filter — this is
        the cross-patient retrieval challenge.
        """
        from modules.query_expansion import generate_query_variants, reciprocal_rank_fusion

        variants    = generate_query_variants(q.query)
        ranked_lists = [self.store.query(query_vec, n_results=self.top_k)]  # original
        for variant in variants[1:]:                                          # LLM variants
            vec = self.embedder.embed_query(variant)
            ranked_lists.append(self.store.query(vec, n_results=self.top_k))

        return reciprocal_rank_fusion(ranked_lists, n_results=self.top_k)

    def _routed_retrieve(self, q: EvalQuery) -> tuple[str | None, list[dict]]:
        """
        If the query maps to a known section slug, fetch all matching chunks
        directly by metadata (deterministic, no embedding). Returns (section, chunks).
        Returns (None, []) if no route matches.
        """
        section = route_query(q.query)
        if not section:
            return None, []

        raw = self.collection.get(
            where={"$and": [{"hadm_id": q.hadm_id}, {"section": section}]},
            include=["documents", "metadatas"],
        )
        chunks = [
            {
                "chunk_id": cid,
                "text":     doc,
                "metadata": meta,
                "score":    1.0,
            }
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        return section, chunks

    def _keyword_retrieve(self, q: EvalQuery) -> list[dict]:
        """Rank all chunks by keyword hit count, return top-k."""
        keywords = [kw.lower() for kw in q.expected_answer_keywords]

        scored: list[tuple[int, dict]] = []
        for chunk in self._all_chunks:
            text_lower = chunk["text"].lower()
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                scored.append((hits, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self.top_k]]

    def _is_tp(self, chunk: dict, q: EvalQuery) -> bool:
        meta = chunk["metadata"]
        return (
            str(meta.get("hadm_id")) == str(q.hadm_id)
            and meta.get("section") == q.expected_section
        )

    def _count_relevant(self, q: EvalQuery) -> int:
        """Count ground-truth relevant chunks already in the collection."""
        return sum(
            1 for c in self._all_chunks
            if str(c["metadata"].get("hadm_id")) == str(q.hadm_id)
            and c["metadata"].get("section") == q.expected_section
        )

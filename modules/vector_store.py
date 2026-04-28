# modules/vector_store.py
import chromadb
from chromadb.config import Settings
from modules.base import BaseModule


class VectorStoreModule(BaseModule):
    """
    Stores embedded chunks in a persistent ChromaDB collection.
    Each chunk stored with its full metadata for retrieval.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        persist_dir     = self.config.get("persist_dir", "./chroma_db")
        collection_name = self.config.get("collection_name", "cdss_chunks")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[{self.name}] Collection '{collection_name}' — {self.collection.count()} existing docs")

    def process(self, input_data: list[dict]) -> int:
        """
        input_data: list of embedded chunk dicts from EmbeddingModule
        returns: total docs stored
        """
        ids         = []
        embeddings  = []
        documents   = []
        metadatas   = []

        for chunk in input_data:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append({
                "hadm_id":    str(chunk["hadm_id"]),
                "patient_id": str(chunk["patient_id"]),
                "section":    chunk["section"],
                "medications": ", ".join(chunk["entities"].get("medications", [])),
                "diseases":    ", ".join(chunk["entities"].get("diseases", [])),
                "anatomy":     ", ".join(chunk["entities"].get("anatomy", [])),
                "procedures":  ", ".join(chunk["entities"].get("procedures", [])),
            })

        # upsert so re-runs don't duplicate
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total = self.collection.count()
        print(f"[{self.name}] Stored {len(ids)} chunks — total in collection: {total}")
        return total

    def query(self, query_embedding: list[float], n_results: int = 10, hadm_id: str = None) -> list[dict]:
        """
        Retrieve top-k chunks by cosine similarity.
        Optionally filter by hadm_id to scope results to one patient.
        """
        where = {"hadm_id": hadm_id} if hadm_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # cosine similarity
            })
        return chunks

    def hybrid_query(
        self,
        query_embedding: list[float],
        keywords: list[str],
        n_results: int = 10,
        hadm_id: str | None = None,
        alpha: float = 0.5,
        overretrieve_k: int = 30,
    ) -> list[dict]:
        """
        Hybrid retrieval: retrieve a larger candidate set via vector search,
        then re-rank by blending cosine similarity with keyword hit score.

        final_score = alpha * vector_score + (1 - alpha) * norm_keyword_score

        alpha=1.0 → pure vector, alpha=0.0 → pure keyword.
        """
        where = {"hadm_id": hadm_id} if hadm_id else None
        fetch_k = min(overretrieve_k, self.collection.count())

        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        candidates = [
            {
                "chunk_id":    res["ids"][0][i],
                "text":        res["documents"][0][i],
                "metadata":    res["metadatas"][0][i],
                "vector_score": 1 - res["distances"][0][i],
            }
            for i in range(len(res["ids"][0]))
        ]

        kws = [kw.lower() for kw in keywords]
        for c in candidates:
            text_lower = c["text"].lower()
            c["kw_hits"] = sum(1 for kw in kws if kw in text_lower)

        max_hits = max((c["kw_hits"] for c in candidates), default=1) or 1
        for c in candidates:
            norm_kw = c["kw_hits"] / max_hits
            c["score"] = alpha * c["vector_score"] + (1 - alpha) * norm_kw

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:n_results]

    def get_patient_entity_map(self) -> dict[str, set[str]]:
        """
        Returns {hadm_id: set[str]} mapping every patient to the union of all
        clinical entities stored across their chunks (medications, diseases,
        anatomy, procedures). Used by the patient matcher for cross-patient
        entity-overlap scoring.
        """
        raw = self.collection.get(include=["metadatas"])
        patient_entities: dict[str, set[str]] = {}
        for meta in raw["metadatas"]:
            hadm_id = meta.get("hadm_id", "")
            if not hadm_id:
                continue
            if hadm_id not in patient_entities:
                patient_entities[hadm_id] = set()
            for field in ("medications", "diseases", "anatomy", "procedures"):
                for entity in meta.get(field, "").split(", "):
                    entity = entity.strip().lower()
                    if entity:
                        patient_entities[hadm_id].add(entity)
        return patient_entities

    def rag_fusion_query(
        self,
        query_variants: list[str],
        embedder,
        n_results: int = 10,
        hadm_id: str | None = None,
    ) -> list[dict]:
        """
        Run a separate vector search for each query variant, then merge the
        result lists using Reciprocal Rank Fusion.

        query_variants: list of query strings (original + rephrased variants)
        embedder: EmbeddingModule instance with .embed_query()
        """
        from modules.query_expansion import reciprocal_rank_fusion

        ranked_lists = []
        for variant in query_variants:
            vec = embedder.embed_query(variant)
            ranked_lists.append(self.query(vec, n_results=n_results, hadm_id=hadm_id))

        return reciprocal_rank_fusion(ranked_lists, n_results=n_results)

    def get_by_section(self, section: str, hadm_id: str) -> list[dict]:
        """
        Fetch all chunks for a given section and patient directly by metadata.
        Used by the section router as a deterministic fallback when the query
        maps to a structured section (e.g. allergies, discharge_medications).
        """
        raw = self.collection.get(
            where={"$and": [{"hadm_id": hadm_id}, {"section": section}]},
            include=["documents", "metadatas"],
        )
        return [
            {
                "chunk_id": cid,
                "text":     doc,
                "metadata": meta,
                "score":    1.0,
            }
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]

    def reset(self):
        self.collection.delete(where={"hadm_id": {"$ne": ""}})
        print(f"[{self.name}] Collection cleared")
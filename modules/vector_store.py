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

    def query(self, query_embedding: list[float], n_results: int = 5, hadm_id: str = None) -> list[dict]:
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

    def get_patient_entities(self, hadm_id: str) -> dict:
        """
        Aggregate all NER entities across every chunk for a patient.
        Uses ChromaDB get() (no embedding needed) with a metadata filter.
        """
        results = self.collection.get(
            where={"hadm_id": hadm_id},
            include=["metadatas"],
        )
        buckets: dict[str, set] = {
            "medications": set(),
            "diseases":    set(),
            "procedures":  set(),
            "anatomy":     set(),
        }
        for meta in results.get("metadatas") or []:
            for key in buckets:
                for item in meta.get(key, "").split(", "):
                    item = item.strip()
                    if item:
                        buckets[key].add(item)
        return {k: sorted(v) for k, v in buckets.items()}

    def reset(self):
        self.collection.delete(where={"hadm_id": {"$ne": ""}})
        print(f"[{self.name}] Collection cleared")
from sentence_transformers import SentenceTransformer


class EmbeddingModule:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    def process(self, chunks):
        texts = []
        items = []

        for chunk in chunks:
            texts.append(chunk.text)
            items.append(chunk)

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedded_chunks = []
        for chunk, embedding in zip(items, embeddings.tolist()):
            embedded_chunks.append({
                "chunk_id": chunk.chunk_id,
                "hadm_id": str(chunk.hadm_id),
                "patient_id": str(chunk.patient_id),
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "entities": getattr(chunk, "entities", {}),
                "embedding": embedding,
            })

        return embedded_chunks

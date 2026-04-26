from sentence_transformers import SentenceTransformer

class EmbeddingModule:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def process(self, records):
        texts = []
        ids = []

        for hadm_id, record in records.items():
            text = record.note if hasattr(record, "note") else str(record)
            texts.append(text)
            ids.append(hadm_id)

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return {
            "ids": ids,
            "texts": texts,
            "embeddings": embeddings.tolist()
        }
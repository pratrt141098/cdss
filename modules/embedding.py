# modules/embedding.py
import ollama
from modules.base import BaseModule
from modules.ner import AnnotatedChunk


class EmbeddingModule(BaseModule):
    """
    Embeds annotated chunks using nomic-embed-text via Ollama.
    Each chunk's text + entity context is embedded into a dense vector.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.model = self.config.get("embedding_model", "nomic-embed-text")

    def process(self, input_data: list) -> list[dict]:
        texts = [self._build_embed_text(chunk) for chunk in input_data]

        # batch embed all at once
        response = ollama.embed(model=self.model, input=texts)
        vectors  = response["embeddings"]

        results = []
        for i, (chunk, vector) in enumerate(zip(input_data, vectors)):
            results.append({
                "chunk_id":   chunk.chunk_id,
                "hadm_id":    chunk.hadm_id,
                "patient_id": chunk.patient_id,
                "section":    chunk.section,
                "text":       chunk.text,
                "entities":   chunk.entities,
                "embedding":  vector,
            })
            if i % 20 == 0:
                print(f"[{self.name}] Embedded {i}/{len(input_data)} chunks")

        print(f"[{self.name}] Done — {len(results)} embeddings")
        return results

    def embed_query(self, query: str) -> list[float]:
        response = ollama.embed(model=self.model, input=f"search_query: {query}")
        return response["embeddings"][0]

    def _build_embed_text(self, chunk: AnnotatedChunk) -> str:
        all_entities = (
            chunk.entities.get("medications", []) +
            chunk.entities.get("diseases", []) +
            chunk.entities.get("anatomy", []) +
            chunk.entities.get("procedures", [])
        )  # closing paren was missing
        entity_str = f"Entities: {', '.join(all_entities[:10])}. " if all_entities else ""
        return f"search_document: Section: {chunk.section}. {entity_str}{chunk.text}"
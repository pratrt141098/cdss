# modules/generation.py
import ollama
from modules.base import BaseModule


class GenerationModule(BaseModule):
    """
    Generates clinical answers from retrieved chunks using llama3.2:3b via Ollama.
    Strictly grounded — only uses provided context, no external knowledge.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.model = self.config.get("generation_model", "llama3.2:3b")

    def process(self, input_data: dict) -> str:
        """
        input_data: {
            "query":          str,
            "context_chunks": list[dict]  — from VectorStoreModule.query()
        }
        returns: str response
        """
        query   = input_data["query"]
        chunks  = input_data["context_chunks"]
        context = self._build_context(chunks)
        prompt  = self._build_prompt(query, context)

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta    = chunk["metadata"]
            section = meta.get("section", "unknown").replace("_", " ").title()
            score   = chunk.get("score", 0)
            parts.append(
                f"[{i}] Section: {section} (relevance: {score:.2f})\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a clinical decision support assistant helping clinicians quickly understand a patient's condition.

Use ONLY the patient records below to answer. Do not use external medical knowledge.
If the answer is not in the records, say: "This information is not available in the provided records."
Be concise and clinical in tone.

PATIENT RECORDS:
{context}

QUESTION: {query}

ANSWER:"""
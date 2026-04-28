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

    def _build_messages(self, input_data: dict) -> list[dict]:
        """
        Build the message list for the Ollama chat call.

        If input_data contains a non-empty "history" list of prior
        {"role": ..., "content": ...} turns, they are prepended before
        the current user prompt so the LLM has conversation context.
        """
        query   = input_data["query"]
        chunks  = input_data["context_chunks"]
        history = input_data.get("history", [])

        context = self._build_context(chunks)
        prompt  = self._build_prompt(query, context)

        return list(history) + [{"role": "user", "content": prompt}]

    def process(self, input_data: dict) -> str:
        """
        input_data: {
            "query":          str,
            "context_chunks": list[dict],
            "history":        list[dict]  — optional prior turns
        }
        returns: str response
        """
        response = ollama.chat(
            model=self.model,
            messages=self._build_messages(input_data),
        )
        return response["message"]["content"]

    def stream(self, input_data: dict):
        """
        Streaming variant of process(). Yields text tokens as they are generated
        so the caller can render them incrementally (e.g. via st.write_stream).
        Accepts the same optional "history" key as process().
        """
        for chunk in ollama.chat(
            model=self.model,
            messages=self._build_messages(input_data),
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                yield token

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
Important: these records are from a de-identified dataset. Patient ages, names, and dates have been removed and may appear as blank spaces or truncated phrases (e.g. "year old female"). Treat such fragments as valid clinical content and answer from whatever information is present.
Only say "This information is not available in the provided records." if the topic is genuinely absent — not because a sentence looks incomplete.
Be concise and clinical in tone.

PATIENT RECORDS:
{context}

QUESTION: {query}

ANSWER:"""
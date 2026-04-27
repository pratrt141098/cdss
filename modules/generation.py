from vllm import LLM, SamplingParams


class GenerationModule:
    def __init__(self):
        self.llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
        )
        self.params = SamplingParams(
            temperature=0.2,
            max_tokens=512,
        )

    def _build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        context_lines = []
        for i, chunk in enumerate(context_chunks, start=1):
            section = chunk.get("metadata", {}).get("section", "unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            context_lines.append(
                f"[{i}] section={section} score={score:.3f}\n{text}"
            )

        context = "\n\n".join(context_lines) if context_lines else "No context provided."
        return (
            "You are a clinical assistant. Answer the question only from the provided "
            "patient record context. If the answer is not supported by the context, "
            "say you cannot determine it from the available record.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def generate(self, prompt: str):
        outputs = self.llm.generate([prompt], self.params)
        return outputs[0].outputs[0].text

    def process(self, input_data: dict):
        query = input_data.get("query", "")
        context_chunks = input_data.get("context_chunks", [])
        prompt = self._build_prompt(query, context_chunks)
        return self.generate(prompt)

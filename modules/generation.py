from vllm import LLM, SamplingParams

class GenerationModule:
    def __init__(self):
        self.llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16"
        )
        self.params = SamplingParams(
            temperature=0.2,
            max_tokens=512
        )

    def generate(self, prompt):
        outputs = self.llm.generate([prompt], self.params)
        return outputs[0].outputs[0].text
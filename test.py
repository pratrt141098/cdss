# test.py
import time
from modules.ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.ner import NERModule



records   = DataIngestionModule().process(input_data=10)
chunks    = PreprocessingModule().process(records)

t0 = time.time()
annotated = NERModule().process(chunks)
print(f"NER took {time.time() - t0:.1f}s")

for c in annotated:
    if c.section.lower() in ("history_of_present_illness", "discharge_diagnosis", "past_medical_history"):
        if any(c.entities[k] for k in c.entities):
            print(f"\nSample annotated chunk:")
            print(f"  hadm_id:     {c.hadm_id}")
            print(f"  section:     {c.section}")
            print(f"  medications: {c.entities['medications'][:5]}")
            print(f"  diseases:    {c.entities['diseases'][:5]}")
            print(f"  anatomy:     {c.entities['anatomy'][:3]}")
            print(f"  text[:200]:  {c.text[:200]}")
            break


        # add this at the bottom of test.py temporarily
sections = set(c.section for c in annotated)
print("\nAll sections found:")
for s in sorted(sections):
    print(f"  '{s}'")

from modules.embedding import EmbeddingModule

embedded = EmbeddingModule().process(annotated)

print(f"\nSample embedding:")
print(f"  chunk_id:      {embedded[0]['chunk_id']}")
print(f"  section:       {embedded[0]['section']}")
print(f"  embedding dim: {len(embedded[0]['embedding'])}")
print(f"  embedding[:5]: {embedded[0]['embedding'][:5]}")

from modules.vector_store import VectorStoreModule

store = VectorStoreModule()
store.process(embedded)

# test a retrieval
from modules.embedding import EmbeddingModule
embedder = EmbeddingModule()
query_vec = embedder.embed_query("What are the patient's current medications?")
results = store.query(query_vec, n_results=3)

print(f"\nTop 3 retrieved chunks:")
for r in results:
    print(f"  [{r['score']:.3f}] {r['metadata']['section']} — {r['text'][:100]}")

from modules.generation import GenerationModule

generator = GenerationModule()
answer = generator.process({
    "query": "What medications is the patient currently on?",
    "context_chunks": results,
})
print(f"\nGenerated answer:\n{answer}")
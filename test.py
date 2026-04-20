from modules.ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule

records = DataIngestionModule().process(input_data=10)
chunks = PreprocessingModule().process(records)

print(f"Total chunks: {len(chunks)}")
print(f"\nSample chunk:")
print(f"  hadm_id:  {chunks[0].hadm_id}")
print(f"  section:  {chunks[0].section}")
print(f"  text[:150]: {chunks[0].text[:150]}")
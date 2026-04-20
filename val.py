# val.py
from modules.ingestion import DataIngestionModule

records = DataIngestionModule().process(input_data=10)

print(f"Total patients: {len(records)}")
print(f"Available hadm_ids: {list(records.keys())}")

# check if 22927623 exists
if "22927623" in records:
    patient = records["22927623"]
    print(f"\nPatient: {patient.hadm_id}")
    print(f"Notes count: {len(patient.discharge_notes)}")
    if patient.discharge_notes:
        print(f"\nFirst note (first 2000 chars):\n{patient.discharge_notes[0][:2000]}")
    else:
        print("No discharge notes attached to this patient")
else:
    print("hadm_id 22927623 not found in first 10 rows")
    print("Try increasing input_data limit or check which hadm_ids loaded")
import pandas as pd
from fastapi import APIRouter, HTTPException

from api import pipeline
from api.schemas import (
    AdmissionInfo, Demographics, Diagnosis, Medication, PatientEntities,
    PatientOverview, Procedure, SummaryResponse,
)

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _los(admit: str | None, discharge: str | None) -> float | None:
    try:
        a = pd.Timestamp(admit)
        d = pd.Timestamp(discharge)
        return round((d - a).total_seconds() / 86400, 1)
    except Exception:
        return None


def _summary_prompt(record) -> str:
    diag_lines = "\n".join(
        f"  {i+1}. {d['description']} (ICD-{d['icd_version']} {d['icd_code']})"
        for i, d in enumerate(record.diagnoses[:12])
    ) or "  (none recorded)"

    # Deduplicate medications by drug name for the prompt
    seen: set[str] = set()
    unique_meds = []
    for m in record.medications:
        if m["drug"] not in seen:
            seen.add(m["drug"])
            unique_meds.append(m)

    med_lines = "\n".join(
        f"  - {m['drug']}"
        + (f" {m['dose_val_rx']} {m['dose_unit_rx']}" if m.get("dose_val_rx") else "")
        + (f" ({m['route']})" if m.get("route") else "")
        for m in unique_meds[:15]
    ) or "  (none recorded)"

    proc_lines = "\n".join(
        f"  {i+1}. {p['description']}"
        for i, p in enumerate(record.procedures[:8])
    ) or "  (none recorded)"

    demo    = record.demographics
    gender  = demo.get("gender", "")
    age     = demo.get("anchor_age")
    pronoun = "she/her" if gender == "F" else "he/him" if gender == "M" else "they/them"
    demo_str = f"{gender or 'Unknown'}" + (f", {age} years old" if age else "")

    return f"""You are a clinical documentation assistant.

Generate a 3-sentence clinical summary of the following hospital admission for a reviewing physician.

Patient: {demo_str} (use pronouns: {pronoun})
Admission type: {record.admission_type or 'Unknown'}
Discharge to: {record.discharge_location or 'Unknown'}

Diagnoses (by priority sequence):
{diag_lines}

Medications prescribed:
{med_lines}

Procedures performed:
{proc_lines}

Instructions:
- Sentence 1: primary reason for admission and key diagnosis
- Sentence 2: main treatments (medications and procedures)
- Sentence 3: discharge disposition and clinical outcome implied

Write exactly 3 sentences. Be concise and clinical. No bullet points or headers."""


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/patients", response_model=list[str])
def list_patients() -> list[str]:
    """Return the sorted list of loaded HADM IDs."""
    return pipeline.state["hadm_ids"]


@router.get("/patients/{hadm_id}/overview", response_model=PatientOverview)
def get_overview(hadm_id: str) -> PatientOverview:
    """Return all structured data for a single admission."""
    records = pipeline.state.get("records", {})
    if hadm_id not in records:
        raise HTTPException(status_code=404, detail=f"HADM ID {hadm_id!r} not found.")

    record = records[hadm_id]
    store  = pipeline.state["store"]

    empty_entities = {"medications": [], "diseases": [], "anatomy": [], "procedures": []}
    entities = pipeline.state.get("ner_entities", {}).get(hadm_id, empty_entities)

    demo = record.demographics
    return PatientOverview(
        hadm_id    = hadm_id,
        patient_id = record.patient_id,
        demographics = Demographics(
            gender     = demo.get("gender", "Unknown"),
            anchor_age = demo.get("anchor_age"),
        ),
        admission = AdmissionInfo(
            admission_time     = record.admission_time,
            discharge_time     = record.discharge_time,
            admission_type     = record.admission_type,
            admission_location = record.admission_location,
            discharge_location = record.discharge_location,
            length_of_stay_days = _los(record.admission_time, record.discharge_time),
        ),
        diagnoses   = [Diagnosis(**d)   for d in record.diagnoses],
        medications = [Medication(**m)  for m in record.medications],
        procedures  = [Procedure(**p)   for p in record.procedures],
        ner_entities = PatientEntities(**entities),
    )


@router.post("/patients/{hadm_id}/summary", response_model=SummaryResponse)
def get_summary(hadm_id: str) -> SummaryResponse:
    """Generate a 3-sentence LLM clinical summary for the admission."""
    records = pipeline.state.get("records", {})
    if hadm_id not in records:
        raise HTTPException(status_code=404, detail=f"HADM ID {hadm_id!r} not found.")

    record    = records[hadm_id]
    generator = pipeline.state["generator"]
    summary   = generator.generate_raw(_summary_prompt(record))
    return SummaryResponse(hadm_id=hadm_id, summary=summary)

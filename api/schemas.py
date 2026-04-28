from typing import Optional
from pydantic import BaseModel


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    hadm_id: str
    query: str
    n_results: int = 5


class SourceChunk(BaseModel):
    text: str
    section: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    elapsed_s: float
    model: str
    n_chunks: int


# ── Patient overview ───────────────────────────────────────────────────────────

class Medication(BaseModel):
    drug: str
    dose_val_rx: Optional[str] = None
    dose_unit_rx: Optional[str] = None
    route: Optional[str] = None
    starttime: Optional[str] = None
    stoptime: Optional[str] = None


class Diagnosis(BaseModel):
    icd_code: str
    icd_version: int
    description: str
    seq_num: int


class Procedure(BaseModel):
    icd_code: str
    icd_version: int
    description: str
    seq_num: int


class Demographics(BaseModel):
    gender: str
    anchor_age: Optional[int] = None


class AdmissionInfo(BaseModel):
    admission_time: Optional[str]
    discharge_time: Optional[str]
    admission_type: Optional[str]
    admission_location: Optional[str]
    discharge_location: Optional[str]
    length_of_stay_days: Optional[float]


class PatientEntities(BaseModel):
    medications: list[str]
    diseases: list[str]
    procedures: list[str]
    anatomy: list[str]


class PatientOverview(BaseModel):
    hadm_id: str
    patient_id: str
    demographics: Demographics
    admission: AdmissionInfo
    diagnoses: list[Diagnosis]
    medications: list[Medication]
    procedures: list[Procedure]
    ner_entities: PatientEntities


# ── Summary ────────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    hadm_id: str
    summary: str

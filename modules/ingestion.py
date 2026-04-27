# modules/ingestion.py

import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from modules.base import BaseModule
from config.settings import settings


@dataclass
class PatientRecord:
    patient_id: str
    hadm_id: str
    admission_time: Optional[str] = None
    discharge_time: Optional[str] = None
    discharge_notes: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)


class DataIngestionModule(BaseModule):
    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.mimic_dir = Path(self.config.get("mimic_dir", settings.mimic_dir))

    def process(self, input_data: Optional[int] = None) -> dict[str, PatientRecord]:
        """
        Loads MIMIC-IV CSVs and returns a dict of hadm_id -> PatientRecord.
        Pass input_data as a row limit for dev (e.g. 147). None = full dataset.
        """
        admissions = self._load_admissions(input_data)
        hadm_ids = set(admissions["hadm_id"].astype(str))
        notes = self._load_notes(hadm_ids)
        diagnoses = self._load_diagnoses()
        medications = self._load_medications()

        records = {}

        for _, row in admissions.iterrows():
            hadm_id = str(row["hadm_id"])
            records[hadm_id] = PatientRecord(
                patient_id=str(row["subject_id"]),
                hadm_id=hadm_id,
                admission_time=str(row.get("admittime", "")),
                discharge_time=str(row.get("dischtime", "")),
            )

        # Attach discharge notes
        for _, row in notes.iterrows():
            hadm_id = str(row.get("hadm_id", ""))
            if hadm_id in records:
                records[hadm_id].discharge_notes.append(str(row.get("text", "")))

        # Attach diagnoses (ICD descriptions)
        for hadm_id, group in diagnoses.groupby("hadm_id"):
            hadm_id = str(hadm_id)
            if hadm_id in records:
                records[hadm_id].diagnoses = group["icd_code"].tolist()

        # Attach medications
        for hadm_id, group in medications.groupby("hadm_id"):
            hadm_id = str(hadm_id)
            if hadm_id in records:
                records[hadm_id].medications = group["drug"].dropna().tolist()

        print(f"[{self.name}] Loaded {len(records)} patient admissions")
        return records

    def _load_admissions(self, limit: Optional[int]) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "admissions.csv"
        df = pd.read_csv(path, nrows=limit)
        n_unique = df["subject_id"].nunique()
        print(f"[{self.name}] admissions: {len(df)} rows ({n_unique} unique patients)")
        return df

    def _load_notes(self, hadm_ids: set) -> pd.DataFrame:
        path = self.mimic_dir / "note" / "discharge.csv"
        chunks = pd.read_csv(path, chunksize=10_000, usecols=["hadm_id", "text"])
        matched = [chunk[chunk["hadm_id"].astype(str).isin(hadm_ids)] for chunk in chunks]
        df = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=["hadm_id", "text"])
        print(f"[{self.name}] discharge notes: {len(df)} rows for {df['hadm_id'].nunique()} admissions")
        return df

    def _load_diagnoses(self) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "diagnoses_icd.csv"
        df = pd.read_csv(path)
        return df

    def _load_medications(self) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "prescriptions.csv"
        df = pd.read_csv(path, usecols=["hadm_id", "drug"])
        return df
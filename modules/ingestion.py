# modules/ingestion.py

import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from modules.base import BaseModule
from config.settings import settings


def _safe_str(val) -> Optional[str]:
    """Return stripped string or None for NaN/empty values."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s else None


@dataclass
class PatientRecord:
    patient_id: str
    hadm_id: str
    admission_time: Optional[str] = None
    discharge_time: Optional[str] = None
    admission_type: Optional[str] = None
    admission_location: Optional[str] = None
    discharge_location: Optional[str] = None
    discharge_notes: list[str] = field(default_factory=list)
    diagnoses: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    procedures: list[dict] = field(default_factory=list)
    demographics: dict = field(default_factory=dict)


class DataIngestionModule(BaseModule):
    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.mimic_dir = Path(self.config.get("mimic_dir", settings.mimic_dir))

    def process(self, input_data: Optional[int] = None) -> dict[str, PatientRecord]:
        """
        Loads MIMIC-IV CSVs and returns a dict of hadm_id -> PatientRecord.
        Pass input_data as a row limit for dev (e.g. 147). None = full dataset.
        """
        admissions   = self._load_admissions(input_data)
        hadm_ids     = set(admissions["hadm_id"].astype(str))
        subject_ids  = set(admissions["subject_id"].astype(str))

        notes        = self._load_notes(hadm_ids)
        icd_diag_lkp = self._load_icd_diagnoses_lookup()
        icd_proc_lkp = self._load_icd_procedures_lookup()
        diagnoses_df = self._load_diagnoses(hadm_ids)
        medications_df = self._load_medications(hadm_ids)
        procedures_df  = self._load_procedures(hadm_ids)
        demographics   = self._load_demographics(subject_ids)

        records: dict[str, PatientRecord] = {}

        for _, row in admissions.iterrows():
            hadm_id    = str(row["hadm_id"])
            subject_id = str(row["subject_id"])
            records[hadm_id] = PatientRecord(
                patient_id        = subject_id,
                hadm_id           = hadm_id,
                admission_time    = _safe_str(row.get("admittime")),
                discharge_time    = _safe_str(row.get("dischtime")),
                admission_type    = _safe_str(row.get("admission_type")),
                admission_location= _safe_str(row.get("admission_location")),
                discharge_location= _safe_str(row.get("discharge_location")),
                demographics      = demographics.get(subject_id, {}),
            )

        for _, row in notes.iterrows():
            hadm_id = str(row.get("hadm_id", ""))
            if hadm_id in records:
                records[hadm_id].discharge_notes.append(str(row.get("text", "")))

        for hadm_id, group in diagnoses_df.groupby("hadm_id"):
            hadm_id = str(hadm_id)
            if hadm_id in records:
                records[hadm_id].diagnoses = [
                    self._icd_row(r, icd_diag_lkp)
                    for _, r in group.sort_values("seq_num").iterrows()
                ]

        for hadm_id, group in medications_df.groupby("hadm_id"):
            hadm_id = str(hadm_id)
            if hadm_id in records:
                records[hadm_id].medications = [
                    {
                        "drug":         _safe_str(r.get("drug")) or "Unknown",
                        "dose_val_rx":  _safe_str(r.get("dose_val_rx")),
                        "dose_unit_rx": _safe_str(r.get("dose_unit_rx")),
                        "route":        _safe_str(r.get("route")),
                        "starttime":    _safe_str(r.get("starttime")),
                        "stoptime":     _safe_str(r.get("stoptime")),
                    }
                    for _, r in group.iterrows()
                ]

        for hadm_id, group in procedures_df.groupby("hadm_id"):
            hadm_id = str(hadm_id)
            if hadm_id in records:
                records[hadm_id].procedures = [
                    self._icd_row(r, icd_proc_lkp)
                    for _, r in group.sort_values("seq_num").iterrows()
                ]

        print(f"[{self.name}] Loaded {len(records)} patient admissions")
        return records

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _icd_row(r: pd.Series, lookup: dict) -> dict:
        code    = str(r["icd_code"]).strip()
        version = int(r["icd_version"]) if pd.notna(r.get("icd_version")) else 9
        return {
            "icd_code":    code,
            "icd_version": version,
            "description": lookup.get((code, version), code),
            "seq_num":     int(r["seq_num"]) if pd.notna(r.get("seq_num")) else 0,
        }

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_admissions(self, limit: Optional[int]) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "admissions.csv"
        usecols = [
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "admission_location", "discharge_location",
        ]
        df = pd.read_csv(path, nrows=limit, usecols=usecols)
        print(f"[{self.name}] admissions: {len(df)} rows ({df['subject_id'].nunique()} unique patients)")
        return df

    def _load_notes(self, hadm_ids: set) -> pd.DataFrame:
        path   = self.mimic_dir / "note" / "discharge.csv"
        chunks = pd.read_csv(path, chunksize=10_000, usecols=["hadm_id", "text"])
        matched = [c[c["hadm_id"].astype(str).isin(hadm_ids)] for c in chunks]
        df = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=["hadm_id", "text"])
        print(f"[{self.name}] discharge notes: {len(df)} rows")
        return df

    def _load_icd_diagnoses_lookup(self) -> dict:
        path = self.mimic_dir / "hosp" / "d_icd_diagnoses.csv"
        df   = pd.read_csv(path, dtype={"icd_code": str})
        return {(str(r["icd_code"]).strip(), int(r["icd_version"])): str(r["long_title"])
                for _, r in df.iterrows()}

    def _load_icd_procedures_lookup(self) -> dict:
        path = self.mimic_dir / "hosp" / "d_icd_procedures.csv"
        df   = pd.read_csv(path, dtype={"icd_code": str})
        return {(str(r["icd_code"]).strip(), int(r["icd_version"])): str(r["long_title"])
                for _, r in df.iterrows()}

    def _load_diagnoses(self, hadm_ids: set) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "diagnoses_icd.csv"
        df   = pd.read_csv(path, dtype={"icd_code": str})
        return df[df["hadm_id"].astype(str).isin(hadm_ids)]

    def _load_medications(self, hadm_ids: set) -> pd.DataFrame:
        path    = self.mimic_dir / "hosp" / "prescriptions.csv"
        usecols = ["hadm_id", "drug", "dose_val_rx", "dose_unit_rx", "route", "starttime", "stoptime"]
        chunks  = pd.read_csv(path, chunksize=50_000, usecols=usecols)
        matched = [c[c["hadm_id"].astype(str).isin(hadm_ids)] for c in chunks]
        df = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=usecols)
        print(f"[{self.name}] medications: {len(df)} prescription rows")
        return df

    def _load_procedures(self, hadm_ids: set) -> pd.DataFrame:
        path = self.mimic_dir / "hosp" / "procedures_icd.csv"
        df   = pd.read_csv(path, dtype={"icd_code": str})
        return df[df["hadm_id"].astype(str).isin(hadm_ids)]

    def _load_demographics(self, subject_ids: set) -> dict:
        path = self.mimic_dir / "hosp" / "patients.csv"
        df   = pd.read_csv(path)
        df   = df[df["subject_id"].astype(str).isin(subject_ids)]
        result = {}
        for _, row in df.iterrows():
            result[str(row["subject_id"])] = {
                "gender":     _safe_str(row.get("gender")) or "Unknown",
                "anchor_age": int(row["anchor_age"]) if pd.notna(row.get("anchor_age")) else None,
            }
        return result

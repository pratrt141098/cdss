// ── Pipeline status ───────────────────────────────────────────────────────────

export interface PipelineStatus {
  ready: boolean
  building: boolean
  logs: string[]
  error: string | null
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export interface SourceChunk {
  text: string
  section: string
  score: number
}

export interface QueryResponse {
  answer: string
  sources: SourceChunk[]
  elapsed_s: number
  model: string
  n_chunks: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceChunk[]
  elapsed_s?: number
  n_chunks?: number
  loading?: boolean
}

// ── Patient overview ──────────────────────────────────────────────────────────

export interface Medication {
  drug: string
  dose_val_rx: string | null
  dose_unit_rx: string | null
  route: string | null
  starttime: string | null
  stoptime: string | null
}

export interface Diagnosis {
  icd_code: string
  icd_version: number
  description: string
  seq_num: number
}

export interface Procedure {
  icd_code: string
  icd_version: number
  description: string
  seq_num: number
}

export interface Demographics {
  gender: string
  anchor_age: number | null
}

export interface AdmissionInfo {
  admission_time: string | null
  discharge_time: string | null
  admission_type: string | null
  admission_location: string | null
  discharge_location: string | null
  length_of_stay_days: number | null
}

export interface PatientEntities {
  medications: string[]
  diseases: string[]
  procedures: string[]
  anatomy: string[]
}

export interface PatientOverview {
  hadm_id: string
  patient_id: string
  demographics: Demographics
  admission: AdmissionInfo
  diagnoses: Diagnosis[]
  medications: Medication[]
  procedures: Procedure[]
  ner_entities: PatientEntities
}

export interface SummaryResponse {
  hadm_id: string
  summary: string
}

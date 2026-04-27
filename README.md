# CDSS — Clinical Decision Support System

## Abstract

A locally-run Retrieval-Augmented Generation (RAG) system designed to help
clinicians under time pressure quickly access consolidated patient information
from discharge notes. Built on MIMIC-IV data, the system extracts clinical
entities from unstructured notes, embeds them into a semantic vector store,
and answers natural language queries grounded strictly in the patient record —
no external knowledge, no data leaving the machine.

---

## Architecture

Modular pipeline, each stage independently testable:

```
DataIngestionModule → PreprocessingModule → NERModule → EmbeddingModule → VectorStoreModule → GenerationModule
```

| Module | Description |
|---|---|
| `DataIngestionModule` | Loads MIMIC-IV admissions, discharge notes, diagnoses, medications into `PatientRecord` objects. Streams discharge notes in chunks filtered by `hadm_id` to avoid loading the full 99M-line CSV |
| `PreprocessingModule` | Parses discharge notes by clinical section (16 MIMIC-IV sections), chunks into ~512 token overlapping segments |
| `NERModule` | Extracts clinical entities (medications, diseases, anatomy, procedures) using `en_core_sci_sm` |
| `EmbeddingModule` | Embeds chunks using `nomic-embed-text` via Ollama with `search_document` / `search_query` prefixes for asymmetric RAG retrieval |
| `VectorStoreModule` | Persists embeddings in ChromaDB with cosine similarity; supports global and `hadm_id`-scoped retrieval |
| `GenerationModule` | Generates grounded clinical answers using `llama3.2:3b` via Ollama; prompt handles MIMIC-IV de-identification artefacts |

---

## Stack

- **Models**: `en_core_sci_sm` (NER), `nomic-embed-text` (embeddings), `llama3.2:3b` (generation)
- **Vector store**: ChromaDB (persistent, local)
- **Runtime**: Ollama (fully local, no API keys)
- **Frontend**: Streamlit
- **Data**: MIMIC-IV (PhysioNet credentialed access)

---

## Running Locally

```bash
# prerequisites
conda activate cdss
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# launch app
streamlit run app.py
```

The sidebar **"Rows to load"** slider controls how many rows are read from
`admissions.csv` (default 147 rows = ~50 unique patients). Discharge notes are
loaded by filtering on the resulting `hadm_id` set — not by a separate `nrows`
limit — ensuring every ingested admission has its notes attached.

---

## Evaluation Framework

A full evaluation suite lives in `eval/` and a walkthrough notebook in
`notebooks/CDSS_Evaluation_Walkthrough.ipynb`.

### Running the eval

```bash
conda activate cdss
# Full run (retrieval + generation, ~15 min)
python -m eval.run_eval

# Retrieval only (fast, no LLM calls, ~30 sec)
python -m eval.run_eval --retrieval-only

# Generation only
python -m eval.run_eval --generation-only
```

Results are saved to `eval/results/eval_results.json` and
`eval/results/eval_summary.csv`.

### Query set

20 clinical queries derived from 10 held-out patients (last 10 `subject_id`s
by sorted order in the `cdss_147` ChromaDB collection). Sections covered:
`discharge_diagnosis`, `discharge_medications`, `medications_on_admission`,
`allergies`, `history_of_present_illness`.

### Retrieval results (cdss_147, 1,193 chunks, 30 patients)

| Strategy | Mean Precision | Mean Recall |
|---|---|---|
| RAG — global (no patient filter) | 0.02 | 0.10 |
| **RAG — scoped to `hadm_id`** | **0.19** | **0.95** |
| Keyword baseline (global) | 0.16 | 0.80 |

Scoped retrieval mirrors the app's "Single patient" mode and achieves 0.95
recall. Global RAG fails because patient-specific queries contain no identifier,
so the embedder matches semantically similar chunks from other patients.

### Generation results (scoped RAG vs no-RAG baseline)

Reference text: full ground-truth section chunk from ChromaDB (20/20 queries
resolved to a real chunk — no keyword fallbacks).

| Metric | RAG (scoped) | No-RAG |
|---|---|---|
| ROUGE-1 | **0.38** | 0.03 |
| ROUGE-2 | **0.28** | 0.00 |
| ROUGE-L | **0.34** | 0.02 |
| BERTScore F1 | **0.87** | 0.79 |
| Faithfulness | **0.84** | — |

No-RAG produces generic clinical answers (instructed to answer from general
medical knowledge); RAG produces patient-specific answers grounded in the
retrieved discharge note text. The gap is +0.35 ROUGE-1, +0.08 BERTScore.

One query remains flagged (Q19 — Corgard/Vasotec allergy entry; 5-word chunk
too short for reliable semantic retrieval).

---

## Repository Layout

```
cdss/
├── app.py                        # Streamlit UI
├── config/
│   └── settings.py               # Pydantic settings (reads .env)
├── modules/
│   ├── base.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── ner.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── generation.py
├── eval/
│   ├── query_set.py              # 20 ground-truth queries
│   ├── retrieval_eval.py         # RAG global / scoped / keyword baseline
│   ├── generation_eval.py        # ROUGE, BERTScore, faithfulness
│   └── run_eval.py               # Master script
├── notebooks/
│   └── CDSS_Evaluation_Walkthrough.ipynb
└── requirements.txt
```

---

## Pending Work

### Short Term
- [ ] Section-routing fallback: directly fetch `allergies` chunk when query contains "allerg" (fixes Q19 miss)
- [ ] Hybrid retrieval: blend scoped vector scores with keyword hit counts for structured vs narrative sections
- [ ] Scale eval to 500+ patients for statistically robust metrics
- [ ] Add streaming to `GenerationModule` so responses render token-by-token in Streamlit

### Production (Flask + React + BU SCC)
- [ ] Swap Streamlit for Flask REST API (`/api/query`, `/api/ingest`, `/api/patients`)
- [ ] Build React/Node.js frontend with patient selector and chat interface
- [ ] Deploy on BU SCC with GPU node for `llama3.1:70b` and `vllm` serving
- [ ] Swap `en_core_sci_sm` for `en_ner_bc5cdr_md` for typed clinical NER labels
- [ ] Add HIPAA compliance layer — audit logging, role-based access, de-identification check

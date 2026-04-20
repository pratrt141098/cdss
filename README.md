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
DataIngestionModule → PreprocessingModule → NERModule → EmbeddingModule → VectorStoreModule → GenerationModule

text

| Module | Description |
|---|---|
| `DataIngestionModule` | Loads MIMIC-IV admissions, discharge notes, diagnoses, medications into `PatientRecord` objects |
| `PreprocessingModule` | Parses discharge notes by clinical section, chunks into ~500 token segments |
| `NERModule` | Extracts clinical entities (medications, diseases, anatomy, procedures) using `en_core_sci_sm` |
| `EmbeddingModule` | Embeds chunks using `nomic-embed-text` via Ollama with `search_document` prefix for RAG |
| `VectorStoreModule` | Persists embeddings in ChromaDB with cosine similarity, supports patient-scoped retrieval |
| `GenerationModule` | Generates grounded clinical answers using `llama3.2:3b` via Ollama |

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

---

## Validated Results

- Zero hallucinations on manual ground truth check against raw MIMIC-IV discharge notes
- 86 chunks from 10 patients, retrieval latency < 1s, generation latency ~11s on MacBook Air M-series
- Cosine similarity scores 0.60–0.78 on clinical queries (medications, diagnosis, hospital course)

---

## Pending Work

### Short Term
- [ ] Add streaming to `GenerationModule` so responses render token by token in Streamlit
- [ ] Add `faithfulness_score` evaluation module for automated hallucination detection
- [ ] Scale ingestion to 100–200 patients via `ingest.py` with persistent ChromaDB

### Production (Flask + React + BU SCC)
- [ ] Swap Streamlit for Flask REST API (`/api/query`, `/api/ingest`, `/api/patients`)
- [ ] Build React/Node.js frontend with patient selector and chat interface
- [ ] Deploy on BU SCC with GPU node for `llama3.1:70b` and `vllm` serving
- [ ] Swap `en_core_sci_sm` for `en_ner_bc5cdr_md` for typed clinical NER labels
- [ ] Add HIPAA compliance layer — audit logging, role-based access, de-identification check
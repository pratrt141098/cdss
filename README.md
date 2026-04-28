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
| `VectorStoreModule` | Persists embeddings in ChromaDB with cosine similarity; supports global, `hadm_id`-scoped, section-routed, hybrid, and RAG-Fusion retrieval |
| `GenerationModule` | Streams grounded clinical answers using `llama3.2:3b` via Ollama; prompt handles MIMIC-IV de-identification artefacts |

---

## Stack

- **Models**: `en_core_sci_sm` (NER), `nomic-embed-text` (embeddings), `llama3.2:3b` (generation)
- **Vector store**: ChromaDB (persistent, local)
- **Runtime**: Ollama (fully local, no API keys)
- **Frontend**: Streamlit (streaming responses)
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

The **RAG-Fusion (cross-patient)** toggle enables entity-extraction-based patient
matching: the LLM extracts clinical entities from the query, scores all patients
by entity overlap against their stored NER metadata, and scopes retrieval to the
best match — no `hadm_id` required from the user.

---

## Retrieval Strategies

The system implements five retrieval strategies, selectable per query:

| Strategy | Mechanism | When it fires |
|---|---|---|
| **Section router** | Metadata `WHERE` fetch by section slug | Query contains structured keywords (allergies, medications, diagnosis) |
| **Scoped vector search** | Cosine similarity filtered to `hadm_id` | Single-patient mode, non-structured queries |
| **Hybrid** | `α · vector + (1-α) · keyword` re-rank | Scoped mode; α swept at eval time |
| **Entity-match cross-patient** | LLM entity extraction → patient scoring → scoped retrieval | Cross-patient mode, entity matches found |
| **RAG-Fusion fallback** | 3 query variants + RRF, unscoped | Cross-patient mode, no entity matches |

---

## Evaluation Framework

A full evaluation suite lives in `eval/` and a walkthrough notebook in
`notebooks/CDSS_Evaluation_Walkthrough.ipynb`.

### Running the eval

```bash
conda activate cdss
# Full run (retrieval + generation, ~15 min)
python -m eval.run_eval

# Retrieval only (fast, no LLM generation calls, ~5 min)
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

### Retrieval results (cdss_147, 1,193 chunks, 30 patients, top-k = 10)

| Strategy | Mean Precision | Mean Recall | Notes |
|---|---|---|---|
| RAG — global (no filter) | 0.02 | 0.20 | Baseline; fails cross-patient |
| Keyword baseline (global) | 0.09 | 0.90 | Strong on structured sections |
| **RAG — scoped to `hadm_id`** | **0.10** | **1.00** | Perfect recall; precision low at k=10 |
| **Section router** | **1.00** | **1.00** | 17/20 queries routed; deterministic |
| Hybrid (α=0.3–0.7, scoped) | 0.10 | 1.00 | α-invariant at this scale |
| RAG-Fusion (unscoped) | 0.01 | 0.05 | Worse than global RAG — see note |

> **RAG-Fusion negative result**: query variants are equally patient-agnostic as
> the original query. RRF merges lists that each surface *different* patients'
> chunks at rank 1, so the target patient's chunk never concentrates enough rank
> signal to surface. Entity-extraction matching solves this correctly.

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

---

## Repository Layout

```
cdss/
├── app.py                        # Streamlit UI (streaming, cross-patient mode)
├── config/
│   └── settings.py               # Pydantic settings (reads .env)
├── modules/
│   ├── base.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── ner.py
│   ├── embedding.py
│   ├── vector_store.py           # hybrid_query, rag_fusion_query, get_by_section
│   ├── generation.py             # streaming via ollama stream=True
│   ├── query_router.py           # regex section router (17/20 eval queries routed)
│   ├── query_expansion.py        # generate_query_variants + reciprocal_rank_fusion
│   └── patient_matcher.py        # cross-patient entity extraction + patient scoring
├── eval/
│   ├── query_set.py              # 20 ground-truth queries
│   ├── retrieval_eval.py         # 5 strategies: RAG / scoped / KW / routed / hybrid / fusion
│   ├── generation_eval.py        # ROUGE, BERTScore, faithfulness
│   └── run_eval.py               # Master script with full summary table
├── notebooks/
│   └── CDSS_Evaluation_Walkthrough.ipynb
└── requirements.txt
```

---

## Pending Work

### Short Term
- [ ] Section-router coverage: extend patterns to `history_of_present_illness` keyword triggers (currently 3/20 queries fall through to vector search by design)
- [ ] Hybrid α selection: run eval in unscoped mode or on multi-chunk sections to find a meaningful α signal
- [ ] Conversation memory: maintain patient context across multi-turn queries in Streamlit

### Production (Flask + React + BU SCC)
- [ ] Swap Streamlit for Flask REST API (`/api/query`, `/api/ingest`, `/api/patients`)
- [ ] Build React/Node.js frontend with patient selector and chat interface
- [ ] Deploy on BU SCC with GPU node for `llama3.1:70b` and `vllm` serving
- [ ] Swap `en_core_sci_sm` for `en_ner_bc5cdr_md` for typed clinical NER labels
- [ ] Add HIPAA compliance layer — audit logging, role-based access, de-identification check
- [ ] Scale eval to 500+ patients for statistically robust metrics

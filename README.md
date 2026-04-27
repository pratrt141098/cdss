# CDSS

Clinical Decision Support System for patient-grounded question answering over MIMIC-IV discharge notes.

This project builds a fully local retrieval-augmented generation pipeline that:

1. Loads MIMIC-IV admissions, discharge notes, diagnoses, and prescriptions.
2. Cleans and chunks discharge notes into section-aware text chunks.
3. Runs clinical named entity recognition over each chunk.
4. Creates embeddings for chunks and queries.
5. Stores the embedded chunks in a persistent local vector database.
6. Retrieves the most relevant patient context for a question.
7. Generates an answer grounded in the retrieved record context.

The primary user interface today is the Streamlit app in [`app.py`](/Users/sonalps/cdss/app.py).

## What This Repository Is For

The repo is meant to be a reproducible local CDSS prototype, not a general-purpose medical chatbot. The design is intentionally constrained:

- answers should come from the patient record, not external medical knowledge
- all data stays local
- the pipeline is modular so each stage can be inspected independently
- the same code path can be run end-to-end or step-by-step during development

## High-Level Architecture

The main pipeline is:

`DataIngestionModule` -> `PreprocessingModule` -> `NERModule` -> `EmbeddingModule` -> `VectorStoreModule` -> `GenerationModule`

Each module has one job:

- [`modules/ingestion.py`](/Users/sonalps/cdss/modules/ingestion.py) loads MIMIC-IV CSVs into `PatientRecord` objects.
- [`modules/preprocessing.py`](/Users/sonalps/cdss/modules/preprocessing.py) splits notes into clinical sections and chunks them.
- [`modules/ner.py`](/Users/sonalps/cdss/modules/ner.py) extracts clinical entities from chunk text.
- [`modules/embedding.py`](/Users/sonalps/cdss/modules/embedding.py) embeds chunks and query strings.
- [`modules/vector_store.py`](/Users/sonalps/cdss/modules/vector_store.py) persists the chunk corpus in ChromaDB and performs retrieval.
- [`modules/generation.py`](/Users/sonalps/cdss/modules/generation.py) turns a query plus retrieved chunks into a grounded answer.

## Repository Layout

- [`app.py`](/Users/sonalps/cdss/app.py) - Streamlit application
- [`config/settings.py`](/Users/sonalps/cdss/config/settings.py) - Default paths and runtime settings
- [`modules/`](/Users/sonalps/cdss/modules) - Core pipeline modules
- [`pipeline/`](/Users/sonalps/cdss/pipeline) - Reserved for future orchestration helpers
- [`api/`](/Users/sonalps/cdss/api) - Reserved API scaffolding
- [`scripts/`](/Users/sonalps/cdss/scripts) - Reserved CLI/script entry points
- [`tests/`](/Users/sonalps/cdss/tests) - Test placeholders and smoke checks
- [`test.py`](/Users/sonalps/cdss/test.py) - Manual end-to-end pipeline walkthrough
- [`val.py`](/Users/sonalps/cdss/val.py) - Manual ingestion sanity check

## Requirements

### Python

Use Python 3.10 or 3.11. The project depends on modern versions of:

- `streamlit`
- `chromadb`
- `spacy`
- `sentence-transformers`
- `vllm`
- `pysqlite3-binary`

### System

The stack is easiest to run on:

- Linux with a CUDA-capable GPU for `vllm`
- macOS or Linux for development and inspection, if you are willing to adapt the generation backend

### Data

The pipeline expects a local MIMIC-IV directory structure under `./data/mimic` by default. You need the relevant PhysioNet-approved files available locally.

### Model Downloads

The pipeline uses:

- `en_core_sci_sm` for spaCy-based clinical NER
- `all-MiniLM-L6-v2` for embeddings
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for generation

The embedding model is downloaded automatically by `sentence-transformers` on first use. The spaCy model must be installed explicitly.

## Reproducible Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

Install the pinned requirements first, then the additional runtime packages used directly by the current code.

```bash
pip install -r requirements.txt
pip install sentence-transformers pysqlite3-binary vllm
```

### 3. Install the spaCy clinical model

```bash
python -m spacy download en_core_sci_sm
```

### 4. Prepare the local data directory

By default, the code reads from:

```text
data/mimic/
  hosp/
    admissions.csv
    diagnoses_icd.csv
    prescriptions.csv
  note/
    discharge.csv
```

If your files live elsewhere, update [`config/settings.py`](/Users/sonalps/cdss/config/settings.py) or set the corresponding environment variables in a `.env` file.

## Configuration

[`config/settings.py`](/Users/sonalps/cdss/config/settings.py) defines the default runtime values:

- `mimic_dir = ./data/mimic`
- `synthetic_dir = ./data/synthetic`
- `chroma_persist_dir = ./chroma_db`
- `chunk_size = 512`
- `chunk_overlap = 50`
- `retrieval_top_k = 5`

The settings class reads `.env` if present.

## Running The Project

### Streamlit application

```bash
streamlit run app.py
```

What the app does:

- loads the pipeline once and caches it in Streamlit
- lets you choose how many admissions to ingest for the session
- lets you switch between all-patient retrieval and single-patient retrieval
- shows source chunks for each generated answer

The app clears and rebuilds the cached pipeline when you press the reload button in the sidebar.

### Ingestion sanity check

```bash
python val.py
```

This prints:

- how many patient admissions were loaded
- which `hadm_id` values are available
- a sample lookup for a known admission if present

### End-to-end walkthrough

```bash
python test.py
```

This script walks through:

- ingestion
- preprocessing
- NER
- embedding
- vector store indexing
- retrieval
- generation

It is useful when you want to inspect intermediate output without the Streamlit UI.

## Module Details

### Ingestion

[`modules/ingestion.py`](/Users/sonalps/cdss/modules/ingestion.py) loads the following files:

- `hosp/admissions.csv`
- `note/discharge.csv`
- `hosp/diagnoses_icd.csv`
- `hosp/prescriptions.csv`

It returns a dictionary keyed by `hadm_id`, with each value holding:

- `patient_id`
- `hadm_id`
- admission/discharge timestamps when present
- discharge notes
- diagnoses
- medications

The `input_data` argument acts as a development-time row limit for admissions and discharge notes.

### Preprocessing

[`modules/preprocessing.py`](/Users/sonalps/cdss/modules/preprocessing.py) performs two main steps:

1. It detects common MIMIC-IV discharge note section headers.
2. It splits text into overlapping word-level chunks.

Important defaults:

- chunk size is 512 words
- chunk overlap is 50 words
- if no known section headers are found, the module falls back to a `general` section

### NER

[`modules/ner.py`](/Users/sonalps/cdss/modules/ner.py) uses spaCy with the `en_core_sci_sm` model.

Behavior worth knowing:

- chunks shorter than a small threshold are passed through with empty entity lists
- entity labels from typed models are bucketed into medications, diseases, anatomy, and procedures
- flat `ENTITY` labels from smaller models are routed heuristically by section name

### Embeddings

[`modules/embedding.py`](/Users/sonalps/cdss/modules/embedding.py) uses `sentence-transformers`.

Current default model:

- `all-MiniLM-L6-v2`

The module provides:

- `process(chunks)` for embedding a list of chunk objects
- `embed_query(text)` for embedding a query string

### Vector Store

[`modules/vector_store.py`](/Users/sonalps/cdss/modules/vector_store.py) stores embedded chunks in ChromaDB.

Current behavior:

- uses a persistent local store in `./chroma_db`
- uses cosine similarity
- supports optional filtering by `hadm_id`
- returns chunk text, metadata, and similarity score for each retrieval result

### Generation

[`modules/generation.py`](/Users/sonalps/cdss/modules/generation.py) builds a prompt from:

- the user question
- the retrieved source chunks

It then calls a local `vllm` model to produce an answer.

The answer prompt explicitly instructs the model to:

- answer only from the provided patient context
- say it cannot determine the answer if the context is insufficient

## Reproducibility Notes

To reproduce the project as closely as possible:

1. Use the exact dependency versions in `requirements.txt`.
2. Install the extra runtime packages listed above.
3. Download the spaCy model before running NER.
4. Keep MIMIC-IV in the expected directory structure.
5. Use the same ChromaDB persist directory if you want retrieval results to survive reruns.
6. Be aware that `vllm` is the most hardware-sensitive dependency in the stack.

The generated answers will still vary slightly across runs because the model is probabilistic, but the retrieval corpus and index should be reproducible if the data and environment stay fixed.

## Troubleshooting

### `en_core_sci_sm` is missing

Install it with:

```bash
python -m spacy download en_core_sci_sm
```

### `sqlite3` import problems

The project patches SQLite to use `pysqlite3`. Make sure `pysqlite3-binary` is installed in the active environment.

### No data is loaded

Check that the local MIMIC-IV files exist under `./data/mimic` and that the file names match the expected layout.

### `vllm` fails to start

This is usually an environment or hardware issue rather than an application bug. `vllm` is most reliable on Linux with a compatible GPU.

### Retrieval returns empty or poor results

Confirm that:

- the vector store was built successfully
- the query is being embedded with the same embedding model used during indexing
- the `hadm_id` filter is not excluding the relevant patient

## Testing And Validation

The repo currently includes manual smoke-check scripts and placeholder test files.

Suggested validation sequence:

1. Run `python val.py` to confirm ingestion.
2. Run `python test.py` to confirm the full pipeline.
3. Run `streamlit run app.py` to confirm the UI and retrieval loop.

If you add new logic, the most valuable checks are the ones that compare retrieved chunks and generated answers against the original discharge notes.

## Privacy And Data Handling

This project is intended for local use with credentialed MIMIC-IV data.

Do not commit:

- raw MIMIC-IV CSVs
- ChromaDB persistence files
- generated embeddings
- patient-level outputs or notes

Keep everything local and handle the dataset according to the access and governance requirements that apply to your environment.

## Current Status

The codebase is already organized around a modular local RAG workflow, but some folders are still scaffolding for future expansion:

- `api/`
- `pipeline/`
- `scripts/`

The current working path is the Streamlit application and the modules under `modules/`.

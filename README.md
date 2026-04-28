# CDSS — Clinical Decision Support System

## Abstract

A locally run **Retrieval-Augmented Generation (RAG)** system that helps clinicians pull patient-level information from unstructured discharge notes. The project ingests **MIMIC-IV** data, segments notes by clinical section, runs **spaCy** clinical NER, embeds chunks with **Ollama** (`nomic-embed-text`), stores vectors in **ChromaDB**, and answers questions with **Ollama** (`llama3.2:3b`) using only retrieved context.

The primary user interface is a **React 18 + TypeScript** single-page app backed by a **FastAPI** server. A **Streamlit** app remains available for lightweight, all-in-one demos. All inference and data stay on the machine; no external LLM APIs are required for the default stack.

---

## Architecture

### RAG pipeline (core)

Each stage is implemented as a module and can be exercised independently:

```
DataIngestionModule → PreprocessingModule → NERModule → EmbeddingModule → VectorStoreModule → GenerationModule
```

| Module | Description |
| --- | --- |
| `DataIngestionModule` | Loads MIMIC-IV **admissions**, **discharge notes**, **diagnoses** (ICD + descriptions), **procedures**, **prescriptions** (with route/dose where present), and **`patients`** demographics into `PatientRecord` objects. Streams large CSVs and filters notes by the active `hadm_id` set |
| `PreprocessingModule` | Parses discharge summaries into MIMIC clinical **sections** (~16 types), chunks overlapping segments (~512 tokens) |
| `NERModule` | Entity extraction with **`en_core_sci_sm`** (medications, diseases, anatomy, procedures) |
| `EmbeddingModule` | Query/document embeddings via **Ollama** `nomic-embed-text`, with `search_document` / `search_query` prefixes |
| `VectorStoreModule` | **ChromaDB** persistent store, cosine distance; `query(…, hadm_id=…)` for single-admission scoping |
| `GenerationModule` | Grounded answers via `llama3.2:3b`; includes **`generate_raw(prompt)`** for non-RAG summaries (e.g. dashboard clinical brief) |

Collection name pattern: `cdss_<N>` where **N** is the number of admissions loaded (e.g. `cdss_147`).

### Web application (optional but recommended)

```
Browser (Vite dev server or static build)
    ↓ HTTP JSON
FastAPI (`api/main.py`) — CORS, lifespan, background pipeline build
    ↓
Routers: patients, query, summary → shared `api.pipeline` state
```

On startup, **`pipeline.build(N)`** runs in a **daemon thread** so the server can accept connections immediately. Clients **poll `GET /status`** until **`ready: true`**. Data endpoints that need an indexed store return **503** until the pipeline is ready (`api.deps.require_ready`).

---

## Stack

| Layer | Technology |
| --- | --- |
| **NER** | spaCy `en_core_sci_sm` (installed from wheel URL in `requirements.txt`) |
| **Embeddings / generation** | Ollama (`nomic-embed-text`, `llama3.2:3b`) |
| **Vector store** | ChromaDB 1.x, local persistent directory (`chroma_db/`) |
| **Backend** | FastAPI, Uvicorn, Pydantic schemas (`api/schemas.py`) |
| **Primary UI** | React 18, TypeScript, Vite 5, Tailwind CSS 3, **Recharts**, **Lucide React** |
| **Legacy UI** | Streamlit (`app.py`) |
| **Evaluation** | ROUGE, BERTScore (`eval/`), Jupyter notebook walkthrough |
| **Data** | MIMIC-IV (PhysioNet credentialed access) |

Configuration is centralized in **`config/settings.py`** (Pydantic Settings, optional `.env`). `gemini_*` fields in settings are legacy placeholders; the running stack uses **Ollama** as above.

---

## Running locally

### Prerequisites

```bash
conda activate cdss   # or your preferred env with Python 3.11+
pip install -r requirements.txt   # installs en_core_sci_sm from the SciSpaCy release URL
# Optional — notebooks / evaluation plotting (matplotlib, Jupyter kernel):
pip install -r requirements-dev.txt

ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Place **MIMIC-IV** CSVs under `data/mimic/` (or set `MIMIC_DIR` / project env to match `config.settings`). The ingestion module expects at minimum the subset used by the app (e.g. admissions, patients, prescriptions, diagnoses ICD, procedures ICD, discharge text — see `modules/ingestion.py`).

### Option A — React + FastAPI (full dashboard)

Terminal 1 — API (default **147** admissions, override with `CDSS_N_PATIENTS`):

```bash
cd /path/to/cdss
export CDSS_N_PATIENTS=147          # optional
export CDSS_CORS_ORIGINS=http://localhost:5173   # optional; comma-separated for multiple origins

uvicorn api.main:app --reload
```

- **`GET /status`** — Pipeline build state: `ready`, `building`, ordered `logs`, `error`. Safe to poll every second while the UI shows a status banner.
- Data routes (`/patients`, `/query`) return **503** until `ready` is true.

Terminal 2 — frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`). The UI polls `/status`, then loads `/patients` and per-patient views.

**Production build:**

```bash
cd frontend && npm run build
```

Serve the `frontend/dist/` assets behind any static host; point `CDSS_CORS_ORIGINS` at that origin.

#### Deploying the frontend to Vercel (static app only)

**What fits Vercel:** The React UI is a **`npm run build`** output (`frontend/dist/`). Vercel detects **Vite** and deploys static assets + edge CDN. Set the Vercel project **Root Directory** to **`frontend`** (this repo is a monolith; the app is not at the repo root).

**What does *not* belong on Vercel (with the current architecture):**

| Piece | Why |
| --- | --- |
| **FastAPI / Uvicorn** | Long-lived process; Vercel Functions are short-lived, CPU/memory/time limits, cold starts — a poor match for a multi-minute pipeline build and Chroma queries. |
| **Ollama** | Runs as a daemon with local models; not available in Vercel’s runtime. |
| **ChromaDB** | Persistent on-disk store; serverless has no durable local volume for this use case. |
| **spaCy + MIMIC ingestion** | Heavy images and long startup; ill-suited to function bundles. |

So this is a **hybrid** deployment: **Vercel = UI only**; **API + RAG** run on a VM, container host, or managed service that supports persistent disk and your chosen inference stack (see [Roadmap](#roadmap-and-known-gaps)).

**Steps (UI on Vercel, API elsewhere):**

1. Deploy the API to a host that can run Python + Ollama + Chroma + data (e.g. a dedicated server, **Fly.io** / **Railway** / **Render** with a volume, AWS/GCP VM with GPU if you scale models). Expose HTTPS, e.g. `https://api.example.com`.
2. In the Vercel project for **`frontend`**, add an environment variable:  
   **`VITE_API_URL`=`https://api.example.com`**  
   (no trailing slash; see `frontend/src/api/client.ts`).
3. On the API server, set **`CDSS_CORS_ORIGINS`** to include your Vercel origin, e.g. `https://your-app.vercel.app` (and preview URLs if you use them: `https://your-app-*.vercel.app` or list each).
4. Trigger a redeploy so the client bundle picks up `VITE_*` at build time.

**HTTPS:** Browsers require **HTTPS** on the page for many features; mixed-content rules may block `http://` API URLs from an `https://` Vercel app — put TLS on the API or use a tunnel for demos.

**Previews:** Each Vercel preview URL is a different origin; either add wildcard CORS carefully or use a single staging API with fixed preview domains.

### Option B — Streamlit (single-process)

```bash
streamlit run app.py
```

The sidebar **“Rows to load”** slider sets how many rows are read from `admissions.csv` (default **147**). Notes are loaded for the resulting `hadm_id` set only. **Retrieved chunks** (3–10) controls RAG **`n_results`**.

The Streamlit path does **not** expose the REST API or the React dashboard; use it for quick experiments.

---

## Web API reference

Base URL: `http://127.0.0.1:8000` when using default Uvicorn.

| Method | Path | Guard | Description |
| --- | --- | --- | --- |
| `GET` | `/status` | none | Pipeline status and build log lines |
| `GET` | `/patients` | ready | Sorted list of **`hadm_id`** strings loaded in memory |
| `GET` | `/patients/{hadm_id}/overview` | ready | Structured **`PatientOverview`**: demographics, admission/discharge meta, diagnoses, medications, procedures, aggregated **NER entities** (from pipeline build, not Chroma `$where` filtering) |
| `POST` | `/patients/{hadm_id}/summary` | ready | **`SummaryResponse`**: LLM-written 3-sentence clinical brief; prompt includes **gender / age** for correct pronoun use |
| `POST` | `/query` | ready | **`QueryRequest`**: `{ "query", "hadm_id", "n_results" }` → **`QueryResponse`** with grounded answer + **source chunks** (truncated text, section, score) |
| `POST` | `/pipeline/reload` | none | Returns **202**; clears state and rebuilds pipeline in background; poll `/status` |

OpenAPI docs: **`/docs`** (Swagger UI).

---

## Frontend features (`frontend/`)

| Area | Behavior |
| --- | --- |
| **Pipeline status** | Top banner connects while building; shows expandable logs; hides when healthy or surfaces errors |
| **Patient selector** | Populated after pipeline ready |
| **`n_results`** | Slider mirrored to **`QueryRequest.n_results`** (scoped retrieval) |
| **Overview tab** | Stat cards (LOS, diagnosis/medication counts, procedures); **NER donut** (Recharts); **AI clinical summary** (from `/patients/{id}/summary`) |
| **Medications tab** | List of orders; **route donut**; **frequency bar chart**; **timeline (Gantt-style)** bars vs admission time when timestamps exist |
| **Diagnoses tab** | ICD-backed labels; **diagnosis chapter bar chart** (`lib/icd.ts`); procedure list |
| **Chat tab** | Prior RAG QA: `/query` with streaming-free full response + sources |

Styling uses **Tailwind** utility classes and **Lucide** icons throughout.

---

## Data model highlights (`PatientRecord`)

Extended beyond raw notes:

- **`diagnoses`**: ICD code, version, long description (`d_icd_diagnoses` join)
- **`medications`**: drug, dose, route, start/stop when present in prescriptions
- **`procedures`**: ICD procedures + descriptions
- **`demographics`**: anchor age, gender (from **`patients`**) — used for API overview and summary prompts

---

## Evaluation framework (`eval/` + notebook)

### CLI

```bash
conda activate cdss

python -m eval.run_eval                     # retrieval + generation (~15 min depending on GPU/CPU)

python -m eval.run_eval --retrieval-only   # fast, no LLM

python -m eval.run_eval --generation-only
```

Artifacts:

- **`eval/results/eval_results.json`**
- **`eval/results/eval_summary.csv`**

### Query set

**20** questions over **10** held-out **`subject_id`s** (`eval/query_set.py`), two per patient across sections such as **`discharge_diagnosis`**, **`discharge_medications`**, **`medications_on_admission`**, **`allergies`**, **`history_of_present_illness`**.

### Retrieval benchmarks (representative **`cdss_147`**, scoped = filter by **`hadm_id`**)

Approximate aggregates at default evaluator settings (see **`eval/retrieval_eval.py`**):

| Strategy | Mean precision | Mean recall |
| --- | --- | --- |
| RAG global (no patient filter) | ~0.02 | ~0.10 |
| **RAG scoped** | **~0.19** | **~0.95** |
| Keyword global baseline | ~0.16 | ~0.80 |

Scoped retrieval matches the dashboard’s single-patient chat. Global retrieval fails without a patient anchor because queries are semantically similar across many admissions.

### Generation benchmarks (scoped RAG vs no-RAG)

Ground truth: full ground-truth **section chunk** text from Chroma for each query where available.

Illustrative aggregate ranges from saved runs:

| Metric | RAG (scoped) | No-RAG |
| --- | --- | --- |
| ROUGE-1 | ~0.32–0.38 | ~0.02–0.03 |
| BERTScore F1 | ~0.86–0.87 | ~0.78–0.79 |
| Faithfulness | ~0.82–0.84 | — |

Exact numbers vary slightly by pipeline version and prompts; regenerate with `eval.run_eval` for your machine.

### Notebook: `notebooks/CDSS_Evaluation_Walkthrough.ipynb`

The notebook is the **authoritative qualitative / exploratory** supplement to the CLI metrics. It includes:

1. **Setup** — Chroma connectivity, corpus inventory  
2. **Retrieval evaluation** — per-query table, aggregates  
3. **Recall @ k sweep (Section 2b)** — k ∈ `{1,2,3,5,7,10}`; recall/precision curves; section-level bar charts; guidance on keeping **k = 5** for scoped RAG  
4. **Advanced strategies (Section 2c)** — documented comparison with experiments from the **`pratik`** branch (**RAG-Fusion** + **RRF**, **deterministic section router**, **hybrid vector+keyword**, **`pratik`** **patient matcher** prototype): metrics are **embedded as static reproductions** because those modules live on a branch  
5. **Live query demos**, **generation eval tables/plots**, **results CSV summary**  
6. **Key observations & updated “Known limitations”** — e.g. Q19 allergy chunk length; closed items (**top‑k**, **hybrid**) vs open items (**eval scale**, **structured allergy lookup**, **Patient Matcher eval**)

Generated figures under `notebooks/` (e.g. `recall_precision_vs_k.png`, `strategy_comparison.png`) are produced when you execute the plotting cells.

---

## Repository layout

```
cdss/
├── api/
│   ├── main.py              # FastAPI app, CORS, lifespan, /status
│   ├── pipeline.py          # Background build, ner_entities aggregation
│   ├── deps.py              # require_ready → 503
│   ├── schemas.py           # Pydantic API models
│   └── routes/
│       ├── patients.py      # GET /patients, overview, POST summary
│       ├── query.py         # POST /query
│       └── summary.py       # POST /pipeline/reload
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── api/client.ts
│       ├── types/api.ts
│       ├── components/
│       │   ├── StatusBanner.tsx
│       │   ├── Sidebar.tsx
│       │   └── tabs/        # Overview, Medications, Diagnoses, Chat
│       └── lib/             # icd.ts, utils
├── config/
│   └── settings.py
├── eval/
│   ├── query_set.py
│   ├── retrieval_eval.py
│   ├── generation_eval.py
│   └── run_eval.py
├── modules/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── ner.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── generation.py
├── notebooks/
│   └── CDSS_Evaluation_Walkthrough.ipynb
├── app.py                   # Streamlit (legacy / alternative UI)
├── requirements.txt
├── requirements-dev.txt        # Jupyter + matplotlib (evaluation notebook)
└── README.md
```

---

## Roadmap and known gaps

Aligned with the latest **“Known limitations & next steps”** in the evaluation notebook:

| Topic | Notes |
| --- | --- |
| **Q19-style allergy misses** | Very short allergy chunks are hard to retrieve by embedding alone; **structured** allergy fields or minimum chunk length policies are more reliable than ad-hoc regex routing |
| **Evaluation scale** | 20 queries / 10 patients is minimal; expand for statistical confidence |
| **Cross-patient search** | **Patient Matcher** (entity overlap) exists on branch **`pratik`** only; needs a dedicated query set and metrics before production claims |
| **Production hardening** | Audit logging, auth, PHI handling, and deployment topology (e.g. reverse proxy, GPU nodes) remain out of scope for this research repo |

Older README bullets that are **obsolete** *as open tasks*: “try higher **k** blindly”, “add hybrid as a blind next step”, “replace Streamlit with Flask/React” — the stack now ships **FastAPI + React** as above; retrieval **k** and **hybrid** trade-offs are analyzed in the notebook.

---

## License and data use

**MIMIC-IV** use is subject to PhysioNet credentialing and your institution’s agreements. Do not redistribute raw data paths or identifiers in public artifacts.

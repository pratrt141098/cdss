# app.py
import time
import streamlit as st
from modules.ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.ner import NERModule
from modules.embedding import EmbeddingModule
from modules.vector_store import VectorStoreModule
from modules.generation import GenerationModule

st.set_page_config(
    page_title="CDSS — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    n_patients  = st.slider("Patients to load", 5, 50, 10, step=5)
    n_results   = st.slider("Retrieved chunks", 3, 10, 5)

    if st.button("🔄 Reload pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Running fully local — no data leaves this machine.")


# ── Pipeline (cached, only reruns if n_patients changes) ─────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(n: int):
    records   = DataIngestionModule().process(input_data=n)   # dict[str, PatientRecord]
    chunks    = PreprocessingModule().process(records)         # pass dict directly
    annotated = NERModule().process(chunks)
    embedded  = EmbeddingModule().process(annotated)
    store     = VectorStoreModule(config={"collection_name": f"cdss_{n}"})
    store.process(embedded)
    embedder  = EmbeddingModule()
    generator = GenerationModule()
    hadm_ids  = sorted(records.keys())  # keys are already hadm_id strings
    return store, embedder, generator, hadm_ids


with st.spinner("Loading pipeline... this may take a minute on first run."):
    store, embedder, generator, hadm_ids = load_pipeline(n_patients)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏥 Clinical Decision Support System")
st.caption("Ask questions about patient records. Answers are grounded in discharge notes only.")

# ── Patient selector (instant — no pipeline dependency) ───────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    filter_mode = st.radio(
        "Retrieval scope",
        ["All patients", "Single patient"],
        horizontal=True,
    )
with col2:
    hadm_filter = None
    if filter_mode == "Single patient":
        hadm_filter = st.selectbox("Select patient (HADM ID)", hadm_ids)

st.divider()

# ── Chat interface ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source chunks"):
                for src in msg["sources"]:
                    section = src["metadata"]["section"].replace("_", " ").title()
                    st.markdown(f"**{section}** (score: `{src['score']:.3f}`)")
                    st.code(src["text"][:300])

if query := st.chat_input("Ask about this patient (e.g. What medications is the patient on?)"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant records..."):
            query_vec = embedder.embed_query(query)
            results   = store.query(
                query_vec,
                n_results=n_results,
                hadm_id=hadm_filter,
            )

        with st.spinner("Generating answer..."):
            t0     = time.time()
            answer = generator.process({
                "query":          query,
                "context_chunks": results,
            })
            elapsed = time.time() - t0

        st.markdown(answer)
        st.caption(f"Generated in {elapsed:.1f}s · llama3.2:3b · {len(results)} chunks retrieved")

        with st.expander("📄 Source chunks"):
            for src in results:
                section = src["metadata"]["section"].replace("_", " ").title()
                st.markdown(f"**{section}** (score: `{src['score']:.3f}`)")
                st.code(src["text"][:300])

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": results,
    })
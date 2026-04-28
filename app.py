# app.py
import time
import streamlit as st
from modules.ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.ner import NERModule
from modules.embedding import EmbeddingModule
from modules.vector_store import VectorStoreModule
from modules.generation import GenerationModule
from modules.query_router import route_query
from modules.query_expansion import generate_query_variants
from modules.patient_matcher import find_candidate_patients

st.set_page_config(
    page_title="CDSS — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    n_patients  = st.slider("Rows to load (admissions)", 50, 500, 147, step=50)
    n_results   = st.slider("Retrieved chunks", 3, 15, 10)

    st.divider()
    st.subheader("Retrieval mode")
    use_rag_fusion = st.toggle(
        "RAG-Fusion (cross-patient)",
        value=False,
        help=(
            "Generates 3 query variants and merges results via Reciprocal Rank Fusion. "
            "Disables patient scoping — retrieves from the full corpus."
        ),
    )
    if use_rag_fusion:
        st.caption("⚠️ Patient filter is disabled in cross-patient mode.")

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
if "active_hadm_id" not in st.session_state:
    st.session_state.active_hadm_id = None

# Reset conversation when the user switches patient or retrieval mode
current_context = (hadm_filter, use_rag_fusion)
if st.session_state.get("_last_context") != current_context:
    st.session_state.messages       = []
    st.session_state.active_hadm_id = hadm_filter
    st.session_state["_last_context"] = current_context

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
            if use_rag_fusion:
                # On follow-up turns reuse the matched patient from the first turn
                locked_hadm = st.session_state.active_hadm_id
                if locked_hadm:
                    routed_section = route_query(query)
                    if routed_section:
                        results        = store.get_by_section(routed_section, locked_hadm)
                        retrieval_mode = f"locked → {locked_hadm} · section-routed"
                    else:
                        query_vec      = embedder.embed_query(query)
                        results        = store.query(query_vec, n_results=n_results, hadm_id=locked_hadm)
                        retrieval_mode = f"locked → {locked_hadm} · vector search"
                else:
                    with st.status("Identifying patient from query…", expanded=True) as status:
                        patient_map = store.get_patient_entity_map()
                        candidates  = find_candidate_patients(query, patient_map)
                        if candidates:
                            best_hadm, overlap = candidates[0]
                            st.session_state.active_hadm_id = best_hadm
                            status.update(
                                label=f"Matched patient {best_hadm} ({overlap} entity overlaps)",
                                state="complete",
                            )
                            routed_section = route_query(query)
                            if routed_section:
                                results        = store.get_by_section(routed_section, best_hadm)
                                retrieval_mode = (
                                    f"entity-match → {best_hadm} "
                                    f"({overlap} overlaps) · section-routed"
                                )
                            else:
                                query_vec      = embedder.embed_query(query)
                                results        = store.query(
                                    query_vec, n_results=n_results, hadm_id=best_hadm
                                )
                                retrieval_mode = (
                                    f"entity-match → {best_hadm} "
                                    f"({overlap} overlaps) · vector search"
                                )
                        else:
                            status.update(
                                label="No entity matches — falling back to RAG-Fusion",
                                state="complete",
                            )
                            variants       = generate_query_variants(query)
                            results        = store.rag_fusion_query(
                                variants, embedder, n_results=n_results
                            )
                            retrieval_mode = f"RAG-Fusion fallback ({len(variants)} variants)"
            else:
                routed_section = route_query(query) if hadm_filter else None
                if routed_section:
                    results        = store.get_by_section(routed_section, hadm_filter)
                    retrieval_mode = f"section-routed → {routed_section.replace('_', ' ')}"
                else:
                    query_vec      = embedder.embed_query(query)
                    results        = store.query(query_vec, n_results=n_results, hadm_id=hadm_filter)
                    retrieval_mode = "vector search"

        # Build conversation history for multi-turn context (role/content pairs only)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        t0      = time.time()
        answer  = st.write_stream(generator.stream({
            "query":          query,
            "context_chunks": results,
            "history":        history,
        }))
        elapsed = time.time() - t0

        st.caption(f"Generated in {elapsed:.1f}s · llama3.2:3b · {len(results)} chunks · {retrieval_mode}")

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
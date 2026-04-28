"""
Module-level pipeline singleton shared across all route handlers.

build() is designed to run in a background thread so the server accepts
connections immediately. Poll status / get_status() for progress.
"""
import logging
import threading

from modules.ingestion import DataIngestionModule
from modules.preprocessing import PreprocessingModule
from modules.ner import NERModule
from modules.embedding import EmbeddingModule
from modules.vector_store import VectorStoreModule
from modules.generation import GenerationModule

logger = logging.getLogger("cdss.pipeline")

state: dict = {}

# Guarded by _lock; safe to read from any thread
status: dict = {
    "ready":    False,
    "building": False,
    "logs":     [],
    "error":    None,
}
_lock = threading.Lock()


def _log(msg: str) -> None:
    logger.info(msg)
    with _lock:
        status["logs"].append(msg)


def get_status() -> dict:
    with _lock:
        return {
            "ready":    status["ready"],
            "building": status["building"],
            "logs":     list(status["logs"]),
            "error":    status.get("error"),
        }


def build(n: int) -> None:
    """Build (or rebuild) the full RAG pipeline for `n` admissions."""
    with _lock:
        status.update(ready=False, building=True, logs=[], error=None)

    try:
        _log(f"Loading {n} admissions from MIMIC-IV…")
        records = DataIngestionModule().process(input_data=n)
        _log(
            f"Loaded {len(records)} records · "
            f"{sum(len(r.diagnoses) for r in records.values())} diagnoses · "
            f"{sum(len(r.medications) for r in records.values())} medication orders"
        )

        _log("Preprocessing discharge notes into text chunks…")
        chunks = PreprocessingModule().process(records)
        _log(f"Produced {len(chunks)} text chunks")

        _log("Running clinical NER (en_core_sci_sm)…")
        annotated = NERModule().process(chunks)
        _log(f"NER complete — {len(annotated)} chunks annotated")

        _log("Generating embeddings via Ollama (nomic-embed-text)…")
        embedded = EmbeddingModule().process(annotated)
        _log(f"Embedded {len(embedded)} chunks")

        _log("Indexing vectors in ChromaDB…")
        store = VectorStoreModule(config={"collection_name": f"cdss_{n}"})
        store.process(embedded)
        _log("Vector index ready")

        _log("Aggregating NER entity index…")
        ner_lookup: dict = {}
        for chunk in embedded:
            hid = str(chunk.get("hadm_id", ""))
            if not hid:
                continue
            if hid not in ner_lookup:
                ner_lookup[hid] = {
                    "medications": set(),
                    "diseases":    set(),
                    "anatomy":     set(),
                    "procedures":  set(),
                }
            for key in ner_lookup[hid]:
                for item in chunk.get("entities", {}).get(key, []):
                    item = item.strip()
                    if item:
                        ner_lookup[hid][key].add(item)

        ner_entities = {
            hid: {k: sorted(v) for k, v in buckets.items()}
            for hid, buckets in ner_lookup.items()
        }

        embedder  = EmbeddingModule()
        generator = GenerationModule()
        hadm_ids  = sorted(records.keys())

        state.update(
            store=store,
            embedder=embedder,
            generator=generator,
            hadm_ids=hadm_ids,
            records=records,
            ner_entities=ner_entities,
            n_patients=n,
        )

        _log(f"Pipeline ready — {len(hadm_ids)} admissions loaded ✓")
        with _lock:
            status.update(ready=True, building=False)

    except Exception as exc:
        logger.error("Pipeline build failed: %s", exc, exc_info=True)
        with _lock:
            status.update(ready=False, building=False, error=str(exc))

import time

from fastapi import APIRouter, HTTPException
from api import pipeline
from api.schemas import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run RAG for a specific patient and return the answer with source chunks."""
    if req.hadm_id not in pipeline.state["hadm_ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"HADM ID {req.hadm_id!r} not found in loaded admissions.",
        )

    store     = pipeline.state["store"]
    embedder  = pipeline.state["embedder"]
    generator = pipeline.state["generator"]

    query_vec = embedder.embed_query(req.query)
    results   = store.query(query_vec, n_results=req.n_results, hadm_id=req.hadm_id)

    t0      = time.time()
    answer  = generator.process({"query": req.query, "context_chunks": results})
    elapsed = time.time() - t0

    sources = [
        SourceChunk(
            text=r["text"][:400],
            section=r["metadata"].get("section", "unknown").replace("_", " ").title(),
            score=round(r.get("score", 0.0), 3),
        )
        for r in results
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        elapsed_s=round(elapsed, 2),
        model="llama3.2:3b",
        n_chunks=len(results),
    )

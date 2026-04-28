"""
FastAPI backend for the CDSS React frontend.

Run:  uvicorn api.main:app --reload
Env:  CDSS_N_PATIENTS   — admissions to load (default 147)
      CDSS_CORS_ORIGINS  — comma-separated allowed origins (default http://localhost:5173)
"""
import os
import threading
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from api import pipeline
from api.deps import require_ready
from api.routes import patients, query, summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

N_PATIENTS   = int(os.getenv("CDSS_N_PATIENTS", "147"))
CORS_ORIGINS = os.getenv("CDSS_CORS_ORIGINS", "http://localhost:5173").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build pipeline in a background thread so the server accepts connections
    # immediately. Clients should poll GET /status until ready=true.
    thread = threading.Thread(target=pipeline.build, args=(N_PATIENTS,), daemon=True)
    thread.start()
    yield
    pipeline.state.clear()


app = FastAPI(
    title="CDSS API",
    description="Clinical Decision Support System — RAG over MIMIC-IV discharge notes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def get_status() -> dict:
    """Return pipeline build status and log messages. Safe to poll at any time."""
    return pipeline.get_status()


# All data routes require a ready pipeline
app.include_router(patients.router, dependencies=[Depends(require_ready)])
app.include_router(query.router,    dependencies=[Depends(require_ready)])
app.include_router(summary.router)  # reload manages its own state

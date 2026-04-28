import threading
from fastapi import APIRouter
from api import pipeline

router = APIRouter()


@router.post("/pipeline/reload", status_code=202)
def reload_pipeline() -> dict:
    """Kick off a background pipeline rebuild. Poll GET /status for progress."""
    n = pipeline.state.get("n_patients", 147)
    pipeline.state.clear()
    thread = threading.Thread(target=pipeline.build, args=(n,), daemon=True)
    thread.start()
    return {"status": "rebuilding", "n_patients": n}

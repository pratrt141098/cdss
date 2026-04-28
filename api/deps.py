from fastapi import HTTPException
from api import pipeline


def require_ready() -> None:
    """FastAPI dependency — raises 503 if the pipeline hasn't finished building."""
    if not pipeline.status.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Pipeline is still building. Poll GET /status for progress.",
        )

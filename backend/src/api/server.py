import logging
import uuid
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from backend.src.api.telemetry import setup_telemetry
from backend.src.config import ConfigError, get_settings
from backend.src.graph.nodes import is_youtube_url
from backend.src.graph.state import ComplianceIssue

logger = logging.getLogger("frameguard.api")


class AuditRequest(BaseModel):
    video_url: str
    video_id: str | None = None

    @field_validator("video_url")
    @classmethod
    def youtube_only(cls, v: str) -> str:
        if not is_youtube_url(v):
            raise ValueError("video_url must be a YouTube link")
        return v


class AuditResponse(BaseModel):
    session_id: str
    video_id: str
    status: Literal["PASS", "FAIL", "ERROR"]
    final_report: str
    compliance_results: list[ComplianceIssue]
    errors: list[str]


@lru_cache
def _azure_graph():
    from backend.src.graph.workflow import build_azure_graph

    return build_azure_graph()


def get_graph():
    try:
        return _azure_graph()
    except ConfigError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def create_app() -> FastAPI:
    setup_telemetry(get_settings())
    app = FastAPI(title="FrameGuard", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/audit", response_model=AuditResponse)
    def audit(request: AuditRequest, graph: Annotated[Any, Depends(get_graph)]) -> AuditResponse:
        session_id = str(uuid.uuid4())
        video_id = request.video_id or f"vid_{session_id[:8]}"
        logger.info("audit %s started for %s", session_id, request.video_url)
        result = graph.invoke({"video_url": request.video_url, "video_id": video_id})
        return AuditResponse(
            session_id=session_id,
            video_id=video_id,
            status=result.get("final_status", "ERROR"),
            final_report=result.get("final_report", ""),
            compliance_results=result.get("compliance_results", []),
            errors=result.get("errors", []),
        )

    return app


app = create_app()

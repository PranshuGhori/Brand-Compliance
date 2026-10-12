import logging
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from backend.src.graph.state import AuditReport, VideoAuditState

logger = logging.getLogger("frameguard")

MAX_QUERY_CHARS = 4000

SYSTEM_PROMPT = """You are a brand safety and advertising compliance officer.
Audit the video transcript and on-screen text (OCR) against the guidelines below.
Only report a violation if a guideline below supports it.

Guidelines:
{guidelines}

Return a single JSON object, no prose:
{{
  "compliance_issues": [
    {{
      "category": "short label, e.g. Disclosure, Claim Validation, Ad Specs",
      "severity": "High | Medium | Low",
      "description": "what in the video breaks the guideline",
      "timestamp": "start time from the transcript or OCR, or null",
      "guideline_quote": "verbatim sentence copied from the guidelines above"
    }}
  ],
  "final_report": "two or three sentence summary of the audit"
}}
If there are no violations, return an empty compliance_issues list."""


class VideoIndexer(Protocol):
    def index(self, video_url: str, video_id: str) -> dict[str, Any]: ...


class GuidelineRetriever(Protocol):
    def search(self, query: str, k: int) -> list[str]: ...


def is_youtube_url(url: str) -> bool:
    return any(host in url for host in ("youtube.com/", "youtu.be/"))


def index_video_node(state: VideoAuditState, indexer: VideoIndexer) -> dict[str, Any]:
    video_url = state.get("video_url", "")
    video_id = state.get("video_id") or "vid_demo"
    logger.info("[indexer] processing %s", video_url)

    if not is_youtube_url(video_url):
        return _error(f"not a YouTube URL: {video_url!r}")
    try:
        return indexer.index(video_url, video_id)
    except Exception as e:
        logger.exception("video indexing failed")
        return _error(f"video indexing failed: {e}")


def audit_content_node(
    state: VideoAuditState, retriever: GuidelineRetriever, llm: BaseChatModel, k: int = 3
) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", [])
    if not transcript and not ocr_text:
        return _error("no transcript or on-screen text to audit")

    query = f"{transcript}\n{' '.join(ocr_text)}"[:MAX_QUERY_CHARS]
    try:
        guidelines = retriever.search(query, k)
    except Exception as e:
        logger.exception("guideline retrieval failed")
        return _error(f"guideline retrieval failed: {e}")

    user_message = (
        f"VIDEO METADATA: {state.get('video_metadata', {})}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"ON-SCREEN TEXT (OCR):\n" + "\n".join(ocr_text)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(guidelines="\n\n".join(guidelines))),
        HumanMessage(content=user_message),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        logger.exception("auditor model call failed")
        return {**_error(f"auditor model call failed: {e}"), "retrieved_guidelines": guidelines}

    usage = dict(getattr(response, "usage_metadata", None) or {})
    try:
        report = parse_report(response.content)
    except (ValueError, ValidationError) as e:
        logger.error("invalid auditor output: %s\n%s", e, response.content)
        return {
            **_error(f"auditor returned invalid JSON: {e}"),
            "retrieved_guidelines": guidelines,
            "token_usage": usage,
        }

    return {
        "retrieved_guidelines": guidelines,
        "compliance_results": report.compliance_issues,
        "final_status": "FAIL" if report.compliance_issues else "PASS",
        "final_report": report.final_report,
        "token_usage": usage,
    }


def parse_report(content: str) -> AuditReport:
    """Validate model output, tolerating markdown fences or text around the JSON object."""
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model output")
    return AuditReport.model_validate_json(content[start : end + 1])


def _error(message: str) -> dict[str, Any]:
    return {"errors": [message], "final_status": "ERROR", "final_report": message}

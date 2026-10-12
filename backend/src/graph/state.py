import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ComplianceIssue(BaseModel):
    category: str
    description: str
    severity: Literal["High", "Medium", "Low"]
    timestamp: str | None = None
    guideline_quote: str = Field(description="Verbatim text of the guideline that is violated.")


class AuditReport(BaseModel):
    """The JSON object the auditor model must return."""

    compliance_issues: list[ComplianceIssue] = []
    final_report: str


class VideoAuditState(TypedDict, total=False):
    # inputs
    video_id: str
    video_url: str

    # extraction
    video_metadata: dict[str, Any]
    transcript: str
    ocr_text: list[str]

    # retrieval and analysis
    retrieved_guidelines: list[str]
    compliance_results: Annotated[list[ComplianceIssue], operator.add]
    token_usage: dict[str, int]

    # report
    final_status: Literal["PASS", "FAIL", "ERROR"]
    final_report: str

    errors: Annotated[list[str], operator.add]

import pytest
from pydantic import ValidationError

from backend.src.graph.nodes import parse_report
from tests.helpers import DISCLOSURE_ISSUE, reply


def test_parses_plain_json():
    report = parse_report(reply([DISCLOSURE_ISSUE]))
    assert report.compliance_issues[0].category == "Disclosure"
    assert report.compliance_issues[0].severity == "High"


def test_parses_fenced_json():
    report = parse_report(f"```json\n{reply()}\n```")
    assert report.compliance_issues == []
    assert report.final_report == "Audit complete."


def test_timestamp_is_optional():
    issue = {k: v for k, v in DISCLOSURE_ISSUE.items() if k != "timestamp"}
    assert parse_report(reply([issue])).compliance_issues[0].timestamp is None


@pytest.mark.parametrize("field,value", [("severity", "Critical"), ("guideline_quote", None)])
def test_rejects_invalid_issue(field, value):
    with pytest.raises(ValidationError):
        parse_report(reply([{**DISCLOSURE_ISSUE, field: value}]))


def test_rejects_missing_report():
    with pytest.raises(ValidationError):
        parse_report('{"compliance_issues": []}')


def test_rejects_non_json():
    with pytest.raises(ValueError):
        parse_report("I could not find any issues.")

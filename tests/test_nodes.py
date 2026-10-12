from backend.src.graph.nodes import audit_content_node, index_video_node
from backend.src.services.fakes import FakeChatModel, FakeIndexer, FakeRetriever
from tests.helpers import DISCLOSURE_ISSUE, GUIDELINES, YOUTUBE_URL, fake_llm, reply

STATE = {"transcript": "[0:00:03] free serum, cures acne", "ocr_text": ["[0:00:05] #ad"]}


def test_indexer_returns_extracted_content(indexer):
    out = index_video_node({"video_url": YOUTUBE_URL, "video_id": "v1"}, indexer)
    assert out["transcript"].startswith("[0:00:03]")
    assert indexer.calls == [(YOUTUBE_URL, "v1")]


def test_indexer_rejects_non_youtube_url(indexer):
    out = index_video_node({"video_url": "https://vimeo.com/1"}, indexer)
    assert out["final_status"] == "ERROR"
    assert "not a YouTube URL" in out["errors"][0]
    assert indexer.calls == []


def test_indexer_failure_is_reported():
    out = index_video_node({"video_url": YOUTUBE_URL}, FakeIndexer(error=RuntimeError("quota")))
    assert out["final_status"] == "ERROR"
    assert "quota" in out["errors"][0]


def test_auditor_reports_violations(retriever):
    out = audit_content_node(STATE, retriever, fake_llm(reply([DISCLOSURE_ISSUE])), k=2)
    assert out["final_status"] == "FAIL"
    assert out["compliance_results"][0].category == "Disclosure"
    assert len(out["retrieved_guidelines"]) == 2
    assert out["token_usage"]["total_tokens"] > 0


def test_auditor_passes_clean_video(retriever):
    out = audit_content_node(STATE, retriever, fake_llm(reply()))
    assert out["final_status"] == "PASS"
    assert out["compliance_results"] == []


def test_auditor_puts_retrieved_guidelines_in_prompt():
    seen = []
    llm = FakeChatModel(respond=lambda messages: seen.append(messages) or reply())
    audit_content_node(STATE, FakeRetriever(GUIDELINES), llm, k=3)
    system_prompt = seen[0][0].content
    assert all(g in system_prompt for g in GUIDELINES)
    assert "free serum" in seen[0][1].content


def test_auditor_skips_empty_content(retriever):
    out = audit_content_node({"transcript": "", "ocr_text": []}, retriever, fake_llm(reply()))
    assert out["final_status"] == "ERROR"


def test_auditor_handles_invalid_model_output(retriever):
    out = audit_content_node(STATE, retriever, fake_llm("sorry, no JSON today"))
    assert out["final_status"] == "ERROR"
    assert "invalid JSON" in out["errors"][0]


def test_auditor_handles_retrieval_failure():
    retriever = FakeRetriever([], error=ConnectionError("search down"))
    out = audit_content_node(STATE, retriever, fake_llm(reply()))
    assert out["final_status"] == "ERROR"
    assert "search down" in out["errors"][0]


def test_auditor_handles_model_failure(retriever):
    def boom(messages):
        raise TimeoutError("model timeout")

    out = audit_content_node(STATE, retriever, FakeChatModel(respond=boom))
    assert out["final_status"] == "ERROR"
    assert "model timeout" in out["errors"][0]

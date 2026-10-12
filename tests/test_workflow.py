from backend.src.graph.workflow import create_graph
from backend.src.services.fakes import FakeChatModel, FakeIndexer
from tests.helpers import DISCLOSURE_ISSUE, YOUTUBE_URL, fake_llm, reply


def test_end_to_end_with_fakes(indexer, retriever):
    graph = create_graph(indexer, retriever, fake_llm(reply([DISCLOSURE_ISSUE], "One issue.")))
    result = graph.invoke({"video_url": YOUTUBE_URL, "video_id": "v1"})

    assert result["final_status"] == "FAIL"
    assert result["final_report"] == "One issue."
    assert [i.category for i in result["compliance_results"]] == ["Disclosure"]
    assert "free" in result["transcript"]
    assert result["retrieved_guidelines"]
    assert result.get("errors", []) == []


def test_indexer_error_skips_auditor(retriever):
    calls = []
    llm = FakeChatModel(respond=lambda m: calls.append(m) or reply())
    graph = create_graph(FakeIndexer(error=RuntimeError("upload failed")), retriever, llm)
    result = graph.invoke({"video_url": YOUTUBE_URL})

    assert result["final_status"] == "ERROR"
    assert result["errors"] == ["video indexing failed: upload failed"]
    assert calls == []

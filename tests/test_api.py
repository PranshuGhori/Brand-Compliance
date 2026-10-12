import pytest
from fastapi.testclient import TestClient

from backend.src.api.server import app, get_graph
from backend.src.graph.workflow import create_graph
from backend.src.services.fakes import FakeIndexer
from tests.helpers import DISCLOSURE_ISSUE, YOUTUBE_URL, fake_llm, reply


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_graph(graph):
    app.dependency_overrides[get_graph] = lambda: graph


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_audit_returns_report(client, indexer, retriever):
    use_graph(create_graph(indexer, retriever, fake_llm(reply([DISCLOSURE_ISSUE]))))
    resp = client.post("/audit", json={"video_url": YOUTUBE_URL, "video_id": "v1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FAIL"
    assert body["video_id"] == "v1"
    assert body["compliance_results"][0]["severity"] == "High"
    assert body["errors"] == []


def test_audit_reports_pipeline_error(client, retriever):
    use_graph(create_graph(FakeIndexer(error=RuntimeError("boom")), retriever, fake_llm(reply())))
    body = client.post("/audit", json={"video_url": YOUTUBE_URL}).json()

    assert body["status"] == "ERROR"
    assert body["video_id"].startswith("vid_")
    assert "boom" in body["errors"][0]


def test_audit_rejects_non_youtube_url(client, indexer, retriever):
    use_graph(create_graph(indexer, retriever, fake_llm(reply())))
    resp = client.post("/audit", json={"video_url": "https://example.com/video.mp4"})
    assert resp.status_code == 422


def test_audit_without_configuration_returns_503(client, monkeypatch):
    for name in ("AZURE_VI_ACCOUNT_ID", "AZURE_OPENAI_API_KEY", "AZURE_SEARCH_API_KEY"):
        monkeypatch.setenv(name, "")
    from backend.src.api import server
    from backend.src.config import get_settings

    get_settings.cache_clear()
    server._azure_graph.cache_clear()
    resp = client.post("/audit", json={"video_url": YOUTUBE_URL})
    get_settings.cache_clear()

    assert resp.status_code == 503
    assert "missing environment variables" in resp.json()["detail"]

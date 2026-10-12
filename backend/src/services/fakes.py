"""In-memory stand-ins for Video Indexer, Azure AI Search and Azure OpenAI.

Used by the tests and by `eval/run_eval.py --offline`.
"""

import re
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeIndexer:
    def __init__(
        self,
        transcript: str = "",
        ocr_text: list[str] | None = None,
        error: Exception | None = None,
    ):
        self.transcript = transcript
        self.ocr_text = ocr_text or []
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def index(self, video_url: str, video_id: str) -> dict[str, Any]:
        self.calls.append((video_url, video_id))
        if self.error:
            raise self.error
        return {
            "transcript": self.transcript,
            "ocr_text": self.ocr_text,
            "video_metadata": {"source": "fake"},
        }


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9#]+", text.lower()))


class FakeRetriever:
    """Ranks passages by word overlap with the query."""

    def __init__(self, passages: list[str], error: Exception | None = None):
        self.passages = passages
        self.error = error

    def search(self, query: str, k: int) -> list[str]:
        if self.error:
            raise self.error
        q = _words(query)
        ranked = sorted(self.passages, key=lambda p: len(q & _words(p)), reverse=True)
        return ranked[:k]


class FakeChatModel(BaseChatModel):
    """Chat model whose reply is computed by `respond`. Token counts are whitespace word counts."""

    respond: Callable[[list[BaseMessage]], str]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self.respond(messages)
        n_in = sum(len(str(m.content).split()) for m in messages)
        n_out = len(text.split())
        usage = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        message = AIMessage(content=text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])

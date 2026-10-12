import pytest

from backend.src.services.fakes import FakeIndexer, FakeRetriever
from tests.helpers import GUIDELINES


@pytest.fixture
def indexer():
    return FakeIndexer(
        transcript="[0:00:03] This serum was sent to me free and it cures acne in two days.",
        ocr_text=["[0:00:05] Link in bio"],
    )


@pytest.fixture
def retriever():
    return FakeRetriever(GUIDELINES)

"""Chunk the guideline PDFs in backend/data and upload them to Azure AI Search.

uv run python -m backend.scripts.index_documents
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.src.config import get_settings
from backend.src.services.azure import build_vector_store

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("frameguard.index_documents")


def index_docs(data_dir: Path = DATA_DIR) -> int:
    settings = get_settings()
    store = build_vector_store(settings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {data_dir}")

    total = 0
    for pdf in pdfs:
        chunks = splitter.split_documents(PyPDFLoader(str(pdf)).load())
        for chunk in chunks:
            chunk.metadata["source"] = pdf.name
        store.add_documents(chunks)
        total += len(chunks)
        logger.info("indexed %s: %d chunks", pdf.name, len(chunks))

    logger.info("done: %d chunks into index %r", total, settings.azure_search_index_name)
    return total


if __name__ == "__main__":
    index_docs()

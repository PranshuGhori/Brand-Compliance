import os
import glob
import logging
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearchVectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger("indexer")

def index_docs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../../backend/data")

logger.info("="*50)
logger.info("Environment configuration check:")
logger.info(f'AZURE_OPENAI_ENDPOINT: {os.getenv("AZURE_OPENAI_ENDPOINT")}')
logger.info(f'AZURE_OPENAI_API_VERSION: {os.getenv("AZURE_OPENAI_API_VERSION")}')
logger.info(f'AZURE_OPENAI_CHAT_DEPLOYMENT: {os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")}')
logger.info(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')}")
logger.info(f'AZURE_SEARCH_ENDPOINT: {os.getenv("AZURE_SEARCH_ENDPOINT")}')
logger.info(f'AZURE_SEARCH_API_KEY: {os.getenv("AZURE_SEARCH_API_KEY")}')
logger.info(f'AZURE_SEARCH_INDEX_NAME: {os.getenv("AZURE_SEARCH_INDEX_NAME")}')
logger.info(f'AZURE_SEARCH_LANGUAGE_KEY: {os.getenv("AZURE_SEARCH_LANGUAGE_KEY")}')
logger.info(f'AZURE_SEARCH_LANGUAGE_NAME: {os.getenv("AZURE_SEARCH_LANGUAGE_NAME")}')
logger.info(f'AZURE_SEARCH_LANGUAGE_VERSION: {os.getenv("AZURE_SEARCH_LANGUAGE_VERSION")}')

logger.info("="*50)


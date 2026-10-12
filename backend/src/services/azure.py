from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from backend.src.config import Settings

OPENAI_FIELDS = ("azure_openai_endpoint", "azure_openai_api_key", "azure_openai_api_version")


def build_chat_model(settings: Settings) -> AzureChatOpenAI:
    settings.require(*OPENAI_FIELDS, "azure_openai_chat_deployment")
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def build_vector_store(settings: Settings) -> AzureSearch:
    settings.require(
        *OPENAI_FIELDS,
        "azure_openai_embedding_deployment",
        "azure_search_endpoint",
        "azure_search_api_key",
        "azure_search_index_name",
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )
    return AzureSearch(
        azure_search_endpoint=settings.azure_search_endpoint,
        azure_search_key=settings.azure_search_api_key,
        index_name=settings.azure_search_index_name,
        embedding_function=embeddings.embed_query,
    )


class AzureSearchRetriever:
    def __init__(self, store: AzureSearch):
        self.store = store

    def search(self, query: str, k: int) -> list[str]:
        return [doc.page_content for doc in self.store.similarity_search(query, k=k)]

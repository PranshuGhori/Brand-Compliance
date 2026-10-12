from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from backend.src.config import Settings, get_settings
from backend.src.graph.nodes import (
    GuidelineRetriever,
    VideoIndexer,
    audit_content_node,
    index_video_node,
)
from backend.src.graph.state import VideoAuditState


def create_graph(
    indexer: VideoIndexer, retriever: GuidelineRetriever, llm: BaseChatModel, k: int = 3
):
    workflow = StateGraph(VideoAuditState)
    workflow.add_node("indexer", partial(index_video_node, indexer=indexer))
    workflow.add_node("auditor", partial(audit_content_node, retriever=retriever, llm=llm, k=k))

    workflow.set_entry_point("indexer")
    workflow.add_conditional_edges(
        "indexer", lambda s: END if s.get("errors") else "auditor", ["auditor", END]
    )
    workflow.add_edge("auditor", END)
    return workflow.compile()


def build_azure_graph(settings: Settings | None = None):
    from backend.src.services.azure import (
        AzureSearchRetriever,
        build_chat_model,
        build_vector_store,
    )
    from backend.src.services.video_indexer import VideoIndexerService

    settings = settings or get_settings()
    return create_graph(
        indexer=VideoIndexerService(settings),
        retriever=AzureSearchRetriever(build_vector_store(settings)),
        llm=build_chat_model(settings),
        k=settings.retrieval_k,
    )

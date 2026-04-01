import os 
import json 
import logging
import re
from typing import List, Dict, Any

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from lanfchain_community.vectorstores import AzureSearchVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#import state schema
from backend.src.graph.state import VideoAuditState, ComplianceIssue

#import service
from backend.src.services.azure_video_indexer import VideoIndexerService

#configure logging
logger = logging.getLogger("brand-guardian")
logging.basicConfig(
    level=logging.INFO)

def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    video_url = state.get("video_url")
    video_id_input = state.get("video_id","vid_demo")

    logger.info(f"---- [Node: Indexer] Processing: {video_url}")

    local_filename = "temp_audit_video.mp4"
    
    try:
        vi_service = VideoIndexerService()
        
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid youtube url"
            )
        azure_video_id = vi_service.upload_and_index(local_path, video_id_input)
        logger.info(f"Uploaded video to Azure Video Indexer with ID: {azure_video_id}")

        if os.path.exists(local_path):
            os.remove(local_path)
            
        raw_insights = vi_service.wait_for_processing(azure_video_id)
        
        clean_data = vi_service.extract_clean_data(raw_insights)
        
        logger.info(f"---- [Node: Indexer] Extraction Complete -----------")
        
        return clean_data
    
    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAILED",
            "transcript": "",
            "ocr_text": [],
        }
        
def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    logger.info("---- [Node: Audio Content] Processing Audio Content ----")
    
    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("No transcript available for audio content analysis")
        return {
            "final_status": "FAILED",
            "final_report": "Audit skipped because no transcript was available",
        }
    
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    vector_store = AzureSearchVectorStore(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query)

    ocr_text = state.get("ocr_text",[])
    quert_text = f"{transcript} {''.join(ocr_text)}"
    docs = vector_store.similarity_search(query=query_text, k=3)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""
    You are a Brand Safety Compliance Officer.
    Analyze the following video transcript and OCR text against the retrieved brand safety rules.
    
    Retrieved Rules:
    {retrieved_rules}
    
    Instructions:
    1. Analyze the video transcript and OCR text against the retrieved brand safety rules.
    2. Identify any violations of the brand safety rules in the video content.
    3. Return the violations in JSON format with the following structure:
        {{
    "compliance_issues": [
        {{
            "category": "Claim Validation",
            "severity": "High|Medium|Low",
            "description": "Description of the violation"
        }}
    ],
    
    "status": "PASS|FAIL"
    "final_report": "Final report of the video content"
    }}
    
    if no violations are found, return an empty list of compliance issues and status as PASS and "compliance_results" to [].
    """    
    
    user_message = f"""
    VIDEO_METADATA:{state.get("video_metadata",{})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR):{ocr_text}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
        content = response.content
        if "```" in content:
            content = re.search(r"```(.*?)```", content, re.DOTALL).group(1)


        audit_data = json.loads(content.strip())
        return {
            "compliance_results": audit_data.get("compliance_issues", []),
            "final_status": audit_data.get("status", "FAILED"),
            "final_report": audit_data.get("final_report", "no report generated")
        }

    except Exception as e:
        logger.error(f"system error in auditor node: {str(e)}")
        
        logger.error(f"Raw Response: {response.content if 'response' in locals() else 'No Response'}")
        
        return {
            "final_status": "FAILED",
            "final_report": "Audit skipped because audio content analysis failed",
        }
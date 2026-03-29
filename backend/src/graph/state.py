import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

#define the schema for single compliance result

class ComplianceIssue(TypedDict):
    category: str
    description: str
    severity: str
    timestamp: Optional[str]

#define global graph state    

class VideoAuditState(TypedDict):
    #video inputs
    video_id: str
    video_url: str

    #ingestion and extraction data
    video_metadata: Optional[Dict[str, Any]]
    local_file_path: Optional[str]
    transcript: Optional[str]
    ocr_text: List[str]

    #analysis output
    compliance_results: Annotated[List[ComplianceResult], operator.add]

    #final report
    final_status: str
    final_report: str
    
    #system observability
    errors: Annotated[List[str], operator.add]
    
    
    
    


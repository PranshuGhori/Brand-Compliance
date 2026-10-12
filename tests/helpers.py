import json

from backend.src.services.fakes import FakeChatModel

GUIDELINES = [
    "If making an endorsement in a video, the disclosure should be in the video and not just "
    "in the description uploaded with the video.",
    "You can't make up claims about a product that would require proof the advertiser doesn't "
    "have - such as scientific proof that a product can treat a health condition.",
    "Bumper ads must be 6 seconds or shorter.",
]

YOUTUBE_URL = "https://www.youtube.com/watch?v=abc123"

DISCLOSURE_ISSUE = {
    "category": "Disclosure",
    "severity": "High",
    "description": "Sponsored product with no disclosure in the video.",
    "timestamp": "0:00:03",
    "guideline_quote": "the disclosure should be in the video and not just in the description",
}


def reply(issues=(), report="Audit complete."):
    return json.dumps({"compliance_issues": list(issues), "final_report": report})


def fake_llm(text: str) -> FakeChatModel:
    return FakeChatModel(respond=lambda messages: text)

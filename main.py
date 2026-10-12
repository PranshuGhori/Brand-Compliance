"""Audit one video from the command line.

uv run python main.py https://youtu.be/<id>
"""

import argparse
import json
import logging

from backend.src.graph.workflow import build_azure_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YouTube video for compliance issues.")
    parser.add_argument("video_url")
    parser.add_argument("--video-id", default="vid_cli")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = build_azure_graph().invoke({"video_url": args.video_url, "video_id": args.video_id})
    print(
        json.dumps(
            {
                "status": result.get("final_status"),
                "final_report": result.get("final_report"),
                "compliance_results": [
                    i.model_dump() for i in result.get("compliance_results", [])
                ],
                "errors": result.get("errors", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

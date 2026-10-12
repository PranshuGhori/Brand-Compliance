import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from backend.src.config import Settings

logger = logging.getLogger("frameguard.indexer")

ARM_SCOPE = "https://management.azure.com/.default"
VI_API = "https://api.videoindexer.ai"


class VideoIndexerError(RuntimeError):
    pass


class VideoIndexerService:
    """Downloads a YouTube video and extracts transcript and OCR with Azure Video Indexer."""

    def __init__(
        self,
        settings: Settings,
        credential=None,
        poll_seconds: int = 30,
        timeout_seconds: int = 1800,
    ):
        settings.require(
            "azure_vi_account_id",
            "azure_vi_location",
            "azure_vi_name",
            "azure_subscription_id",
            "azure_resource_group",
        )
        if credential is None:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
        self.settings = settings
        self.credential = credential
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    @property
    def _base(self) -> str:
        s = self.settings
        return f"{VI_API}/{s.azure_vi_location}/Accounts/{s.azure_vi_account_id}"

    def index(self, video_url: str, video_id: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = self.download_youtube_video(video_url, Path(tmp) / "video.mp4")
            azure_video_id = self.upload_and_index(local_path, video_id)
        logger.info("uploaded %s to Video Indexer as %s", video_id, azure_video_id)
        return self.extract_clean_data(self.wait_for_processing(azure_video_id))

    def download_youtube_video(self, url: str, output_path: Path) -> Path:
        import yt_dlp

        opts = {"format": "best[ext=mp4]/best", "outtmpl": str(output_path), "quiet": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        if not output_path.exists():
            raise VideoIndexerError(f"download failed for {url}")
        return output_path

    def get_access_token(self) -> str:
        s = self.settings
        arm_token = self.credential.get_token(ARM_SCOPE).token
        url = (
            f"https://management.azure.com/subscriptions/{s.azure_subscription_id}"
            f"/resourceGroups/{s.azure_resource_group}/providers/Microsoft.VideoIndexer"
            f"/accounts/{s.azure_vi_name}/generateAccessToken?api-version=2024-01-01"
        )
        resp = requests.post(
            url,
            json={"permissionType": "Contributor", "scope": "Account"},
            headers={"Authorization": f"Bearer {arm_token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["accessToken"]

    def upload_and_index(self, local_path: Path, video_name: str) -> str:
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self._base}/Videos",
                params={
                    "accessToken": self.get_access_token(),
                    "name": video_name,
                    "privacy": "Private",
                    "indexingPreset": "Default",
                },
                files={"file": f},
                timeout=600,
            )
        if resp.status_code != 200:
            raise VideoIndexerError(f"upload failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json()["id"]

    def wait_for_processing(self, azure_video_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            resp = requests.get(
                f"{self._base}/Videos/{azure_video_id}/Index",
                params={"accessToken": self.get_access_token()},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state")
            if state == "Processed":
                return data
            if state == "Failed":
                raise VideoIndexerError(f"indexing failed for {azure_video_id}")
            logger.info("video %s state=%s, waiting", azure_video_id, state)
            time.sleep(self.poll_seconds)
        raise VideoIndexerError(f"indexing timed out for {azure_video_id}")

    @staticmethod
    def extract_clean_data(insights: dict[str, Any]) -> dict[str, Any]:
        video = (insights.get("videos") or [{}])[0].get("insights", {})

        def start(item: dict) -> str:
            instances = item.get("instances") or [{}]
            return instances[0].get("start", "")

        transcript = "\n".join(
            f"[{start(t)}] {t['text']}" for t in video.get("transcript", []) if t.get("text")
        )
        ocr_text = [f"[{start(o)}] {o['text']}" for o in video.get("ocr", []) if o.get("text")]
        metadata = {
            "duration": insights.get("summarizedInsights", {}).get("duration", {}).get("seconds"),
            "language": video.get("sourceLanguage"),
        }
        return {"transcript": transcript, "ocr_text": ocr_text, "video_metadata": metadata}

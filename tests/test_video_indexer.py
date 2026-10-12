from backend.src.services.video_indexer import VideoIndexerService

INSIGHTS = {
    "state": "Processed",
    "summarizedInsights": {"duration": {"seconds": 31.5}},
    "videos": [
        {
            "insights": {
                "sourceLanguage": "en-US",
                "transcript": [
                    {"text": "Hey everyone", "instances": [{"start": "0:00:00", "end": "0:00:02"}]},
                    {"text": "this cream cures acne", "instances": [{"start": "0:00:02.5"}]},
                ],
                "ocr": [{"text": "#ad", "instances": [{"start": "0:00:01"}]}],
            }
        }
    ],
}


def test_extract_clean_data():
    data = VideoIndexerService.extract_clean_data(INSIGHTS)
    assert data["transcript"] == "[0:00:00] Hey everyone\n[0:00:02.5] this cream cures acne"
    assert data["ocr_text"] == ["[0:00:01] #ad"]
    assert data["video_metadata"] == {"duration": 31.5, "language": "en-US"}


def test_extract_clean_data_tolerates_empty_insights():
    data = VideoIndexerService.extract_clean_data({})
    assert data["transcript"] == "" and data["ocr_text"] == []

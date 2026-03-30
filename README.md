# Compliance QA Pipeline

## Overview
Compliance QA Pipeline is an AI-driven, automated workflow designed to analyze video content for branding and compliance issues. Built with **LangGraph** and **LangChain**, the system orchestrates the ingestion of video data, extracts transcripts and OCR text, and evaluates the content against compliance guidelines. The final output is a structured compliance report detailing any violations, including category, description, severity, and timestamps.

## Key Features
- **Video Ingestion & Processing:** Uses `yt-dlp` to fetch videos and extracts relevant metadata.
- **Multimodal Extraction:** Gathers audio transcripts and on-screen text (OCR) for comprehensive analysis.
- **Compliance Analysis Workflow:** Powered by a LangGraph state machine (`VideoAuditState`) to methodically step through metadata extraction to result generation.
- **Reporting:** Generates detailed compliance results outlining errors, severity, and status.
- **Service Integrations:** Uses Azure services (Search, Blob, Monitor) and OpenAI models for AI-powered evaluation.
- **API & UI:** Provides a fast, scalable backend via FastAPI and an interactive frontend via Streamlit.

## Tech Stack
- **Python 3.13+**
- **LangChain & LangGraph**
- **FastAPI**
- **Streamlit**
- **Azure Search, OpenAI APIs**

## Setup
1. Ensure Python >= 3.13 is installed.
2. Install dependencies (e.g., via `uv` or standard package managers as specified in `pyproject.toml`).
3. Set your environment variables (like API keys) in a `.env` file based on `.env.example`.
4. Run the API server using FastAPI/Uvicorn or the UI using Streamlit.
##

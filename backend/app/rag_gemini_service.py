"""
Gemini text-generation service for the live pedagogical RAG condition.

This file is additive only. It does not modify the existing baseline code.

Role in the study:
- The File Search retrieval stage happens separately in rag_file_search_service.py.
- Retrieved instructional evidence is injected into the existing pedagogical RAG prompt.
- This service sends that final enriched prompt to Gemini 2.5 Flash.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-2.5-flash"


class RagGeminiServiceError(Exception):
    """Raised when live Gemini text generation cannot complete."""


def load_project_env() -> None:
    """Load project-level .env without modifying it."""
    load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    """Return GEMINI_API_KEY from .env or environment."""
    load_project_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RagGeminiServiceError(
            "GEMINI_API_KEY is missing. Your existing baseline used this key, "
            "so confirm it is still present in your .env file."
        )

    return api_key


def get_model_name(model_name: Optional[str] = None) -> str:
    """Resolve the Gemini model, defaulting to gemini-2.5-flash."""
    load_project_env()
    return model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def build_client(api_key: Optional[str] = None) -> genai.Client:
    """Create a Gemini Developer API client."""
    return genai.Client(api_key=api_key or get_api_key())


def extract_text_from_response(response: Any) -> str:
    """Extract non-empty text from a Gemini response."""
    text = getattr(response, "text", None)

    if text and str(text).strip():
        return str(text).strip()

    raise RagGeminiServiceError(
        "Gemini returned a response object, but response.text was empty."
    )


def generate_rag_tutor_response(
    prompt_text: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate the final pedagogical RAG tutor response.

    This call intentionally does NOT attach File Search as a tool.
    Retrieval already occurred in the prior stage, and the retrieved chunks
    are already embedded inside the final RAG prompt.
    """
    if not prompt_text or not prompt_text.strip():
        raise RagGeminiServiceError("Final RAG prompt text is empty.")

    client = build_client()
    resolved_model = get_model_name(model_name)

    response = client.models.generate_content(
        model=resolved_model,
        contents=prompt_text,
    )

    return extract_text_from_response(response)

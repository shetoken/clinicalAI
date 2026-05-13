"""Gemini client factory.

Two modes:

  USE_VERTEX_AI=true   -> Vertex AI (production path, BAA-covered, PHI-safe).
                          Requires GOOGLE_CLOUD_PROJECT and ADC credentials
                          (gcloud auth application-default login, or a workload
                          identity / service account JSON via GOOGLE_APPLICATION_CREDENTIALS).

  USE_VERTEX_AI=false  -> Direct Gemini API (dev only, never with real PHI).
                          Requires GEMINI_API_KEY.

Per the CenterWell Home Health roadmap, Gemini 2.5 Pro / Flash / Flash-Lite
all run via Vertex AI with the Google Cloud BAA when handling clinical text.
"""

from __future__ import annotations

import os
from functools import lru_cache

from google import genai


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Return a cached google-genai client configured per environment."""
    use_vertex = _truthy(os.getenv("USE_VERTEX_AI"))

    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "USE_VERTEX_AI=true but GOOGLE_CLOUD_PROJECT is not set. "
                "Set it (and ensure ADC is configured) before calling get_client()."
            )
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No GEMINI_API_KEY found. Either set USE_VERTEX_AI=true with a GCP project, "
            "or set GEMINI_API_KEY for direct API access (dev only — never with real PHI)."
        )
    return genai.Client(api_key=api_key)


def is_vertex_mode() -> bool:
    return _truthy(os.getenv("USE_VERTEX_AI"))

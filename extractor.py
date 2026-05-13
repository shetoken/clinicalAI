"""Visit-note feature extractor using Gemini structured output."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable

from google.genai import types
from pydantic import ValidationError

from .client import get_client
from .prompts import SYSTEM_PROMPT, render_user_prompt
from .schema import VisitNoteFeatures

logger = logging.getLogger(__name__)

# Default model: Flash for cost/latency. Override to gemini-2.5-pro for harder notes
# (long multi-page records, ambiguous documentation, or low-confidence retries).
DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class ExtractionResult:
    """Wrapper carrying the parsed features plus metadata about the call."""
    features: VisitNoteFeatures
    model: str
    note_id: str | None = None
    raw_response_text: str | None = None
    usage: dict | None = None


def _build_config(temperature: float, max_output_tokens: int | None) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=VisitNoteFeatures,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _parse(response) -> VisitNoteFeatures:
    """Prefer SDK-parsed object; fall back to text JSON parse if needed."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, VisitNoteFeatures):
        return parsed
    # Some SDK versions hand back a dict; coerce.
    if isinstance(parsed, dict):
        return VisitNoteFeatures.model_validate(parsed)
    # Last resort: parse the text payload.
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response had no parsed payload and no text.")
    try:
        return VisitNoteFeatures.model_validate_json(text)
    except ValidationError as e:
        logger.error("Pydantic validation failed. Raw text:\n%s", text)
        raise RuntimeError(f"Could not parse Gemini response as VisitNoteFeatures: {e}") from e


def extract_features(
    note_text: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int | None = 4096,
    note_id: str | None = None,
) -> ExtractionResult:
    """Extract structured features from a single visit note.

    Args:
        note_text:  Raw clinician visit note text.
        model:      Gemini model id (default gemini-2.5-flash).
        temperature: 0.0 is recommended for extraction; raises hallucination otherwise.
        max_output_tokens: Cap on response length.
        note_id:    Optional caller-supplied identifier for traceability.

    Returns:
        ExtractionResult with parsed VisitNoteFeatures and call metadata.
    """
    if not note_text or not note_text.strip():
        raise ValueError("note_text is empty")

    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=render_user_prompt(note_text),
        config=_build_config(temperature, max_output_tokens),
    )
    features = _parse(response)

    usage = None
    if getattr(response, "usage_metadata", None) is not None:
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
        }

    return ExtractionResult(
        features=features,
        model=model,
        note_id=note_id,
        raw_response_text=getattr(response, "text", None),
        usage=usage,
    )


async def _extract_async(note: tuple[str, str], model: str, temperature: float) -> ExtractionResult:
    note_id, text = note
    # google-genai exposes async on client.aio; if unavailable, fall back to a thread.
    client = get_client()
    aio = getattr(client, "aio", None)
    if aio is not None:
        response = await aio.models.generate_content(
            model=model,
            contents=render_user_prompt(text),
            config=_build_config(temperature, max_output_tokens=4096),
        )
        features = _parse(response)
        return ExtractionResult(features=features, model=model, note_id=note_id,
                                raw_response_text=getattr(response, "text", None))
    # Fallback: run the sync path in a worker thread.
    return await asyncio.to_thread(extract_features, text, model=model,
                                   temperature=temperature, note_id=note_id)


async def extract_batch_async(
    notes: Iterable[tuple[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    concurrency: int = 4,
) -> list[ExtractionResult]:
    """Concurrently extract features for many notes.

    Args:
        notes: iterable of (note_id, note_text) tuples.
        concurrency: max in-flight requests; tune to your Vertex quota.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(note):
        async with sem:
            try:
                return await _extract_async(note, model=model, temperature=temperature)
            except Exception as e:  # noqa: BLE001
                logger.exception("Extraction failed for note %s", note[0])
                return e

    return await asyncio.gather(*[_bounded(n) for n in notes])


def extract_batch(
    notes: Iterable[tuple[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    concurrency: int = 4,
) -> list[ExtractionResult]:
    """Sync wrapper around extract_batch_async."""
    return asyncio.run(
        extract_batch_async(notes, model=model, temperature=temperature, concurrency=concurrency)
    )

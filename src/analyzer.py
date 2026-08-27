"""LLM-based highlight detection and scoring."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

HIGHLIGHT_PROMPT = """\
You are a viral content analyst. Given a video transcript with timestamps, identify the most viral-worthy highlight clips.

For each highlight, provide:
- start: start time in seconds
- end: end time in seconds
- score: virality score 0-100
- reason: brief reason why this moment is viral-worthy
- title: catchy short title for the clip

CRITICAL RULE: Each clip MUST be between 20 and 40 seconds long. (end - start) must be >= 20 and <= 40.

Virality criteria: hook moments, emotional peaks, hot takes, revelations, conflict, quotables, story peaks, practical value, surprising facts.

Video content type: {content_type}

Transcript:
{transcript}

Return a JSON array of highlights sorted by score descending. Return at most {max_clips} clips.
Overlap rule: if two clips overlap by more than 5 seconds, keep only the higher-scored one.

Return ONLY valid JSON array, no markdown fences.\
"""

CONTENT_TYPE_PROMPT = """\
Classify this video into ONE category: podcast, interview, tutorial, vlog, speech, debate, documentary, comedy, other.

Also describe the pacing: fast, moderate, or slow.

Return JSON only: {{"content_type": "...", "pacing": "..."}}

Transcript (first 2000 chars):
{transcript}\
"""


@dataclass
class Highlight:
    start: float
    end: float
    score: float
    reason: str
    title: str


def _format_transcript(segments: list) -> str:
    lines = []
    for seg in segments:
        mm = int(seg.start // 60)
        ss = int(seg.start % 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg.text}")
    return "\n".join(lines)


def _detect_content_type(segments: list) -> str:
    transcript = _format_transcript(segments[:80])
    prompt = CONTENT_TYPE_PROMPT.format(transcript=transcript)
    resp = _llm_call(prompt)
    try:
        match = re.search(r"\{[^}]+\}", resp)
        if match:
            data = json.loads(match.group())
            return data.get("content_type", "other")
    except (json.JSONDecodeError, AttributeError):
        pass
    return "other"


def _llm_call(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return _call_openai(prompt)
    elif provider == "gemini":
        return _call_gemini(prompt)
    elif provider == "muapi":
        return _call_muapi(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


def _call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    chat = client.chats.create(model=model_name)
    resp = chat.send_message(prompt)
    return resp.text or ""


def _call_muapi(prompt: str) -> str:
    api_key = os.getenv("MUAPI_API_KEY")
    resp = requests.post(
        "https://api.muapi.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def detect_highlights(segments: list, max_clips: int = 3) -> list[Highlight]:
    content_type = _detect_content_type(segments)
    transcript = _format_transcript(segments)
    prompt = HIGHLIGHT_PROMPT.format(
        content_type=content_type,
        transcript=transcript,
        max_clips=max_clips,
    )
    resp = _llm_call(prompt)

    try:
        match = re.search(r"\[.*\]", resp, re.DOTALL)
        if not match:
            return []
        raw = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return []

    highlights = []
    for item in raw:
        try:
            start = float(item["start"])
            end = float(item["end"])
            
            # Enforce 20-40 seconds rule
            duration = end - start
            if duration < 20:
                # Extend to 30 seconds, shifting start slightly if needed
                new_start = max(0, start - 5)
                new_end = new_start + 30
                start = new_start
                end = new_end
            elif duration > 40:
                # Trim to 40 seconds
                end = start + 40
            
            # Ensure end is within reasonable bounds of the segment
            # If the adjustment pushed end past the original segment end + buffer, 
            # and original was short, it might be better to shift start.
            # But for simplicity, we stick to the current logic.

            highlights.append(Highlight(
                start=start,
                end=end,
                score=float(item["score"]),
                reason=str(item.get("reason", "")),
                title=str(item.get("title", "clip")),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    highlights = _dedupe(highlights)
    highlights.sort(key=lambda h: h.score, reverse=True)
    return highlights[:max_clips]


def _dedupe(highlights: list[Highlight], overlap_sec: float = 5.0) -> list[Highlight]:
    highlights.sort(key=lambda h: h.score, reverse=True)
    kept: list[Highlight] = []
    for h in highlights:
        if all(not _overlaps(h, k, overlap_sec) for k in kept):
            kept.append(h)
    return kept


def _overlaps(a: Highlight, b: Highlight, min_overlap: float) -> bool:
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)
    return (overlap_end - overlap_start) >= min_overlap

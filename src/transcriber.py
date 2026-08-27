"""Transcribe video with faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(
    video_path: Path,
    model_size: str = "base",
    device: str = "auto",
    language: str | None = None,
) -> list[Segment]:
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")

    segments_iter, info = model.transcribe(
        str(video_path),
        language=language if language else None,
        beam_size=5,
        vad_filter=True,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))

    return segments

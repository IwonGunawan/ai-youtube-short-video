"""Cache pipeline results to avoid reprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from .analyzer import Highlight
from .transcriber import Segment


def _cache_dir(video_path: Path, output_dir: Path) -> Path:
    stem = video_path.stem
    d = output_dir / f".cache_{stem}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_segments(video_path: Path, output_dir: Path) -> list[Segment] | None:
    p = _cache_dir(video_path, output_dir) / "segments.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return [Segment(**s) for s in data]


def save_segments(video_path: Path, output_dir: Path, segments: list[Segment]) -> None:
    p = _cache_dir(video_path, output_dir) / "segments.json"
    p.write_text(json.dumps([vars(s) for s in segments], indent=2))


def load_highlights(video_path: Path, output_dir: Path, max_clips: int) -> list[Highlight] | None:
    p = _cache_dir(video_path, output_dir) / "highlights.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("max_clips") != max_clips:
        return None
    return [Highlight(**h) for h in data["highlights"]]


def save_highlights(video_path: Path, output_dir: Path, highlights: list[Highlight], max_clips: int) -> None:
    p = _cache_dir(video_path, output_dir) / "highlights.json"
    p.write_text(json.dumps({
        "max_clips": max_clips,
        "highlights": [vars(h) for h in highlights],
    }, indent=2))

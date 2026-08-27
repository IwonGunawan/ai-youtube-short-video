"""Full pipeline: download → transcribe → analyze → render."""

from __future__ import annotations

import re
from pathlib import Path

from .downloader import download, is_youtube_url
from .transcriber import transcribe
from .analyzer import detect_highlights
from .face_detect import detect_face_center_x
from .renderer import render_clip
from .cache import load_segments, save_segments, load_highlights, save_highlights


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:80]


def run_pipeline(
    target: str,
    n: int = 3,
    ratio: str = "9:16",
    resolution: int = 720,
    language: str | None = None,
    hook: bool = True,
    output_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    output_dir = output_dir or Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading video...")
    if is_youtube_url(target):
        video_path = download(target, resolution=resolution, out_dir=output_dir)
    else:
        video_path = Path(target)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {target}")

    print(f"[2/4] Transcribing (whisper)...")
    cached_segments = None if force else load_segments(video_path, output_dir)
    if cached_segments is not None:
        segments = cached_segments
        print(f"  -> {len(segments)} segments (cached)")
    else:
        segments = transcribe(video_path, language=language)
        save_segments(video_path, output_dir, segments)
        print(f"  -> {len(segments)} segments")

    print(f"[3/4] Analyzing highlights (LLM)...")
    cached_highlights = None if force else load_highlights(video_path, output_dir, max_clips=n)
    if cached_highlights is not None:
        highlights = cached_highlights
        print(f"  -> {len(highlights)} highlights found (cached)")
    else:
        highlights = detect_highlights(segments, max_clips=n)
        save_highlights(video_path, output_dir, highlights, max_clips=n)
        print(f"  -> {len(highlights)} highlights found")

    if not highlights:
        print("  No highlights found. Try different video or adjust settings.")
        return []

    print(f"[4/4] Rendering clips...")
    rendered: list[Path] = []
    for i, hl in enumerate(highlights, 1):
        title = _sanitize_filename(hl.title)
        out_file = output_dir / f"clip_{i:02d}_{title}.mp4"
        print(f"  [{i}/{len(highlights)}] {hl.title} ({hl.start:.1f}s - {hl.end:.1f}s) score={hl.score}")
        try:
            face_x = None
            if ratio == "9:16":
                face_x = detect_face_center_x(video_path, hl.start, hl.end)
            render_clip(video_path, hl.start, hl.end, out_file, ratio=ratio, resolution=resolution, face_center_x=face_x)
            rendered.append(out_file)
            print(f"    -> saved {out_file.name}")
        except Exception as e:
            print(f"    -> FAILED: {e}")

    print(f"\nDone. {len(rendered)}/{len(highlights)} clips saved to {output_dir}/")
    return rendered

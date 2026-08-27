"""Download YouTube videos via yt-dlp."""

from __future__ import annotations

import re
from pathlib import Path

import yt_dlp


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def is_youtube_url(target: str) -> bool:
    return bool(_YT_ID_RE.search(target) or "youtube.com" in target or "youtu.be" in target)


def download(target: str, resolution: int = 720, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            base = Path(filename).with_suffix("")
            filename = str(base) + ".mp4"

    return Path(filename)

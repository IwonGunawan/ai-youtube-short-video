"""Render short video clips using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    ratio: str = "9:16",
    resolution: int = 720,
    face_center_x: float | None = None,
) -> Path:
    """Cut a segment from source video, auto-crop to target ratio, scale to resolution."""

    duration = end - start
    w_ratio, h_ratio = (int(x) for x in ratio.split(":"))

    if w_ratio / h_ratio < 1:
        out_h = resolution
        out_w = int(resolution * w_ratio / h_ratio)
    else:
        out_w = resolution
        out_h = int(resolution * h_ratio / w_ratio)

    if out_w % 2 != 0:
        out_w += 1
    if out_h % 2 != 0:
        out_h += 1

    if face_center_x is not None:
        crop_w = f"ih*{w_ratio}/{h_ratio}"
        crop_h = "ih"
        crop_x = f"max(0\\,min(iw-{crop_w}\\,iw*{face_center_x:.3f}-{crop_w}/2))"
        crop_y = "0"
        crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    else:
        crop_filter = f"crop=ih*{w_ratio}/{h_ratio}:ih"

    vf = (
        f"{crop_filter},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    return output_path

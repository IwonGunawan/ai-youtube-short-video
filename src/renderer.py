"""Render short video clips using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .subtitles import CAPTION_BOTTOM_GAP, render_caption_overlays


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    ratio: str = "9:16",
    resolution: int = 720,
    face_center_x: float | None = None,
    subtitles_path: Path | None = None,
) -> Path:
    """Cut a segment from source video, auto-crop to target ratio, scale to resolution.

    If subtitles_path is given, burn the SRT captions into the final frame
    as overlay composites.
    """

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

    base = (
        f"[0:v]{crop_filter},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    overlays = render_caption_overlays(subtitles_path, out_w, out_h) if subtitles_path is not None else []

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
    ]

    filter_parts = [f"{base}[base]"]
    for i in range(len(overlays)):
        cmd += ["-loop", "1", "-i", str(overlays[i][2])]

    last = "base"
    for i, (s, e, _) in enumerate(overlays):
        label = f"ov{i}"
        enable = f"between(t\\,{s:.3f}\\,{e:.3f})" if s < e else "lte(t\\,0)"
        filter_parts.append(
            f"[{last}][{i + 1}:v]"
            f"overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h-{CAPTION_BOTTOM_GAP}"
            f":enable='{enable}'[{label}]"
        )
        last = label

    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{last}]",
        "-map", "0:a",
        "-t", f"{duration:.3f}",
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
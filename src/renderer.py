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
    face_keyframes: list[tuple[float, float]] | None = None,
    subtitles_path: Path | None = None,
) -> Path:
    """Cut a segment from source video, auto-crop to target ratio, scale to resolution.

    face_keyframes: [(t_seconds_relative_to_clip_start, normalized_center_x)]
    keyframes; crop window follows largest face across the clip (time-interpolated).
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

    if face_keyframes:
        crop_w = f"ih*{w_ratio}/{h_ratio}"
        crop_h = "ih"
        crop_x_expr = _build_face_crop_expr(face_keyframes, duration)
        if crop_x_expr is not None:
            crop_x = f"max(0\\,min(iw-{crop_w}\\,iw*({crop_x_expr})-{crop_w}/2))"
            crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:0"
        else:
            crop_filter = f"crop={crop_w}:{crop_h}"
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
        "-map", "0:a?",
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


HOLD_MAX_GAP = 2.5


def _build_face_crop_expr(keyframes: list[tuple[float, float]], duration: float) -> str | None:
    """Build a time-based ffmpeg expression returning normalized crop center (0..1).

    Interpolates linearly between nearby keyframes; holds the last value when a
    detection gap exceeds HOLD_MAX_GAP seconds. Commas are backslash-escaped for
    embedding inside a -vf filter chain. Returns None when no usable keyframes.
    """
    if not keyframes:
        return None

    pts = sorted(keyframes)
    if len(pts) == 1:
        return f"{pts[0][1]:.4f}"

    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < duration:
        pts.append((duration, pts[-1][1]))

    pieces: list[tuple[float, float, float | None, float | None]] = []
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        delta = t1 - t0
        if 0 < delta <= HOLD_MAX_GAP:
            pieces.append((t1, v0, v1, t0))
        else:
            pieces.append((t1, v0, None, None))

    expr = _piece_expr(pieces[-1])
    for t_end, v0, v1, t0 in reversed(pieces[:-1]):
        piece = _piece_expr((t_end, v0, v1, t0))
        expr = f"if(lt(t\\,{t_end:.3f})\\,{piece}\\,{expr})"

    return expr


def _piece_expr(piece: tuple[float, float, float | None, float | None]) -> str:
    t_end, v0, v1, t0 = piece
    if v1 is None or t0 is None:
        return f"({v0:.4f})"
    frac = f"(t-{t0:.3f})/({t_end - t0:.3f})"
    return f"({v0:.4f}+({v1:.4f}-{v0:.4f})*{frac})"
"""Build SRT subtitle files and caption overlay images from transcript segments."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .transcriber import Segment

MAX_CAPTION_DURATION = 6.0
MAX_GAP = 0.8
CAPTION_MARGIN_H = 24
CAPTION_BOTTOM_GAP = 48
BOX_PADDING = 10
BOX_BG = (0, 0, 0, 140)
TEXT_FILL = (255, 255, 255, 255)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_font_path = None


def build_srt(segments: list[Segment], start: float, end: float, path: Path) -> Path:
    """Slice segments overlapping [start, end), cluster short ones, write SRT."""
    clipped = _clip(segments, start, end)
    blocks = _cluster(clipped, start)

    lines: list[str] = []
    for i, (s, e, text) in enumerate(blocks, 1):
        lines.append(str(i))
        lines.append(f"{_ts(s)} --> {_ts(e)}")
        lines.append(text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_caption_overlays(srt_path: Path, out_w: int, out_h: int) -> list[tuple[float, float, Path]]:
    """Render each SRT cue into a PNG overlay; return [(start, end, image)]."""
    cues = _parse_srt(srt_path)
    if not cues:
        return []

    font = ImageFont.truetype(_find_font(), max(8, int(out_h * 18 / 720)))
    line_h = int(font.size * 1.35)
    max_text_w = out_w - 2 * CAPTION_MARGIN_H
    tmp_dir = Path(tempfile.mkdtemp(prefix="captions_"))

    overlays: list[tuple[float, float, Path]] = []
    for i, (s, e, text) in enumerate(cues):
        lines = _wrap(text, font, max_text_w)
        text_img = Image.new("RGBA", (max_text_w, line_h * len(lines)), (0, 0, 0, 0))
        td = ImageDraw.Draw(text_img)
        y = 0
        for ln in lines:
            td.text((0, y), ln, font=font, fill=TEXT_FILL)
            y += line_h

        max_w = max(td.textlength(ln, font=font) for ln in lines)
        w = int(max_w) + 2 * BOX_PADDING
        h = line_h * len(lines) + 2 * BOX_PADDING

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=6, fill=BOX_BG)
        img.alpha_composite(text_img, (BOX_PADDING, BOX_PADDING))

        png = tmp_dir / f"cue_{i:03d}.png"
        img.save(png)
        overlays.append((s, e, png))

    return overlays


def _clip(segments: list[Segment], start: float, end: float) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        out.append(Segment(
            start=max(seg.start, start),
            end=min(seg.end, end),
            text=seg.text,
        ))
    return out


def _cluster(segments: list[Segment], start: float) -> list[tuple[float, float, str]]:
    if not segments:
        return []

    blocks: list[tuple[float, float, str]] = []
    cur_start = segments[0].start
    cur_end = segments[0].end
    cur_text = segments[0].text

    for seg in segments[1:]:
        gap = seg.start - cur_end
        if gap < MAX_GAP and (seg.end - cur_start) <= MAX_CAPTION_DURATION:
            cur_end = seg.end
            cur_text = f"{cur_text} {seg.text}".strip()
        else:
            blocks.append((cur_start - start, cur_end - start, cur_text))
            cur_start, cur_end, cur_text = seg.start, seg.end, seg.text

    blocks.append((cur_start - start, cur_end - start, cur_text))
    return blocks


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            start_t, end_t = line.split("-->")
            start = _from_ts(start_t.strip())
            end = _from_ts(end_t.strip())
            text: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text.append(lines[i].strip())
                i += 1
            cues.append((start, end, " ".join(text).strip()))
            continue
        i += 1
    return cues


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        candidate = f"{cur} {word}"
        if probe.textlength(candidate, font=font) <= max_w:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _find_font() -> str:
    global _font_path
    if _font_path:
        return _font_path
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            _font_path = cand
            return cand
    raise RuntimeError("No usable TTF font found for caption rendering. Install fontconfig fonts or set one in "
                       "src/subtitles.py _FONT_CANDIDATES.")


def _ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _from_ts(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
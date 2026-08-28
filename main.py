from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="shorts-generator",
        description="Turn YouTube videos into viral short clips.",
    )
    p.add_argument("target", help="YouTube URL or local video file path")
    p.add_argument("--n", type=int, default=3, dest="num_clips", help="Number of clips to render (default: 3)")
    p.add_argument("--ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    p.add_argument("--resolution", type=int, default=720, help="Source video resolution: 360/480/720/1080 (default: 720)")
    p.add_argument("--language", default="", help="Force Whisper language code, e.g. en (default: auto-detect)")
    p.add_argument("--no-hook", action="store_true", help="Exclude AI-generated hook from clip start")
    p.add_argument("--no-subtitles", action="store_true", help="Do not burn speech captions into clips")
    p.add_argument("--face-tracking", default="largest", choices=["largest", "best", "average", "closest"],
                   help="Multi-face crop strategy: largest/best/average/closest (default: largest)")
    p.add_argument("--force", action="store_true", help="Ignore cache, re-run transcription and analysis")

    args = p.parse_args(argv)

    lang = args.language.strip() or None

    try:
        rendered = run_pipeline(
            target=args.target,
            n=args.num_clips,
            ratio=args.ratio,
            resolution=args.resolution,
            language=lang,
            hook=not args.no_hook,
            force=args.force,
            subtitles=not args.no_subtitles,
            face_tracking=args.face_tracking,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rendered:
        return 1

    print(f"\nOutput files:")
    for f in rendered:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shorts generator package.

macOS-only mitigation: both PyAV (faster-whisper) and opencv-python bundle
their own FFmpeg dylibs. Loading the two bundled ``libavdevice`` copies in one
process makes the objc runtime print duplicate-class warnings
(AVFFrameReceiver / AVFAudioReceiver) at stderr. Those classes are internal
AVFoundation *device-capture* helpers never reached when demuxing regular
files, so the collision is benign. This module preloads both libraries with
fd 2 temporarily redirected so the one-shot warning is swallowed once per
process.
"""

from __future__ import annotations

import os
import sys

# OpenCV 5's new ONNX graph engine warns once that setPreferableTarget is
# unsupported (YuNet still runs fine on CPU). Read at cv2 import time.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

if sys.platform == "darwin" and not getattr(sys, "_objc_duplicates_silenced", False):
    sys._objc_duplicates_silenced = True
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        import av  # noqa: F401
        import cv2  # noqa: F401
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull)
        os.close(saved_stderr)
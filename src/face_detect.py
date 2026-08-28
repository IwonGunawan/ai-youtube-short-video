"""Face detection for smart crop positioning using YuNet DNN."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx")

SAMPLE_INTERVAL = 0.5


def _create_detector() -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        _MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000,
    )


def detect_face_centers(video_path: Path, start: float, end: float) -> list[tuple[float, float]]:
    """Track largest-face horizontal center across the clip.

    Samples every SAMPLE_INTERVAL seconds across [start, end). Returns
    keyframes as [(t_seconds_relative_to_start, normalized_center_x)] sorted by
    time. Empty list when no face is found anywhere in the clip.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or total_frames == 0:
        cap.release()
        return []

    detector = _create_detector()

    centers: list[tuple[float, float]] = []
    t_abs = start
    while t_abs < end:
        frame_idx = int(t_abs * fps)
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            if faces is not None and len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                face_cx = (largest[0] + largest[2] / 2) / w
                centers.append((t_abs - start, float(face_cx)))

        t_abs += SAMPLE_INTERVAL

    cap.release()
    return centers
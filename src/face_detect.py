"""Face detection for smart crop positioning using YuNet DNN."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx")


def _create_detector() -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        _MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000,
    )


def detect_face_center_x(video_path: Path, start: float, end: float) -> float:
    """Detect face in video segment and return normalized horizontal center (0.0-1.0).

    Samples up to 5 frames across the clip. Returns 0.5 (center) if no face found.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.5

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or total_frames == 0:
        cap.release()
        return 0.5

    sample_times = [start + i * (end - start) / 4 for i in range(5)]
    detector = _create_detector()

    centers: list[float] = []
    for t in sample_times:
        frame_idx = int(t * fps)
        if frame_idx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is None or len(faces) == 0:
            continue

        # Largest face (area = col2 * col3)
        largest = max(faces, key=lambda f: f[2] * f[3])
        face_cx = (largest[0] + largest[2] / 2) / w
        centers.append(face_cx)

    cap.release()

    if not centers:
        return 0.5

    return float(np.mean(centers))

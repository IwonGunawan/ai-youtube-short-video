"""Face detection for smart crop positioning using YuNet DNN."""

from __future__ import annotations

from pathlib import Path

import cv2

_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx")

SAMPLE_INTERVAL = 0.5

STRATEGIES = ("largest", "best", "average", "closest")


def _create_detector() -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        _MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000,
    )


def detect_face_centers(
    video_path: Path,
    start: float,
    end: float,
    strategy: str = "largest",
) -> list[tuple[float, float]]:
    """Track the crop-target face center across the clip.

    Samples every SAMPLE_INTERVAL seconds across [start, end). After detection,
    reduces all faces in a frame to one center using `strategy`:
      - largest: biggest face area (default, single-speaker framing)
      - best:    highest-confidence face (choose the cleanest detection)
      - average: mean center of all faces (both subjects in frame)
      - closest: face nearest the frame center (most prominent subject)

    Returns keyframes as [(t_seconds_relative_to_start, normalized_center_x)]
    sorted by time. Empty list when no face is found anywhere in the clip.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown face tracking strategy: {strategy}. Expected one of {STRATEGIES}")

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
                cx = _reduce_center(faces, w, strategy)
                centers.append((t_abs - start, float(cx)))

        t_abs += SAMPLE_INTERVAL

    cap.release()
    return centers


def _reduce_center(faces: object, w: int, strategy: str) -> float:
    """Reduce detected faces (array [x,y,w,h,..,score]) to one center_x."""
    if strategy == "average":
        xs = [(f[0] + f[2] / 2) / w for f in faces]
        return sum(xs) / len(xs)

    if strategy == "best":
        chosen = max(faces, key=lambda f: f[14])
    elif strategy == "closest":
        chosen = min(faces, key=lambda f: abs((f[0] + f[2] / 2) / w - 0.5))
    else:  # largest
        chosen = max(faces, key=lambda f: f[2] * f[3])

    return (chosen[0] + chosen[2] / 2) / w
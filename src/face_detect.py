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
    for t_rel, w, faces in _sample_frames(cap, detector, start, end, fps, total_frames):
        if len(faces) > 0:
            centers.append((t_rel, _reduce_center(faces, w, strategy)))

    cap.release()
    return centers


def detect_all_faces(
    video_path: Path,
    start: float,
    end: float,
) -> list[tuple[float, list[tuple[float, float]]]]:
    """Detect every face per sampled frame; return [(t_rel, [(cx, area), ...])].

    Used by the `speaker` strategy so each speaker can be mapped to their own
    face position instead of collapsing all faces at frame level.
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

    samples: list[tuple[float, list[tuple[float, float]]]] = []
    for t_rel, w, faces in _sample_frames(cap, detector, start, end, fps, total_frames):
        boxes = [(float((f[0] + f[2] / 2) / w), float(f[2] * f[3])) for f in faces]
        samples.append((t_rel, boxes))

    cap.release()
    return samples


def speaker_keyframes(
    faces: list[tuple[float, list[tuple[float, float]]]],
    turns: list[tuple[float, float, int]],
    clip_start: float,
) -> list[tuple[float, float]]:
    """Resolve per-frame faces + speaker turns into crop keyframes for `speaker`.

    For each speaker, anchor = median center of the biggest face seen while that
    speaker talks. When a speaker is active at a sampled frame, crop to the face
    nearest their anchor; on silence pick the biggest face; drop frames with no
    faces so the renderer interpolates/holds.
    """
    anchor: dict[int, float] = {}
    for _t, boxes in faces:
        if not boxes:
            continue
        t_abs = clip_start + _t
        sid = _active_speaker(t_abs, turns)
        if sid is None:
            continue
        biggest_cx = max(boxes, key=lambda b: b[1])[0]
        anchor.setdefault(sid, []).append(biggest_cx)
    for sid in anchor:
        xs = sorted(anchor[sid])
        anchor[sid] = xs[len(xs) // 2]

    keyframes: list[tuple[float, float]] = []
    for t_rel, boxes in faces:
        if not boxes:
            continue
        t_abs = clip_start + t_rel
        sid = _active_speaker(t_abs, turns)
        if sid is not None and sid in anchor:
            cx = min(boxes, key=lambda b: abs(b[0] - anchor[sid]))[0]
        else:
            cx = max(boxes, key=lambda b: b[1])[0]
        keyframes.append((t_rel, cx))

    return keyframes


def _sample_frames(cap, detector, start, end, fps, total_frames):
    """Yield (t_relative, frame_width, raw_faces_array) per SAMPLE_INTERVAL sample."""
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
            if faces is None:
                faces = []
            yield (t_abs - start, w, faces)
        t_abs += SAMPLE_INTERVAL


def _reduce_center(faces: object, frame_w: int, strategy: str) -> float:
    """Reduce detected faces (array [x,y,w,h,..,score]) to one center_x."""
    if strategy == "average":
        xs = [(f[0] + f[2] / 2) / frame_w for f in faces]
        return sum(xs) / len(xs)

    if strategy == "best":
        chosen = max(faces, key=lambda f: f[14])
    elif strategy == "closest":
        chosen = min(faces, key=lambda f: abs((f[0] + f[2] / 2) / frame_w - 0.5))
    else:  # largest
        chosen = max(faces, key=lambda f: f[2] * f[3])

    return (chosen[0] + chosen[2] / 2) / frame_w


def _active_speaker(t_abs: float, turns: list[tuple[float, float, int]]) -> int | None:
    for t0, t1, sid in turns:
        if t0 <= t_abs < t1:
            return sid
    return None
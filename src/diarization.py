"""Lightweight speaker diarization from audio using spectral clustering.

Zero extra dependencies (numpy only). Decodes the clip audio as mono 16 kHz
PCM via ffmpeg, computes per-window log-mel statistics as speaker features,
and clusters windows into speaker turns. Not pyannote-grade, but adequate for
binary-turn podcasts/interviews and cheap (no HF token, no model download).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
N_MELS = 26
WINDOW_SEC = 0.75
HOP_SEC = 0.25
SILENCE_RMS = 0.0015
MEL_FMIN = 40.0
MEL_FMAX = 7800.0

_feat_cache: dict[tuple[str, float, float], np.ndarray] = {}


def diarize(
    video_path: Path,
    start: float,
    end: float,
    n_speakers: int | None = None,
) -> list[tuple[float, float, int]]:
    """Return speaker turns as [(t_start_abs, t_end_abs, speaker_id)] within [start, end).

    speaker_id is 0-based, ordered by total speech time. Silence segments are
    dropped, leaving gaps in the turn timeline.
    """
    feats, rms = _features(video_path, start, end)
    labels = _cluster_labels(feats, rms, n_speakers)
    return _turns(labels, start, end)


def _features(video_path: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    key = (str(video_path), round(start, 3), round(end, 3))
    if key in _feat_cache:
        return _feat_cache[key]

    audio = _read_pcm(video_path, start, end)
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP]
    if len(frames) == 0:
        raise RuntimeError("Diarization: no audio decoded from clip")

    win = frames * np.hanning(N_FFT)
    power = np.abs(np.fft.rfft(win, n=N_FFT, axis=1)) ** 2
    log_mel = np.log(power @ _mel_matrix() + 1e-10)

    win_len = int(WINDOW_SEC * SAMPLE_RATE / HOP)
    win_step = int(HOP_SEC * SAMPLE_RATE / HOP)
    n_windows = max(0, (len(log_mel) - win_len) // win_step + 1)

    feats: list[np.ndarray] = []
    rms: list[float] = []
    for i in range(n_windows):
        block = log_mel[i * win_step: i * win_step + win_len]
        feats.append(_normalize(block.mean(axis=0)))
        a = audio[i * win_step * HOP: i * win_step * HOP + win_len * HOP]
        rms.append(float(np.sqrt(np.mean(a ** 2))))

    _feat_cache[key] = (np.vstack(feats), np.array(rms))
    return _feat_cache[key]


def _read_pcm(video_path: Path, start: float, end: float) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-i", str(video_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(pcm) == 0:
        raise RuntimeError("Diarization: no audio signal in clip (silent or no audio track)")
    return pcm


def _mel_matrix() -> np.ndarray:
    n_freqs = N_FFT // 2 + 1
    freqs = np.linspace(0, SAMPLE_RATE / 2, n_freqs)
    mel = np.linspace(_hz_to_mel(MEL_FMIN), _hz_to_mel(MEL_FMAX), N_MELS + 2)
    hz = _mel_to_hz(mel)

    matrix = np.zeros((n_freqs, N_MELS))
    for m in range(N_MELS):
        f_lo, f_mid, f_hi = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs >= f_lo) & (freqs < f_mid)
        down = (freqs >= f_mid) & (freqs <= f_hi)
        matrix[up, m] = (freqs[up] - f_lo) / max(f_mid - f_lo, 1e-9)
        matrix[down, m] = (f_hi - freqs[down]) / max(f_hi - f_mid, 1e-9)
    return matrix


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    std = vec.std()
    if std < 1e-9:
        return vec * 0.0
    return (vec - vec.mean()) / std


def _cluster_labels(
    feats: np.ndarray,
    rms: np.ndarray,
    n_speakers: int | None,
) -> list[int]:
    speech_ids = np.where(rms >= SILENCE_RMS)[0]
    if len(speech_ids) < 2:
        return [-1] * len(feats)

    speech = feats[speech_ids]
    if n_speakers and n_speakers >= 1:
        labels_sub = _kmeans(speech, n_speakers)
    else:
        labels_sub = _online_cluster(speech)

    labels: list[int] = [-1] * len(feats)
    for j, sid in zip(speech_ids, labels_sub):
        labels[j] = sid
    return labels


def _online_cluster(feats: np.ndarray, sim_thresh: float = 0.55) -> list[int]:
    centroids: list[np.ndarray] = []
    cluster_sizes: list[int] = []
    labels: list[int] = []
    for v in feats:
        if not centroids:
            centroids.append(v.copy())
            cluster_sizes.append(1)
            labels.append(0)
            continue
        sims = [(v @ c) / max(np.linalg.norm(v) * np.linalg.norm(c), 1e-9) for c in centroids]
        j = int(np.argmax(sims))
        if sims[j] >= sim_thresh:
            labels.append(j)
            cluster_sizes[j] += 1
            centroids[j] += (v - centroids[j]) / cluster_sizes[j]
        else:
            centroids.append(v.copy())
            cluster_sizes.append(1)
            labels.append(len(centroids) - 1)

    merged = _merge_close(centroids, labels)
    return _remap(labels, merged)


def _merge_close(centroids: list[np.ndarray], labels: list[int], sim_thresh: float = 0.8) -> dict[int, int]:
    merged: dict[int, int] = {}
    for a in range(len(centroids)):
        hit = None
        for b in range(a):
            sim = (centroids[a] @ centroids[b]) / max(
                np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]), 1e-9)
            if sim >= sim_thresh:
                hit = b
                break
        merged[a] = merged[hit] if hit is not None else a
    return merged


def _kmeans(feats: np.ndarray, k: int, iters: int = 30, restarts: int = 3) -> list[int]:
    if k >= len(feats):
        return list(range(len(feats)))
    rng = np.random.default_rng(0)
    best = None
    best_inertia = float("inf")
    for _ in range(restarts):
        idx = rng.choice(len(feats), k, replace=False)
        centroids = feats[idx].copy()
        labels = np.zeros(len(feats), dtype=int)
        for _ in range(iters):
            d = feats @ centroids.T
            new_labels = np.argmax(d, axis=1)
            new_centroids = np.array([
                feats[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centroids[j]
                for j in range(k)
            ])
            if np.array_equal(new_labels, labels):
                centroids = new_centroids
                break
            labels, centroids = new_labels, new_centroids
        inertia = sum(np.linalg.norm(feats[j] - centroids[labels[j]]) ** 2 for j in range(len(feats)))
        if inertia < best_inertia:
            best_inertia, best = inertia, labels
    return _remap(best.tolist(), {i: i for i in range(k)})


def _remap(labels: list[int], mapping: dict[int, int]) -> list[int]:
    raw = [mapping.get(l, l) for l in labels]
    order: dict[int, int] = {}
    for uid in sorted(set(raw), key=lambda u: -raw.count(u)):
        order[uid] = len(order)
    return [order.get(uid, uid) for uid in raw]


def _turns(labels: list[int], start: float, end: float) -> list[tuple[float, float, int]]:
    turns: list[tuple[float, float, int]] = []
    cur_sid: int | None = None
    cur_start = start
    win_len = WINDOW_SEC
    for i, sid in enumerate(labels):
        t0 = min(start + i * HOP_SEC, end)
        t1 = min(t0 + win_len, end)
        if sid != cur_sid:
            if cur_sid is not None and cur_sid >= 0:
                turns.append((round(cur_start, 3), round(t0, 3), cur_sid))
            cur_sid = sid
            cur_start = t0 if sid >= 0 else t1
    if cur_sid is not None and cur_sid >= 0:
        turns.append((round(cur_start, 3), round(end, 3), cur_sid))
    return turns
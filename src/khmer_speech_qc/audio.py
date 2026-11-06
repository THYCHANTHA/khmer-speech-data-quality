"""WAV loading, resampling and level helpers (NumPy/SciPy only)."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

TARGET_SR = 16_000
EPS = 1e-12


def read_wav(path: str | Path) -> tuple[np.ndarray, int, int, str]:
    """Return (samples as float32 in [-1, 1], sample rate, channels, original dtype)."""
    sr, data = wavfile.read(str(path))
    dtype = str(data.dtype)
    channels = 1 if data.ndim == 1 else data.shape[1]
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        x = (data.astype(np.float64) - (info.max + info.min + 1) / 2) / (info.max + 1)
    else:
        x = data.astype(np.float64)
    return x.astype(np.float32), int(sr), int(channels), dtype


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def load_mono(path: str | Path, target: int = TARGET_SR) -> np.ndarray:
    x, sr, _, _ = read_wav(path)
    return resample(to_mono(x), sr, target)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS))


def dbfs(value: float) -> float:
    return float(20 * np.log10(max(value, EPS)))


def frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Frames as rows (n_frames, frame); the tail shorter than a frame is zero-padded."""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame + hop - 1) // hop
    x = np.pad(x, (0, (n - 1) * hop + frame - len(x)))
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10) -> np.ndarray:
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    f = frame_signal(x, frame, hop)
    return 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + EPS)

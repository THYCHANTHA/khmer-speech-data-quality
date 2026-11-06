"""MFCC and summary features in NumPy (no librosa), so the project adds no audio dependency.

MFCC recipe: pre-emphasis 0.97, 25 ms Hamming frames, 10 ms hop, 512-point FFT power spectrum,
26 triangular mel filters (HTK mel scale), log, DCT-II (orthonormal), first 13 coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .audio import EPS, frame_signal

N_MFCC = 13


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int = 512, n_mels: int = 26, fmin: float = 0.0, fmax: float | None = None
) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular filters."""
    fmax = fmax or sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c > lo:
            fb[m - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c:
            fb[m - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return fb


def power_spectrum(x: np.ndarray, sr: int, n_fft: int = 512) -> np.ndarray:
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    frames = frame_signal(x, int(0.025 * sr), int(0.010 * sr)) * np.hamming(int(0.025 * sr))
    return np.abs(np.fft.rfft(frames, n_fft)) ** 2 / n_fft


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """(n_frames, n_mfcc)."""
    pspec = power_spectrum(x, sr)
    mel = np.log(pspec @ mel_filterbank(sr).T + EPS)
    return dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def feature_vector(x: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean and std (26 values, the report's feature family) plus spectral flatness and
    zero-crossing-rate mean/std (4 values): 30 features per segment."""
    m = mfcc(x, sr)
    pspec = power_spectrum(x, sr) + EPS
    flat = np.exp(np.mean(np.log(pspec), axis=1)) / np.mean(pspec, axis=1)
    frames = frame_signal(x, int(0.025 * sr), int(0.010 * sr))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return np.concatenate([m.mean(0), m.std(0), [flat.mean(), flat.std(), zcr.mean(), zcr.std()]])


FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc{i}_std" for i in range(N_MFCC)]
    + ["flatness_mean", "flatness_std", "zcr_mean", "zcr_std"]
)

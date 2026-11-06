"""Controlled degradation of clean speech: noise generators and mixing at a target SNR.

No external noise corpus is used (none with a suitable licence fits on the small disk): noise is
synthetic (white, pink, brown, mains hum) or babble built from *other speakers* in the same
corpus. This is a documented limitation: real environmental noise (traffic, rooms, phones) is
not represented.
"""

from __future__ import annotations

import numpy as np

from .audio import rms

NOISE_TYPES = ("white", "pink", "brown", "hum", "babble")


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """White (flat), pink (1/f power) or brown (1/f^2 power) Gaussian noise, unit RMS."""
    white = rng.standard_normal(n)
    if kind == "white":
        y = white
    else:
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n)
        f[0] = f[1] if n > 1 else 1.0
        exponent = {"pink": 0.5, "brown": 1.0}[kind]
        y = np.fft.irfft(spec / f**exponent, n)
    return (y / (rms(y) or 1.0)).astype(np.float32)


def hum(n: int, sr: int, rng: np.random.Generator, base_hz: float = 50.0) -> np.ndarray:
    """Mains hum: fundamental plus decaying odd/even harmonics, random phase, unit RMS."""
    t = np.arange(n) / sr
    y = sum(
        (0.6**k) * np.sin(2 * np.pi * base_hz * (k + 1) * t + rng.uniform(0, 2 * np.pi))
        for k in range(6)
    )
    y = y + 0.05 * rng.standard_normal(n)
    return (y / rms(y)).astype(np.float32)


def babble(n: int, others: list[np.ndarray], rng: np.random.Generator, k: int = 4) -> np.ndarray:
    """Sum of k random excerpts from other speakers' clips, unit RMS."""
    if not others:
        raise ValueError("babble needs clips from other speakers")
    y = np.zeros(n, dtype=np.float64)
    for idx in rng.choice(len(others), size=min(k, len(others)), replace=False):
        src = others[idx]
        if len(src) < n:
            src = np.tile(src, int(np.ceil(n / len(src))))
        start = rng.integers(0, len(src) - n + 1)
        seg = src[start : start + n]
        y += seg / (rms(seg) or 1.0)
    return (y / (rms(y) or 1.0)).astype(np.float32)


def make_noise(
    kind: str, n: int, sr: int, rng: np.random.Generator, others: list[np.ndarray] | None = None
) -> np.ndarray:
    if kind in ("white", "pink", "brown"):
        return colored_noise(n, kind, rng)
    if kind == "hum":
        return hum(n, sr, rng)
    if kind == "babble":
        return babble(n, others or [], rng)
    raise ValueError(f"unknown noise type {kind!r}")


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """speech + g*noise with g chosen so 20*log10(rms(speech)/rms(g*noise)) == snr_db.
    The result is scaled down only if it would clip."""
    if len(noise) != len(speech):
        raise ValueError("speech and noise must have the same length")
    g = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
    y = speech + g * noise
    peak = np.max(np.abs(y))
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)

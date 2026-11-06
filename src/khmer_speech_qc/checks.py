"""Per-clip validation checks for an ASR corpus: audio integrity and transcript sanity.

Each check returns measurements; `flag_clip` turns them into named flags with documented
thresholds (see THRESHOLDS). Thresholds are screening rules, not ground truth: flagged clips
should be listened to before being removed.
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

import numpy as np

from .audio import dbfs, frame_energy_db, rms

KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))

THRESHOLDS = {
    "min_duration_s": 0.5,
    "max_duration_s": 20.0,
    "clipping_ratio": 0.001,  # share of samples at |x| >= 0.999
    "min_rms_dbfs": -40.0,  # very quiet recording
    "max_edge_silence_s": 1.5,  # leading or trailing silence
    "min_est_snr_db": 15.0,  # energy-percentile SNR estimate
    "min_khmer_ratio": 0.9,  # share of Khmer letters among non-space characters
    "speech_rate_robust_z": 4.0,  # |robust z| of characters per second (needs the corpus)
}


@dataclass
class AudioStats:
    duration_s: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    leading_silence_s: float
    trailing_silence_s: float
    est_snr_db: float


def audio_stats(x: np.ndarray, sr: int, channels: int = 1) -> AudioStats:
    """`x` mono float in [-1, 1]."""
    e = frame_energy_db(x, sr)
    hop = 0.010
    # Frames within 35 dB of the loudest 5 % count as active (simple energy VAD).
    active = e > np.percentile(e, 95) - 35
    idx = np.flatnonzero(active)
    lead = idx[0] * hop if len(idx) else len(x) / sr
    trail = (len(e) - 1 - idx[-1]) * hop if len(idx) else len(x) / sr
    # SNR estimate: loud frames (90th percentile) against the quietest (10th percentile).
    snr = float(np.percentile(e, 90) - np.percentile(e, 10))
    return AudioStats(
        duration_s=len(x) / sr,
        sample_rate=sr,
        channels=channels,
        peak_dbfs=dbfs(float(np.max(np.abs(x))) if len(x) else 0.0),
        rms_dbfs=dbfs(rms(x)),
        clipping_ratio=float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0,
        leading_silence_s=float(lead),
        trailing_silence_s=float(trail),
        est_snr_db=snr,
    )


def is_khmer_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in KHMER_RANGES)


@dataclass
class TextStats:
    n_chars: int
    khmer_ratio: float
    latin_chars: int
    digits: int
    has_zero_width: bool


def text_stats(text: str) -> TextStats:
    text = unicodedata.normalize("NFC", text)
    chars = [c for c in text if not c.isspace()]
    zero_width = any(c in "​‌‍﻿" for c in text)
    visible = [c for c in chars if c not in "​‌‍﻿"]
    khmer = sum(is_khmer_char(c) for c in visible)
    return TextStats(
        n_chars=len(visible),
        khmer_ratio=khmer / len(visible) if visible else 0.0,
        latin_chars=sum(c.isascii() and c.isalpha() for c in visible),
        digits=sum(c.isdigit() for c in visible),
        has_zero_width=zero_width,
    )


def robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); 0 where MAD is 0."""
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med))
    return np.zeros_like(values) if mad == 0 else (values - med) / mad


def flag_clip(a: AudioStats, t: TextStats, rate_z: float | None = None) -> list[str]:
    th = THRESHOLDS
    flags = []
    if a.duration_s < th["min_duration_s"]:
        flags.append("too_short")
    if a.duration_s > th["max_duration_s"]:
        flags.append("too_long")
    if a.channels != 1:
        flags.append("not_mono")
    if a.clipping_ratio > th["clipping_ratio"]:
        flags.append("clipping")
    if a.rms_dbfs < th["min_rms_dbfs"]:
        flags.append("too_quiet")
    if max(a.leading_silence_s, a.trailing_silence_s) > th["max_edge_silence_s"]:
        flags.append("long_edge_silence")
    if a.est_snr_db < th["min_est_snr_db"]:
        flags.append("low_est_snr")
    if t.n_chars == 0:
        flags.append("empty_transcript")
    elif t.khmer_ratio < th["min_khmer_ratio"]:
        flags.append("non_khmer_text")
    if t.has_zero_width:
        flags.append("zero_width_chars")
    if rate_z is not None and abs(rate_z) > th["speech_rate_robust_z"]:
        flags.append("speech_rate_outlier")
    return flags


def as_row(a: AudioStats, t: TextStats) -> dict:
    return {**asdict(a), **asdict(t)}

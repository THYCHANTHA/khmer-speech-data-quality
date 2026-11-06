"""OpenSLR SLR42 (Google Khmer TTS corpus) index loader.

The archive holds WAV files and `line_index.tsv` (file id <TAB> transcript). Each file id starts
with a speaker identifier, e.g. `khm_0308_0011865966` -> speaker `khm_0308`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

SPEAKER_RE = re.compile(r"^([A-Za-z]+_\d+)_")


def data_dir() -> Path:
    """Folder containing the extracted corpus (set KHMER_ASR_DATA_DIR)."""
    return Path(
        os.environ.get("KHMER_ASR_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "raw")
    )


def speaker_of(file_id: str) -> str:
    m = SPEAKER_RE.match(file_id)
    if not m:
        raise ValueError(f"cannot parse a speaker id from {file_id!r}")
    return m.group(1)


def load_index(root: str | Path) -> pd.DataFrame:
    """One row per indexed utterance: file_id, transcript, speaker, path, exists."""
    root = Path(root)
    tsvs = sorted(root.rglob("line_index.tsv"))
    if len(tsvs) != 1:
        raise FileNotFoundError(f"expected one line_index.tsv under {root}, found {len(tsvs)}")
    rows = []
    with open(tsvs[0], encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"line {n}: expected 'file_id<TAB>transcript'")
            file_id, text = parts[0].strip(), "\t".join(parts[1:]).strip()
            rows.append((file_id, text))
    df = pd.DataFrame(rows, columns=["file_id", "transcript"])
    wavs = {p.stem: p for p in root.rglob("*.wav")}
    df["speaker"] = df["file_id"].map(speaker_of)
    df["path"] = df["file_id"].map(lambda i: str(wavs[i]) if i in wavs else "")
    df["exists"] = df["path"] != ""
    df.attrs["unindexed_wavs"] = sorted(set(wavs) - set(df["file_id"]))
    return df

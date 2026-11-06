"""Validate the SLR42 corpus: audio integrity, levels, silence, SNR estimate, transcript script and
speech-rate outliers. Writes reports/validation_summary.json and reports/flagged_clips.csv
(file ids, speakers, flags and measurements; no audio and no transcripts).

Usage:  KHMER_ASR_DATA_DIR=<extracted corpus> python scripts/validate_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmer_speech_qc.audio import read_wav, to_mono  # noqa: E402
from khmer_speech_qc.checks import THRESHOLDS, as_row, audio_stats, flag_clip, robust_z, text_stats  # noqa: E402
from khmer_speech_qc.corpus import data_dir, load_index  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> int:
    root = data_dir()
    if not root.is_dir():
        print(f"corpus not found at {root}; set KHMER_ASR_DATA_DIR", file=sys.stderr)
        return 2
    idx = load_index(root)
    rows, hashes, dtypes = [], {}, Counter()
    for r in idx[idx["exists"]].itertuples():
        x, sr, ch, dtype = read_wav(r.path)
        dtypes[dtype] += 1
        hashes.setdefault(hashlib.sha256(Path(r.path).read_bytes()).hexdigest(), []).append(
            r.file_id
        )
        a, t = audio_stats(to_mono(x), sr, ch), text_stats(r.transcript)
        active = max(a.duration_s - a.leading_silence_s - a.trailing_silence_s, 0.1)
        rows.append(
            {
                "file_id": r.file_id,
                "speaker": r.speaker,
                **as_row(a, t),
                "chars_per_active_s": t.n_chars / active,
                "_a": a,
                "_t": t,
            }
        )
    df = pd.DataFrame(rows)
    df["speech_rate_z"] = robust_z(df["chars_per_active_s"].to_numpy())
    df["flags"] = [
        ";".join(flag_clip(a, t, z))
        for a, t, z in zip(df["_a"], df["_t"], df["speech_rate_z"], strict=True)
    ]
    df = df.drop(columns=["_a", "_t"])

    flag_counts = Counter(f for fl in df["flags"] for f in fl.split(";") if f)
    dup_text = idx["transcript"].duplicated(keep=False)
    by_spk = df.groupby("speaker").agg(clips=("file_id", "size"), hours=("duration_s", "sum"))
    by_spk["hours"] = by_spk["hours"] / 3600
    num = [
        "duration_s",
        "rms_dbfs",
        "peak_dbfs",
        "est_snr_db",
        "leading_silence_s",
        "trailing_silence_s",
        "chars_per_active_s",
        "n_chars",
    ]
    summary = {
        "corpus": "OpenSLR SLR42 km_kh_male (CC BY-SA 4.0)",
        "indexed_utterances": int(len(idx)),
        "indexed_without_wav": idx.loc[~idx["exists"], "file_id"].tolist(),
        "wavs_without_index_entry": idx.attrs["unindexed_wavs"],
        "clips_checked": int(len(df)),
        "total_hours": float(df["duration_s"].sum() / 3600),
        "speakers": int(df["speaker"].nunique()),
        "clips_per_speaker": {k: int(v) for k, v in by_spk["clips"].items()},
        "hours_per_speaker": {k: round(float(v), 3) for k, v in by_spk["hours"].items()},
        "sample_rates": {str(k): int(v) for k, v in df["sample_rate"].value_counts().items()},
        "channels": {str(k): int(v) for k, v in df["channels"].value_counts().items()},
        "sample_formats": dict(dtypes),
        "distributions": {
            c: {
                q: float(df[c].quantile(p))
                for q, p in (("min", 0), ("p05", 0.05), ("median", 0.5), ("p95", 0.95), ("max", 1))
            }
            for c in num
        },
        "thresholds": THRESHOLDS,
        "flag_counts": dict(flag_counts),
        "clips_with_any_flag": int((df["flags"] != "").sum()),
        "exact_duplicate_audio_groups": [v for v in hashes.values() if len(v) > 1],
        "utterances_sharing_a_transcript": int(dup_text.sum()),
        "distinct_transcripts": int(idx["transcript"].nunique()),
        "khmer_ratio_min": float(df["khmer_ratio"].min()),
        "transcripts_with_latin": int((df["latin_chars"] > 0).sum()),
        "transcripts_with_digits": int((df["digits"] > 0).sum()),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    cols = ["file_id", "speaker", "flags", *num, "speech_rate_z", "clipping_ratio", "khmer_ratio"]
    df.loc[df["flags"] != "", cols].sort_values("file_id").to_csv(
        REPORTS / "flagged_clips.csv", index=False, float_format="%.4g"
    )
    short = {
        k: v
        for k, v in summary.items()
        if k not in ("distributions", "clips_per_speaker", "hours_per_speaker")
    }
    print(json.dumps(short, indent=2, ensure_ascii=False)[:3500])
    print(
        "speech-rate, est SNR medians:",
        float(np.median(df["chars_per_active_s"])),
        float(np.median(df["est_snr_db"])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

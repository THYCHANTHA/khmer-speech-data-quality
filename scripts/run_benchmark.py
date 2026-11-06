"""Clean-vs-degraded benchmark on SLR42 for several seeds. Writes reports/benchmark.json.

Usage:  KHMER_ASR_DATA_DIR=<extracted corpus> python scripts/run_benchmark.py [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmer_speech_qc.audio import TARGET_SR, load_mono  # noqa: E402
from khmer_speech_qc.benchmark import (  # noqa: E402
    Clip,
    build_original_style_dataset,
    build_quality_dataset,
    detection_by_condition,
    evaluate_grouped,
    evaluate_random_split,
    evaluate_unseen_noise,
)
from khmer_speech_qc.corpus import data_dir, load_index  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
BABBLE_POOL_PER_SPEAKER = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    idx = load_index(data_dir())
    idx = idx[idx["exists"]].sort_values("file_id")
    clips = [Clip(r.file_id, r.speaker, path=r.path) for r in idx.itertuples()]

    results = {
        "corpus": "OpenSLR SLR42 km_kh_male",
        "n_clips": len(clips),
        "n_speakers": int(idx["speaker"].nunique()),
        "sample_rate": TARGET_SR,
        "segment_s": 2.0,
        "seeds": {},
    }
    t0 = time.time()
    for seed in args.seeds:
        rng = np.random.default_rng(1000 + seed)
        pool = []
        for _spk, g in idx.groupby("speaker"):
            for r in g.sample(
                min(BABBLE_POOL_PER_SPEAKER, len(g)), random_state=int(rng.integers(1e9))
            ).itertuples():
                pool.append(Clip(r.file_id, r.speaker, data=load_mono(r.path)))
        quality = build_quality_dataset(clips, TARGET_SR, pool, seed)
        original = build_original_style_dataset(clips, TARGET_SR, seed)
        q_grouped, q_oof = evaluate_grouped(quality, seed)
        o_grouped, _ = evaluate_grouped(original, seed)
        res = {
            "segments": {
                "quality": int(len(quality)),
                "original_style": int(len(original)),
                "quality_degraded_share": float(quality["label"].mean()),
            },
            "original_style": {
                "random_split": evaluate_random_split(original, seed),
                "speaker_grouped_cv": o_grouped,
            },
            "quality_task": {
                "random_split": evaluate_random_split(quality, seed),
                "speaker_grouped_cv": q_grouped,
            },
            "quality_unseen_noise_type": evaluate_unseen_noise(quality, seed),
            "quality_detection_by_condition": {
                m: detection_by_condition(q_oof, m)
                for m in ("random_forest", "svm_rbf", "snr_threshold_baseline")
            },
        }
        results["seeds"][str(seed)] = res
        print(f"seed {seed} done after {time.time() - t0:.0f}s", flush=True)
        for task in ("original_style", "quality_task"):
            for design in ("random_split", "speaker_grouped_cv"):
                row = {
                    m: round(v["roc_auc"], 3)
                    for m, v in res[task][design].items()
                    if not m.startswith("_")
                }
                print(f"  {task:15s} {design:18s} AUC {row}")
    results["runtime_s"] = round(time.time() - t0, 1)
    results["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    import sklearn

    results["environment"]["scikit_learn"] = sklearn.__version__
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

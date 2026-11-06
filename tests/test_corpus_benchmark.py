import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from khmer_speech_qc.benchmark import (
    Clip,
    assign_conditions,
    build_original_style_dataset,
    build_quality_dataset,
    detection_by_condition,
    evaluate_grouped,
    evaluate_random_split,
    evaluate_unseen_noise,
    segments,
    snr_threshold,
)
from khmer_speech_qc.corpus import load_index, speaker_of
from khmer_speech_qc.features import FEATURE_NAMES

SR = 8_000  # small rate keeps the synthetic benchmark fast


def fake_speech(seconds, f0, rng):
    """Amplitude-modulated harmonic tone: crude but speech-like (voiced, syllable rhythm)."""
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)))
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * env
    return (0.3 * x / np.max(np.abs(x)) + 0.0005 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="module")
def clips():
    rng = np.random.default_rng(0)
    out = []
    for s, f0 in enumerate([110, 130, 150, 170, 190, 210]):
        for k in range(6):
            out.append(Clip(f"spk{s}_{k}", f"spk{s}", fake_speech(4.0, f0 + 3 * k, rng)))
    return out


# ---------------------------------------------------------------- corpus
def test_speaker_parsing():
    assert speaker_of("khm_0308_0011865966") == "khm_0308"
    with pytest.raises(ValueError):
        speaker_of("nospeaker")


def test_load_index_links_wavs_and_reports_gaps(tmp_path):
    (tmp_path / "wavs").mkdir()
    for fid in ["khm_0001_a1", "khm_0002_b2", "khm_0003_extra"]:
        wavfile.write(tmp_path / "wavs" / f"{fid}.wav", SR, np.zeros(SR, dtype=np.int16))
    (tmp_path / "line_index.tsv").write_text(
        "khm_0001_a1\tសួស្តី\nkhm_0002_b2\tអរគុណ\nkhm_0009_missing\tបាទ\n\n", encoding="utf-8"
    )
    df = load_index(tmp_path)
    assert list(df["speaker"]) == ["khm_0001", "khm_0002", "khm_0009"]
    assert list(df["exists"]) == [True, True, False]
    assert df.attrs["unindexed_wavs"] == ["khm_0003_extra"]


def test_load_index_rejects_malformed_lines(tmp_path):
    (tmp_path / "line_index.tsv").write_text("khm_0001_a1 no tab here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_index(tmp_path)


# ---------------------------------------------------------------- dataset construction
def test_segments_drop_partial_tail():
    assert [len(s) for s in segments(np.zeros(int(5.5 * SR)), SR)] == [2 * SR, 2 * SR]


def test_conditions_are_per_clip_balanced_per_speaker_and_deterministic(clips):
    cond = assign_conditions(clips, seed=3)
    assert cond == assign_conditions(clips, seed=3)
    per_spk = (
        pd.Series({c.clip_id: cond[c.clip_id][0] for c in clips})
        .groupby(lambda cid: cid.split("_")[0])
        .mean()
    )
    assert (per_spk == 0.5).all()
    degraded = [v for v in cond.values() if v[0] == 1]
    assert all(v[2] in (0, 5, 10, 15, 20) for v in degraded)


def test_quality_dataset_never_puts_one_clip_in_both_classes(clips):
    df = build_quality_dataset(clips, SR, babble_pool=clips[:12], seed=0)
    assert (df.groupby("clip_id")["label"].nunique() == 1).all()
    assert set(df["label"]) == {0, 1}
    assert df[FEATURE_NAMES].notna().all().all()
    # degraded segments have lower estimated SNR on average
    assert df.loc[df.label == 1, "est_snr_db"].mean() < df.loc[df.label == 0, "est_snr_db"].mean()


def test_original_style_is_easier_than_quality_task(clips):
    easy = build_original_style_dataset(clips, SR, seed=0)
    hard = build_quality_dataset(clips, SR, babble_pool=clips[:12], seed=0)
    easy_auc = evaluate_grouped(easy, seed=0, n_splits=3)[0]["logistic_regression"]["roc_auc"]
    hard_auc = evaluate_grouped(hard, seed=0, n_splits=3)[0]["logistic_regression"]["roc_auc"]
    assert easy_auc >= 0.95
    assert easy_auc >= hard_auc


# ---------------------------------------------------------------- evaluation
def test_grouped_cv_keeps_speakers_apart_and_returns_all_models(clips):
    df = build_quality_dataset(clips, SR, babble_pool=clips[:12], seed=0)
    res, oof = evaluate_grouped(df, seed=0, n_splits=3)
    assert set(res) == {
        "svm_rbf",
        "random_forest",
        "knn_k3",
        "gradient_boosting",
        "logistic_regression",
        "snr_threshold_baseline",
    }
    assert all(0 <= r["roc_auc"] <= 1 and r["n"] == len(df) for r in res.values())
    cond = detection_by_condition(oof, "logistic_regression")
    assert set(cond["recall_by_snr_db"]) <= {"0", "5", "10", "15", "20"}
    assert 0 <= cond["false_alarm_rate_clean"] <= 1


def test_random_split_reports_clip_leakage(clips):
    df = build_quality_dataset(clips, SR, babble_pool=clips[:12], seed=0)
    res = evaluate_random_split(df, seed=0)
    assert res["_clips_in_both_train_and_test"] > 0  # segments of one clip on both sides


def test_unseen_noise_evaluation_covers_every_noise_type(clips):
    df = build_quality_dataset(clips, SR, babble_pool=clips[:12], seed=0)
    res = evaluate_unseen_noise(df, seed=0, n_splits=3)
    present = set(df.loc[df.label == 1, "noise"])
    assert set(res) == {"white", "pink", "brown", "hum", "babble"}
    for held in present:
        r = res[held]["logistic_regression"]
        assert 0 <= r["recall_unseen_noise"] <= 1 and 0 <= r["false_alarm_rate_clean"] <= 1


def test_snr_threshold_finds_separating_value():
    snr = np.array([5.0, 6, 7, 30, 31, 32])
    y = np.array([1, 1, 1, 0, 0, 0])
    t = snr_threshold(snr, y)
    assert 7 < t <= 30

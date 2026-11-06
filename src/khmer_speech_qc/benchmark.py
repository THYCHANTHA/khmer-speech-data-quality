"""Clean-vs-degraded classification benchmark.

Two framings are compared on the same clean speech:

* original-style (mirrors the 2024 course report): class 1 is *noise alone* (no speech). This
  tests speech-vs-non-speech, which is easy, so high accuracy says little about data quality.
* quality task: class 1 is *speech with noise mixed in* at 0-20 dB SNR. This is the question a
  data-collection pipeline actually has to answer.

And two evaluation designs:

* random 80/20 split of 2-second segments (segments of one recording land on both sides);
* GroupKFold by speaker (no speaker in both training and test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .audio import TARGET_SR, load_mono
from .checks import audio_stats
from .degrade import NOISE_TYPES, make_noise, mix_at_snr
from .features import FEATURE_NAMES, feature_vector

SEGMENT_S = 2.0
SNR_LEVELS = (0, 5, 10, 15, 20)
NON_SPEECH_NOISE = ("white", "pink", "brown", "hum")


def segments(x: np.ndarray, sr: int, seconds: float = SEGMENT_S) -> list[np.ndarray]:
    n = int(seconds * sr)
    return [x[i : i + n] for i in range(0, len(x) - n + 1, n)]


@dataclass
class Clip:
    """A clip held in memory (`data`) or loaded from `path` on each access (keeps memory low)."""

    clip_id: str
    speaker: str
    data: np.ndarray | None = None
    path: str | None = None
    sr: int = TARGET_SR

    @property
    def audio(self) -> np.ndarray:
        if self.data is not None:
            return self.data
        if self.path is None:
            raise ValueError(f"{self.clip_id}: no audio data or path")
        return load_mono(self.path, self.sr)


def _rows(clip: Clip, x: np.ndarray, sr: int, label: int, noise: str, snr) -> list[dict]:
    out = []
    for k, seg in enumerate(segments(x, sr)):
        out.append(
            {
                "clip_id": clip.clip_id,
                "speaker": clip.speaker,
                "segment": k,
                "label": label,
                "noise": noise,
                "snr_db": snr,
                "est_snr_db": audio_stats(seg, sr).est_snr_db,
                **dict(zip(FEATURE_NAMES, feature_vector(seg, sr), strict=True)),
            }
        )
    return out


def assign_conditions(clips: list[Clip], seed: int) -> dict[str, tuple[int, str, float | None]]:
    """Half of each speaker's clips stay clean, half are degraded with a random noise type and
    SNR. Whole clips are assigned, so one recording never appears in both classes."""
    rng = np.random.default_rng(seed)
    by_spk: dict[str, list[str]] = {}
    for c in clips:
        by_spk.setdefault(c.speaker, []).append(c.clip_id)
    cond = {}
    for spk in sorted(by_spk):
        ids = sorted(by_spk[spk])
        rng.shuffle(ids)
        for i, cid in enumerate(ids):
            if i % 2 == 0:
                cond[cid] = (0, "none", None)
            else:
                cond[cid] = (1, str(rng.choice(NOISE_TYPES)), float(rng.choice(SNR_LEVELS)))
    return cond


def build_quality_dataset(
    clips: list[Clip], sr: int, babble_pool: list[Clip], seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    cond = assign_conditions(clips, seed)
    rows: list[dict] = []
    for c in clips:
        label, kind, snr = cond[c.clip_id]
        x = c.audio
        if label == 1:
            others = [b.audio for b in babble_pool if b.speaker != c.speaker]
            noise = make_noise(kind, len(x), sr, rng, others=others)
            x = mix_at_snr(x, noise, snr)
        rows.extend(_rows(c, x, sr, label, kind, snr))
    return pd.DataFrame(rows)


def build_original_style_dataset(clips: list[Clip], sr: int, seed: int) -> pd.DataFrame:
    """Class 0: clean speech segments. Class 1: non-speech noise segments of the same length
    and level (one noise clip per degraded-assigned speech clip)."""
    rng = np.random.default_rng(seed + 2)
    cond = assign_conditions(clips, seed)
    rows: list[dict] = []
    for c in clips:
        label = cond[c.clip_id][0]
        x = c.audio
        if label == 0:
            rows.extend(_rows(c, x, sr, 0, "none", None))
        else:
            kind = str(rng.choice(NON_SPEECH_NOISE))
            level = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
            noise = make_noise(kind, len(x), sr, rng) * level
            rows.extend(_rows(c, np.clip(noise, -0.99, 0.99), sr, 1, kind, None))
    return pd.DataFrame(rows)


def models(seed: int) -> dict:
    """The four model families of the course report, plus logistic regression."""
    return {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed)),
        "random_forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        ),
        "knn_k3": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "gradient_boosting": make_pipeline(
            StandardScaler(), GradientBoostingClassifier(random_state=seed)
        ),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def metrics(y, pred, score) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "n": int(len(y)),
    }


def snr_threshold(train_snr: np.ndarray, y_train: np.ndarray) -> float:
    """Non-ML baseline: flag as degraded when estimated SNR < t; t maximises training balanced
    accuracy."""
    cands = np.unique(np.round(train_snr, 1))
    best = max(cands, key=lambda t: balanced_accuracy_score(y_train, (train_snr < t).astype(int)))
    return float(best)


def evaluate_random_split(df: pd.DataFrame, seed: int) -> dict:
    X, y = df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy()
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y
    )
    out = {}
    for name, m in models(seed).items():
        m.fit(X[idx_tr], y[idx_tr])
        out[name] = metrics(y[idx_te], m.predict(X[idx_te]), _score(m, X[idx_te]))
    t = snr_threshold(df["est_snr_db"].to_numpy()[idx_tr], y[idx_tr])
    s = df["est_snr_db"].to_numpy()[idx_te]
    out["snr_threshold_baseline"] = {
        **metrics(y[idx_te], (s < t).astype(int), -s),
        "threshold_db": t,
    }
    shared = set(df["clip_id"].to_numpy()[idx_tr]) & set(df["clip_id"].to_numpy()[idx_te])
    out["_clips_in_both_train_and_test"] = len(shared)
    return out


def evaluate_grouped(df: pd.DataFrame, seed: int, n_splits: int = 5) -> tuple[dict, pd.DataFrame]:
    """GroupKFold by speaker. Returns pooled out-of-fold metrics (with per-fold spread) and the
    out-of-fold predictions for per-condition analysis."""
    X, y = df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy()
    groups = df["speaker"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    names = [*models(seed), "snr_threshold_baseline"]
    pred = {n: np.zeros(len(df), dtype=int) for n in names}
    score = {n: np.zeros(len(df)) for n in names}
    fold_metrics: dict[str, list[dict]] = {n: [] for n in names}
    for tr, te in cv.split(X, y, groups):
        for name, m in models(seed).items():
            m.fit(X[tr], y[tr])
            pred[name][te], score[name][te] = m.predict(X[te]), _score(m, X[te])
            fold_metrics[name].append(metrics(y[te], pred[name][te], score[name][te]))
        snr = df["est_snr_db"].to_numpy()
        t = snr_threshold(snr[tr], y[tr])
        pred["snr_threshold_baseline"][te] = (snr[te] < t).astype(int)
        score["snr_threshold_baseline"][te] = -snr[te]
        fold_metrics["snr_threshold_baseline"].append(
            metrics(y[te], pred["snr_threshold_baseline"][te], score["snr_threshold_baseline"][te])
        )
    out = {}
    for n in names:
        pooled = metrics(y, pred[n], score[n])
        folds = pd.DataFrame(fold_metrics[n])
        out[n] = {
            **pooled,
            "fold_sd": {
                k: float(folds[k].std(ddof=1))
                for k in ("accuracy", "balanced_accuracy", "f1", "roc_auc")
            },
        }
    oof = df[["clip_id", "speaker", "label", "noise", "snr_db"]].copy()
    for n in names:
        oof[f"pred_{n}"] = pred[n]
    return out, oof


def evaluate_unseen_noise(df: pd.DataFrame, seed: int, n_splits: int = 5) -> dict:
    """Leave-one-noise-type-out, grouped by speaker: train on clean + four noise types, test on
    held-out speakers' clean segments + the fifth (unseen) noise type. Measures whether a
    detector generalises to noise it was not trained on, which is the situation with real data."""
    X, y = df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy()
    noise, groups = df["noise"].to_numpy(), df["speaker"].to_numpy()
    snr = df["est_snr_db"].to_numpy()
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    out: dict[str, dict] = {}
    for held in NOISE_TYPES:
        pred = {n: [] for n in [*models(seed), "snr_threshold_baseline"]}
        truth: list[np.ndarray] = []
        for tr, te in cv.split(X, y, groups):
            tr = tr[noise[tr] != held]
            te = te[(y[te] == 0) | (noise[te] == held)]
            truth.append(y[te])
            for name, m in models(seed).items():
                m.fit(X[tr], y[tr])
                pred[name].append(m.predict(X[te]))
            t = snr_threshold(snr[tr], y[tr])
            pred["snr_threshold_baseline"].append((snr[te] < t).astype(int))
        yt = np.concatenate(truth)
        out[held] = {
            name: {
                "recall_unseen_noise": float(np.concatenate(p)[yt == 1].mean()),
                "false_alarm_rate_clean": float(np.concatenate(p)[yt == 0].mean()),
                "balanced_accuracy": float(balanced_accuracy_score(yt, np.concatenate(p))),
            }
            for name, p in pred.items()
        }
    return out


def detection_by_condition(oof: pd.DataFrame, model: str) -> dict:
    """Share of degraded segments detected, by SNR and by noise type; false-alarm rate on clean."""
    col = f"pred_{model}"
    deg = oof[oof["label"] == 1]
    return {
        "recall_by_snr_db": {
            str(int(k)): float(v) for k, v in deg.groupby("snr_db")[col].mean().items()
        },
        "recall_by_noise": {k: float(v) for k, v in deg.groupby("noise")[col].mean().items()},
        "false_alarm_rate_clean": float(oof.loc[oof["label"] == 0, col].mean()),
    }

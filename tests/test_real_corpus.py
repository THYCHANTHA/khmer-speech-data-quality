"""Checks against the real SLR42 corpus and the recorded validation summary. Skipped when absent."""

import json
from pathlib import Path

import pytest

from khmer_speech_qc.corpus import data_dir, load_index

ROOT = data_dir()
REPORTS = Path(__file__).resolve().parents[1] / "reports"
needs_data = pytest.mark.skipif(
    not (ROOT / "line_index.tsv").exists(), reason="SLR42 corpus not present"
)


@pytest.mark.data
@needs_data
def test_index_is_complete_and_matches_published_layout():
    idx = load_index(ROOT)
    assert len(idx) == 2906 and idx["exists"].all()
    assert idx.attrs["unindexed_wavs"] == []
    assert idx["speaker"].nunique() == 16
    assert idx["file_id"].str.match(r"^khm_\d{4}_\d+$").all()


@pytest.mark.data
@needs_data
def test_recorded_summary_matches_index():
    summary = json.loads((REPORTS / "validation_summary.json").read_text(encoding="utf-8"))
    idx = load_index(ROOT)
    assert summary["indexed_utterances"] == len(idx)
    assert summary["speakers"] == idx["speaker"].nunique()
    assert summary["distinct_transcripts"] == idx["transcript"].nunique()

# Consent-first design for a Khmer speech collection platform

**Status: design only. No platform has been built or deployed in this repository.**

A voice recording is personal data: it can identify a speaker and may reveal age, gender, region,
health or emotional state. A collection platform must therefore be designed around consent,
minimal data and a clear licence *before* the first recording is accepted. This document sets out
the requirements a platform (for example one modelled on Mozilla Common Voice) should meet, and
how the checks in this repository fit into it.

## 1. Consent and licence

| Requirement | Implementation note |
|---|---|
| Informed consent before recording | Plain-language Khmer and English explanation: purpose, who can access the data, licence, retention, how to withdraw |
| Explicit licence choice | Recordings released under a stated open licence (e.g. CC0 like Common Voice, or CC BY 4.0); the contributor ticks a separate box for it |
| Age check | Contributors under the age of consent need guardian consent or are excluded |
| Withdrawal | A contributor ID (not a name) lets a person request deletion; removed clips are excluded from the next release |
| Third-party content | Contributors read the provided sentences only; uploading other people's audio, broadcasts or app content is not accepted |

## 2. Data minimisation

* Store a random contributor ID, never a name, phone number or e-mail next to the audio.
* Optional demographics (age band, gender, province) are coarse, optional and stored separately.
* Strip file metadata (device, location) from uploaded audio on arrival.
* Sentences to read come from public-domain or openly licensed text, so transcripts carry no
  third-party rights.

## 3. Record schema

| Field | Type | Notes |
|---|---|---|
| `clip_id` | UUID | Random |
| `contributor_id` | UUID | Random, links clips of one speaker for speaker-grouped splits |
| `sentence_id`, `sentence_text` | str | From the prompt list; source and licence recorded |
| `consent_version`, `licence` | str | Exact consent text version and licence accepted |
| `recorded_at` | date | Day precision is enough |
| `sample_rate`, `duration_s`, `channels` | numbers | Filled by the validation step |
| `auto_flags` | list | Output of `khmer_speech_qc.checks.flag_clip` |
| `votes_up`, `votes_down` | int | Community validation, as in Common Voice |
| `status` | enum | pending, accepted, rejected, withdrawn |

## 4. Validation pipeline

1. **Automatic checks on upload** (this repository): duration, sample rate, mono, clipping,
   level, edge silence, estimated SNR, transcript script, speech-rate outliers.
2. **Automatic quality score**: the clean-vs-degraded classifier ranks clips for review; it does
   not reject clips on its own (see the false-alarm rates in the README).
3. **Human validation**: at least two independent listeners confirm that the audio matches the
   sentence, as in Common Voice.
4. **Release splits by contributor**: train, development and test sets never share a speaker.

## 5. Security

* Uploads through HTTPS only; parameterised database queries; file-type checks by content, not by
  extension; size limits; rate limiting.
* Audio stored outside the web root; access to raw clips limited to the maintainers.
* Regular backups; a documented deletion procedure for withdrawals.

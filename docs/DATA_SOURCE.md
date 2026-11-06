# Data source

| Field | Value |
|---|---|
| Corpus | OpenSLR **SLR42**: high-quality transcribed Khmer speech (multi-speaker TTS data), male speakers |
| Page | https://www.openslr.org/42/ |
| Publisher | Google LLC (copyright 2016-2018), hosted by OpenSLR |
| Licence | **CC BY-SA 4.0** (LICENSE file shipped with the corpus, read 2026-11-06) |
| Archive | `km_kh_male.zip`, 866,086,951 bytes, SHA-256 `c0ec9c0494c57f04cf1f2d8d2668d517598375f24e34de07272ecd637c332591` |
| Contents (verified) | 2,906 WAV files (48 kHz, 16-bit, mono) + `line_index.tsv`; 16 speakers; 3.97 hours |
| Stored in this repository | **No.** No audio and no transcripts are committed; reports contain file ids, speakers and measurements only |

## Citation required by the publisher

Keshan Sodimana, Knot Pipatsrisawat, Linne Ha, Martin Jansche, Oddur Kjartansson, Pasindu De
Silva, Supheakmungkol Sarin. *A Step-by-Step Process for Building TTS Voices Using Open Source Data
and Framework for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese.* Proc. SLTU 2018,
Gurugram, India, pp. 66-70. https://doi.org/10.21437/SLTU.2018-14

## Notes

* The corpus was recorded for speech synthesis, so it is clean studio-style speech read from
  prompts. It is a good *clean reference* for degradation experiments, but it does not represent
  phone recordings, spontaneous speech, dialects or female speakers (the female part of SLR42 was
  not downloaded).
* `line_index.tsv` separates the file id and the transcript with **two** tab characters; the
  loader joins every field after the first, so the transcript is read correctly.
* The publisher states that the data were manually quality-checked but may still contain errors.
* CC BY-SA 4.0 covers the corpus. This repository's MIT licence covers only its own code; derived
  datasets (for example degraded audio) would have to be shared under CC BY-SA 4.0 with attribution.

## Local layout used for the recorded run

`C:\Users\<user>\portfolio-data\khmer-asr\` with `download/` (zip and LICENSE) and `raw/` (extracted).
Scripts read the corpus folder from `KHMER_ASR_DATA_DIR`.

# Provenance and credits

This project grew out of a 2024 course project at the Institute of Technology of Cambodia (ITC):

> *Khmer ASR Data Collection.* Suon Senchey, Thy Chantha, Taing Thaitheang. Data Ethics final
> project, Group 6, AMS-I4, Department of Applied Mathematics and Statistics (Data Science), ITC.
> Supervisor: Touch Sopheak.

The course project explored three parts of building Khmer speech data: gathering audio, a web
platform for contributors to upload and review recordings, and machine-learning models that sort
recordings into "clean" and "noisy".

## What this repository is

An **independent re-implementation** by Thy Chantha that takes up the same questions with openly
licensed data and a stricter evaluation design. It contains **no code, audio, text or results from
the course project**; nothing was copied from the team's materials. The course report is
credited here as the origin of the idea; its contents are not reproduced.

## What changed, and why

| Topic | This repository |
|---|---|
| Data | Only openly licensed speech: Google's Khmer corpus on OpenSLR (SLR42, CC BY-SA 4.0). Material whose licence does not allow reuse is not used |
| Quality task | "Degraded" means *speech with noise mixed in* at a known signal-to-noise ratio, so the models must judge recording quality rather than tell speech from non-speech |
| Evaluation | Cross-validation grouped by speaker, compared side by side with a random segment split, so leakage between related recordings is measured rather than assumed away |
| Baseline | A non-machine-learning rule (estimated-SNR threshold) is reported next to every model |
| Validation | Systematic per-clip checks of audio integrity and Khmer transcripts, with documented thresholds |
| Collection platform | A consent-first design document ([COLLECTION_DESIGN.md](COLLECTION_DESIGN.md)) instead of re-building the upload site |

The team members of the course project are acknowledged for the original idea. Any errors in this
repository are the author's own.

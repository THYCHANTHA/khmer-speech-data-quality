import numpy as np
import pytest
from scipy.io import wavfile

from khmer_speech_qc.audio import frame_signal, load_mono, read_wav, resample, rms
from khmer_speech_qc.checks import (
    THRESHOLDS,
    audio_stats,
    flag_clip,
    robust_z,
    text_stats,
)
from khmer_speech_qc.degrade import colored_noise, make_noise, mix_at_snr
from khmer_speech_qc.features import (
    FEATURE_NAMES,
    feature_vector,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    mfcc,
)

SR = 16_000
RNG = np.random.default_rng(0)


def tone(seconds=1.0, hz=440.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# ---------------------------------------------------------------- audio
def test_read_wav_scales_int16_and_reports_format(tmp_path):
    data = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    wavfile.write(tmp_path / "a.wav", 48_000, data)
    x, sr, ch, dtype = read_wav(tmp_path / "a.wav")
    assert (sr, ch, dtype) == (48_000, 1, "int16")
    assert x[1] == pytest.approx(0.5, abs=1e-4) and x[2] == -1.0 and x[3] < 1.0


def test_load_mono_resamples_and_downmixes(tmp_path):
    stereo = np.stack([tone(1, sr=48_000), tone(1, sr=48_000)], axis=1)
    wavfile.write(tmp_path / "s.wav", 48_000, (stereo * 32767).astype(np.int16))
    x = load_mono(tmp_path / "s.wav")
    assert len(x) == SR and x.ndim == 1
    assert rms(x) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_resample_keeps_tone_frequency():
    y = resample(tone(1, 1000, sr=48_000), 48_000)
    peak = np.argmax(np.abs(np.fft.rfft(y))) * SR / len(y)
    assert peak == pytest.approx(1000, abs=2)


def test_frame_signal_shapes_and_padding():
    f = frame_signal(np.arange(10.0), frame=4, hop=3)
    assert f.shape == (3, 4) and list(f[1]) == [3, 4, 5, 6] and f[-1, -1] == 9


# ---------------------------------------------------------------- features
def test_mel_scale_round_trip_and_filterbank_shape():
    assert mel_to_hz(hz_to_mel(1234.0)) == pytest.approx(1234.0)
    fb = mel_filterbank(SR)
    assert fb.shape == (26, 257) and fb.max() <= 1.0 and (fb.sum(axis=1) > 0).all()


def test_mfcc_shape_and_feature_vector_length():
    m = mfcc(tone(), SR)
    # 1 s at 16 kHz, 400-sample frames, 160-sample hop: 1 + ceil((16000 - 400) / 160) = 99 frames
    assert m.shape == (99, 13) and np.isfinite(m).all()
    v = feature_vector(tone(), SR)
    assert v.shape == (len(FEATURE_NAMES),) == (30,)


def test_spectral_flatness_separates_tone_from_white_noise():
    flat_tone = feature_vector(tone(), SR)[FEATURE_NAMES.index("flatness_mean")]
    flat_noise = feature_vector(0.1 * colored_noise(SR, "white", RNG), SR)[
        FEATURE_NAMES.index("flatness_mean")
    ]
    assert flat_noise > 10 * flat_tone


# ---------------------------------------------------------------- degradation
@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_mix_at_snr_hits_target(snr):
    s = tone()
    n = colored_noise(len(s), "white", RNG)
    y = mix_at_snr(s, n, snr)
    residual = y - s * (y @ s) / (s @ s)  # remove the speech component (scale-invariant)
    measured = 20 * np.log10(rms(y - residual) / rms(residual))
    assert measured == pytest.approx(snr, abs=0.3)


def test_mix_never_clips():
    y = mix_at_snr(tone(amp=0.99), colored_noise(SR, "brown", RNG), 0.0)
    assert np.max(np.abs(y)) <= 0.99 + 1e-6


def test_noise_colours_have_expected_spectral_slope():
    def slope(kind):
        p = np.abs(np.fft.rfft(colored_noise(2**16, kind, np.random.default_rng(1)))) ** 2
        f = np.fft.rfftfreq(2**16)
        band = (f > 0.01) & (f < 0.4)
        return np.polyfit(np.log(f[band]), np.log(p[band]), 1)[0]

    assert slope("white") == pytest.approx(0, abs=0.2)
    assert slope("pink") == pytest.approx(-1, abs=0.2)
    assert slope("brown") == pytest.approx(-2, abs=0.2)


def test_babble_requires_other_speakers_and_unknown_kind_fails():
    with pytest.raises(ValueError):
        make_noise("babble", 100, SR, RNG, others=[])
    with pytest.raises(ValueError):
        make_noise("rain", 100, SR, RNG)
    y = make_noise("babble", SR, SR, RNG, others=[tone(0.3), tone(2, 300)])
    assert rms(y) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------- checks
def speech_like(seconds=3.0, lead=0.5, trail=0.5):
    """Tone burst with silent edges and a faint noise floor (-60 dBFS)."""
    x = 0.001 * RNG.standard_normal(int(seconds * SR)).astype(np.float32)
    a, b = int(lead * SR), int((seconds - trail) * SR)
    x[a:b] += tone(seconds, 220)[a:b]
    return x


def test_audio_stats_measure_edges_and_snr():
    a = audio_stats(speech_like(lead=0.5, trail=1.0), SR)
    assert a.leading_silence_s == pytest.approx(0.5, abs=0.05)
    assert a.trailing_silence_s == pytest.approx(1.0, abs=0.05)
    assert a.est_snr_db > 40
    assert a.duration_s == pytest.approx(3.0)


def test_noisy_clip_gets_low_snr_flag():
    s = speech_like()
    noisy = mix_at_snr(s, colored_noise(len(s), "white", RNG), 0.0)
    assert "low_est_snr" in flag_clip(audio_stats(noisy, SR), text_stats("សួស្តី"))
    assert flag_clip(audio_stats(s, SR), text_stats("សួស្តី")) == []


def test_clipping_quiet_short_and_silence_flags():
    clipped = np.clip(speech_like() * 4, -1, 1)
    assert "clipping" in flag_clip(audio_stats(clipped, SR), text_stats("ក"))
    quiet = speech_like() * 0.001
    assert "too_quiet" in flag_clip(audio_stats(quiet, SR), text_stats("ក"))
    assert "too_short" in flag_clip(audio_stats(tone(0.2), SR), text_stats("ក"))
    assert "long_edge_silence" in flag_clip(
        audio_stats(speech_like(6, lead=3), SR), text_stats("ក")
    )


def test_text_stats_and_text_flags():
    t = text_stats("ខ្ញុំ ស្រលាញ់ កម្ពុជា")
    assert t.khmer_ratio == 1.0 and t.latin_chars == 0 and not t.has_zero_width
    a = audio_stats(speech_like(), SR)
    assert "non_khmer_text" in flag_clip(a, text_stats("hello world"))
    assert "empty_transcript" in flag_clip(a, text_stats("   "))
    assert "zero_width_chars" in flag_clip(a, text_stats("ខ្ញុំ​ទៅ"))


def test_robust_z_and_rate_flag():
    z = robust_z(np.array([10, 11, 9, 10, 10, 50.0]))
    assert abs(z[-1]) > THRESHOLDS["speech_rate_robust_z"] and abs(z[0]) < 1
    assert (robust_z(np.ones(5)) == 0).all()
    a = audio_stats(speech_like(), SR)
    assert "speech_rate_outlier" in flag_clip(a, text_stats("ក"), rate_z=9.0)

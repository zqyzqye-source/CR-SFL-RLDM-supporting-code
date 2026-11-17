"""Explicit audio loading and peak normalization for the waveform example."""

from pathlib import Path
import numpy as np
import soundfile as sf


def load_speech(audio_path, n_samples=100000):
    """Read a specified audio file and return normalized, trimmed/padded mono audio.

    No resampling is performed. Stereo channels are averaged before peak
    normalization. The sample rate is returned with the audio array.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    speech, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if len(speech) == 0 or not np.all(np.isfinite(speech)):
        raise ValueError("audio must be nonempty and finite")
    speech = speech.mean(axis=1)
    peak = np.max(np.abs(speech))
    # Match librosa's normal handling of silent or numerically tiny input.
    if peak >= np.finfo(float).tiny:
        speech = speech / peak
    result = np.zeros(n_samples)
    copied = min(n_samples, len(speech))
    result[:copied] = speech[:copied]
    return result, int(sample_rate)

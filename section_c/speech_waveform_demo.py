"""Portable waveform example extracted from read data11.py."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

if __package__:
    from .speech_preprocessing import load_speech
else:
    from speech_preprocessing import load_speech


def generate_waveform(speech, gain=2.5, threshold=1.0, noise_std=0.1, seed=2026):
    """Apply the polynomial/Gaussian-bump echo example and symmetric saturation.

    Defaults preserve the waveform-generation settings in read data11.py.
    The noise is background Gaussian noise. No adaptive filter is run here.
    """
    speech = np.asarray(speech, dtype=float)
    if speech.ndim != 1 or speech.size == 0 or not np.all(np.isfinite(speech)):
        raise ValueError("speech must be a finite, nonempty mono sequence")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be positive")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    x = gain * speech
    dc_bias = 0.8 * np.exp(-16.0 * (0.0 - 0.2) ** 2)
    nonlinear = (0.2 * x - 0.5 * x ** 2 + 0.3 * x ** 3
                 + 0.8 * np.exp(-16.0 * (x - 0.2) ** 2) - dc_bias)
    main = lfilter([1.0, 0.6, -0.4, -0.2], [1.0], nonlinear)
    mismatch = lfilter([0.0, 0.8, -0.5, 0.1], [1.0], x ** 3)
    clean_echo = main + mismatch
    noise = np.random.RandomState(seed).normal(0, noise_std, len(x))
    observation = np.clip(clean_echo + noise, -threshold, threshold)
    return {"amplified_speech": x, "clean_echo": clean_echo,
            "noise": noise, "observation": observation}


def plot_waveform(observation, output_path, show=False):
    """Save the waveform with sample index expressed in units of 10000."""
    import matplotlib.pyplot as plt

    observation = np.asarray(observation, dtype=float)
    if observation.ndim != 1 or observation.size == 0 or not np.all(np.isfinite(observation)):
        raise ValueError("observation must be finite, nonempty and one-dimensional")
    figure, axes = plt.subplots(figsize=(10, 4))
    try:
        sample_axis = np.arange(len(observation)) / 10000.0
        axes.plot(sample_axis, observation, color="#1f77b4", linewidth=0.6)
        axes.set(xlabel=r"Sample ($\times 10^4$)", ylabel="Amplitude")
        axes.set_xlim(0, max(len(observation) / 10000.0, 0.0001))
        axes.grid(True, linestyle="--", alpha=0.5)
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio", type=Path,
        default=Path(r"C:\Users\lenovo\Desktop\Claude\real data\train-clean-100\LibriSpeech\train-clean-100\103\1240\103-1240-0000.flac"),
        help="WAV/FLAC input (default: author local LibriSpeech 103-1240-0000)",
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/section_c"))
    parser.add_argument("--gain", type=float, default=2.5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}. Supply an existing WAV/FLAC file with --audio.")
    speech, sample_rate = load_speech(args.audio, args.samples)
    arrays = generate_waveform(speech, args.gain, args.threshold, args.noise_std, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "waveform_data.npz", speech=speech, **arrays)
    plot_waveform(arrays["observation"], args.output_dir / "waveform.pdf", args.show)
    metadata = {"audio_filename": args.audio.name,
                "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                "sample_rate": sample_rate, "samples": args.samples, "gain": args.gain,
                "threshold": args.threshold, "noise_std": args.noise_std,
                "seed": args.seed, "model": "read data11 polynomial/Gaussian-bump waveform example"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Waveform example saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

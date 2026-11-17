# Section C speech and waveform utilities

The auxiliary waveform pipeline is extracted from `read data11.py`.
It reads one explicitly specified audio file, normalizes its peak, trims or
zero-pads the sequence, generates a nonlinear echo waveform, adds background
Gaussian noise and applies symmetric saturation.

## Run

From the repository root, the author can run with the verified local default:

```powershell
python section_c/speech_waveform_demo.py
```

The default is the original script's currently selected file,
`103-1240-0000.flac` (16 kHz, mono), located at:

```text
C:\Users\lenovo\Desktop\Claude\real data\train-clean-100\LibriSpeech\train-clean-100\103\1240\103-1240-0000.flac
```

Other users can supply an existing WAV or FLAC file:

```powershell
python section_c/speech_waveform_demo.py --audio "D:\speech\example.flac"
```

Optional arguments:

```powershell
python section_c/speech_waveform_demo.py --audio "D:\speech\example.flac" --samples 100000 --gain 2.5 --threshold 1.0 --noise-std 0.1 --seed 2026 --output-dir outputs/section_c
```

WAV and FLAC inputs are supported by SoundFile. An explicit file replaces the
original selection of the first file in a directory. No resampling is
performed. Stereo channels are averaged to mono. Peak normalization is
applied before trimming/padding, as in the original mono-audio pipeline.
NumPy handles normalization without requiring librosa.

## Waveform configuration

The defaults preserve the waveform settings in `read data11.py`:

| Parameter | Value |
| --- | --- |
| Sample count | 100000 |
| Input amplification | 2.5 |
| Saturation bounds | -1 and +1 |
| Background Gaussian noise standard deviation | 0.1, corresponding to variance 0.01 |
| Random seed | 2026 |
| Main path | `[1.0, 0.6, -0.4, -0.2]` |
| Additional cubic path | `[0.0, 0.8, -0.5, 0.1]` |

For amplified audio `x`, the main nonlinear signal is

```text
0.2*x - 0.5*x**2 + 0.3*x**3
+ 0.8*exp(-16*(x - 0.2)**2) - 0.8*exp(-16*0.2**2).
```

The main path filters this signal. The additional path filters `x**3`.
Their outputs are added before Gaussian noise and saturation. The constant
subtraction makes the static nonlinear response zero at zero input.
This is the polynomial/Gaussian-bump waveform illustration from the source
script. The exported functions cover speech preprocessing, waveform generation
and plotting. Adaptive-filter comparisons and ERLE calculations are outside
the scope of this auxiliary module.

## Outputs

- `waveform.pdf`: saturated microphone waveform with sample indices in units of 10000.
- `waveform_data.npz`: normalized input, amplified input, clean echo, noise and saturated observation.
- `metadata.json`: input filename/hash, sample rate and generation parameters.

Generated figures are saved without opening an interactive window. Python
users can call `plot_waveform(..., show=True)` to display a figure.

Audio files are not included. Obtain LibriSpeech from
[OpenSLR](https://www.openslr.org/12) and specify the exact utterance file
used in a run. The input filename and SHA-256 are recorded for reference.

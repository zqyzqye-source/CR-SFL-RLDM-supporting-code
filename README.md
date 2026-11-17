# Supporting Code for CR-SFL-RLDM

Selected experimental code and supporting modules related to **Robust Spline
Functional Link Adaptive Filter for Nonlinear Censored Regression**.

## Contents

| Directory | Contents |
| --- | --- |
| `section_a` | Existing experimental script associated with Fig. 2(a), supplied separately by the author. |
| `section_b` | Data-recovery functions, ensemble MSE calculation, linear-domain smoothing and visualization. |
| `section_c` | Speech loading, peak normalization, waveform generation and visualization. |

The B and C directories provide auxiliary functions and runnable usage examples.
Their individual READMEs describe inputs, numerical settings and outputs.

## Installation

Use Python 3.10 or later. From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

The utilities were checked with Python 3.12, NumPy 2.3.4, SciPy 1.16.3,
Matplotlib 3.10.7 and SoundFile 0.13.1.

## Section B example

```bash
python section_b/demo_auxiliary.py
```

This writes `outputs/section_b/recovery_example.csv`, `synthetic_errors.npz`
and `mse_example.pdf`. The example uses explicitly generated synthetic errors
to demonstrate the utility interfaces.

## Section C example

On the author's computer, the verified default input is LibriSpeech utterance
`103-1240-0000.flac`. Run:

```bash
python section_c/speech_waveform_demo.py
```

Other users should supply their own local WAV or FLAC path. Put quotation marks
around a path that contains spaces. For example, on Windows:

```powershell
python section_c/speech_waveform_demo.py --audio "D:\speech\example.flac"
```

On other platforms, replace the audio path with your local file path.
The command writes `outputs/section_c/waveform.pdf`, `waveform_data.npz`
and `metadata.json`. The metadata records the input filename and hash,
sample rate, seed and generation settings. No audio dataset is distributed.

LibriSpeech audio can be obtained from its [official data page](https://www.openslr.org/12).
Specify the utterance file explicitly when sharing an experiment configuration.

## Checks

```bash
python -m unittest discover -s tests -v
```

Section A is retained as supplied. The commands above exercise the B and C
utilities. Generated output files are excluded from version control.

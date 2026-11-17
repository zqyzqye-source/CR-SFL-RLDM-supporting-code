# Section B auxiliary modules

## Data recovery

`data_recovery.py` extracts the recovery helpers used in `Untitled-1.py`
and `Untitled-3.py`. It accepts a prediction and a supplied noise standard
deviation, performs conditional Gaussian recovery at the observation bounds,
and passes uncensored samples through unchanged.

```python
from section_b.data_recovery import em_recover_asym

recovered = em_recover_asym(
    d_obs=1.5, y_pred=0.0, r_pos=1.5, r_neg=0.5, sigma_bg=0.1
)
```

`r_pos` and `r_neg` are positive magnitudes. They define the signed interval
`[-r_neg, r_pos]`. `sigma_bg` is a standard deviation, not a variance.
The lower-level `recover_censored` function accepts signed `lower` and `upper`
bounds. Scalar and broadcast-compatible array arguments are supported.

The original numerical treatment is retained: an additive `1e-15` protects
the Gaussian tail denominator, and the upper-tail ratio is approximated by
`z` for `z > 5`. This module performs recovery only. The caller supplies the
prediction and variance information.

## MSE statistics and visualization

`mse_visualization.py` provides:

- `ensemble_mse(errors, trial_axis=0)` for averaging squared error magnitudes.
- `ema_smooth(values, alpha)` for smoothing a linear-MSE sequence.
- `mse_to_db(mse, alpha=None)` for optional smoothing followed by dB conversion.
- `plot_mse(curves, output_path, alpha=0.01)` for saving a labeled figure.

The order is **ensemble average, optional smoothing in the linear domain,
then `10*log10(MSE + 1e-12)`**. This preserves the processing used in the
white-input and asymmetric-threshold source scripts. Smoothing uses the first
MSE value as its initial value. Trial counts are inferred from the provided
error array instead of being hard-coded as a global variable.

```python
from section_b.mse_visualization import ensemble_mse, plot_mse

# errors has shape (number_of_trials, number_of_samples).
mse = ensemble_mse(errors)
plot_mse({"My experiment": mse}, "outputs/my_mse.pdf", alpha=0.01)
```

Use `alpha=0.015` for the smoothing value in `Untitled-1.py`, `alpha=0.01`
for `Untitled-3.py`, or `alpha=None` to disable smoothing. The specialized
split smoothing of the tracking script is not part of this generic utility.

## Runnable example

From the repository root:

```bash
python section_b/demo_auxiliary.py --output-dir outputs/section_b
```

The CSV demonstrates recovery calls. The NPZ stores synthetic trial errors
and their unsmoothed ensemble MSE, and the PDF visualizes this synthetic
sequence. These example data demonstrate the auxiliary interfaces rather than
the paper's algorithm comparison curves.

"""MSE averaging and visualization based on the Section B source scripts."""

from pathlib import Path
import numpy as np


def _mse_array(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("MSE must be a nonempty array of finite nonnegative values")
    return values


def ensemble_mse(errors, trial_axis=0):
    """Average squared error magnitudes over trials in the linear domain."""
    errors = np.asarray(errors)
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("errors must be nonempty and finite")
    return np.mean(np.abs(errors) ** 2, axis=trial_axis)


def ema_smooth(values, alpha=0.01):
    """Smooth a one-dimensional linear-MSE sequence, preserving its first value."""
    values = _mse_array(values)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1 - alpha) * result[index - 1]
    return result


def mse_to_db(mse, alpha=None, epsilon=1e-12):
    """Optionally smooth linear MSE, then compute 10*log10(MSE + epsilon)."""
    mse = _mse_array(mse)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if alpha is not None:
        mse = ema_smooth(mse, alpha)
    return 10 * np.log10(mse + epsilon)


def plot_mse(curves, output_path, alpha=0.01, start_iteration=0, show=False):
    """Plot a mapping of labels to linear MSE arrays and save one figure."""
    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("curves cannot be empty")
    figure, axes = plt.subplots(figsize=(9, 5))
    try:
        for label, values in curves.items():
            values = _mse_array(values)
            if values.ndim != 1:
                raise ValueError("each MSE curve must be one-dimensional")
            axis = np.arange(start_iteration, start_iteration + len(values))
            axes.plot(axis, mse_to_db(values, alpha=alpha), label=label)
        axes.set(xlabel="Iteration", ylabel="MSE (dB)")
        axes.grid(True, linestyle=":", alpha=0.5)
        axes.legend()
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path

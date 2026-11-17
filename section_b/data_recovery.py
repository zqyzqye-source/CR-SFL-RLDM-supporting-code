"""Gaussian recovery helpers extracted from Untitled-3.py.

The caller supplies the prediction and positive noise standard deviation.
The right-tail ratio uses the original approximation ratio(z) = z for z > 5.
These functions perform recovery only, without updating a filter or detector.
"""

import numpy as np
from scipy.stats import norm


def _finite_array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def imr_right(z):
    """Return the original numerically protected upper-tail ratio."""
    z = _finite_array(z, "z")
    return np.where(z > 5.0, z, norm.pdf(z) / (norm.sf(z) + 1e-15))


def imr_left(z):
    """Return the original numerically protected lower-tail ratio."""
    return imr_right(-_finite_array(z, "z"))


def recover_censored(d_obs, y_pred, lower, upper, sigma_bg):
    """Recover observations at two saturation bounds.

    lower and upper are the signed observation bounds, for example -0.5 and
    1.5. sigma_bg is a standard deviation, rather than a variance. Scalar or
    broadcast-compatible array inputs are supported. Interior observations
    pass through unchanged.
    """
    obs, pred, lo, hi, sigma = np.broadcast_arrays(
        *[_finite_array(v, n) for v, n in zip(
            [d_obs, y_pred, lower, upper, sigma_bg],
            ["d_obs", "y_pred", "lower", "upper", "sigma_bg"]
        )]
    )
    if np.any(lo >= hi):
        raise ValueError("lower must be smaller than upper")
    if np.any(sigma <= 0):
        raise ValueError("sigma_bg must be positive")
    recovered = np.where(
        obs >= hi,
        pred + sigma * imr_right((hi - pred) / sigma),
        np.where(obs <= lo, pred - sigma * imr_left((lo - pred) / sigma), obs),
    )
    return float(recovered) if recovered.ndim == 0 else recovered


def em_recover(d_obs, y_pred, r_thresh, sigma_bg):
    """Compatibility wrapper for the symmetric helper in Untitled-1.py."""
    threshold = _finite_array(r_thresh, "r_thresh")
    if np.any(threshold <= 0):
        raise ValueError("r_thresh must be positive")
    return recover_censored(d_obs, y_pred, -threshold, threshold, sigma_bg)


def em_recover_asym(d_obs, y_pred, r_pos, r_neg, sigma_bg):
    """Compatibility wrapper: r_pos and r_neg are positive magnitudes.

    The signed observation interval is [-r_neg, r_pos].
    """
    positive = _finite_array(r_pos, "r_pos")
    negative = _finite_array(r_neg, "r_neg")
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("r_pos and r_neg must be positive magnitudes")
    return recover_censored(d_obs, y_pred, -negative, positive, sigma_bg)

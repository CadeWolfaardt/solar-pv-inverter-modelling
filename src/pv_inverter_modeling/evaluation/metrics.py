# thirdpartylib
import numpy as np
# projectlib
from pv_inverter_modeling.utils.typing import ArrayLike1D

def safe_mape(
        y_true: ArrayLike1D, 
        y_pred: ArrayLike1D, 
        eps: float = 1e-6
    ) -> float:
    """
    Numerically safe Mean Absolute Percentage Error (MAPE).

    This variant clamps the denominator to `eps` to avoid division
    by zero and excessive inflation when y_true is near zero.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    eps : float, default=1e-6
        Minimum value for the denominator.

    Returns
    -------
    float
        MAPE expressed as a percentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.mean(
            np.abs((y_true - y_pred) / np.maximum(eps, y_true))
        ) * 100.0
    )

def strict_r2(
    y_true: ArrayLike1D,
    y_pred: ArrayLike1D,
    eps: float = 1e-12
) -> float:
    """
    Compute the strict coefficient of determination (R²).

    This implementation follows the textbook definition of R²:

        R² = 1 - Σ(y_true - y_pred)² / Σ(y_true - ȳ_true)²

    Unlike :func:`sklearn.metrics.r2_score`, this function does **not**
    apply fallback conventions when the variance of ``y_true`` is zero
    or near-zero. In such degenerate cases, the R² score is 
    mathematically undefined, and this function returns ``NaN`` instead 
    of coercing the result to 0.0 or 1.0.

    This behavior is intentional and designed for analytical and 
    research workflows where undefined metrics should be surfaced 
    explicitly rather than silently masked.

    Parameters
    ----------
    y_true : ArrayLike1D
        Ground-truth target values.
    y_pred : ArrayLike1D
        Predicted target values. Must have the same shape as ``y_true``.
    eps : float, default=1e-12
        Minimum allowable denominator value. If the total variance of
        ``y_true`` is less than or equal to ``eps``, the function 
        returns ``NaN``.

    Returns
    -------
    float
        The R² score. Returns ``NaN`` if the variance of ``y_true`` is
        zero or near-zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= eps:
        return float("nan")

    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom
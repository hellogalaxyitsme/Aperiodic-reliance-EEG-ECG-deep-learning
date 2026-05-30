from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedAperiodicFit:
    """Fixed 1/f fits for log-PSD epochs.

    The model is:

    log_power(f) = offset - exponent * log(f)
    """

    log_psd: np.ndarray
    fitted_log_psd: np.ndarray
    residual_log_psd: np.ndarray
    offset: np.ndarray
    exponent: np.ndarray
    r_squared: np.ndarray

    @property
    def params(self) -> np.ndarray:
        return np.stack([self.offset, self.exponent], axis=-1)


def fit_fixed_aperiodic(
    psd: np.ndarray,
    freqs: np.ndarray,
    eps: float = 1e-30,
) -> FixedAperiodicFit:
    """Fit a fixed aperiodic background for each epoch and channel.

    Parameters
    ----------
    psd:
        Array with shape ``(epochs, channels, frequencies)``.
    freqs:
        Frequency grid with shape ``(frequencies,)``.
    eps:
        Lower bound before log transform.
    """
    psd = np.asarray(psd)
    freqs = np.asarray(freqs)
    if psd.ndim != 3:
        raise ValueError(f"Expected PSD shape (epochs, channels, freqs), got {psd.shape}")
    if freqs.ndim != 1 or freqs.shape[0] != psd.shape[-1]:
        raise ValueError(
            f"Frequency grid shape {freqs.shape} is incompatible with PSD shape {psd.shape}"
        )
    if np.any(freqs <= 0):
        raise ValueError("All frequencies must be positive for log-frequency fitting.")

    log_psd = np.log(np.maximum(psd.astype("float64", copy=False), eps))
    design = np.column_stack([np.ones_like(freqs), -np.log(freqs)])
    pinv = np.linalg.pinv(design)

    n_epochs, n_channels, n_freqs = log_psd.shape
    observations = log_psd.transpose(2, 0, 1).reshape(n_freqs, -1)
    coefficients = pinv @ observations

    fitted = design @ coefficients
    fitted_log_psd = fitted.reshape(n_freqs, n_epochs, n_channels).transpose(1, 2, 0)
    residual_log_psd = log_psd - fitted_log_psd

    params = coefficients.T.reshape(n_epochs, n_channels, 2)
    offset = params[:, :, 0]
    exponent = params[:, :, 1]

    centered = log_psd - log_psd.mean(axis=-1, keepdims=True)
    ss_total = np.sum(centered**2, axis=-1)
    ss_residual = np.sum(residual_log_psd**2, axis=-1)
    r_squared = 1.0 - ss_residual / np.maximum(ss_total, np.finfo(float).eps)

    return FixedAperiodicFit(
        log_psd=log_psd.astype("float32"),
        fitted_log_psd=fitted_log_psd.astype("float32"),
        residual_log_psd=residual_log_psd.astype("float32"),
        offset=offset.astype("float32"),
        exponent=exponent.astype("float32"),
        r_squared=r_squared.astype("float32"),
    )


def flatten_spectral_features(array: np.ndarray) -> np.ndarray:
    """Flatten epoch x channel x frequency arrays into tabular features."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D spectral array, got {array.shape}")
    return array.reshape(array.shape[0], -1)


def flatten_param_features(params: np.ndarray) -> np.ndarray:
    """Flatten epoch x channel x parameter arrays into tabular features."""
    if params.ndim != 3:
        raise ValueError(f"Expected 3D parameter array, got {params.shape}")
    return params.reshape(params.shape[0], -1)


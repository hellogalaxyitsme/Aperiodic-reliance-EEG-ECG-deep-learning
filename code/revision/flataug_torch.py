"""Training-time flattening augmentation (GPU implementation).

Batched torch re-implementation of the paper's fixed linear-fit phase-preserving
flattening operator, so it can be applied on the fly to training batches:

  variant="eeg"  run_tuab_biot_intervention.make_phase_preserving_inputs
                 (log10-power ~ a - b*log10 f fit over the band; divide band FFT
                 by the centred fitted amplitude shape; mean-centre; std-match)
  variant="ecg"  run_ptbxl_raw_cnn_interventions.intervention_batch
                 (log-amplitude ~ a - b*log f fit; residual + mean fit; RMS-match;
                 mean-centre)

The two fits differ only by a constant factor in log space, so the removed
shape is identical; the variants reproduce each pipeline's exact rescaling.
Equivalence with the numpy reference is checked by `selftest()`.
"""
from __future__ import annotations

import torch


def flatten_linear(x: torch.Tensor, sfreq: float, band=(1.0, 45.0), variant: str = "eeg") -> torch.Tensor:
    """x: (batch, channels, time) float tensor in raw units. Returns flattened tensor, same shape."""
    n = x.shape[-1]
    xd = x.double()
    fft = torch.fft.rfft(xd, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sfreq).to(x.device, torch.float64)
    hi = band[1] if variant == "eeg" else min(band[1], sfreq / 2.0 - 1e-6)
    mask = (freqs >= band[0]) & (freqs <= hi)
    bf = freqs[mask]
    band_fft = fft[..., mask]
    amp = band_fft.abs().clamp_min(1e-12)
    log_amp = amp.log()
    design = torch.stack([torch.ones_like(bf), -bf.log()], dim=1)  # (n_bins, 2)
    pinv = torch.linalg.pinv(design)  # (2, n_bins)
    coef = log_amp @ pinv.T  # (..., 2)
    fitted = coef @ design.T  # (..., n_bins), natural-log amplitude
    shape = fitted - fitted.mean(dim=-1, keepdim=True)
    new_fft = fft.clone()
    new_fft[..., mask] = band_fft * torch.exp(-shape)
    rec = torch.fft.irfft(new_fft, n=n, dim=-1)
    if variant == "eeg":
        rec = rec - rec.mean(dim=-1, keepdim=True)
        rec = rec * (xd.std(dim=-1, keepdim=True) / rec.std(dim=-1, keepdim=True).clamp_min(1e-6))
    else:
        rms_x = xd.pow(2).mean(dim=-1, keepdim=True).sqrt()
        rms_r = rec.pow(2).mean(dim=-1, keepdim=True).sqrt()
        rec = rec * (rms_x / rms_r.clamp_min(1e-8))
        rec = rec - rec.mean(dim=-1, keepdim=True)
    return rec.to(x.dtype)


def augment_batch(x: torch.Tensor, sfreq: float, prob: float, generator: torch.Generator, variant: str):
    """Replace each window by its flattened version with probability `prob`."""
    if prob <= 0:
        return x, 0
    pick = torch.rand(x.shape[0], generator=generator) < prob
    k = int(pick.sum())
    if k == 0:
        return x, 0
    idx = pick.nonzero().squeeze(1).to(x.device)
    out = x.clone()
    out[idx] = flatten_linear(x[idx], sfreq, variant=variant)
    return out, k


def selftest():
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_ptbxl_raw_cnn_interventions as ecg
    import run_tuab_biot_intervention as fm

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((6, 4, 2000)), axis=-1).astype("float32") * 5
    ref = fm.make_phase_preserving_inputs(x, sfreq=100.0, band_min=1.0, band_max=45.0)["phase_flattened"]
    got = flatten_linear(torch.from_numpy(x), 100.0, variant="eeg").numpy()
    e_eeg = float(np.abs(ref - got).max() / np.abs(ref).max())
    xe = np.cumsum(rng.standard_normal((6, 12, 1000)), axis=-1).astype("float32")
    ref = ecg.intervention_batch(xe, 100.0)["phase_flattened"]
    got = flatten_linear(torch.from_numpy(xe), 100.0, variant="ecg").numpy()
    e_ecg = float(np.abs(ref - got).max() / np.abs(ref).max())
    print(f"selftest max rel err: eeg {e_eeg:.2e}  ecg {e_ecg:.2e}")
    assert e_eeg < 1e-4 and e_ecg < 1e-4


if __name__ == "__main__":
    selftest()

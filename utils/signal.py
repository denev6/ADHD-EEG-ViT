import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from sktime.libs.vmdpy import VMD


def _local_mean_envelope(signal):
    """Computes the local mean and envelope of a signal based on extrema interpolation."""
    x = np.arange(len(signal))

    # Find local maxima and minima
    max_idx = argrelextrema(signal, np.greater)[0]
    min_idx = argrelextrema(signal, np.less)[0]

    # Interpolate maxima and minima to get upper and lower envelopes
    if len(max_idx) > 1 and len(min_idx) > 1:
        upper_env = interp1d(
            max_idx, signal[max_idx], kind="cubic", fill_value="extrapolate"
        )(x)
        lower_env = interp1d(
            min_idx, signal[min_idx], kind="cubic", fill_value="extrapolate"
        )(x)
    else:
        upper_env = lower_env = np.zeros_like(signal)

    # Compute local mean
    local_mean = (upper_env + lower_env) / 2
    return local_mean


def apply_rlmd(signal, max_imfs=5, tol=1e-6):
    """Performs Robust Local Mean Decomposition (RLMD) to extract IMFs."""
    imfs = []
    residue = signal.copy()

    for _ in range(max_imfs):
        local_mean = _local_mean_envelope(residue)
        pf = residue - local_mean  # Extract product function

        # Stopping criterion: If the change is too small, break
        if np.linalg.norm(pf) < tol:
            break

        imfs.append(pf)
        residue -= pf

    return imfs


def apply_vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-6):
    """Applies Variational Mode Decomposition (VMD) to extract modes."""

    # Perform VMD decomposition
    imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    return imfs


# Example Usage
if __name__ == "__main__":
    np.random.seed(42)
    sample_signal = np.sin(
        2 * np.pi * 5 * np.linspace(0, 1, 500)
    ) + 0.5 * np.random.randn(500)

    rlmd_imfs = apply_rlmd(sample_signal)
    vmd_imfs = apply_vmd(sample_signal)

    print("RLMD IMFs Shape:", np.array(rlmd_imfs).shape)
    print("VMD IMFs Shape:", np.array(vmd_imfs).shape)

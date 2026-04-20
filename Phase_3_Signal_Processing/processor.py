"""
Phase 3 — Signal Processing and Conditioning

Public API
----------
    lowpass_filter(signal, cutoff_hz, sample_rate, order=4)  -> np.ndarray
    resample_signal(signal, src_rate, tgt_rate)               -> np.ndarray
    normalize(signal, method='zscore')                        -> np.ndarray
    condition_signal(sig_dict, **kwargs)                      -> Signal dict (conditioned copy)
    condition_dataset(signals, **kwargs)                      -> list[Signal dict]

Conditioning pipeline (default)
--------------------------------
    1. Optional resampling   — resample_to_hz=None (keep original rate)
    2. Low-pass filter       — cutoff = filter_cutoff_fraction * Nyquist  (default 0.8)
    3. Z-score normalization — per channel, per signal

Design rationale
----------------
    Motors are captured at 20 000 Hz (60 ms window, Nyquist 10 kHz).
    Chargers are captured at 5 000 Hz (240 ms window, Nyquist 2.5 kHz).
    Forcing a common rate would destroy motor high-frequency EMI content.
    Instead, filtering and normalization are applied relative to each
    signal's own Nyquist so the conditioning is always proportionally
    equivalent across devices.  Resampling is available as an option for
    tasks that explicitly need a unified rate (e.g. raw-waveform comparison).
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_2_Ingestion_Pipeline"))

# ── Primitive operations ──────────────────────────────────────────────────────

def lowpass_filter(signal: np.ndarray, cutoff_hz: float, sample_rate: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal      : 1-D voltage array
    cutoff_hz   : filter cutoff in Hz (must be < sample_rate / 2)
    sample_rate : samples per second
    order       : filter order (4 is a good default)
    """
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        warnings.warn(f"cutoff_hz ({cutoff_hz}) >= Nyquist ({nyquist}); filter skipped.")
        return signal.copy()
    if len(signal) < 3 * order:
        warnings.warn("Signal too short to filter; returning copy.")
        return signal.copy()
    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


def resample_signal(signal: np.ndarray, src_rate: float, tgt_rate: float) -> np.ndarray:
    """
    Resample a signal from src_rate to tgt_rate using polyphase filtering.
    The up/down integers are derived from the reduced fraction tgt/src.
    """
    if src_rate == tgt_rate:
        return signal.copy()
    src_i, tgt_i = int(src_rate), int(tgt_rate)
    common = gcd(src_i, tgt_i)
    up, down = tgt_i // common, src_i // common
    return resample_poly(signal, up, down).astype(signal.dtype)


def normalize(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a 1-D signal.

    method : "zscore"  — zero mean, unit std
             "minmax"  — scale to [0, 1]
             "rms"     — divide by RMS amplitude
    """
    if method == "zscore":
        mu, sigma = np.mean(signal), np.std(signal)
        return (signal - mu) / sigma if sigma > 0 else signal - mu
    if method == "minmax":
        lo, hi = signal.min(), signal.max()
        return (signal - lo) / (hi - lo) if hi > lo else np.zeros_like(signal)
    if method == "rms":
        rms = np.sqrt(np.mean(signal ** 2))
        return signal / rms if rms > 0 else signal.copy()
    raise ValueError(f"Unknown normalization method: {method!r}")


# ── Signal-level conditioning ─────────────────────────────────────────────────

def condition_signal(
    sig_dict: dict,
    resample_to_hz: float | None = None,
    filter_cutoff_fraction: float = 0.8,
    norm_method: str = "zscore",
) -> dict:
    """
    Apply the full conditioning pipeline to one Signal dict.

    Returns a new dict (original is not mutated) with:
      - conditioned ch1 / ch2 arrays
      - updated sample_rate_hz, increment_sec, sample_count, time axis
      - 'conditioned': True flag
      - 'conditioning_params': record of settings used

    Parameters
    ----------
    sig_dict               : Signal dict from Phase 2 loader
    resample_to_hz         : target sample rate in Hz, or None to keep original
    filter_cutoff_fraction : LP cutoff as fraction of each signal's Nyquist (0 < f < 1)
    norm_method            : 'zscore' | 'minmax' | 'rms'
    """
    out = dict(sig_dict)  # shallow copy — arrays replaced below, not mutated
    src_rate = sig_dict["sample_rate_hz"]

    if src_rate is None:
        out["conditioned"] = False
        out["conditioning_notes"] = "skipped: no sample_rate_hz"
        return out

    # ── Step 1: optional resampling ───────────────────────────────────────────
    tgt_rate = resample_to_hz if resample_to_hz else src_rate
    for ch in ("ch1", "ch2"):
        arr = sig_dict[ch]
        if arr is None or len(arr) == 0:
            out[ch] = arr
            continue
        if resample_to_hz and resample_to_hz != src_rate:
            arr = resample_signal(arr, src_rate, tgt_rate)
        out[ch] = arr

    # Update timing metadata after resampling
    out["sample_rate_hz"] = tgt_rate
    out["increment_sec"]  = 1.0 / tgt_rate
    ref = out["ch1"] if out["ch1"] is not None else out["ch2"]
    new_len = len(ref) if ref is not None else sig_dict["sample_count"]
    out["sample_count"] = new_len
    start = sig_dict["start_time"] if sig_dict["start_time"] is not None else 0.0
    out["time"] = start + np.arange(new_len) * out["increment_sec"]

    # ── Step 2: low-pass filter ───────────────────────────────────────────────
    cutoff_hz = filter_cutoff_fraction * (tgt_rate / 2.0)
    for ch in ("ch1", "ch2"):
        arr = out[ch]
        if arr is None or len(arr) == 0:
            continue
        out[ch] = lowpass_filter(arr, cutoff_hz, tgt_rate)

    # ── Step 3: normalization ─────────────────────────────────────────────────
    for ch in ("ch1", "ch2"):
        arr = out[ch]
        if arr is None or len(arr) == 0:
            continue
        out[ch] = normalize(arr, method=norm_method)

    out["conditioned"] = True
    out["conditioning_params"] = {
        "resample_to_hz":         resample_to_hz,
        "original_rate_hz":       src_rate,
        "filter_cutoff_hz":       cutoff_hz,
        "filter_cutoff_fraction": filter_cutoff_fraction,
        "norm_method":            norm_method,
    }
    return out


# ── Dataset-level conditioning ────────────────────────────────────────────────

def condition_dataset(
    signals: list[dict],
    resample_to_hz: float | None = None,
    filter_cutoff_fraction: float = 0.8,
    norm_method: str = "zscore",
) -> list[dict]:
    """
    Apply conditioning to every signal in the dataset.

    Returns a new list of conditioned Signal dicts.
    """
    return [
        condition_signal(
            s,
            resample_to_hz=resample_to_hz,
            filter_cutoff_fraction=filter_cutoff_fraction,
            norm_method=norm_method,
        )
        for s in signals
    ]

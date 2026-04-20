"""
Phase 4 — Feature Engineering

Extracts 51 features per signal and writes features.csv.

Design
------
Features are extracted from the appropriate signal version:

    Amplitude features  -> RAW signals (pre-conditioning)
        mean, std, rms, variance, peak_to_peak, energy
        xc_rms_ratio, xc_energy_ratio

    Shape & spectral features -> CONDITIONED signals (filtered + z-scored)
        skewness, kurtosis, zero_crossing_rate, num_peaks, crest_factor, iqr
        all fd_* frequency-domain features
        xc_correlation, xc_dom_freq_diff_hz, xc_centroid_diff_hz

This split is necessary because z-score normalization destroys amplitude
information — extracting RMS from a z-scored signal always gives ~1.0.
Shape and spectral features are scale-invariant (or benefit from filtering),
so they stay on the conditioned signals.

Feature groups
--------------
Time-domain (12 per channel):
    mean, std, rms, variance, peak_to_peak, energy,         [from raw]
    skewness, kurtosis, zero_crossing_rate, num_peaks,      [from conditioned]
    crest_factor, iqr                                       [from conditioned]

Frequency-domain (11 per channel, from conditioned):
    dominant_freq_hz, dominant_freq_norm,
    spectral_centroid_hz, spectral_centroid_norm,
    spectral_spread_hz, spectral_entropy,
    band_energy_abs_low   (0 – 500 Hz),
    band_energy_abs_mid   (500 – 2 000 Hz),
    band_energy_abs_high  (2 000 Hz+),
    band_energy_rel_low   (0 – 20 % Nyquist),
    band_energy_rel_high  (50 – 100 % Nyquist)

Cross-channel (5, dual-channel only — NaN for single):
    xc_rms_ratio, xc_energy_ratio,                          [from raw]
    xc_correlation, xc_dom_freq_diff_hz, xc_centroid_diff_hz [from conditioned]

All features are prefixed: td_ch1_*, fd_ch1_*, td_ch2_*, fd_ch2_*, xc_*
Single-channel signals have NaN for the missing channel's columns.

Public API
----------
    extract_features(raw_sig, cond_sig)           -> flat feature dict (one row)
    build_feature_table(raw_signals, cond_signals) -> pd.DataFrame
    run(save_csv=True)                             -> pd.DataFrame
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_2_Ingestion_Pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_3_Signal_Processing"))

from loader import load_dataset
from processor import condition_dataset

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.csv")

# ── Primitive feature functions ───────────────────────────────────────────────

def _td_amplitude(signal: np.ndarray, prefix: str) -> dict:
    """Amplitude features — must be extracted from RAW signals."""
    return {
        f"{prefix}mean":         float(np.mean(signal)),
        f"{prefix}std":          float(np.std(signal)),
        f"{prefix}rms":          float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}variance":     float(np.var(signal)),
        f"{prefix}peak_to_peak": float(np.ptp(signal)),
        f"{prefix}energy":       float(np.mean(signal ** 2)),
    }


def _td_shape(signal: np.ndarray, prefix: str) -> dict:
    """Shape features — scale-invariant, extracted from CONDITIONED signals."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    abs_max = float(np.max(np.abs(signal)))
    peaks, _ = find_peaks(signal, height=np.mean(signal) + np.std(signal))
    return {
        f"{prefix}skewness":           float(skew(signal)),
        f"{prefix}kurtosis":           float(sp_kurtosis(signal)),
        f"{prefix}zero_crossing_rate": float(np.sum(np.diff(np.sign(signal)) != 0)) / len(signal),
        f"{prefix}num_peaks":          int(len(peaks)),
        f"{prefix}crest_factor":       float(abs_max / rms) if rms > 0 else 0.0,
        f"{prefix}iqr":                float(np.percentile(signal, 75) - np.percentile(signal, 25)),
    }


def _frequency_domain(signal: np.ndarray, sample_rate: float, prefix: str) -> dict:
    n      = len(signal)
    freqs  = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag    = np.abs(np.fft.rfft(signal))
    nyq    = sample_rate / 2.0

    dom_idx       = int(np.argmax(mag[1:]) + 1)
    dominant_freq = float(freqs[dom_idx])

    total_mag   = float(np.sum(mag)) + 1e-12
    centroid_hz = float(np.sum(freqs * mag) / total_mag)
    spread_hz   = float(np.sqrt(np.sum(((freqs - centroid_hz) ** 2) * mag) / total_mag))

    power      = mag ** 2
    power_norm = power / (power.sum() + 1e-12)
    entropy    = float(-np.sum(power_norm * np.log(power_norm + 1e-12)))

    def _abs_band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(power[mask]))

    def _rel_band(lo_frac, hi_frac):
        mask = (freqs >= lo_frac * nyq) & (freqs < hi_frac * nyq)
        return float(np.sum(power[mask]))

    return {
        f"{prefix}dominant_freq_hz":      dominant_freq,
        f"{prefix}dominant_freq_norm":    dominant_freq / nyq,
        f"{prefix}spectral_centroid_hz":  centroid_hz,
        f"{prefix}spectral_centroid_norm":centroid_hz / nyq,
        f"{prefix}spectral_spread_hz":    spread_hz,
        f"{prefix}spectral_entropy":      entropy,
        f"{prefix}band_energy_abs_low":   _abs_band(0,    500),
        f"{prefix}band_energy_abs_mid":   _abs_band(500,  2000),
        f"{prefix}band_energy_abs_high":  _abs_band(2000, nyq),
        f"{prefix}band_energy_rel_low":   _rel_band(0.0,  0.20),
        f"{prefix}band_energy_rel_high":  _rel_band(0.50, 1.00),
    }


def _xc_amplitude(raw1: np.ndarray, raw2: np.ndarray) -> dict:
    """Amplitude ratios — from RAW signals."""
    rms1 = float(np.sqrt(np.mean(raw1 ** 2)))
    rms2 = float(np.sqrt(np.mean(raw2 ** 2)))
    e1   = float(np.mean(raw1 ** 2))
    e2   = float(np.mean(raw2 ** 2))
    return {
        "xc_rms_ratio":    rms1 / rms2 if rms2 > 0 else float("nan"),
        "xc_energy_ratio": e1 / e2     if e2 > 0   else float("nan"),
    }


def _xc_shape(cond1: np.ndarray, cond2: np.ndarray, sample_rate: float) -> dict:
    """Correlation and spectral diffs — from CONDITIONED signals."""
    corr = float(np.corrcoef(cond1, cond2)[0, 1]) if len(cond1) == len(cond2) else float("nan")

    n     = len(cond1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag1  = np.abs(np.fft.rfft(cond1))
    mag2  = np.abs(np.fft.rfft(cond2))

    dom1  = float(freqs[int(np.argmax(mag1[1:]) + 1)])
    dom2  = float(freqs[int(np.argmax(mag2[1:]) + 1)])
    cent1 = float(np.sum(freqs * mag1) / (np.sum(mag1) + 1e-12))
    cent2 = float(np.sum(freqs * mag2) / (np.sum(mag2) + 1e-12))

    return {
        "xc_correlation":      corr,
        "xc_dom_freq_diff_hz": dom1 - dom2,
        "xc_centroid_diff_hz": cent1 - cent2,
    }


# ── NaN placeholder dicts ─────────────────────────────────────────────────────

_AMP_KEYS   = ["mean","std","rms","variance","peak_to_peak","energy"]
_SHAPE_KEYS = ["skewness","kurtosis","zero_crossing_rate","num_peaks","crest_factor","iqr"]
_FREQ_KEYS  = ["dominant_freq_hz","dominant_freq_norm","spectral_centroid_hz",
               "spectral_centroid_norm","spectral_spread_hz","spectral_entropy",
               "band_energy_abs_low","band_energy_abs_mid","band_energy_abs_high",
               "band_energy_rel_low","band_energy_rel_high"]

def _nan_td(prefix):
    return {f"{prefix}{k}": float("nan") for k in _AMP_KEYS + _SHAPE_KEYS}

def _nan_freq(prefix):
    return {f"{prefix}{k}": float("nan") for k in _FREQ_KEYS}

def _nan_cross():
    return {k: float("nan") for k in
            ["xc_rms_ratio","xc_energy_ratio","xc_correlation",
             "xc_dom_freq_diff_hz","xc_centroid_diff_hz"]}


# ── Per-signal extraction ─────────────────────────────────────────────────────

META_COLS = [
    "file_path","filename","device_family","device_id",
    "state","distance_label","probe_id","channel_mode","sample_rate_hz",
]


def _add_channel_features(row: dict, raw: np.ndarray | None, cond: np.ndarray | None,
                          sr: float, prefix_td: str, prefix_fd: str) -> None:
    """Populate time- and frequency-domain features for one channel, in place."""
    if raw is not None and len(raw) > 0:
        row.update(_td_amplitude(raw, prefix_td))
    else:
        row.update({f"{prefix_td}{k}": float("nan") for k in _AMP_KEYS})

    if cond is not None and len(cond) > 0:
        row.update(_td_shape(cond, prefix_td))
        row.update(_frequency_domain(cond, sr, prefix_fd))
    else:
        row.update({f"{prefix_td}{k}": float("nan") for k in _SHAPE_KEYS})
        row.update(_nan_freq(prefix_fd))


def extract_features(raw_sig: dict, cond_sig: dict) -> dict:
    """
    Extract all features from one (raw, conditioned) signal pair.
    Amplitude features come from raw_sig; shape and spectral from cond_sig.
    Returns a flat dict of metadata + features (one row of features.csv).
    """
    row = {col: cond_sig.get(col) for col in META_COLS}

    row["label_device"] = cond_sig.get("device_id", "unknown")
    row["label_family"] = cond_sig.get("device_family", "unknown")
    row["label_state"]  = cond_sig.get("state", "unknown")
    row["label_coarse"] = f"{cond_sig.get('device_family','?')}_{cond_sig.get('state','?')}"

    sr = cond_sig.get("sample_rate_hz")

    raw_ch1,  raw_ch2  = raw_sig.get("ch1"),  raw_sig.get("ch2")
    cond_ch1, cond_ch2 = cond_sig.get("ch1"), cond_sig.get("ch2")

    has_ch1 = cond_ch1 is not None and len(cond_ch1) > 0
    has_ch2 = cond_ch2 is not None and len(cond_ch2) > 0

    _add_channel_features(row, raw_ch1, cond_ch1, sr, "td_ch1_", "fd_ch1_")
    _add_channel_features(row, raw_ch2, cond_ch2, sr, "td_ch2_", "fd_ch2_")

    # Cross-channel features
    if has_ch1 and has_ch2:
        row.update(_xc_amplitude(raw_ch1, raw_ch2))
        row.update(_xc_shape(cond_ch1, cond_ch2, sr))
    else:
        row.update(_nan_cross())

    # Primary channel (whichever has data) — used for cross-device analysis
    if has_ch1:
        raw_pri, cond_pri, primary_prefix = raw_ch1, cond_ch1, "ch1"
    elif has_ch2:
        raw_pri, cond_pri, primary_prefix = raw_ch2, cond_ch2, "ch2"
    else:
        raw_pri, cond_pri, primary_prefix = None, None, None

    row["primary_channel"] = primary_prefix
    _add_channel_features(row, raw_pri, cond_pri, sr, "td_pri_", "fd_pri_")

    return row


def build_feature_table(raw_signals: list[dict], cond_signals: list[dict]) -> pd.DataFrame:
    if len(raw_signals) != len(cond_signals):
        raise ValueError(f"raw ({len(raw_signals)}) and conditioned ({len(cond_signals)}) "
                         "signal counts differ — they must be in matching order.")
    rows = [extract_features(r, c) for r, c in zip(raw_signals, cond_signals)]
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(save_csv: bool = True) -> pd.DataFrame:
    print("Loading dataset...")
    signals, report = load_dataset(skip_duplicates=True)
    print(f"  {report['loaded']} signals loaded.")

    print("Conditioning signals...")
    conditioned = condition_dataset(signals)

    print("Extracting features (amplitude from raw, shape/spectral from conditioned)...")
    df = build_feature_table(signals, conditioned)
    print(f"  Feature table: {df.shape[0]} rows × {df.shape[1]} columns")

    if save_csv:
        df.to_csv(OUT_CSV, index=False)
        print(f"  Saved → {OUT_CSV}")

    return df


if __name__ == "__main__":
    df = run()
    feat_cols = [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]
    print(f"\nPrimary-channel features ({len(feat_cols)}):")
    print(df[feat_cols].describe().round(4).to_string())

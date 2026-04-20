"""
Phase 3 — Conditioning Comparison Plots

Generates four figures that justify the conditioning choices:
  fig_p3_01_raw_vs_filtered.png      — time-domain before/after low-pass filter
  fig_p3_02_fft_before_after.png     — FFT magnitude before/after filter
  fig_p3_03_normalization.png        — amplitude distribution before/after z-score
  fig_p3_04_cross_device.png         — conditioned signals across all device types

Run from the project root:
    python Phase_3_Signal_Processing/compare_conditioning.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_2_Ingestion_Pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_dataset
from processor import condition_signal, condition_dataset, lowpass_filter, normalize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_fft(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag   = np.abs(np.fft.rfft(signal))
    return freqs, mag


def pick_one(signals, device_id, channel="ch1"):
    """Return first signal for a device that has the requested channel."""
    for s in signals:
        if s["device_id"] == device_id and s.get(channel) is not None and len(s[channel]) > 0:
            return s
    return None


def main():
    print("Loading dataset...")
    raw_signals, report = load_dataset(skip_duplicates=True)
    print(f"  {report['loaded']} signals loaded.")

    print("Conditioning dataset...")
    cond_signals = condition_dataset(raw_signals)

    # ── Fig 1: Raw vs Filtered waveform ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Fig P3-01 — Raw vs Low-Pass Filtered Waveform", fontsize=13)

    examples = [
        (pick_one(raw_signals, "motor_1",   "ch2"), pick_one(cond_signals, "motor_1",   "ch2"), "ch2", "Motor 1 (20 kHz)"),
        (pick_one(raw_signals, "motor_3",   "ch1"), pick_one(cond_signals, "motor_3",   "ch1"), "ch1", "Motor 3 — CH1 distant (20 kHz)"),
        (pick_one(raw_signals, "charger_1", "ch1"), pick_one(cond_signals, "charger_1", "ch1"), "ch1", "Charger 1 OFF (5 kHz)"),
        (pick_one(raw_signals, "charger_2", "ch1"), pick_one(cond_signals, "charger_2", "ch1"), "ch1", "Charger 2 ON distant (5 kHz)"),
    ]

    for ax, (raw, cond, ch, title) in zip(axes.flat, examples):
        if raw is None or cond is None:
            ax.set_visible(False)
            continue
        raw_ch  = raw[ch]
        cond_ch = cond[ch]
        t_raw   = raw["time"]
        t_cond  = cond["time"]
        ax.plot(t_raw  * 1e3, raw_ch,  linewidth=0.7, alpha=0.7, label="Raw")
        ax.plot(t_cond * 1e3, cond_ch, linewidth=0.9, label=f"Filtered + z-scored")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p3_01_raw_vs_filtered.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 2: FFT before and after filtering ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Fig P3-02 — FFT Magnitude: Raw vs Filtered", fontsize=13)

    fft_examples = [
        ("motor_1",   "ch2", "Motor 1 (20 kHz / Nyquist 10 kHz)"),
        ("motor_3",   "ch1", "Motor 3 CH1 (20 kHz / Nyquist 10 kHz)"),
        ("charger_1", "ch1", "Charger 1 (5 kHz / Nyquist 2.5 kHz)"),
        ("charger_2", "ch1", "Charger 2 (5 kHz / Nyquist 2.5 kHz)"),
    ]

    for ax, (dev, ch, title) in zip(axes.flat, fft_examples):
        raw  = pick_one(raw_signals, dev, ch)
        cond = pick_one(cond_signals, dev, ch)
        if raw is None or cond is None:
            ax.set_visible(False)
            continue

        raw_ch   = raw[ch]
        cond_ch  = cond[ch]
        sr_raw   = raw["sample_rate_hz"]
        sr_cond  = cond["sample_rate_hz"]

        f_raw,  m_raw  = compute_fft(raw_ch,  sr_raw)
        f_cond, m_cond = compute_fft(cond_ch, sr_cond)

        cutoff = cond["conditioning_params"]["filter_cutoff_hz"]

        ax.plot(f_raw,  m_raw,  linewidth=0.7, alpha=0.7, label="Raw FFT")
        ax.plot(f_cond, m_cond, linewidth=0.9, label="Filtered FFT")
        ax.axvline(cutoff, color="red", linestyle="--", linewidth=1,
                   label=f"Cutoff {cutoff:.0f} Hz")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, sr_raw / 2)
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p3_02_fft_before_after.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 3: Amplitude distribution before/after normalization ─────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle("Fig P3-03 — Amplitude Distribution: Raw vs Z-Scored", fontsize=13)

    dist_examples = [
        ("motor_1",   "ch2"),
        ("motor_3",   "ch1"),
        ("charger_1", "ch1"),
        ("charger_2", "ch1"),
    ]

    for col, (dev, ch) in enumerate(dist_examples):
        raw  = pick_one(raw_signals,  dev, ch)
        cond = pick_one(cond_signals, dev, ch)
        if raw is None or cond is None:
            continue
        label = f"{dev} ({ch})"
        axes[0, col].hist(raw[ch],  bins=40, color="steelblue", edgecolor="none", alpha=0.8)
        axes[0, col].set_title(f"Raw — {label}", fontsize=8)
        axes[0, col].set_xlabel("Voltage (V)")
        axes[0, col].grid(True, linewidth=0.3)

        axes[1, col].hist(cond[ch], bins=40, color="darkorange", edgecolor="none", alpha=0.8)
        axes[1, col].set_title(f"Z-Scored — {label}", fontsize=8)
        axes[1, col].set_xlabel("Std devs")
        axes[1, col].grid(True, linewidth=0.3)

    for ax in axes.flat:
        ax.set_ylabel("Count", fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p3_03_normalization.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 4: Cross-device conditioned signals ───────────────────────────────
    device_channel_map = [
        ("motor_1",   "ch2", "Motor 1 ON"),
        ("motor_2",   "ch1", "Motor 2 ON CH1-far"),
        ("motor_2",   "ch2", "Motor 2 ON CH2-close"),
        ("motor_3",   "ch1", "Motor 3 ON CH1-distant"),
        ("charger_1", "ch1", "Charger 1 OFF"),
        ("charger_1", "ch1", "Charger 1 ON close"),
        ("charger_2", "ch1", "Charger 2 OFF"),
        ("charger_2", "ch1", "Charger 2 ON distant"),
    ]

    # For charger_1, pick one OFF and one ON explicitly
    charger1_off = next((s for s in cond_signals
                         if s["device_id"] == "charger_1" and s["state"] == "off"), None)
    charger1_on  = next((s for s in cond_signals
                         if s["device_id"] == "charger_1" and s["state"] == "on"), None)
    charger2_off = next((s for s in cond_signals
                         if s["device_id"] == "charger_2" and s["state"] == "off"), None)
    charger2_on  = next((s for s in cond_signals
                         if s["device_id"] == "charger_2" and s["state"] == "on"), None)

    specific = [
        (pick_one(cond_signals, "motor_1", "ch2"), "ch2", "Motor 1 ON (CH2)"),
        (pick_one(cond_signals, "motor_2", "ch1"), "ch1", "Motor 2 ON CH1-far"),
        (pick_one(cond_signals, "motor_2", "ch2"), "ch2", "Motor 2 ON CH2-close"),
        (pick_one(cond_signals, "motor_3", "ch1"), "ch1", "Motor 3 ON CH1-distant"),
        (charger1_off, "ch1", "Charger 1 OFF"),
        (charger1_on,  "ch1", "Charger 1 ON close"),
        (charger2_off, "ch1", "Charger 2 OFF"),
        (charger2_on,  "ch1", "Charger 2 ON distant"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("Fig P3-04 — Conditioned Signals Across All Device Types", fontsize=13)

    for ax, (sig, ch, label) in zip(axes.flat, specific):
        if sig is None or sig.get(ch) is None:
            ax.set_visible(False)
            continue
        t = sig["time"] * 1e3
        ax.plot(t, sig[ch], linewidth=0.7)
        sr = sig["sample_rate_hz"]
        ax.set_title(f"{label}  |  {sr:.0f} Hz", fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (z-scored)")
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p3_04_cross_device.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n  Conditioning summary:")
    for rate in [5000.0, 20000.0]:
        subset = [s for s in cond_signals if s["conditioning_params"]["original_rate_hz"] == rate]
        if not subset:
            continue
        cutoff = subset[0]["conditioning_params"]["filter_cutoff_hz"]
        print(f"    {rate:.0f} Hz signals ({len(subset)}): "
              f"LP cutoff = {cutoff:.0f} Hz  ({cutoff / (rate/2) * 100:.0f}% of Nyquist)")

    print("\nDone. All figures saved to Phase_3_Signal_Processing/")


if __name__ == "__main__":
    main()

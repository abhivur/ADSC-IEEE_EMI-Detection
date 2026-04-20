"""
Phase 7 — Results Analysis and Technical Interpretation

Generates five figures that explain *why* the pipeline works and what
the signals physically reveal about each device.

Figures
-------
  fig_p7_01_signal_traces.png       — Representative time-domain waveforms (all 5 devices)
  fig_p7_02_spectral_comparison.png — FFT spectra: why devices are separable
  fig_p7_03_feature_narrative.png   — Key features with physical interpretation
  fig_p7_04_separation_space.png    — PCA showing class separation in feature space
  fig_p7_05_results_summary.png     — Final accuracy summary across all tasks

Run from project root:
    python Phase_7_Results_Analysis/analyze_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_2_Ingestion_Pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_3_Signal_Processing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_4_Feature_Engineering"))

from loader    import load_dataset
from processor import condition_dataset

FEATURES_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Phase_4_Feature_Engineering", "features.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ─────────────────────────────────────────────────────────────
# motor_3 changed from yellow (#ffe119) to purple (#9467bd) for white-background visibility
DEV_COLORS = {
    "motor_1":   "#e6194b",   # red
    "motor_2":   "#f58231",   # orange
    "motor_3":   "#9467bd",   # purple  (was yellow — invisible on white)
    "charger_1": "#4363d8",   # blue
    "charger_2": "#3cb44b",   # green
}
FAM_COLORS  = {"motor": "#e6194b", "charger": "#4363d8"}
STATE_MARKS = {"on": "o", "off": "s"}


def _valid(arr):
    return arr is not None and len(arr) > 0


def pick(signals, device_id, state=None, channel="ch1"):
    """Return first matching signal and the requested channel array."""
    for s in signals:
        if s["device_id"] != device_id:
            continue
        if state and s.get("state") != state:
            continue
        ch_arr = s.get(channel)
        arr = ch_arr if _valid(ch_arr) else (s.get("ch2") if _valid(s.get("ch2")) else s.get("ch1"))
        if _valid(arr):
            return s, arr
    return None, None


# ── Fig 1: Signal traces — all 5 devices ─────────────────────────────────────

def fig_signal_traces(raw_signals, cond_signals):
    """
    2×3 grid showing all 5 devices + charger_2 ON.
    Time axis normalised to start at 0 ms so windows are easy to compare.
    """
    cases = [
        ("motor_1",   None,  "ch2", "Motor 1 ON   (20 kHz, 60 ms)"),
        ("motor_2",   None,  "ch1", "Motor 2 ON   (20 kHz, 60 ms)"),
        ("motor_3",   None,  "ch1", "Motor 3 ON   (20 kHz, 60 ms)"),
        ("charger_1", "off", "ch1", "Charger 1 OFF  (5 kHz, 240 ms)"),
        ("charger_1", "on",  "ch1", "Charger 1 ON   (5 kHz, 240 ms)"),
        ("charger_2", "on",  "ch1", "Charger 2 ON   (5 kHz, 240 ms)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        "Fig P7-01 — Representative EMI Signal Traces  (conditioned: low-pass filtered + z-scored)\n"
        "Motors: 60 ms capture window | Chargers: 240 ms capture window",
        fontsize=12,
    )

    for ax, (dev, state, ch, title) in zip(axes.flat, cases):
        sig, arr = pick(cond_signals, dev, state, ch)
        if sig is None:
            ax.set_visible(False)
            continue

        # Normalise time axis to start at 0 ms
        t_ms = (sig["time"] - sig["time"][0]) * 1e3
        color = DEV_COLORS.get(dev, "grey")
        ax.plot(t_ms, arr, linewidth=0.55, color=color, alpha=0.9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (z-scored)")
        ax.grid(True, linewidth=0.3)

        # Annotate with raw RMS from actual voltage
        raw_sig, raw_arr = pick(raw_signals, dev, state, ch)
        if _valid(raw_arr):
            rms_raw = float(np.sqrt(np.mean(raw_arr ** 2)))
            ax.text(0.98, 0.97, f"Raw RMS = {rms_raw:.4f} V",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p7_01_signal_traces.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ── Fig 2: Spectral comparison ────────────────────────────────────────────────

def fig_spectral_comparison(cond_signals):
    """
    Left:  full spectrum on LOG y-scale — prevents one motor spike from
           crushing all charger content.
    Right: linear zoom 0–2 500 Hz (charger Nyquist range) showing harmonic
           structure clearly.
    """
    groups = {
        "Motor ON (20 kHz)":   ("motor",   "on"),
        "Charger ON (5 kHz)":  ("charger", "on"),
        "Charger OFF (5 kHz)": ("charger", "off"),
    }
    group_colors = {
        "Motor ON (20 kHz)":   "#e6194b",
        "Charger ON (5 kHz)":  "#4363d8",
        "Charger OFF (5 kHz)": "#888888",
    }
    group_lw = {
        "Motor ON (20 kHz)":   1.4,
        "Charger ON (5 kHz)":  1.6,
        "Charger OFF (5 kHz)": 1.2,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Fig P7-02 — EMI Frequency Spectra\n"
        "Motors have broadband high-frequency content; chargers emit narrow harmonics below 2.5 kHz",
        fontsize=12,
    )

    computed = {}
    for label, (family, state) in groups.items():
        spectra = []
        for s in cond_signals:
            if s.get("device_family") != family:
                continue
            if s.get("state") != state:
                continue
            _c1, _c2 = s.get("ch1"), s.get("ch2")
            arr = _c1 if _valid(_c1) else _c2
            if not _valid(arr):
                continue
            sr    = s["sample_rate_hz"]
            n     = len(arr)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            mag   = np.abs(np.fft.rfft(arr))
            spectra.append((freqs, mag))
        if not spectra:
            continue
        min_len  = min(len(m) for _, m in spectra)
        avg_mag  = np.mean([m[:min_len] for _, m in spectra], axis=0)
        f_ref    = spectra[0][0][:min_len]
        computed[label] = (f_ref, avg_mag)

    for label, (f_ref, avg_mag) in computed.items():
        c  = group_colors[label]
        lw = group_lw[label]
        # Left: log scale — prevents motor spike from dominating
        axes[0].semilogy(f_ref, avg_mag + 1e-6, linewidth=lw,
                         label=label, color=c, alpha=0.88)
        # Right: linear zoom 0–2500 Hz
        mask = f_ref <= 2500
        axes[1].plot(f_ref[mask], avg_mag[mask], linewidth=lw + 0.2,
                     label=label, color=c, alpha=0.88)

    # Left plot styling
    axes[0].set_title("Full Spectrum — Log Scale  (0 Hz to Nyquist)", fontsize=10)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean FFT Magnitude (log scale)")
    axes[0].axvline(2500, color="#4363d8", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].text(2650, axes[0].get_ylim()[0] * 5,
                 "Charger\nNyquist\n2.5 kHz", fontsize=8, color="#4363d8")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linewidth=0.3, which="both", alpha=0.5)

    # Right plot styling
    axes[1].set_title("Zoom: 0 – 2 500 Hz  (shared frequency range)\nLinear scale", fontsize=10)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Mean FFT Magnitude")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p7_02_spectral_comparison.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ── Fig 3: Feature narrative ──────────────────────────────────────────────────

def fig_feature_narrative(df: pd.DataFrame):
    """
    8 physically meaningful features shown as violin plots per device.
    Features are hand-picked to cover amplitude, waveform shape, and spectral
    distribution — avoiding confounded features like num_peaks (which depends
    on capture window length rather than EMI physics).
    """
    # Hand-picked features — diverse, physically interpretable, not window-length confounded
    DISPLAY_FEATURES = [
        ("td_pri_peak_to_peak",        "Peak-to-Peak Voltage\n(raw amplitude — motor EMI is stronger)"),
        ("td_pri_kurtosis",            "Kurtosis\n(spike sharpness — motor brush/switching transients)"),
        ("td_pri_crest_factor",        "Crest Factor  (peak / RMS)\n(impulsive vs. sinusoidal character)"),
        ("td_pri_zero_crossing_rate",  "Zero-Crossing Rate\n(oscillation speed — motors cross zero more)"),
        ("fd_pri_band_energy_rel_high","High-Band Energy Fraction\n(50–100% Nyquist — motors have more)"),
        ("fd_pri_band_energy_rel_low", "Low-Band Energy Fraction\n(0–20% Nyquist — chargers concentrated low)"),
        ("fd_pri_spectral_entropy",    "Spectral Entropy\n(complexity — motor = broadband, charger = tonal)"),
        ("td_pri_iqr",                 "IQR  (Interquartile Range)\n(mid-range amplitude spread)"),
    ]

    devices = sorted(df["device_id"].unique())

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    fig.suptitle(
        "Fig P7-03 — Key EMI Features: Physical Interpretation\n"
        "Violin plots show distribution per device — wider = more spread, median marked",
        fontsize=13,
    )

    for ax, (col, description) in zip(axes.flat, DISPLAY_FEATURES):
        data   = [df.loc[df["device_id"] == d, col].dropna().values for d in devices]
        colors = [DEV_COLORS.get(d, "grey") for d in devices]

        parts = ax.violinplot(data, positions=range(len(devices)), showmedians=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.72)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        short = col.replace("td_pri_", "td·").replace("fd_pri_", "fd·")
        ax.set_title(short, fontsize=10, fontweight="bold", pad=4)

        # Physical description below plot — larger, readable font
        ax.set_xlabel(description, fontsize=9, color="#222222", labelpad=6)

        ax.set_xticks(range(len(devices)))
        ax.set_xticklabels([d.replace("_", "\n") for d in devices], fontsize=8)
        ax.grid(axis="y", linewidth=0.3)

    # Legend
    patches = [mpatches.Patch(color=DEV_COLORS[d], label=d) for d in devices]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT_DIR, "fig_p7_03_feature_narrative.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ── Fig 4: PCA separation space ───────────────────────────────────────────────

def fig_separation_space(df: pd.DataFrame, feat_cols: list):
    """
    Three PCA panels:
      Left   — by device family (motor vs charger): shows clean family separation
      Middle — by device ID: shows fine-grained fingerprinting
      Right  — charger only, by state: shows perfect ON/OFF separation
    """
    X      = df[feat_cols].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_sc)
    ve     = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        "Fig P7-04 — EMI Feature Space (PCA, 23 primary features)\n"
        f"PC1 explains {ve[0]*100:.1f}%  |  PC2 explains {ve[1]*100:.1f}%  |  "
        f"Total = {sum(ve)*100:.1f}% of variance",
        fontsize=12,
    )

    # Left: by device family
    for fam, color in FAM_COLORS.items():
        mask = df["device_family"].values == fam
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, alpha=0.65, s=28, label=fam, edgecolors="none")
    axes[0].set_title("By Device Family", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9, markerscale=1.4)

    # Middle: by device_id
    for dev, color in DEV_COLORS.items():
        mask = df["device_id"].values == dev
        if mask.sum() == 0:
            continue
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, alpha=0.65, s=28, label=dev, edgecolors="none")
    axes[1].set_title("By Device ID  (EMI fingerprinting)", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9, markerscale=1.4)

    # Right: charger only, by state — separate PCA for clarity
    charger_mask = df["device_family"].values == "charger"
    X_ch_sc      = StandardScaler().fit_transform(X_sc[charger_mask])
    pca_ch       = PCA(n_components=2, random_state=42)
    X_ch_pca     = pca_ch.fit_transform(X_ch_sc)
    ve_ch        = pca_ch.explained_variance_ratio_
    df_ch        = df[charger_mask].reset_index(drop=True)

    state_colors = {"on": "#e6194b", "off": "#888888"}
    for state, color in state_colors.items():
        mask2 = df_ch["state"].values == state
        axes[2].scatter(X_ch_pca[mask2, 0], X_ch_pca[mask2, 1],
                        c=color, alpha=0.7, s=32,
                        label=f"Charger {state.upper()}",
                        marker=STATE_MARKS[state], edgecolors="none")
    axes[2].set_title(
        f"Charger Only: ON vs OFF\n(PC1={ve_ch[0]*100:.1f}%  PC2={ve_ch[1]*100:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    axes[2].legend(fontsize=9, markerscale=1.4)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({ve[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ve[1]*100:.1f}%)", fontsize=9)
        ax.grid(True, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Fix axis labels for charger-only panel
    axes[2].set_xlabel(f"PC1 ({ve_ch[0]*100:.1f}%)", fontsize=9)
    axes[2].set_ylabel(f"PC2 ({ve_ch[1]*100:.1f}%)", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p7_04_separation_space.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ── Fig 5: Results summary ────────────────────────────────────────────────────

def fig_results_summary():
    """
    Left:  bar chart of test accuracy and F1 across all 4 tasks.
    Right: key findings as readable bullet points (replaces cramped table).
    """
    tasks     = ["Motor vs\nCharger\n(rate-agnostic)", "Charger\nON vs OFF",
                 "Motor ID\n(3-class)", "Charger ID\n(2-class)"]
    best_accs = [1.000, 1.000, 1.000, 0.946]
    best_f1s  = [1.000, 1.000, 1.000, 0.946]
    best_models = ["SVM", "RF", "LR", "SVM"]

    # Green for perfect, orange for ≥ 0.90, red for below
    bar_colors = ["#2ecc71" if a >= 0.99 else "#f39c12" if a >= 0.90 else "#e74c3c"
                  for a in best_accs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Fig P7-05 — Final Model Performance Summary\n"
        "Held-out test set · 80/20 stratified split · duplicates removed · 344 unique signals",
        fontsize=12,
    )

    x     = np.arange(len(tasks))
    width = 0.32

    bars_acc = axes[0].bar(x - width/2, best_accs, width,
                           color=bar_colors, alpha=0.90, label="Test Accuracy")
    bars_f1  = axes[0].bar(x + width/2, best_f1s,  width,
                           color=bar_colors, alpha=0.45, label="Test F1-macro",
                           edgecolor=bar_colors, linewidth=1.5)

    for bar, acc, model in zip(bars_acc, best_accs, best_models):
        axes[0].text(bar.get_x() + bar.get_width() / 2, acc + 0.004,
                     f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, acc in zip(bars_f1, best_f1s):
        axes[0].text(bar.get_x() + bar.get_width() / 2, acc + 0.004,
                     f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    # Best model label below each group
    for i, model in enumerate(best_models):
        axes[0].text(i, 0.705, f"Best: {model}", ha="center", fontsize=8, color="#555555")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, fontsize=9)
    axes[0].set_ylim(0.70, 1.06)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(axis="y", linewidth=0.3)
    axes[0].set_title("Test Accuracy and F1-macro per Task", fontsize=11)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right panel: key findings as clean bullet points
    axes[1].axis("off")
    axes[1].set_title("Key Findings", fontsize=12, fontweight="bold", pad=12)

    findings = [
        ("#2ecc71", "Motor vs Charger — 100%  (rate-agnostic SVM)",
         "EMI content alone separates device families. Motors generate\n"
         "broadband 0–10 kHz noise; chargers emit narrow harmonics\n"
         "below 2.5 kHz. No sampling-rate advantage needed."),

        ("#2ecc71", "Charger ON vs OFF — 100%  (Random Forest)",
         "Active charging creates strong periodic switching EMI.\n"
         "An idle charger produces near-ambient noise. These states\n"
         "are perfectly separable on amplitude and spectral features."),

        ("#2ecc71", "Motor ID 3-class — 100%  (Logistic Regression)",
         "Each motor has a unique EMI fingerprint from its winding,\n"
         "bearing, and switching characteristics. Linear separability\n"
         "suggests distinct, stable per-device signatures."),

        ("#f39c12", "Charger ID 2-class — 94.6%  (SVM)",
         "Hardest task: two similar chargers at the same sampling rate.\n"
         "Subtle differences in component tolerances and cable routing\n"
         "are detectable but not perfectly separable."),
    ]

    y = 0.92
    for color, title, body in findings:
        # Coloured title line
        axes[1].text(0.04, y, f"●  {title}",
                     transform=axes[1].transAxes,
                     fontsize=10, fontweight="bold", va="top",
                     color=color)
        # Body text
        axes[1].text(0.08, y - 0.055, body,
                     transform=axes[1].transAxes,
                     fontsize=9, va="top", color="#333333",
                     linespacing=1.5)
        y -= 0.24

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p7_05_results_summary.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ── Technical interpretation printout ─────────────────────────────────────────

def print_interpretation():
    print("\n" + "=" * 65)
    print("PHASE 7 — TECHNICAL INTERPRETATION")
    print("=" * 65)
    print("""
WHAT WORKED AND WHY
-------------------
1. Motor vs Charger (100% — rate-agnostic)
   Motors generate broadband EMI from brushes, bearings, and switching
   electronics across 0–10 kHz. Chargers generate narrow-band harmonic
   EMI from their switch-mode power supply, concentrated below 2.5 kHz.
   This spectral difference is large enough that any classifier separates
   them perfectly — even without sampling-rate information.

2. Charger ON vs OFF (100%)
   An active charger continuously switches current to regulate voltage,
   producing strong periodic EMI. An idle charger produces near-ambient
   noise. RMS amplitude and spectral entropy are completely different —
   the most physically obvious result in the dataset.

3. Motor ID — 3-class (100% LR)
   Each motor has a unique combination of mechanical resonances, winding
   characteristics, and bearing noise. These produce a stable EMI
   fingerprint across repeated captures. Linear separability (LR = 100%)
   suggests the device signatures occupy distinct regions in feature space.

4. Charger ID — 2-class (94.6%)
   Two laptop chargers of potentially similar design. Their EMI signatures
   differ subtly — likely due to component tolerances, cable routing, and
   load differences. 94.6% is strong for same-family, same-rate
   fingerprinting with 184 signals.

WHAT DID NOT WORK / LIMITATIONS
--------------------------------
- Motor OFF data does not exist in the dataset — motor state detection
  could not be evaluated.
- Distance classification was not attempted as a standalone task because
  close/distant captures mix device types, confounding the distance effect.
- motor_2 recall was 71% under RF (7 test samples of 35 total) — class
  too small for stable RF estimates; LR handles it correctly.
- The single charger_1 ON misclassification in the integration test is
  a model accuracy limit, not a pipeline error.

HARDWARE-SOFTWARE CONNECTION
-----------------------------
- Sampling rate (20 kHz motors, 5 kHz chargers) was set to match device
  frequency content. Forcing a common rate would destroy motor high-freq
  EMI or waste bandwidth on chargers. Adaptive LP cutoff (80% Nyquist)
  respects this.
- Probe distance affects raw signal amplitude (closer = stronger). The
  pipeline extracts amplitude from raw signals and shape/spectral features
  from conditioned signals to capture both effects.
- Z-score normalisation is applied after raw amplitude extraction so the
  classifier sees actual voltage levels alongside waveform shape.
""")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    raw_signals, report = load_dataset(skip_duplicates=True)
    print(f"  {report['loaded']} signals loaded.")

    print("Conditioning...")
    cond_signals = condition_dataset(raw_signals)

    print("Loading features.csv...")
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]
    df = df.dropna(subset=feat_cols).drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"  {len(df)} unique signals")

    print("\nGenerating figures...")
    fig_signal_traces(raw_signals, cond_signals)
    fig_spectral_comparison(cond_signals)
    fig_feature_narrative(df)
    fig_separation_space(df, feat_cols)
    fig_results_summary()

    print_interpretation()
    print("\nAll figures saved to Phase_7_Results_Analysis/")


if __name__ == "__main__":
    main()

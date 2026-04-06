# =============================================================================
# Group B - Exploration: MOTOR_1_ON
# Tasks: FFT from time-domain, basic features (RMS, variance, peaks),
#        first simple classifier
#
# Required pip installs (run once):
#   pip install numpy pandas matplotlib scipy scikit-learn
# =============================================================================



import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =============================================================================
# 1. PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#file paths
MOTOR_1_ON_DIR = os.path.join(BASE_DIR, "Motor_Data", "MOTOR_1_ON", "time domain")

# Used as the "OFF/background" class for the classifier since dedicated
# Charger OFF captures are not yet available. Same 1200-sample length.
CHRGR_OFF_DIR = os.path.join(BASE_DIR, "Laptop_1", "LAPTOP_CHRGR_1_OFF", "time domain")

# =============================================================================
# 2. DATA LOADER
# =============================================================================

def load_oscilloscope_csv(filepath):
    """
    Reads a CSV with two header rows.

    Row 0: channel label + metadata column names  (e.g. X, CH2, Start, Increment)
    Row 1: units + numeric metadata               (e.g. Sequence, Volt, -0.03, 5e-05)
    Rows 2+: sample index, voltage value

    Returns
    -------
    signal : np.ndarray  – voltage samples
    sample_rate : float  – samples per second derived from Increment
    """

    with open(filepath, "r") as f:
        lines = f.readlines()


    # Row 1 contains Start and Increment in columns 2 and 3
    meta = lines[1].strip().split(",")
    increment = float(meta[3])
    sample_rate = 1.0 / increment

    # Data starts at row 2; only the second column (voltage) is needed
    data_lines = lines[2:]
    signal = []
    for line in data_lines:
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[1] != "":
            try:
                signal.append(float(parts[1]))
            except ValueError:
                pass

    return np.array(signal), sample_rate


def load_all_files(directory, label):
    """Load every CSV in *directory*, return list of (signal, sample_rate, label)."""
    pattern = os.path.join(directory, "*.csv")
    files = sorted(glob.glob(pattern))
    records = []
    for fp in files:
        sig, sr = load_oscilloscope_csv(fp)
        records.append({"signal": sig, "sample_rate": sr, "label": label,
                         "filename": os.path.basename(fp)})
    return records


# =============================================================================
# 3. LOAD MOTOR_1_ON DATA
# =============================================================================

print("Loading MOTOR_1_ON time-domain files...")
motor_records = load_all_files(MOTOR_1_ON_DIR, label="MOTOR_1_ON")
print(f"  Loaded {len(motor_records)} files")

# Quick sanity check on one file
ex = motor_records[0]
print(f"  Example -> samples: {len(ex['signal'])}, sample_rate: {ex['sample_rate']:.0f} Hz")

# =============================================================================
# 4. VISUALISE RAW WAVEFORMS  (first 5 captures)
# =============================================================================

fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=False)
fig.suptitle("MOTOR_1_ON — Raw Waveforms (first 5 captures)", fontsize=13)

for i, ax in enumerate(axes):
    rec = motor_records[i]
    sr = rec["sample_rate"]
    t = np.arange(len(rec["signal"])) / sr
    ax.plot(t, rec["signal"], linewidth=0.8)
    ax.set_ylabel("Volt")
    ax.set_title(rec["filename"], fontsize=9)
    ax.grid(True, linewidth=0.4)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_01_raw_waveforms.png"), dpi=120)
plt.show()

# =============================================================================
# 5. FFT — SINGLE CAPTURE (annotated)
# =============================================================================

def compute_fft(signal, sample_rate):
    """Return one-sided frequency array and magnitude spectrum."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag = np.abs(np.fft.rfft(signal))
    return freqs, mag


rec0 = motor_records[0]
freqs0, mag0 = compute_fft(rec0["signal"], rec0["sample_rate"])

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle("MOTOR_1_ON — Capture 1: Time Domain vs FFT", fontsize=13)

t0 = np.arange(len(rec0["signal"])) / rec0["sample_rate"]
axes[0].plot(t0, rec0["signal"], linewidth=0.8)
axes[0].set_title("Time Domain")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude (V)")
axes[0].grid(True, linewidth=0.4)

axes[1].plot(freqs0, mag0, linewidth=0.8)
axes[1].set_title("FFT Magnitude Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_xlim(0, rec0["sample_rate"] / 2)
axes[1].grid(True, linewidth=0.4)

# Mark dominant frequency
dom_idx = np.argmax(mag0[1:]) + 1
axes[1].axvline(freqs0[dom_idx], color="red", linestyle="--", linewidth=1,
                label=f"Dominant: {freqs0[dom_idx]:.1f} Hz")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_02_fft_single.png"), dpi=120)
plt.show()

# =============================================================================
# 6. FFT — OVERLAY OF 5 CAPTURES
# =============================================================================

plt.figure(figsize=(14, 5))
for rec in motor_records[:5]:
    freqs, mag = compute_fft(rec["signal"], rec["sample_rate"])
    plt.plot(freqs, mag, alpha=0.7, linewidth=0.8, label=rec["filename"])

plt.title("MOTOR_1_ON — FFT Overlay (5 captures)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
nyq = motor_records[0]["sample_rate"] / 2
plt.xlim(0, nyq)
plt.legend(fontsize=7)
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_03_fft_overlay.png"), dpi=120)
plt.show()

# =============================================================================
# 7. BASIC FEATURE EXTRACTION
# =============================================================================

def extract_features(signal, sample_rate):
    """
    Returns a dict of handcrafted time-domain and frequency-domain features.

    Time domain
    -----------
    rms          : root-mean-square amplitude  (energy proxy)
    variance     : spread of voltage values
    peak_to_peak : max - min  (dynamic range)
    num_peaks    : count of local maxima above mean + 1 std
    skewness     : asymmetry of the amplitude distribution
    kurtosis     : tail heaviness

    Frequency domain (from FFT)
    ---------------------------
    dominant_freq      : frequency of the highest magnitude bin
    dominant_mag       : magnitude at that bin
    spectral_centroid  : amplitude-weighted average frequency
    spectral_spread    : amplitude-weighted standard deviation of frequency
    band_energy_low    : energy in 0–1 kHz band
    band_energy_mid    : energy in 1–5 kHz band
    band_energy_high   : energy in 5 kHz – Nyquist band
    """
    # --- time domain ---
    rms = np.sqrt(np.mean(signal ** 2))
    variance = np.var(signal)
    ptp = np.ptp(signal)
    threshold = np.mean(signal) + np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    num_peaks = len(peaks)
    skewness = float(skew(signal))
    kurt = float(kurtosis(signal))

    # --- frequency domain ---
    freqs, mag = compute_fft(signal, sample_rate)
    dom_idx = np.argmax(mag[1:]) + 1
    dominant_freq = freqs[dom_idx]
    dominant_mag = mag[dom_idx]

    total_mag = np.sum(mag) + 1e-12
    spectral_centroid = np.sum(freqs * mag) / total_mag
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag) / total_mag)

    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(mag[mask] ** 2))

    nyq = sample_rate / 2.0
    energy_low  = band_energy(0,    1000)
    energy_mid  = band_energy(1000, 5000)
    energy_high = band_energy(5000, nyq)

    return {
        "rms":             rms,
        "variance":        variance,
        "peak_to_peak":    ptp,
        "num_peaks":       num_peaks,
        "skewness":        skewness,
        "kurtosis":        kurt,
        "dominant_freq":   dominant_freq,
        "dominant_mag":    dominant_mag,
        "spectral_centroid": spectral_centroid,
        "spectral_spread":   spectral_spread,
        "band_energy_low":   energy_low,
        "band_energy_mid":   energy_mid,
        "band_energy_high":  energy_high,
    }


# Build feature table for MOTOR_1_ON
print("\nExtracting features from MOTOR_1_ON...")
motor_features = []
for rec in motor_records:
    feats = extract_features(rec["signal"], rec["sample_rate"])
    feats["label"] = rec["label"]
    motor_features.append(feats)

motor_df = pd.DataFrame(motor_features)
print(motor_df.describe().round(4))

# =============================================================================
# 8. FEATURE DISTRIBUTIONS — MOTOR_1_ON
# =============================================================================

feature_cols = [c for c in motor_df.columns if c != "label"]

fig, axes = plt.subplots(3, 5, figsize=(18, 9))
fig.suptitle("MOTOR_1_ON — Feature Distributions", fontsize=13)

for ax, col in zip(axes.flat, feature_cols):
    ax.hist(motor_df[col], bins=15, edgecolor="black", linewidth=0.5)
    ax.set_title(col, fontsize=8)
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3)

# Hide any extra subplots
for ax in axes.flat[len(feature_cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_04_feature_distributions.png"), dpi=120)
plt.show()

# =============================================================================
# 9. SIMPLE CLASSIFIER
#    Two classes:
#      MOTOR_1_ON      — 75 captures from Motor_Data/MOTOR_1_ON/time domain
#      BACKGROUND_OFF  — 50 captures from Laptop_1/LAPTOP_CHRGR_1_OFF/time domain
#                        (used as baseline "no active EMI source" until dedicated
#                         Charger OFF data is collected)
# =============================================================================

print("\nLoading BACKGROUND (Laptop Charger OFF) files for classifier...")
chrgr_records = load_all_files(CHRGR_OFF_DIR, label="BACKGROUND_OFF")
print(f"  Loaded {len(chrgr_records)} files")

# Extract features from background captures
chrgr_features = []
for rec in chrgr_records:
    feats = extract_features(rec["signal"], rec["sample_rate"])
    feats["label"] = rec["label"]
    chrgr_features.append(feats)

chrgr_df = pd.DataFrame(chrgr_features)

# Combine both classes
combined_df = pd.concat([motor_df, chrgr_df], ignore_index=True)
print(f"\nCombined dataset: {combined_df.shape[0]} samples, "
      f"classes: {combined_df['label'].value_counts().to_dict()}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(combined_df["label"])
X = combined_df[feature_cols].values

print(f"Classes: {list(le.classes_)}")

# Train / test split (stratified so both classes are balanced in each split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# =============================================================================
# 10. TRAIN THREE CLASSIFIERS AND COMPARE
# =============================================================================

classifiers = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

results = []
trained = {}

for name, pipe in classifiers.items():
    # 5-fold cross-validation on the training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    # Final fit on full training set, evaluate on held-out test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    trained[name] = pipe
    results.append({
        "model": name,
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
        "test_acc": test_acc,
    })
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

results_df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
print("\n=== Accuracy Summary ===")
print(results_df.to_string(index=False))

# =============================================================================
# 11. ACCURACY BAR CHART
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results_df))
bars = ax.bar(x, results_df["test_acc"], width=0.5, label="Test accuracy",
              color=["tab:blue", "tab:orange", "tab:green"][:len(results_df)])
ax.errorbar(x, results_df["cv_mean"], yerr=results_df["cv_std"],
            fmt="D", color="black", capsize=5, label="5-fold CV mean ± std")
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], fontsize=10)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Classifier Accuracy: MOTOR_1_ON vs BACKGROUND_OFF")
ax.legend()
ax.grid(axis="y", linewidth=0.4)
for bar, val in zip(bars, results_df["test_acc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_05_classifier_accuracy.png"), dpi=120)
plt.show()

# =============================================================================
# 12. CONFUSION MATRICES
# =============================================================================

fig, axes = plt.subplots(1, len(trained), figsize=(5 * len(trained), 4))
fig.suptitle("Confusion Matrices", fontsize=13)

if len(trained) == 1:
    axes = [axes]

for ax, (name, pipe) in zip(axes, trained.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_06_confusion_matrices.png"), dpi=120)
plt.show()

# =============================================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

rf_pipe = trained["Random Forest"]
rf_clf  = rf_pipe.named_steps["clf"]

importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": rf_clf.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n=== Random Forest Feature Importance ===")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1],
         color="steelblue", edgecolor="black", linewidth=0.5)
plt.xlabel("Importance")
plt.title("Random Forest — Feature Importances\n(MOTOR_1_ON vs BACKGROUND_OFF)")
plt.grid(axis="x", linewidth=0.4)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_07_feature_importance.png"), dpi=120)
plt.show()


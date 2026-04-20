"""
Phase 4 — Feature Exploratory Analysis

Generates four figures:
  fig_p4_01_feature_by_class.png      — violin plots of top features per device
  fig_p4_02_correlation_heatmap.png   — correlation matrix of primary features
  fig_p4_03_pca_scatter.png           — PCA coloured by device family and state
  fig_p4_04_motor_vs_charger.png      — key features separating motors from chargers

Run from the project root:
    python Phase_4_Feature_Engineering/explore_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor import run, OUT_CSV

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE_COLORS = {
    "motor_1":   "#e6194b",
    "motor_2":   "#f58231",
    "motor_3":   "#ffe119",
    "charger_1": "#4363d8",
    "charger_2": "#3cb44b",
}

FAMILY_COLORS = {"motor": "#e6194b", "charger": "#4363d8"}


def load_or_build() -> pd.DataFrame:
    if os.path.isfile(OUT_CSV):
        print(f"Loading features from {OUT_CSV}")
        return pd.read_csv(OUT_CSV)
    print("features.csv not found — building now...")
    return run(save_csv=True)


def primary_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]


def main():
    df = load_or_build()
    pri = primary_feat_cols(df)
    df_clean = df.dropna(subset=pri).copy()
    print(f"  {len(df_clean)} / {len(df)} signals have complete primary features.")

    # ── Fig 1: Violin plots of key features by device ─────────────────────────
    key_features = [
        ("td_pri_rms",                  "RMS Amplitude"),
        ("td_pri_kurtosis",             "Kurtosis"),
        ("td_pri_zero_crossing_rate",   "Zero-Crossing Rate"),
        ("fd_pri_dominant_freq_hz",     "Dominant Frequency (Hz)"),
        ("fd_pri_spectral_centroid_hz", "Spectral Centroid (Hz)"),
        ("fd_pri_spectral_entropy",     "Spectral Entropy"),
        ("td_pri_crest_factor",         "Crest Factor"),
        ("fd_pri_band_energy_abs_mid",  "Band Energy 500–2000 Hz"),
    ]

    devices = list(df_clean["device_id"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Fig P4-01 — Key Features by Device", fontsize=13)

    for ax, (col, label) in zip(axes.flat, key_features):
        data   = [df_clean.loc[df_clean["device_id"] == d, col].dropna().values for d in devices]
        colors = [DEVICE_COLORS.get(d, "grey") for d in devices]
        parts  = ax.violinplot(data, positions=range(len(devices)), showmedians=True)
        for pc, col_c in zip(parts["bodies"], colors):
            pc.set_facecolor(col_c)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(devices)))
        ax.set_xticklabels([d.replace("_", "\n") for d in devices], fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p4_01_feature_by_class.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 2: Correlation heatmap ────────────────────────────────────────────
    corr = df_clean[pri].corr()
    short_names = [c.replace("td_pri_", "td·").replace("fd_pri_", "fd·") for c in pri]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(pri)))
    ax.set_yticks(range(len(pri)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title("Fig P4-02 — Feature Correlation Matrix (Primary Channel)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p4_02_correlation_heatmap.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 3: PCA scatter ────────────────────────────────────────────────────
    X = df_clean[pri].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Fig P4-03 — PCA  (PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  "
        f"PC3={var_exp[2]*100:.1f}%)", fontsize=12
    )

    # Left: coloured by device_id
    for dev in devices:
        mask = df_clean["device_id"].values == dev
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=DEVICE_COLORS.get(dev, "grey"), label=dev, alpha=0.65, s=20)
    axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    axes[0].set_title("Coloured by device")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linewidth=0.3)

    # Right: coloured by device_family, marker by state
    state_markers = {"on": "o", "off": "s"}
    for fam in ["motor", "charger"]:
        for state, marker in state_markers.items():
            mask = (df_clean["device_family"].values == fam) & (df_clean["state"].values == state)
            if mask.sum() == 0:
                continue
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=FAMILY_COLORS[fam], marker=marker,
                            label=f"{fam} {state}", alpha=0.65, s=20)
    axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    axes[1].set_title("Coloured by family / state")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p4_03_pca_scatter.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Fig 4: Motor vs Charger — most discriminating features ───────────────
    df_clean["_family"] = df_clean["device_family"]
    discriminating = [
        ("td_pri_rms",                  "RMS"),
        ("td_pri_kurtosis",             "Kurtosis"),
        ("fd_pri_dominant_freq_hz",     "Dominant Freq (Hz)"),
        ("fd_pri_spectral_centroid_hz", "Spectral Centroid (Hz)"),
        ("fd_pri_spectral_entropy",     "Spectral Entropy"),
        ("fd_pri_band_energy_abs_mid",  "Band Energy 500–2000 Hz"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fig P4-04 — Motor vs Charger: Discriminating Features", fontsize=13)

    for ax, (col, label) in zip(axes.flat, discriminating):
        motors   = df_clean.loc[df_clean["_family"] == "motor",   col].dropna()
        chargers = df_clean.loc[df_clean["_family"] == "charger", col].dropna()

        all_vals = pd.concat([motors, chargers]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        if hi > lo:
            bins = np.linspace(lo, hi, 27)
            ax.hist(motors,   bins=bins, alpha=0.6, color="#e6194b", label=f"Motor  (n={len(motors)})",   density=True)
            ax.hist(chargers, bins=bins, alpha=0.6, color="#4363d8", label=f"Charger(n={len(chargers)})", density=True)
        else:
            ax.text(0.5, 0.5, "Zero variance", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_p4_04_motor_vs_charger.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out)}")

    # ── Feature quality summary ───────────────────────────────────────────────
    print("\n  Top 10 features by motor/charger separation (Cohen's d):")
    results = []
    for col in pri:
        m = df_clean.loc[df_clean["device_family"] == "motor",   col].dropna()
        c = df_clean.loc[df_clean["device_family"] == "charger", col].dropna()
        if len(m) < 5 or len(c) < 5:
            continue
        pooled_std = np.sqrt((m.std()**2 + c.std()**2) / 2)
        d = abs(m.mean() - c.mean()) / pooled_std if pooled_std > 0 else 0
        results.append((col.replace("td_pri_","td·").replace("fd_pri_","fd·"), d))
    results.sort(key=lambda x: -x[1])
    for name, d in results[:10]:
        print(f"    {name:<40} d = {d:.3f}")

    print(f"\nFeature table: {df.shape[0]} rows × {len(pri)} primary features "
          f"(+ {df.shape[1] - len(pri) - len([c for c in df.columns if not c.startswith('td_') and not c.startswith('fd_') and not c.startswith('xc_')])} labelled metadata cols)")
    print("Done. All figures saved to Phase_4_Feature_Engineering/")


if __name__ == "__main__":
    main()

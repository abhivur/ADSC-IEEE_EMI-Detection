"""
Phase 5 — Modeling and Evaluation

Five prediction tasks, three baseline models each.

Tasks
-----
  Task 1  — Motor vs Charger (all 23 features)
              establishes the baseline — note sampling-rate confound
  Task 1b — Motor vs Charger (17 rate-agnostic features only)
              removes Hz-denominated spectral features that encode 20 kHz vs
              5 kHz sampling difference rather than EMI content; honest result
  Task 2  — Charger ON vs OFF  (185 signals, rate-homogeneous)
              state detection; most practically meaningful result
  Task 3  — Motor ID 3-class  (186 signals, all 20 kHz)
              motor_1 / motor_2 / motor_3 — fine-grained EMI fingerprinting
              within a single device family, no sampling-rate advantage
  Task 4  — Charger ID 2-class  (185 signals, all 5 kHz)
              charger_1 / charger_2 — same as Task 3 but for chargers

Methodological choices
----------------------
  • 80 / 20 stratified train/test split — confusion matrices shown on the
    held-out TEST set only, not on training data
  • 5-fold stratified CV performed on training set only (not the full dataset)
  • Rate-agnostic feature set for Task 1b excludes:
      fd_pri_dominant_freq_hz, fd_pri_spectral_centroid_hz,
      fd_pri_spectral_spread_hz, fd_pri_band_energy_abs_{low,mid,high}

Figures produced
----------------
  fig_p5_01_cv_scores.png           — CV accuracy + F1, all tasks and models
  fig_p5_02_task1_confusion.png     — Motor vs Charger, all features (test set)
  fig_p5_03_task1b_confusion.png    — Motor vs Charger, rate-agnostic (test set)
  fig_p5_04_task2_confusion.png     — Charger ON vs OFF (test set)
  fig_p5_05_task3_confusion.png     — Motor ID 3-class (test set)
  fig_p5_06_task4_confusion.png     — Charger ID 2-class (test set)
  fig_p5_07_feature_importance.png  — RF feature importances (Tasks 1b, 2, 3, 4)
  fig_p5_08_error_analysis.png      — Error breakdown: per-class rates + metadata

Run from project root:
    python Phase_5_Modeling/classifier.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics         import (confusion_matrix, ConfusionMatrixDisplay,
                                     accuracy_score, f1_score, classification_report)

warnings.filterwarnings("ignore")

FEATURES_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Phase_4_Feature_Engineering", "features.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hz-denominated spectral features that encode absolute frequency — confounded
# with sampling rate when comparing motors (20 kHz) vs chargers (5 kHz).
_HZ_CONFOUNDED = {
    "fd_pri_dominant_freq_hz",
    "fd_pri_spectral_centroid_hz",
    "fd_pri_spectral_spread_hz",
    "fd_pri_band_energy_abs_low",
    "fd_pri_band_energy_abs_mid",
    "fd_pri_band_energy_abs_high",
}


def get_all_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]


def get_rate_agnostic_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in get_all_feat_cols(df) if c not in _HZ_CONFOUNDED]


# ── Model factory ─────────────────────────────────────────────────────────────

def make_models() -> dict[str, Pipeline]:
    return {
        "LR":  Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "RF":  Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                                        class_weight="balanced"))]),
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                     random_state=42, class_weight="balanced"))]),
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_task(df_subset: pd.DataFrame, feat_cols: list[str],
                  label_col: str, class_names: list[str]) -> dict:
    """
    80/20 stratified train/test split.
    5-fold CV on training set → accuracy and macro-F1 estimates.
    Final confusion matrix and classification report on held-out test set.
    Returns indices into df_subset so callers can look up metadata for errors.
    """
    X       = df_subset[feat_cols].values
    y       = df_subset[label_col].values
    indices = np.arange(len(df_subset))

    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, pipe in make_models().items():
        cv = cross_validate(pipe, X_tr, y_tr, cv=skf,
                            scoring=["accuracy", "f1_macro"],
                            return_train_score=False)
        cv_results[name] = {
            "acc_mean": float(cv["test_accuracy"].mean()),
            "acc_std":  float(cv["test_accuracy"].std()),
            "f1_mean":  float(cv["test_f1_macro"].mean()),
            "f1_std":   float(cv["test_f1_macro"].std()),
        }

    # Fit on train, evaluate on TEST set
    test_cms     = {}
    test_preds   = {}
    test_reports = {}
    test_accs    = {}
    test_f1s     = {}
    label_order  = np.unique(y)

    for name, pipe in make_models().items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        test_preds[name]   = y_pred
        test_cms[name]     = confusion_matrix(y_te, y_pred, labels=label_order)
        test_reports[name] = classification_report(
            y_te, y_pred, target_names=class_names, zero_division=0
        )
        test_accs[name] = float(accuracy_score(y_te, y_pred))
        test_f1s[name]  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))

    return {
        "cv":          cv_results,
        "cms":         test_cms,
        "reports":     test_reports,
        "preds":       test_preds,
        "test_accs":   test_accs,
        "test_f1s":    test_f1s,
        "y_test":      y_te,
        "test_idx":    idx_te,
        "classes":     class_names,
        "n_train":     len(y_tr),
        "n_test":      len(y_te),
    }


# ── Figure helpers ────────────────────────────────────────────────────────────

def plot_confusion_matrices(task_result: dict, task_title: str, out_path: str) -> None:
    """Confusion matrices on HELD-OUT TEST SET."""
    classes  = task_result["classes"]
    cms      = task_result["cms"]
    models   = list(cms.keys())
    n_test   = task_result["n_test"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5))
    fig.suptitle(f"{task_title}\n(test set  n={n_test})", fontsize=11)
    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=task_result["cms"][name],
            display_labels=classes,
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        cv_acc  = task_result["cv"][name]["acc_mean"]
        te_acc  = task_result["test_accs"][name]
        te_f1   = task_result["test_f1s"][name]
        ax.set_title(
            f"{name}\nCV acc={cv_acc:.3f}  |  test acc={te_acc:.3f}  F1={te_f1:.3f}",
            fontsize=9
        )
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


def plot_cv_scores(all_results: dict, out_path: str) -> None:
    tasks  = list(all_results.keys())
    models = ["LR", "RF", "SVM"]
    x      = np.arange(len(tasks))
    width  = 0.22
    colors = {"LR": "#4363d8", "RF": "#3cb44b", "SVM": "#e6194b"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Fig P5-01 — CV Performance (training set, 5-fold) Across Tasks and Models",
                 fontsize=12)

    for ax, (metric, ylabel) in zip(axes, [("acc_mean", "Accuracy"), ("f1_mean", "F1-macro")]):
        for i, model in enumerate(models):
            means  = [all_results[t]["cv"][model][metric] for t in tasks]
            stds   = [all_results[t]["cv"][model][metric.replace("mean","std")] for t in tasks]
            offset = (i - 1) * width
            ax.bar(x + offset, means, width, label=model, color=colors[model],
                   alpha=0.85, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.3)
        ax.set_title(ylabel)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


def plot_feature_importance(fi_dict: dict, feat_cols: list[str], out_path: str) -> None:
    """RF feature importances — top 15 per task."""
    tasks = list(fi_dict.keys())
    n     = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    fig.suptitle("Fig P5-07 — Random Forest Feature Importances", fontsize=13)
    if n == 1:
        axes = [axes]

    short = [c.replace("td_pri_", "td·").replace("fd_pri_", "fd·") for c in feat_cols]

    for ax, task in zip(axes, tasks):
        imps = fi_dict[task]
        idx  = np.argsort(imps)[-15:]
        ax.barh([short[i] for i in idx], imps[idx], color="#4363d8", alpha=0.8)
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("Mean decrease in impurity")
        ax.grid(axis="x", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


def plot_error_analysis(error_tasks: list[tuple], out_path: str) -> None:
    """
    error_tasks: list of (task_label, task_result, df_subset, meta_col)
        meta_col — metadata column to break down errors by (e.g. 'device_id', 'distance_label')
    """
    n = len(error_tasks)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))
    fig.suptitle("Fig P5-08 — Error Analysis: Per-Class Error Rates and Error Metadata (RF)",
                 fontsize=12)

    if n == 1:
        axes = [axes]

    for row, (task_label, res, df_sub, meta_col) in enumerate(error_tasks):
        y_test   = res["y_test"]
        y_pred   = res["preds"]["RF"]
        test_idx = res["test_idx"]
        classes  = res["classes"]

        ax_l = axes[row][0]
        ax_r = axes[row][1]

        # Left: per-class error rate
        rates, counts = [], []
        for cls in classes:
            mask = y_test == cls
            n_cls = mask.sum()
            n_err = (y_pred[mask] != y_test[mask]).sum()
            rates.append(n_err / n_cls if n_cls > 0 else 0.0)
            counts.append(n_cls)

        bar_colors = ["#e6194b" if r > 0 else "#3cb44b" for r in rates]
        bars = ax_l.bar(classes, rates, color=bar_colors, alpha=0.8)
        for bar, cnt in zip(bars, counts):
            ax_l.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.01,
                      f"n={cnt}", ha="center", fontsize=7)
        ax_l.set_title(f"{task_label} — Error Rate per True Class", fontsize=9)
        ax_l.set_ylabel("Fraction misclassified")
        ax_l.set_ylim(0, max(rates) * 1.4 + 0.05)
        ax_l.tick_params(axis="x", rotation=25)
        ax_l.grid(axis="y", linewidth=0.3)

        # Right: metadata breakdown of misclassified signals
        error_mask = y_pred != y_test
        error_df   = df_sub.iloc[test_idx[error_mask]]

        if len(error_df) == 0:
            ax_r.text(0.5, 0.5, f"No errors  (test acc=1.00)",
                      ha="center", va="center", transform=ax_r.transAxes, fontsize=10)
            ax_r.set_title(f"{task_label} — Error Metadata", fontsize=9)
        else:
            meta_counts = error_df[meta_col].value_counts()
            ax_r.bar(meta_counts.index, meta_counts.values, color="#f58231", alpha=0.8)
            ax_r.set_title(f"{task_label} — {meta_col} of Misclassified Signals "
                           f"(n_errors={error_mask.sum()})", fontsize=9)
            ax_r.set_ylabel("Count")
            ax_r.tick_params(axis="x", rotation=25)
            ax_r.grid(axis="y", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading features from {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    all_feat_cols  = get_all_feat_cols(df)
    rate_agn_cols  = get_rate_agnostic_cols(df)
    print(f"  {len(df)} signals")
    print(f"  {len(all_feat_cols)} total primary features, "
          f"{len(rate_agn_cols)} rate-agnostic features")

    df_clean = df.dropna(subset=all_feat_cols).copy()
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=all_feat_cols).reset_index(drop=True)
    n_after = len(df_clean)
    print(f"  {n_before} signals with complete features, "
          f"{n_before - n_after} exact-duplicate feature vectors removed → {n_after} unique")

    all_cv_results = {}  # for CV scores figure
    fi_dict        = {}  # for feature importance figure

    # ── Task 1: Motor vs Charger (all features) ───────────────────────────────
    print("\n── Task 1: Motor vs Charger (all features) ──")
    t1_df = df_clean.copy()
    res1  = evaluate_task(t1_df, all_feat_cols, "device_family",
                          sorted(t1_df["device_family"].unique().tolist()))
    all_cv_results["Motor vs\nCharger\n(all feats)"] = res1
    for m, v in res1["cv"].items():
        print(f"  {m}: CV acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}  "
              f"test acc={res1['test_accs'][m]:.3f}  F1={res1['test_f1s'][m]:.3f}")
    plot_confusion_matrices(res1, "Fig P5-02 — Motor vs Charger (all features)",
                            os.path.join(OUT_DIR, "fig_p5_02_task1_confusion.png"))

    # ── Task 1b: Motor vs Charger (rate-agnostic features) ───────────────────
    print("\n── Task 1b: Motor vs Charger (rate-agnostic features only) ──")
    print(f"  Using {len(rate_agn_cols)} features "
          f"(excluded: {sorted(_HZ_CONFOUNDED)})")
    res1b = evaluate_task(t1_df, rate_agn_cols, "device_family",
                          sorted(t1_df["device_family"].unique().tolist()))
    all_cv_results["Motor vs\nCharger\n(rate-agnostic)"] = res1b
    for m, v in res1b["cv"].items():
        print(f"  {m}: CV acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}  "
              f"test acc={res1b['test_accs'][m]:.3f}  F1={res1b['test_f1s'][m]:.3f}")
    plot_confusion_matrices(res1b, "Fig P5-03 — Motor vs Charger (rate-agnostic features only)",
                            os.path.join(OUT_DIR, "fig_p5_03_task1b_confusion.png"))

    rf1b = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf1b.fit(StandardScaler().fit_transform(t1_df[rate_agn_cols].values),
             t1_df["device_family"].values)
    fi_dict["Motor vs\nCharger\n(rate-agnostic)"] = rf1b.feature_importances_

    # ── Task 2: Charger ON vs OFF ─────────────────────────────────────────────
    print("\n── Task 2: Charger ON vs OFF ──")
    t2_df = df_clean[df_clean["device_family"] == "charger"].copy().reset_index(drop=True)
    print(f"  n={len(t2_df)}  |  "
          f"{dict(t2_df['state'].value_counts())}")
    res2 = evaluate_task(t2_df, all_feat_cols, "state",
                         sorted(t2_df["state"].unique().tolist()))
    all_cv_results["Charger\nON vs OFF"] = res2
    for m, v in res2["cv"].items():
        print(f"  {m}: CV acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}  "
              f"test acc={res2['test_accs'][m]:.3f}  F1={res2['test_f1s'][m]:.3f}")
    plot_confusion_matrices(res2, "Fig P5-04 — Charger ON vs OFF",
                            os.path.join(OUT_DIR, "fig_p5_04_task2_confusion.png"))

    rf2 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf2.fit(StandardScaler().fit_transform(t2_df[all_feat_cols].values),
            t2_df["state"].values)
    fi_dict["Charger\nON vs OFF"] = rf2.feature_importances_

    # ── Task 3: Motor ID (3-class, all 20 kHz) ───────────────────────────────
    print("\n── Task 3: Motor ID 3-class (motor_1 / motor_2 / motor_3) ──")
    t3_df = df_clean[df_clean["device_family"] == "motor"].copy().reset_index(drop=True)
    print(f"  n={len(t3_df)}  |  "
          f"{dict(t3_df['device_id'].value_counts())}")
    res3 = evaluate_task(t3_df, all_feat_cols, "device_id",
                         sorted(t3_df["device_id"].unique().tolist()))
    all_cv_results["Motor ID\n(3-class)"] = res3
    for m, v in res3["cv"].items():
        print(f"  {m}: CV acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}  "
              f"test acc={res3['test_accs'][m]:.3f}  F1={res3['test_f1s'][m]:.3f}")
    plot_confusion_matrices(res3, "Fig P5-05 — Motor ID (motor_1 / motor_2 / motor_3)",
                            os.path.join(OUT_DIR, "fig_p5_05_task3_confusion.png"))

    rf3 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf3.fit(StandardScaler().fit_transform(t3_df[all_feat_cols].values),
            t3_df["device_id"].values)
    fi_dict["Motor ID\n(3-class)"] = rf3.feature_importances_

    # ── Task 4: Charger ID (2-class, all 5 kHz) ──────────────────────────────
    print("\n── Task 4: Charger ID 2-class (charger_1 / charger_2) ──")
    t4_df = df_clean[df_clean["device_family"] == "charger"].copy().reset_index(drop=True)
    print(f"  n={len(t4_df)}  |  "
          f"{dict(t4_df['device_id'].value_counts())}")
    res4 = evaluate_task(t4_df, all_feat_cols, "device_id",
                         sorted(t4_df["device_id"].unique().tolist()))
    all_cv_results["Charger ID\n(2-class)"] = res4
    for m, v in res4["cv"].items():
        print(f"  {m}: CV acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}  "
              f"test acc={res4['test_accs'][m]:.3f}  F1={res4['test_f1s'][m]:.3f}")
    plot_confusion_matrices(res4, "Fig P5-06 — Charger ID (charger_1 / charger_2)",
                            os.path.join(OUT_DIR, "fig_p5_06_task4_confusion.png"))

    rf4 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf4.fit(StandardScaler().fit_transform(t4_df[all_feat_cols].values),
            t4_df["device_id"].values)
    fi_dict["Charger ID\n(2-class)"] = rf4.feature_importances_

    # ── Summary figures ───────────────────────────────────────────────────────
    plot_cv_scores(all_cv_results, os.path.join(OUT_DIR, "fig_p5_01_cv_scores.png"))
    plot_feature_importance(fi_dict, all_feat_cols,
                            os.path.join(OUT_DIR, "fig_p5_07_feature_importance.png"))

    # Error analysis: Tasks 2, 3, 4 (honest, rate-homogeneous tasks)
    error_tasks = [
        ("Charger ON vs OFF",     res2, t2_df, "device_id"),
        ("Motor ID (3-class)",    res3, t3_df, "distance_label"),
        ("Charger ID (2-class)",  res4, t4_df, "distance_label"),
    ]
    plot_error_analysis(error_tasks, os.path.join(OUT_DIR, "fig_p5_08_error_analysis.png"))

    # ── Per-class classification reports ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("PER-CLASS CLASSIFICATION REPORTS  (RF, test set)")
    print("=" * 65)
    report_tasks = [
        ("Task 1b — Motor vs Charger (rate-agnostic)", res1b, "RF"),
        ("Task 2  — Charger ON vs OFF",                res2,  "RF"),
        ("Task 3  — Motor ID",                         res3,  "RF"),
        ("Task 4  — Charger ID",                       res4,  "RF"),
    ]
    for label, res, model in report_tasks:
        print(f"\n  {label}")
        print(f"  n_train={res['n_train']}  n_test={res['n_test']}")
        for line in res["reports"][model].splitlines():
            print(f"    {line}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 5 — FINAL EVALUATION SUMMARY")
    print("=" * 65)

    summary_tasks = [
        ("Motor vs Charger (rate-agnostic)", res1b),
        ("Charger ON vs OFF",                res2),
        ("Motor ID (3-class)",               res3),
        ("Charger ID (2-class)",             res4),
    ]
    for task_name, res in summary_tasks:
        best_m = max(res["cv"], key=lambda m: res["cv"][m]["f1_mean"])
        print(f"\n  {task_name}")
        print(f"    Best CV model : {best_m} "
              f"(CV F1={res['cv'][best_m]['f1_mean']:.3f}  "
              f"test acc={res['test_accs'][best_m]:.3f}  "
              f"test F1={res['test_f1s'][best_m]:.3f})")
        print(f"    n_train={res['n_train']}  n_test={res['n_test']}")

    print("\n  Task 1 (all features) vs Task 1b (rate-agnostic):")
    for m in ["LR", "RF", "SVM"]:
        acc_all  = res1["test_accs"][m]
        acc_ragn = res1b["test_accs"][m]
        print(f"    {m}: all-features test acc={acc_all:.3f}  |  "
              f"rate-agnostic test acc={acc_ragn:.3f}  "
              f"(drop={acc_all - acc_ragn:+.3f})")

    print("\n  Recommended model for demo: Random Forest")
    print("    Rationale: interpretable importances, handles imbalanced classes,")
    print("    consistent across all tasks, no hyperparameter tuning needed.")
    print("\n  All figures saved to Phase_5_Modeling/")
    print("=" * 65)


if __name__ == "__main__":
    main()

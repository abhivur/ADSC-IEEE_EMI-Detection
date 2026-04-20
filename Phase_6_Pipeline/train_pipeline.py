"""
Phase 6 — Train and Save Pipeline Models

Trains four production classifiers on all available deduplicated data and
saves them to Phase_6_Pipeline/models/ as .joblib files.

Production training uses the full deduplicated dataset (no held-out split)
because Phase 5 already established generalisation estimates via 5-fold CV
and an 80/20 test split.

Models saved
------------
  models/family_clf.joblib      — Motor vs Charger  (rate-agnostic features)
  models/motor_id_clf.joblib    — Motor ID 3-class  (all features)
  models/charger_state_clf.joblib — Charger ON vs OFF (all features)
  models/charger_id_clf.joblib  — Charger ID 2-class (all features)
  models/feature_cols.json      — feature column lists used by pipeline.py

Run from project root:
    python Phase_6_Pipeline/train_pipeline.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Phase_4_Feature_Engineering"))

FEATURES_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Phase_4_Feature_Engineering", "features.csv"
)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_HZ_CONFOUNDED = {
    "fd_pri_dominant_freq_hz", "fd_pri_spectral_centroid_hz",
    "fd_pri_spectral_spread_hz",
    "fd_pri_band_energy_abs_low", "fd_pri_band_energy_abs_mid", "fd_pri_band_energy_abs_high",
}


def get_feat_cols(df):
    all_cols = [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]
    agn_cols = [c for c in all_cols if c not in _HZ_CONFOUNDED]
    return all_cols, agn_cols


def make_rf():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                                    class_weight="balanced"))])


def make_svm():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                 random_state=42, class_weight="balanced"))])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    all_cols, agn_cols = get_feat_cols(df)
    df = df.dropna(subset=all_cols).drop_duplicates(subset=all_cols).reset_index(drop=True)
    print(f"  {len(df)} unique signals after deduplication")

    # ── Family classifier: Motor vs Charger (rate-agnostic, SVM) ─────────────
    print("\nTraining family classifier (Motor vs Charger, SVM, rate-agnostic) ...")
    X_fam = df[agn_cols].values
    y_fam = df["device_family"].values
    clf_fam = make_svm()
    clf_fam.fit(X_fam, y_fam)
    path = os.path.join(MODELS_DIR, "family_clf.joblib")
    joblib.dump(clf_fam, path)
    print(f"  Saved {path}")

    # ── Charger state: ON vs OFF (RF) ─────────────────────────────────────────
    print("\nTraining charger state classifier (ON vs OFF, RF) ...")
    t2 = df[df["device_family"] == "charger"]
    X_cs = t2[all_cols].values
    y_cs = t2["state"].values
    clf_cs = make_rf()
    clf_cs.fit(X_cs, y_cs)
    path = os.path.join(MODELS_DIR, "charger_state_clf.joblib")
    joblib.dump(clf_cs, path)
    print(f"  Saved {path}")

    # ── Motor ID: 3-class (SVM) ───────────────────────────────────────────────
    print("\nTraining motor ID classifier (3-class, SVM) ...")
    t3 = df[df["device_family"] == "motor"]
    X_mid = t3[all_cols].values
    y_mid = t3["device_id"].values
    clf_mid = make_svm()
    clf_mid.fit(X_mid, y_mid)
    path = os.path.join(MODELS_DIR, "motor_id_clf.joblib")
    joblib.dump(clf_mid, path)
    print(f"  Saved {path}")

    # ── Charger ID: 2-class (SVM) ─────────────────────────────────────────────
    print("\nTraining charger ID classifier (2-class, SVM) ...")
    t4 = df[df["device_family"] == "charger"]
    X_cid = t4[all_cols].values
    y_cid = t4["device_id"].values
    clf_cid = make_svm()
    clf_cid.fit(X_cid, y_cid)
    path = os.path.join(MODELS_DIR, "charger_id_clf.joblib")
    joblib.dump(clf_cid, path)
    print(f"  Saved {path}")

    # ── Save feature column lists ─────────────────────────────────────────────
    feat_meta = {"all_feat_cols": all_cols, "rate_agnostic_cols": agn_cols}
    path = os.path.join(MODELS_DIR, "feature_cols.json")
    with open(path, "w") as f:
        json.dump(feat_meta, f, indent=2)
    print(f"\n  Saved feature column lists → {path}")
    print(f"\nAll models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()

"""
Phase 6 — EMI Detection Pipeline

Single entry point: raw oscilloscope CSV → device predictions.

Pipeline stages
---------------
  1. Load    — parse raw CSV, reconstruct time axis and channel arrays
  2. Condition — low-pass filter + z-score normalisation
  3. Extract — compute 23 time/frequency-domain features
  4. Predict — hierarchical classifier:
                  family_clf        → motor | charger
                  if motor  → motor_id_clf      → motor_1 | motor_2 | motor_3
                  if charger → charger_state_clf → on | off
                              charger_id_clf    → charger_1 | charger_2

Public API
----------
    pipeline = EMIPipeline()              # loads saved models
    result   = pipeline.predict_file(filepath)
    results  = pipeline.predict_batch(filepaths)

Result dict keys
----------------
    file            : basename of the input file
    device_family   : 'motor' | 'charger'
    motor_id        : 'motor_1' | 'motor_2' | 'motor_3'   (motors only)
    charger_state   : 'on' | 'off'                         (chargers only)
    charger_id      : 'charger_1' | 'charger_2'           (chargers only)
    sample_rate_hz  : detected from file header
    primary_channel : 'ch1' | 'ch2'

Run from project root:
    python Phase_6_Pipeline/pipeline.py path/to/signal.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

_PHASE2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_2_Ingestion_Pipeline")
_PHASE3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3_Signal_Processing")
_PHASE4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_4_Feature_Engineering")

for _p in (_PHASE2, _PHASE3, _PHASE4):
    sys.path.insert(0, _p)

from loader    import load_file
from processor import condition_signal
from extractor import extract_features

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class EMIPipeline:
    """End-to-end EMI device identification pipeline."""

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(
                f"Models directory not found: {models_dir}\n"
                "Run `python Phase_6_Pipeline/train_pipeline.py` first."
            )
        self._load_models(models_dir)

    def _load_models(self, models_dir: str) -> None:
        def _load(name):
            path = os.path.join(models_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file missing: {path}")
            return joblib.load(path)

        self.family_clf        = _load("family_clf.joblib")
        self.charger_state_clf = _load("charger_state_clf.joblib")
        self.motor_id_clf      = _load("motor_id_clf.joblib")
        self.charger_id_clf    = _load("charger_id_clf.joblib")

        col_path = os.path.join(models_dir, "feature_cols.json")
        with open(col_path) as f:
            cols = json.load(f)
        self.all_feat_cols  = cols["all_feat_cols"]
        self.rate_agn_cols  = cols["rate_agnostic_cols"]

    def _feature_vector(self, row: dict, col_list: list) -> np.ndarray:
        vec = np.array([row.get(c, np.nan) for c in col_list], dtype=float)
        if np.any(np.isnan(vec)):
            missing = [c for c in col_list if np.isnan(row.get(c, np.nan))]
            raise ValueError(f"NaN in features after extraction: {missing}")
        return vec.reshape(1, -1)

    def predict_file(self, filepath: str) -> dict:
        """
        Full pipeline for a single raw CSV file.
        Returns a prediction dict (see module docstring).
        """
        # Stage 1: Load
        sig = load_file(filepath)
        if sig is None:
            raise ValueError(f"Could not load file: {filepath}")

        # Stage 2: Condition
        cond = condition_signal(sig)
        if not cond.get("conditioned"):
            raise ValueError(f"Conditioning failed for: {filepath}")

        # Stage 3: Extract features
        row = extract_features(sig, cond)

        # Stage 4: Predict — hierarchical
        X_agn = self._feature_vector(row, self.rate_agn_cols)
        X_all = self._feature_vector(row, self.all_feat_cols)

        family = self.family_clf.predict(X_agn)[0]

        result = {
            "file":           os.path.basename(filepath),
            "device_family":  family,
            "sample_rate_hz": cond.get("sample_rate_hz"),
            "primary_channel": row.get("primary_channel"),
        }

        if family == "motor":
            result["motor_id"] = self.motor_id_clf.predict(X_all)[0]
        elif family == "charger":
            result["charger_state"] = self.charger_state_clf.predict(X_all)[0]
            result["charger_id"]    = self.charger_id_clf.predict(X_all)[0]

        return result

    def predict_batch(self, filepaths: list) -> list:
        """Run predict_file on a list of paths. Returns list of result dicts."""
        results = []
        for fp in filepaths:
            try:
                results.append(self.predict_file(fp))
            except Exception as e:
                results.append({"file": os.path.basename(fp), "error": str(e)})
        return results


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Phase_6_Pipeline/pipeline.py <path/to/signal.csv> [...]")
        sys.exit(1)

    pipe = EMIPipeline()
    for filepath in sys.argv[1:]:
        try:
            r = pipe.predict_file(filepath)
            print(f"\n{r['file']}")
            for k, v in r.items():
                if k != "file":
                    print(f"  {k:<20} {v}")
        except Exception as e:
            print(f"\n{os.path.basename(filepath)}  ERROR: {e}")

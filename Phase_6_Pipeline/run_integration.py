"""
Phase 6 — End-to-End Integration Test

Runs 21 known raw CSV files through the full EMIPipeline and checks
predictions against ground-truth labels from dataset_index.csv.

Checks performed
----------------
  1. device_family prediction matches ground truth
  2. motor_id     prediction matches ground truth  (motors only)
  3. charger_state prediction matches ground truth (chargers only)
  4. charger_id   prediction matches ground truth  (chargers only)
  5. No exceptions or NaN errors during processing

Prints a PASS / FAIL report and summary table.

Run from project root:
    python Phase_6_Pipeline/run_integration.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Phase_2_Ingestion_Pipeline"))

from pipeline import EMIPipeline
from loader   import BASE_DIR

INDEX_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Phase_1_Dataset_Audit", "dataset_index.csv"
)

# 21 hand-picked non-duplicate time-domain files covering all device/state combos
TEST_CASES = [
    # (relative_path,                                                    device_family, device_id,  state)
    ("Laptop_1/LAPTOP_CHRGR_1_OFF/time domain/NewFile35.csv",           "charger", "charger_1", "off"),
    ("Laptop_1/LAPTOP_CHRGR_1_OFF/time domain/NewFile2.csv",            "charger", "charger_1", "off"),
    ("Laptop_1/LAPTOP_CHRGR_1_OFF/time domain/NewFile19.csv",           "charger", "charger_1", "off"),
    ("Laptop_1/LAPTOP_CHRGR_1_ON_CLOSE/time domain/NewFile4.csv",       "charger", "charger_1", "on"),
    ("Laptop_1/LAPTOP_CHRGR_1_ON_CLOSE/time domain/NewFile28.csv",      "charger", "charger_1", "on"),
    ("Laptop_1/LAPTOP_CHRGR_1_ON_CLOSE/time domain/NewFile24.csv",      "charger", "charger_1", "on"),
    ("Laptop_2/LAPTOP_CHRGR_2_OFF/time domain/NewFile35.csv",           "charger", "charger_2", "off"),
    ("Laptop_2/LAPTOP_CHRGR_2_OFF/time domain/NewFile2.csv",            "charger", "charger_2", "off"),
    ("Laptop_2/LAPTOP_CHRGR_2_OFF/time domain/NewFile19.csv",           "charger", "charger_2", "off"),
    ("Laptop_2/LAPTOP_CHRGR_2_ON_DISTANT/time domain/NewFile35.csv",    "charger", "charger_2", "on"),
    ("Laptop_2/LAPTOP_CHRGR_2_ON_DISTANT/time domain/NewFile2.csv",     "charger", "charger_2", "on"),
    ("Laptop_2/LAPTOP_CHRGR_2_ON_DISTANT/time domain/NewFile19.csv",    "charger", "charger_2", "on"),
    ("Motor_Data/MOTOR_1_ON/time domain/NewFile57.csv",                  "motor",   "motor_1",   "on"),
    ("Motor_Data/MOTOR_1_ON/time domain/NewFile61.csv",                  "motor",   "motor_1",   "on"),
    ("Motor_Data/MOTOR_1_ON/time domain/NewFile3.csv",                   "motor",   "motor_1",   "on"),
    ("Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe1/time domain (far ch1 close ch2)/NewFile1.csv",
                                                                         "motor",   "motor_2",   "on"),
    ("Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe1/time domain (far ch1 close ch2)/NewFile10.csv",
                                                                         "motor",   "motor_2",   "on"),
    ("Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe1/time domain (far ch1 close ch2)/NewFile11.csv",
                                                                         "motor",   "motor_2",   "on"),
    ("Motor_Data/MOTOR_3_ON/time domain/NewFile57.csv",                  "motor",   "motor_3",   "on"),
    ("Motor_Data/MOTOR_3_ON/time domain/NewFile61.csv",                  "motor",   "motor_3",   "on"),
    ("Motor_Data/MOTOR_3_ON/time domain/NewFile3.csv",                   "motor",   "motor_3",   "on"),
]


def check(pred_val, expected, label):
    """Returns (passed, message)."""
    if pred_val is None:
        return False, f"{label}: not returned (None)"
    if pred_val == expected:
        return True, f"{label}: {pred_val} ✓"
    return False, f"{label}: predicted={pred_val!r}  expected={expected!r} ✗"


def main():
    print("=" * 65)
    print("PHASE 6 — END-TO-END INTEGRATION TEST")
    print("=" * 65)

    pipe = EMIPipeline()
    print("  Models loaded.\n")

    rows   = []
    passed = 0
    failed = 0
    errors = 0

    for rel_path, true_family, true_device, true_state in TEST_CASES:
        abs_path = os.path.join(BASE_DIR, rel_path)
        fname    = os.path.basename(rel_path)

        if not os.path.isfile(abs_path):
            print(f"  [SKIP] File not found: {rel_path}")
            errors += 1
            continue

        try:
            result = pipe.predict_file(abs_path)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            errors += 1
            rows.append({"file": fname, "status": "ERROR", "detail": str(e)})
            continue

        checks = []
        all_pass = True

        ok, msg = check(result.get("device_family"), true_family, "family")
        checks.append(msg); all_pass = all_pass and ok

        if true_family == "motor":
            ok, msg = check(result.get("motor_id"), true_device, "motor_id")
            checks.append(msg); all_pass = all_pass and ok
        elif true_family == "charger":
            ok, msg = check(result.get("charger_state"), true_state, "state")
            checks.append(msg); all_pass = all_pass and ok
            ok, msg = check(result.get("charger_id"), true_device, "charger_id")
            checks.append(msg); all_pass = all_pass and ok

        status = "PASS" if all_pass else "FAIL"
        symbol = "✓" if all_pass else "✗"
        print(f"  [{status}] {fname:<35}  {',  '.join(checks)}")

        if all_pass:
            passed += 1
        else:
            failed += 1

        rows.append({
            "file":        fname,
            "true_family": true_family,
            "true_device": true_device,
            "true_state":  true_state,
            "pred_family": result.get("device_family"),
            "pred_device": result.get("motor_id") or result.get("charger_id"),
            "pred_state":  result.get("charger_state"),
            "status":      status,
        })

    total = passed + failed + errors
    print()
    print("=" * 65)
    print(f"  PASSED : {passed} / {total - errors}")
    print(f"  FAILED : {failed} / {total - errors}")
    if errors:
        print(f"  ERRORS : {errors} (files missing or load failure)")
    pct = passed / (total - errors) * 100 if (total - errors) > 0 else 0
    print(f"  Overall: {pct:.1f}%")
    print("=" * 65)

    if failed == 0 and errors == 0:
        print("\n  Pipeline is STABLE — all integration checks passed.")
    else:
        print("\n  Pipeline has failures — review output above.")

    return passed, failed, errors


if __name__ == "__main__":
    main()

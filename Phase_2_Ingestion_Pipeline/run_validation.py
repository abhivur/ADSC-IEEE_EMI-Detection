"""
Phase 2 — Validation Script
Loads the full time-domain dataset and prints a report.
Run from the project root:   python Phase_2_Ingestion_Pipeline/run_validation.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_dataset, load_file, BASE_DIR

SEP = "=" * 60


def main():
    print(f"\n{SEP}")
    print("PHASE 2 — INGESTION VALIDATION REPORT")
    print(SEP)

    signals, report = load_dataset(skip_duplicates=True, skip_flagged=False)

    print(f"\n  Attempted   : {report['total_attempted']}")
    print(f"  Loaded OK   : {report['loaded']}")
    print(f"  Skipped (freq-domain misplaced): {report['skipped_freq']}")
    print(f"  Skipped (errors/malformed)     : {report['skipped_errors']}")

    print(f"\n  Loaded by device:")
    for k, v in report["by_device"].items():
        print(f"    {k:<15} {v:>4}")

    print(f"\n  Loaded by channel mode:")
    for k, v in report["by_channel_mode"].items():
        print(f"    {k:<20} {v:>4}")

    print(f"\n  Loaded by state:")
    for k, v in report["by_state"].items():
        print(f"    {k:<10} {v:>4}")

    if report["warnings"]:
        print(f"\n  Files with warnings ({len(report['warnings'])}):")
        for w in report["warnings"][:10]:
            print(f"    {w['file']}  ->  {w['reason']}")
        if len(report["warnings"]) > 10:
            print(f"    ... and {len(report['warnings']) - 10} more")

    # ── Signal sanity checks ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SIGNAL SANITY CHECKS")
    print(SEP)

    sample_counts = [s["sample_count"] for s in signals]
    sample_rates  = [s["sample_rate_hz"] for s in signals if s["sample_rate_hz"]]
    dual          = [s for s in signals if s["channel_mode"] == "dual"]
    single_ch1    = [s for s in signals if s["channel_mode"] == "single_ch1"]
    single_ch2    = [s for s in signals if s["channel_mode"] == "single_ch2"]

    print(f"\n  Sample count  min={min(sample_counts)}  max={max(sample_counts)}  "
          f"unique={sorted(set(sample_counts))}")
    print(f"  Sample rates  unique={sorted(set(sample_rates))}")
    print(f"  Dual-channel  : {len(dual)}")
    print(f"  Single CH1    : {len(single_ch1)}")
    print(f"  Single CH2    : {len(single_ch2)}")

    # Spot-check one signal of each type
    print(f"\n{SEP}")
    print("SPOT CHECKS")
    print(SEP)

    for label, subset in [("dual", dual), ("single_ch1", single_ch1), ("single_ch2", single_ch2)]:
        if not subset:
            continue
        s = subset[0]
        ch1_len = len(s["ch1"]) if s["ch1"] is not None else 0
        ch2_len = len(s["ch2"]) if s["ch2"] is not None else 0
        time_len = len(s["time"])
        print(f"\n  [{label}] {s['filename']}")
        print(f"    device={s['device_id']}  state={s['state']}  "
              f"distance={s['distance_label']}  probe={s['probe_id']}")
        print(f"    ch1_samples={ch1_len}  ch2_samples={ch2_len}  time_len={time_len}")
        print(f"    start_time={s['start_time']}  increment={s['increment_sec']}  "
              f"sample_rate={s['sample_rate_hz']:.0f} Hz")
        if s["ch1"] is not None:
            print(f"    CH1 range: [{s['ch1'].min():.4f}, {s['ch1'].max():.4f}] V")
        if s["ch2"] is not None:
            print(f"    CH2 range: [{s['ch2'].min():.4f}, {s['ch2'].max():.4f}] V")
        print(f"    quality_flags={s['quality_flags']}")

    print(f"\n{SEP}")
    print(f"Phase 2 ingestion complete.  {report['loaded']} signals ready for Phase 3.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()

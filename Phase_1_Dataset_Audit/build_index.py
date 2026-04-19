"""
Phase 1 — Dataset Audit
Scans all data folders, parses every CSV header, and writes dataset_index.csv.

Each row in the output represents one file and includes:
  file_path, filename, device_family, device_id, state, distance_label,
  probe_id, domain, channel_mode, channels, start_time, increment_sec,
  sample_count, is_duplicate, quality_flags
"""

import os
import re
import csv
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_index.csv")

# ── Folder → canonical metadata ───────────────────────────────────────────────
# Keys are relative paths from BASE_DIR (forward slashes, exact match).
FOLDER_META = {
    "Motor_Data/MOTOR_1_ON/time domain": {
        "device_family": "motor",   "device_id": "motor_1",   "state": "on",
        "distance_label": "unknown","probe_id": "default",    "domain": "time",
    },
    "Motor_Data/MOTOR_1_ON/freq domain": {
        "device_family": "motor",   "device_id": "motor_1",   "state": "on",
        "distance_label": "unknown","probe_id": "default",    "domain": "freq",
    },
    "Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe1/time domain (far ch1 close ch2)": {
        "device_family": "motor",   "device_id": "motor_2",   "state": "on",
        "distance_label": "ch1_far_ch2_close", "probe_id": "probe_1", "domain": "time",
    },
    "Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe1/freq": {
        "device_family": "motor",   "device_id": "motor_2",   "state": "on",
        "distance_label": "ch1_far_ch2_close", "probe_id": "probe_1", "domain": "freq",
    },
    "Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe2/time domain (distant ch1 close ch2)": {
        "device_family": "motor",   "device_id": "motor_2",   "state": "on",
        "distance_label": "ch1_distant_ch2_close", "probe_id": "probe_2", "domain": "time",
    },
    "Motor_Data/MOTOR_2_ON/MOTOR2_ON_Probe2/freq domain": {
        "device_family": "motor",   "device_id": "motor_2",   "state": "on",
        "distance_label": "ch1_distant_ch2_close", "probe_id": "probe_2", "domain": "freq",
    },
    "Motor_Data/MOTOR_3_ON/time domain": {
        "device_family": "motor",   "device_id": "motor_3",   "state": "on",
        "distance_label": "ch1_distant_ch2_close", "probe_id": "default", "domain": "time",
    },
    # Note: "freq doman" is a typo in the original folder name — preserved intentionally.
    "Motor_Data/MOTOR_3_ON/freq doman": {
        "device_family": "motor",   "device_id": "motor_3",   "state": "on",
        "distance_label": "ch1_distant_ch2_close", "probe_id": "default", "domain": "freq",
    },
    "Laptop_1/LAPTOP_CHRGR_1_OFF/time domain": {
        "device_family": "charger", "device_id": "charger_1", "state": "off",
        "distance_label": "close",  "probe_id": "default",    "domain": "time",
    },
    "Laptop_1/LAPTOP_CHRGR_1_ON_CLOSE/time domain": {
        "device_family": "charger", "device_id": "charger_1", "state": "on",
        "distance_label": "close",  "probe_id": "default",    "domain": "time",
    },
    "Laptop_2/LAPTOP_CHRGR_2_OFF/time domain": {
        "device_family": "charger", "device_id": "charger_2", "state": "off",
        "distance_label": "distant","probe_id": "default",    "domain": "time",
    },
    "Laptop_2/LAPTOP_CHRGR_2_OFF/freq domain": {
        "device_family": "charger", "device_id": "charger_2", "state": "off",
        "distance_label": "distant","probe_id": "default",    "domain": "freq",
    },
    "Laptop_2/LAPTOP_CHRGR_2_ON_DISTANT/time domain": {
        "device_family": "charger", "device_id": "charger_2", "state": "on",
        "distance_label": "distant","probe_id": "default",    "domain": "time",
    },
    "Laptop_2/LAPTOP_CHRGR_2_ON_DISTANT/freq domain": {
        "device_family": "charger", "device_id": "charger_2", "state": "on",
        "distance_label": "distant","probe_id": "default",    "domain": "freq",
    },
}

FIELDNAMES = [
    "file_path", "filename",
    "device_family", "device_id", "state", "distance_label", "probe_id", "domain",
    "channel_mode", "channels", "start_time", "increment_sec",
    "sample_count", "is_duplicate", "quality_flags",
]


def parse_csv_header(filepath):
    """
    Returns (channel_mode, channels, start_time, increment_sec, sample_count, flags).

    Expected layout:
      Row 0: X, [CH1,] [CH2,] Start, Increment, ...
      Row 1: Sequence, Volt, [Volt,] <start_value>, <increment_value>, ...
      Rows 2+: sample data
    """
    flags = []
    try:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return "unknown", "", None, None, 0, [f"read_error:{e}"]

    if len(lines) < 3:
        return "unknown", "", None, None, 0, ["too_few_rows"]

    header = [c.strip() for c in lines[0].split(",")]
    meta   = [c.strip() for c in lines[1].split(",")]

    # Detect channel columns
    channels = [c for c in header if c.startswith("CH")]
    if len(channels) == 0:
        flags.append("no_channel_column")
        channel_mode = "unknown"
    elif len(channels) == 1:
        channel_mode = f"single_{channels[0].lower()}"
    else:
        channel_mode = "dual"

    # Extract Start and Increment from meta row
    try:
        start_idx = header.index("Start")
        increment_idx = header.index("Increment")
        start_time    = float(meta[start_idx])
        increment_sec = float(meta[increment_idx])
    except (ValueError, IndexError):
        start_time    = None
        increment_sec = None
        flags.append("missing_timing_metadata")

    # Count data rows (skip blank / malformed)
    sample_count = 0
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[0] != "" and parts[1] != "":
            try:
                float(parts[1])
                sample_count += 1
            except ValueError:
                pass

    if sample_count == 0:
        flags.append("no_samples")

    return channel_mode, ",".join(channels), start_time, increment_sec, sample_count, flags


def is_duplicate_filename(filename):
    """Returns True if filename contains a parenthesised copy marker like (1)."""
    return bool(re.search(r"\(\d+\)", filename))


def build_index():
    rows = []
    unmapped_dirs = []

    for rel_folder, meta in FOLDER_META.items():
        abs_folder = os.path.join(BASE_DIR, *rel_folder.split("/"))
        if not os.path.isdir(abs_folder):
            print(f"  [WARN] folder not found: {rel_folder}")
            continue

        csv_files = sorted(glob.glob(os.path.join(abs_folder, "*.csv")))
        if not csv_files:
            print(f"  [WARN] no CSV files in: {rel_folder}")
            continue

        for fp in csv_files:
            fname = os.path.basename(fp)
            rel_path = os.path.relpath(fp, BASE_DIR)

            channel_mode, channels, start_time, increment_sec, sample_count, parse_flags = parse_csv_header(fp)

            is_dup = is_duplicate_filename(fname)
            if is_dup:
                parse_flags = ["duplicate"] + parse_flags

            rows.append({
                "file_path":     rel_path,
                "filename":      fname,
                "device_family": meta["device_family"],
                "device_id":     meta["device_id"],
                "state":         meta["state"],
                "distance_label":meta["distance_label"],
                "probe_id":      meta["probe_id"],
                "domain":        meta["domain"],
                "channel_mode":  channel_mode,
                "channels":      channels,
                "start_time":    start_time if start_time is not None else "",
                "increment_sec": increment_sec if increment_sec is not None else "",
                "sample_count":  sample_count,
                "is_duplicate":  is_dup,
                "quality_flags": "|".join(parse_flags) if parse_flags else "ok",
            })

    # Walk remaining CSV files not covered by FOLDER_META
    all_csvs = set(glob.glob(os.path.join(BASE_DIR, "**", "*.csv"), recursive=True))
    indexed  = {os.path.join(BASE_DIR, r["file_path"]) for r in rows}
    for fp in sorted(all_csvs - indexed):
        # Skip phase folders and GP folder
        rel = os.path.relpath(fp, BASE_DIR)
        if rel.startswith("Phase_") or rel.startswith("GP"):
            continue
        fname = os.path.basename(fp)
        channel_mode, channels, start_time, increment_sec, sample_count, parse_flags = parse_csv_header(fp)
        is_dup = is_duplicate_filename(fname)
        if is_dup:
            parse_flags = ["duplicate"] + parse_flags
        parse_flags = ["unmapped_folder"] + parse_flags
        rows.append({
            "file_path":      rel,
            "filename":       fname,
            "device_family":  "unknown",
            "device_id":      "unknown",
            "state":          "unknown",
            "distance_label": "unknown",
            "probe_id":       "unknown",
            "domain":         "unknown",
            "channel_mode":   channel_mode,
            "channels":       channels,
            "start_time":     start_time if start_time is not None else "",
            "increment_sec":  increment_sec if increment_sec is not None else "",
            "sample_count":   sample_count,
            "is_duplicate":   is_dup,
            "quality_flags":  "|".join(parse_flags),
        })

    # Write CSV
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_summary(rows):
    import collections

    total = len(rows)
    duplicates  = sum(1 for r in rows if r["is_duplicate"])
    ok          = sum(1 for r in rows if r["quality_flags"] == "ok")
    time_domain = sum(1 for r in rows if r["domain"] == "time")
    freq_domain = sum(1 for r in rows if r["domain"] == "freq")
    unmapped    = sum(1 for r in rows if "unmapped_folder" in r["quality_flags"])

    print("\n" + "="*60)
    print("DATASET INDEX SUMMARY")
    print("="*60)
    print(f"  Total files indexed : {total}")
    print(f"  Time-domain CSVs    : {time_domain}")
    print(f"  Freq-domain CSVs    : {freq_domain}")
    print(f"  Unmapped files      : {unmapped}")
    print(f"  Duplicate files     : {duplicates}")
    print(f"  Clean (no flags)    : {ok}")

    print("\n  Files per dataset condition:")
    by_condition = collections.Counter(
        f"{r['device_id']}_{r['state']}_{r['distance_label']}_{r['probe_id']}_{r['domain']}"
        for r in rows
    )
    for key, count in sorted(by_condition.items()):
        print(f"    {key:<55} {count:>4}")

    print("\n  Channel modes:")
    by_channel = collections.Counter(r["channel_mode"] for r in rows)
    for k, v in sorted(by_channel.items()):
        print(f"    {k:<30} {v:>4}")

    print("\n  Unique increment values (sampling rates):")
    increments = collections.Counter(r["increment_sec"] for r in rows if r["increment_sec"] != "")
    for inc, cnt in sorted(increments.items(), key=lambda x: float(x[0])):
        sr = 1.0 / float(inc)
        print(f"    increment={inc}  =>  {sr:.0f} Hz  ({cnt} files)")

    print(f"\n  Output written to: {OUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    print(f"Scanning data folders under: {BASE_DIR}")
    rows = build_index()
    print_summary(rows)

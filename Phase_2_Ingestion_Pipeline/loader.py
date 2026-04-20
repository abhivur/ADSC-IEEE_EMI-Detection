"""
Phase 2 — Data Extraction and Ingestion Pipeline

Public API
----------
    load_file(filepath, meta=None)           -> Signal | None
    load_folder(folder_path, ...)            -> list[Signal]
    load_dataset(index_csv, ...)             -> (list[Signal], ValidationReport)

A Signal is a plain dict:
    {
      "ch1"          : np.ndarray | None,   # CH1 voltage samples
      "ch2"          : np.ndarray | None,   # CH2 voltage samples
      "time"         : np.ndarray,          # time[i] = start + i * increment  (seconds)
      "file_path"    : str,
      "filename"     : str,
      "device_family": str,
      "device_id"    : str,
      "state"        : str,
      "distance_label": str,
      "probe_id"     : str,
      "channel_mode" : str,                 # "single_ch1" | "single_ch2" | "dual"
      "start_time"   : float,
      "increment_sec": float,
      "sample_rate_hz": float,
      "sample_count" : int,
      "is_duplicate" : bool,
      "quality_flags": str,
    }

A ValidationReport is a dict with load counts and per-file warning records.
"""

from __future__ import annotations

import os
import glob
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BASE_DIR, "Phase_1_Dataset_Audit", "dataset_index.csv")

# ── Header detection helpers ──────────────────────────────────────────────────

def _is_freq_domain_file(header_row0: str) -> bool:
    """True if the file is a frequency-domain export rather than a time-domain capture."""
    low = header_row0.lower()
    return "frequency" in low or "magnitude" in low or "_fft" in low


def _detect_channels(col_names: list[str]) -> list[str]:
    """Return the channel column names (e.g. ['CH1'] or ['CH1', 'CH2'])."""
    return [c for c in col_names if c.upper().startswith("CH") and "MAGNITUDE" not in c.upper()]


# ── Low-level CSV parser ──────────────────────────────────────────────────────

def _parse_csv(filepath: str):
    """
    Parse one oscilloscope CSV.

    Expected format
    ---------------
    Row 0  : X, [CH1,] [CH2,] Start, Increment, ...
    Row 1  : Sequence, Volt, [Volt,] <start>, <increment>, ...
    Rows 2+: sample_index, voltage [, voltage], ...

    Returns
    -------
    channels   : dict[str, np.ndarray]  e.g. {"CH1": array, "CH2": array}
    start_time : float
    increment  : float
    flags      : list[str]              quality issues found
    """
    flags = []

    try:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return {}, None, None, [f"read_error:{exc}"]

    if len(lines) < 3:
        return {}, None, None, ["too_few_rows"]

    col_names = [c.strip() for c in lines[0].split(",")]
    meta_vals = [c.strip() for c in lines[1].split(",")]

    # Reject frequency-domain files
    if _is_freq_domain_file(lines[0]):
        return {}, None, None, ["freq_domain_file"]

    channel_cols = _detect_channels(col_names)
    if not channel_cols:
        return {}, None, None, ["no_channel_column"]

    # Extract timing metadata
    try:
        start_idx = col_names.index("Start")
        incr_idx  = col_names.index("Increment")
        start_time = float(meta_vals[start_idx])
        increment  = float(meta_vals[incr_idx])
    except (ValueError, IndexError):
        start_time, increment = None, None
        flags.append("missing_timing_metadata")

    # Build per-channel voltage arrays
    ch_indices = {ch: col_names.index(ch) for ch in channel_cols}
    ch_data: dict[str, list] = {ch: [] for ch in channel_cols}

    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        for ch, idx in ch_indices.items():
            if idx < len(parts) and parts[idx] != "":
                try:
                    ch_data[ch].append(float(parts[idx]))
                except ValueError:
                    pass

    channels = {ch: np.array(vals) for ch, vals in ch_data.items()}

    # Sanity check: all channels should have equal length
    lengths = [len(v) for v in channels.values()]
    if len(set(lengths)) > 1:
        flags.append("channel_length_mismatch")

    if all(l == 0 for l in lengths):
        flags.append("no_samples")

    return channels, start_time, increment, flags


# ── Signal factory ────────────────────────────────────────────────────────────

_EMPTY_META = {
    "device_family": "unknown",
    "device_id":     "unknown",
    "state":         "unknown",
    "distance_label":"unknown",
    "probe_id":      "unknown",
    "is_duplicate":  False,
    "quality_flags": "",
}


def load_file(filepath: str, meta: dict | None = None) -> dict | None:
    """
    Load one time-domain CSV and return a Signal dict.

    Parameters
    ----------
    filepath : absolute or relative path to the CSV
    meta     : optional row from dataset_index.csv as a dict (auto-attaches labels)

    Returns None if the file cannot be parsed (e.g. freq-domain, malformed).
    A warning is issued for every skipped file.
    """
    abs_path = os.path.abspath(filepath)
    filename = os.path.basename(abs_path)

    channels, start_time, increment, flags = _parse_csv(abs_path)

    # Skip unloadable files
    skip_flags = {"freq_domain_file", "no_channel_column", "no_samples", "too_few_rows"}
    if any(f in skip_flags for f in flags):
        warnings.warn(f"Skipped {filename}: {', '.join(flags)}", stacklevel=2)
        return None

    channel_names = list(channels.keys())
    if len(channel_names) == 1:
        ch_name = channel_names[0].upper()
        channel_mode = f"single_{ch_name.lower()}"
        ch1 = channels[channel_names[0]] if ch_name == "CH1" else None
        ch2 = channels[channel_names[0]] if ch_name == "CH2" else None
    else:
        channel_mode = "dual"
        ch1 = channels.get("CH1")
        ch2 = channels.get("CH2")

    # Reconstruct time axis from whichever channel has data
    ref_len = len(ch1) if ch1 is not None else len(ch2) if ch2 is not None else 0
    if start_time is not None and increment is not None and ref_len > 0:
        time_axis = start_time + np.arange(ref_len) * increment
    else:
        time_axis = np.arange(ref_len, dtype=float)
        if "missing_timing_metadata" not in flags:
            flags.append("time_axis_estimated")

    sample_rate = (1.0 / increment) if increment else None
    m = _EMPTY_META.copy()
    if meta:
        m.update(meta)

    return {
        "ch1":           ch1,
        "ch2":           ch2,
        "time":          time_axis,
        "file_path":     os.path.relpath(abs_path, BASE_DIR),
        "filename":      filename,
        "device_family": m["device_family"],
        "device_id":     m["device_id"],
        "state":         m["state"],
        "distance_label":m["distance_label"],
        "probe_id":      m["probe_id"],
        "channel_mode":  channel_mode,
        "start_time":    start_time,
        "increment_sec": increment,
        "sample_rate_hz":sample_rate,
        "sample_count":  ref_len,
        "is_duplicate":  bool(m["is_duplicate"]),
        "quality_flags": "|".join(flags) if flags else "ok",
    }


# ── Folder loader ─────────────────────────────────────────────────────────────

def load_folder(
    folder_path: str,
    index_df: pd.DataFrame | None = None,
    skip_duplicates: bool = True,
) -> list[dict]:
    """
    Load all time-domain CSVs in a folder.

    Parameters
    ----------
    folder_path     : path to a data folder
    index_df        : dataset_index DataFrame for metadata lookup (optional)
    skip_duplicates : if True, skip files with '(1)' in the filename
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signals = []

    for fp in csv_files:
        fname = os.path.basename(fp)
        rel   = os.path.relpath(fp, BASE_DIR)

        # Metadata lookup from index
        meta = None
        if index_df is not None:
            rows = index_df[index_df["file_path"] == rel]
            if not rows.empty:
                meta = rows.iloc[0].to_dict()

        if skip_duplicates and meta and bool(meta.get("is_duplicate", False)):
            continue

        sig = load_file(fp, meta=meta)
        if sig is not None:
            signals.append(sig)

    return signals


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(
    index_csv: str = DEFAULT_INDEX,
    domain: str = "time",
    skip_duplicates: bool = True,
    skip_flagged: bool = False,
) -> tuple[list[dict], dict]:
    """
    Load all indexed files matching the given domain.

    Parameters
    ----------
    index_csv       : path to dataset_index.csv (Phase 1 output)
    domain          : "time" or "freq" (default "time")
    skip_duplicates : skip files marked as duplicate
    skip_flagged    : skip files whose quality_flags is not "ok"

    Returns
    -------
    signals : list of Signal dicts (successfully loaded)
    report  : ValidationReport dict
    """
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(f"dataset_index.csv not found at: {index_csv}")

    index_df = pd.read_csv(index_csv)
    subset   = index_df[index_df["domain"] == domain].copy()

    if skip_duplicates:
        subset = subset[~subset["is_duplicate"].astype(bool)]

    if skip_flagged:
        subset = subset[subset["quality_flags"] == "ok"]

    signals      = []
    skipped_freq = 0
    skipped_err  = 0
    warned       = []

    for _, row in subset.iterrows():
        fp = os.path.join(BASE_DIR, row["file_path"])
        if not os.path.isfile(fp):
            warned.append({"file": row["file_path"], "reason": "file_not_found"})
            skipped_err += 1
            continue

        meta = row.to_dict()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sig = load_file(fp, meta=meta)

        if sig is None:
            # Distinguish freq-domain skips from real errors
            reasons = [str(w.message) for w in caught]
            if any("freq_domain_file" in r for r in reasons):
                skipped_freq += 1
            else:
                skipped_err += 1
                warned.append({"file": row["file_path"], "reason": reasons})
        else:
            signals.append(sig)

    report = {
        "total_attempted":  len(subset),
        "loaded":           len(signals),
        "skipped_freq":     skipped_freq,
        "skipped_errors":   skipped_err,
        "warnings":         warned,
        "by_device":        _count_by(signals, "device_id"),
        "by_channel_mode":  _count_by(signals, "channel_mode"),
        "by_state":         _count_by(signals, "state"),
    }
    return signals, report


def _count_by(signals: list[dict], key: str) -> dict:
    counts: dict = {}
    for s in signals:
        k = s.get(key, "unknown")
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))

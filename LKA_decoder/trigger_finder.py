#!/usr/bin/env python3
"""
trigger_finder.py

Automatically find candidate LKA trigger CAN IDs and bytes from CAN bus data.

This version is tolerant of multiple CSV layouts, including:
- "Classic" layout: columns like Identifier, Timestamp, Byte1..Byte8
- "Arb.ID / unlabeled time / 64 bytes" layout (like your Test_1.csv):
    col0 = CAN ID (e.g., 0x160) with a different header name
    col1 = Time (seconds) with no header (often shows up as Unnamed: 1)
    remaining columns = Byte 1 .. Byte 64 (hex strings like 'EC', '0', etc.)

Usage:
    python trigger_finder.py --csv data.csv
    python trigger_finder.py --csv data.csv --min-events 2 --max-unique 10
"""

import argparse
import math
import os
import re
import shutil
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================
# Helpers
# =========================
def clear_directory(dir_path: str) -> None:
    """
    Delete everything inside dir_path (files + subfolders), but keep dir_path itself.
    Safe for re-running so each run produces a clean results folder.
    """
    if not os.path.exists(dir_path):
        return
    if not os.path.isdir(dir_path):
        raise ValueError(f"Outdir is not a directory: {dir_path}")

    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            else:
                shutil.rmtree(p)
        except Exception as e:
            print(f"Warning: could not delete {p}: {e}")

def hex_to_int(x):
    """Convert a hex-ish byte/ID cell to int; returns NaN for invalid values.

    Notes:
    - Your logs store bytes as hex strings (e.g., 'EC', '0', 'FF', sometimes '0x1A').
    - Some CSV readers may coerce values to floats (e.g., 0 -> 0.0). We treat X.0 as int(X).
    """
    if pd.isna(x):
        return np.nan

    # If already numeric, accept integer-like floats.
    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return np.nan
        if float(x).is_integer():
            return int(x)
        return np.nan

    s = str(x).strip()
    if s == "" or s.lower() == "errorframe":
        return np.nan

    # Handle strings like '0.0' if they slip through
    if re.fullmatch(r"\d+\.0+", s):
        return int(float(s))

    # common "0x" prefix
    s = s.replace("0x", "").replace("0X", "")

    # bytes are always hex-ish in your format; treat digit-only as hex too
    if not all(ch in "0123456789abcdefABCDEF" for ch in s):
        return np.nan

    try:
        return int(s, 16)
    except Exception:
        return np.nan


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so we can robustly detect id/time/byte columns.
    - Strips whitespace
    - Converts 'Byte 1' -> 'Byte1'
    - Converts 'Data0' -> 'Byte1', 'Data1' -> 'Byte2', etc. (CAN FD format)
    - Keeps 'Unnamed: x' columns as-is
    """
    rename = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        c0 = c
        c1 = c.strip()

        c1 = re.sub(r"\s+", " ", c1)

        # DataN → Byte(N+1) for CAN FD format
        m = re.match(r"^(?i:data)\s*(\d+)$", c1)
        if m:
            rename[c0] = f"Byte{int(m.group(1)) + 1}"
            continue

        m = re.match(r"^(?i:byte)\s*(\d+)$", c1)
        if m:
            rename[c0] = f"Byte{int(m.group(1))}"
            continue

        rename[c0] = c1

    return df.rename(columns=rename)


def _detect_id_time_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect CAN ID and Timestamp columns.
    Returns: (id_col, time_col)
    """
    cols = list(df.columns)

    # ID: prefer known names
    if "Identifier" in df.columns:
        id_col = "Identifier"
    else:
        # common variations
        candidates = []
        for c in cols:
            if not isinstance(c, str):
                continue
            cl = c.lower()
            if "arb" in cl and "id" in cl:
                candidates.append(c)
            elif cl in {"can_id", "canid", "id", "arb.id", "arb_id", "identifier"}:
                candidates.append(c)
        id_col = candidates[0] if candidates else cols[0]

    # TIME: prefer known names
    if "Timestamp" in df.columns:
        time_col = "Timestamp"
    elif "Time" in df.columns:
        time_col = "Time"
    else:
        # If there's an Unnamed column in position 1, it's often the unlabeled time
        if len(cols) > 1 and isinstance(cols[1], str) and cols[1].lower().startswith("unnamed"):
            time_col = cols[1]
        elif len(cols) > 1:
            time_col = cols[1]
        else:
            # degenerate case; fall back to first
            time_col = cols[0]

    return id_col, time_col


def _detect_byte_cols(df: pd.DataFrame, expected_n_bytes: Optional[int] = None) -> List[str]:
    """
    Detect byte columns (Byte1..ByteN). Returns sorted list.
    If expected_n_bytes is provided, ensures Byte1..Byte{N} exist (adds missing with NaN).
    """
    byte_cols = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        m = re.match(r"^Byte(\d+)$", c)
        if m:
            byte_cols.append(c)

    # Sort by byte index
    def _byte_idx(name: str) -> int:
        m = re.match(r"^Byte(\d+)$", name)
        return int(m.group(1)) if m else 10**9

    byte_cols = sorted(byte_cols, key=_byte_idx)

    if expected_n_bytes is not None:
        # Create any missing Byte{i} columns so downstream loops are consistent
        for i in range(1, expected_n_bytes + 1):
            col = f"Byte{i}"
            if col not in df.columns:
                df[col] = np.nan
        byte_cols = [f"Byte{i}" for i in range(1, expected_n_bytes + 1)]

    return byte_cols


def _find_header_row(csv_path: str, max_scan: int = 15) -> int:
    """Scan first N lines to find the row containing actual column headers."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            if "Time" in line and "Data0" in line:
                return i
    return 0


def load_and_clean_csv(csv_path: str, expected_n_bytes: Optional[int] = None) -> pd.DataFrame:
    """Load and clean CAN CSV data."""
    header_row = _find_header_row(csv_path)
    df = pd.read_csv(csv_path, skiprows=header_row, dtype=str, low_memory=False)

    df = _normalize_columns(df)
    id_col, time_col = _detect_id_time_cols(df)

    # Detect bytes (and optionally force a fixed count)
    byte_cols = _detect_byte_cols(df, expected_n_bytes=expected_n_bytes)

    print(f"Loaded: {csv_path}")
    print(f"Rows (raw): {len(df)}")
    print(f"Detected ID column: {id_col!r}")
    print(f"Detected time column: {time_col!r}")
    print(f"Detected bytes: {len(byte_cols)} columns")

    # Parse CAN ID
    df["CAN_ID"] = df[id_col].apply(hex_to_int)

    # Parse bytes
    for c in byte_cols:
        df[c] = df[c].apply(hex_to_int).astype(float)

    # Parse timestamp
    df["Timestamp"] = pd.to_numeric(df[time_col], errors="coerce")

    # Clean
    df = (
        df.dropna(subset=["Timestamp", "CAN_ID"])
          .sort_values("Timestamp")
          .reset_index(drop=True)
    )

    # Attach detected bytes for downstream functions
    df.attrs["byte_cols"] = byte_cols

    print(f"Rows after cleaning: {len(df)}")
    print(f"Unique CAN IDs: {df['CAN_ID'].nunique()}")
    print(f"Time range: {df['Timestamp'].min():.4f}s - {df['Timestamp'].max():.4f}s")
    print()

    return df


@dataclass
class TriggerCandidate:
    can_id: int
    byte_num: int
    unique_values: List[int]
    dominant_value: int
    dominant_pct: float
    active_values: List[int]
    state_change_count: int
    event_count: int
    median_event_gap: float
    score: float
    trigger_match_rate: float = 0.0
    bit_num: Optional[int] = None
    analysis_type: str = "byte"

    def __str__(self):
        active_hex = [hex(v) for v in self.active_values]
        gap_str = "n/a" if math.isnan(self.median_event_gap) else f"{self.median_event_gap:.2f}s"
        match_str = f", match={self.trigger_match_rate:.0%}" if self.trigger_match_rate > 0 else ""
        if self.bit_num is not None:
            label = f"CAN {hex(self.can_id)} Byte{self.byte_num} Bit{self.bit_num}"
        else:
            label = f"CAN {hex(self.can_id)} Byte{self.byte_num}"
        return (
            f"{label}: "
            f"idle={hex(self.dominant_value)} ({self.dominant_pct:.1f}%), "
            f"active={active_hex}, "
            f"changes={self.state_change_count}, "
            f"events={self.event_count}, "
            f"gap~{gap_str}, "
            f"score={self.score:.1f}"
            f"{match_str}"
        )



def analyze_byte_for_triggers(
    timestamps: np.ndarray,
    values: np.ndarray,
    can_id: int,
    byte_num: int,
    min_events: int = 2,
    max_events: Optional[int] = None,
    min_unique: int = 2,
    max_unique: int = 15,
    min_dominant_pct: float = 50.0,
    test_start: Optional[float] = None,
    test_end: Optional[float] = None,
    edge_seconds: float = 5.0,
    intertrigger_gap: Optional[float] = None,
    intertrigger_tol: float = 0.5,
    known_trigger_times: Optional[List[float]] = None,
    trigger_window: float = 3.0,
    **kwargs,
) -> Optional[TriggerCandidate]:
    """
    Analyze a single byte stream to see if it looks like a trigger signal.
    """
    # Remove NaN values
    mask = ~np.isnan(values)
    values = values[mask].astype(int)
    timestamps = timestamps[mask]

    if len(values) < 100:
        return None

    unique_vals, counts = np.unique(values, return_counts=True)

    # If you know the trigger has a small number of discrete states,
    # you can require a minimum number of unique values.
    if len(unique_vals) < min_unique:
        return None

    if len(unique_vals) > max_unique:
        return None

    if len(unique_vals) < 2:
        return None

    dominant_idx = np.argmax(counts)
    dominant_value = unique_vals[dominant_idx]
    dominant_pct = 100.0 * float(counts[dominant_idx]) / len(values)

    if dominant_pct < min_dominant_pct:
        return None

    active_values = [int(v) for v in unique_vals if v != dominant_value]

    # -------------------------
    # 1) Count ALL state changes
    # -------------------------
    prev_for_changes = np.roll(values, 1)
    prev_for_changes[0] = values[0]  # don't count an artificial "first sample" change
    change_mask = (values != prev_for_changes)
    change_mask[0] = False
    state_changes = int(np.sum(change_mask))

    if state_changes < min_events:
        return None

    if max_events is not None and state_changes > max_events:
        return None

    # -----------------------------------------
    # 2) Event starts: idle -> non-idle
    # -----------------------------------------
    prev_for_events = np.roll(values, 1)
    prev_for_events[0] = dominant_value  # treat first sample as if it came from idle
    event_start_mask = (prev_for_events == dominant_value) & (values != dominant_value)
    event_start_times = timestamps[event_start_mask]
    event_count = int(len(event_start_times))

    # If there's never an idle->non-idle transition, it's probably not a trigger-style signal.
    if event_count == 0:
        return None

    # Median gap between events (for info/debug)
    if event_count >= 2:
        median_gap = float(np.median(np.diff(event_start_times)))
    else:
        median_gap = float("nan")

    # -----------------------------------------
    # 3) Exclude candidates where ALL events
    #    occur within the first OR last N seconds
    # -----------------------------------------
    if edge_seconds and edge_seconds > 0 and test_start is not None and test_end is not None:
        duration = float(test_end) - float(test_start)
        if duration > 2.0 * edge_seconds:
            if np.all(event_start_times <= float(test_start) + edge_seconds):
                return None
            if np.all(event_start_times >= float(test_end) - edge_seconds):
                return None

    # -----------------------------------------
    # 4) Optional: require a consistent inter-trigger gap
    # -----------------------------------------
    if intertrigger_gap is not None:
        if event_count < 2:
            return None
        gaps = np.diff(event_start_times)
        
        # MINIMUM allowed gap (seconds).
        min_allowed = float(intertrigger_gap)  # strict
        if np.any(gaps < min_allowed):
            return None

    # -----------------------------------------
    # 5) Optional: filter by known trigger times
    # -----------------------------------------
    trigger_match_rate = 0.0
    if known_trigger_times is not None and len(known_trigger_times) > 0:
        # For each known trigger time, check if any state change occurs within ±trigger_window
        change_times = timestamps[change_mask]
        matched = 0
        for kt in known_trigger_times:
            if np.any(np.abs(change_times - kt) <= trigger_window):
                matched += 1
        trigger_match_rate = matched / len(known_trigger_times)

        # Require minimum match rate (configurable via min_trigger_match_rate param)
        min_rate = kwargs.get("min_trigger_match_rate", 1.0)
        if trigger_match_rate < min_rate:
            return None

    # Score: favor (reasonable) number of state changes + strong idle dominance + fewer unique states
    score = (
        min(state_changes, 40) * 5
        + dominant_pct * 0.5
        - len(unique_vals) * 2
    )

    if known_trigger_times is not None and len(known_trigger_times) > 0:
        score += trigger_match_rate * 200
        if trigger_match_rate >= 1.0:
            score += 50

    return TriggerCandidate(
        can_id=int(can_id),
        byte_num=int(byte_num),
        unique_values=[int(v) for v in unique_vals],
        dominant_value=int(dominant_value),
        dominant_pct=float(dominant_pct),
        active_values=active_values,
        state_change_count=int(state_changes),
        event_count=int(event_count),
        median_event_gap=float(median_gap),
        score=float(score),
        trigger_match_rate=float(trigger_match_rate),
    )


def analyze_bit_for_triggers(
    timestamps: np.ndarray,
    byte_values: np.ndarray,
    can_id: int,
    byte_num: int,
    bit_num: int,
    min_events: int = 2,
    max_events: Optional[int] = None,
    test_start: Optional[float] = None,
    test_end: Optional[float] = None,
    edge_seconds: float = 5.0,
    intertrigger_gap: Optional[float] = None,
    intertrigger_tol: float = 0.5,
    known_trigger_times: Optional[List[float]] = None,
    trigger_window: float = 3.0,
    **kwargs,
) -> Optional[TriggerCandidate]:
    """
    Extract a single bit from byte_values and analyze as a binary trigger.
    bit_num: 0 (LSB) to 7 (MSB).
    """
    mask = ~np.isnan(byte_values)
    byte_clean = byte_values[mask].astype(int)
    ts_clean = timestamps[mask]

    if len(byte_clean) < 100:
        return None

    # Extract bit: 0 or 1
    bit_values = ((byte_clean >> bit_num) & 1).astype(float)

    # Must have both 0s and 1s
    if len(np.unique(bit_values[~np.isnan(bit_values)].astype(int))) < 2:
        return None

    candidate = analyze_byte_for_triggers(
        timestamps=ts_clean,
        values=bit_values,
        can_id=can_id,
        byte_num=byte_num,
        min_events=min_events,
        max_events=max_events,
        min_unique=2,
        max_unique=2,
        min_dominant_pct=50.0,
        test_start=test_start,
        test_end=test_end,
        edge_seconds=edge_seconds,
        intertrigger_gap=intertrigger_gap,
        intertrigger_tol=intertrigger_tol,
        known_trigger_times=known_trigger_times,
        trigger_window=trigger_window,
        **kwargs,
    )

    if candidate is None:
        return None

    candidate.bit_num = bit_num
    candidate.analysis_type = "bit"

    return candidate


def _get_byte_cols(df: pd.DataFrame) -> List[str]:
    byte_cols = df.attrs.get("byte_cols")
    if not byte_cols:
        # fallback: discover from columns
        byte_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"^Byte\d+$", c)]
        byte_cols = sorted(byte_cols, key=lambda c: int(re.match(r"^Byte(\d+)$", c).group(1)))
    return byte_cols


def find_trigger_candidates(
    df: pd.DataFrame,
    min_events: int = 2,
    max_events: Optional[int] = None,
    min_unique: int = 2,
    max_unique: Optional[int] = None,
    min_dominant_pct: float = 50.0,
    edge_seconds: Optional[float] = None,
    intertrigger_gap: Optional[float] = None,
    intertrigger_tol: float = 0.5,
    top_n: int = 20,
    known_trigger_times: Optional[List[float]] = None,
    trigger_window: float = 3.0,
    enable_bit_analysis: bool = True,
    min_trigger_match_rate: float = 1.0,
) -> List[TriggerCandidate]:
    """
    Scan all CAN IDs and bytes to find potential trigger signals.
    Includes bit-level analysis for CAN FD data.
    """
    candidates: List[TriggerCandidate] = []
    byte_cols = _get_byte_cols(df)
    n_bytes = len(byte_cols)

    # Adaptive max_unique: CAN 2.0 (<=8 bytes) -> 15, CAN FD -> 256
    if max_unique is None:
        max_unique = 15 if n_bytes <= 8 else 256

    # Global test bounds (used by edge filter)
    test_start = float(df["Timestamp"].min())
    test_end = float(df["Timestamp"].max())
    duration = test_end - test_start

    # Adaptive edge_seconds: 5% of test duration, clamped to [1.0, 10.0]
    if edge_seconds is None:
        edge_seconds = max(1.0, min(duration * 0.05, 10.0))

    can_ids = df["CAN_ID"].unique()
    bit_note = " + bit-level" if enable_bit_analysis else ""
    print(f"Analyzing {len(can_ids)} CAN IDs x {n_bytes} bytes = {len(can_ids) * n_bytes} signals{bit_note}...")
    print(f"  max_unique={max_unique}, edge_seconds={edge_seconds:.1f}s (test duration={duration:.1f}s)")

    extra_kwargs = {"min_trigger_match_rate": min_trigger_match_rate}

    for can_id in can_ids:
        can_id = int(can_id)
        g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
        if len(g) < 100:
            continue

        timestamps = g["Timestamp"].values

        for byte_col in byte_cols:
            m = re.match(r"^Byte(\d+)$", byte_col)
            byte_num = int(m.group(1)) if m else -1
            values = g[byte_col].values

            # --- Byte-level analysis ---
            candidate = analyze_byte_for_triggers(
                timestamps=timestamps,
                values=values,
                can_id=can_id,
                byte_num=byte_num,
                min_events=min_events,
                max_events=max_events,
                min_unique=min_unique,
                max_unique=max_unique,
                min_dominant_pct=min_dominant_pct,
                test_start=test_start,
                test_end=test_end,
                edge_seconds=edge_seconds,
                intertrigger_gap=intertrigger_gap,
                intertrigger_tol=intertrigger_tol,
                known_trigger_times=known_trigger_times,
                trigger_window=trigger_window,
                **extra_kwargs,
            )
            if candidate is not None:
                candidates.append(candidate)

            # --- Bit-level analysis ---
            if enable_bit_analysis:
                clean = values[~np.isnan(values)]
                if len(clean) < 100:
                    continue
                n_unique = len(np.unique(clean.astype(int)))
                if n_unique < 2:
                    continue

                # Run bit analysis when trigger times are known OR byte-level was too noisy
                run_bits = (known_trigger_times is not None) or (n_unique > max_unique)
                if run_bits:
                    for bit in range(8):
                        bit_cand = analyze_bit_for_triggers(
                            timestamps=timestamps,
                            byte_values=values,
                            can_id=can_id,
                            byte_num=byte_num,
                            bit_num=bit,
                            min_events=min_events,
                            max_events=max_events,
                            test_start=test_start,
                            test_end=test_end,
                            edge_seconds=edge_seconds,
                            intertrigger_gap=intertrigger_gap,
                            intertrigger_tol=intertrigger_tol,
                            known_trigger_times=known_trigger_times,
                            trigger_window=trigger_window,
                            **extra_kwargs,
                        )
                        if bit_cand is not None:
                            candidates.append(bit_cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def _extract_candidate_values(df: pd.DataFrame, candidate: TriggerCandidate):
    """Extract timestamps and values for a candidate (byte-level or bit-level)."""
    can_id = candidate.can_id
    byte_num = candidate.byte_num
    byte_col = f"Byte{byte_num}"

    g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
    raw_values = g[byte_col].values
    timestamps = g["Timestamp"].values

    mask = ~np.isnan(raw_values)
    byte_clean = raw_values[mask].astype(int)
    ts_clean = timestamps[mask]

    if candidate.bit_num is not None:
        values = (byte_clean >> candidate.bit_num) & 1
    else:
        values = byte_clean

    return ts_clean, values


def print_candidate_details(df: pd.DataFrame, candidate: TriggerCandidate):
    """Print detailed analysis of a trigger candidate."""
    can_id = candidate.can_id
    byte_num = candidate.byte_num

    timestamps, values = _extract_candidate_values(df, candidate)

    print(f"\n{'='*60}")
    if candidate.bit_num is not None:
        print(f"CAN ID: {hex(can_id)} | Byte: {byte_num} | Bit: {candidate.bit_num} (bit-level)")
    else:
        print(f"CAN ID: {hex(can_id)} | Byte: {byte_num}")
    print(f"{'='*60}")

    print(f"\nValue Distribution:")
    unique_vals, counts = np.unique(values, return_counts=True)

    for val, cnt in sorted(zip(unique_vals, counts), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(values)
        marker = " <-- IDLE" if val == candidate.dominant_value else (" <-- ACTIVE" if val in candidate.active_values else "")
        print(f"  {hex(val):>6}: {cnt:6d} ({pct:5.1f}%){marker}")

    print(f"\nTransition Events (into active states):")
    prev_values = np.roll(values, 1)
    prev_values[0] = values[0]

    is_active = np.isin(values, candidate.active_values)
    was_not_active = ~np.isin(prev_values, candidate.active_values)
    transition_mask = is_active & was_not_active

    transition_times = timestamps[transition_mask]
    transition_vals = values[transition_mask]

    for i, (t, v) in enumerate(zip(transition_times[:15], transition_vals[:15])):
        print(f"  {i+1:2d}. t={t:10.4f}s -> {hex(v)}")

    if len(transition_times) > 15:
        print(f"  ... and {len(transition_times) - 15} more")

    if candidate.trigger_match_rate > 0:
        print(f"\nTrigger Match Rate: {candidate.trigger_match_rate:.0%}")

    print(f"\nSuggested usage:")
    if candidate.bit_num is not None:
        # For bit-level: user still needs to use byte + value, but we show the bit info
        print(f"  (Bit-level trigger: Byte{byte_num} Bit{candidate.bit_num})")
        print(f"  --lka-can-id {hex(can_id)} --trigger-byte {byte_num} --trigger-value {hex(candidate.active_values[0])}")
    elif len(candidate.active_values) == 1:
        print(f"  --lka-can-id {hex(can_id)} --trigger-byte {byte_num} --trigger-value {hex(candidate.active_values[0])}")
    else:
        active_hex = " ".join(hex(v) for v in candidate.active_values)
        print(f"  --lka-can-id {hex(can_id)} --trigger-byte {byte_num} --trigger-value {active_hex}")


def plot_trigger_candidate(
    df: pd.DataFrame,
    candidate: TriggerCandidate,
    outpath: Optional[str] = None,
    fig_size: Tuple[int, int] = (16, 4),
):
    """
    Plot the trigger candidate signal over time.
    Shows the byte/bit value with idle and active regions highlighted.
    """
    can_id = candidate.can_id
    byte_num = candidate.byte_num

    timestamps, values = _extract_candidate_values(df, candidate)

    if len(values) < 10:
        label = str(candidate).split(":")[0]
        print(f"Not enough data to plot {label}")
        return

    if candidate.bit_num is not None:
        signal_label = f"{hex(can_id)} Byte{byte_num} Bit{candidate.bit_num}"
        y_label = "Bit Value"
    else:
        signal_label = f"{hex(can_id)} Byte{byte_num}"
        y_label = "Byte Value"

    fig, ax = plt.subplots(figsize=fig_size)

    ax.step(timestamps, values, where="post", linewidth=1.2, color="blue", label=signal_label)

    prev_values = np.roll(values, 1)
    prev_values[0] = values[0]
    is_active = np.isin(values, candidate.active_values)
    was_not_active = ~np.isin(prev_values, candidate.active_values)
    transition_mask = is_active & was_not_active
    transition_times = timestamps[transition_mask]

    for t in transition_times:
        ax.axvline(t, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.axhline(candidate.dominant_value, color="green", linestyle=":", linewidth=1.0, alpha=0.5,
               label=f"Idle ({hex(candidate.dominant_value)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)

    active_hex = ", ".join(hex(v) for v in candidate.active_values)
    ax.set_title(
        f"{signal_label} | Idle={hex(candidate.dominant_value)} ({candidate.dominant_pct:.1f}%) "
        f"| Active={active_hex} | changes={candidate.state_change_count}, events={candidate.event_count}"
    )

    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=4)
    ax.tick_params(axis="x", which="major", length=7)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_can_id_all_bytes(
    df: pd.DataFrame,
    can_id: int,
    outpath: Optional[str] = None,
    ncols: int = 8,
    fig_size: Optional[Tuple[int, int]] = None,
):
    """
    Plot all bytes of a CAN ID in a single figure with subplots.
    For 64-byte logs, this will produce an 8x8 grid by default.
    """
    byte_cols = _get_byte_cols(df)

    g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
    if len(g) < 10:
        print(f"Not enough data to plot {hex(can_id)}")
        return

    timestamps = g["Timestamp"].values

    nbytes = len(byte_cols)
    nrows = int(math.ceil(nbytes / ncols))

    if fig_size is None:
        # heuristic: make each subplot ~3"x2" (clamped a bit)
        w = max(16, min(3 * ncols, 36))
        h = max(8, min(2 * nrows, 40))
        fig_size = (w, h)

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, byte_col in enumerate(byte_cols):
        ax = axes[i]
        m = re.match(r"^Byte(\d+)$", byte_col)
        byte_num = int(m.group(1)) if m else i + 1

        values = g[byte_col].values

        mask = ~np.isnan(values)
        plot_timestamps = timestamps[mask]
        plot_values = values[mask].astype(int)

        if len(plot_values) < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Byte{byte_num}", fontsize=9)
            ax.grid(True, alpha=0.3)
            continue

        unique_vals, counts = np.unique(plot_values, return_counts=True)
        dominant_val = unique_vals[np.argmax(counts)]
        dominant_pct = 100 * np.max(counts) / len(plot_values)

        ax.step(plot_timestamps, plot_values, where="post", linewidth=0.9, color="blue")

        prev = np.roll(plot_values, 1)
        prev[0] = dominant_val
        trans_mask = plot_values != prev
        trans_count = int(np.sum(trans_mask))

        for t in plot_timestamps[trans_mask]:
            ax.axvline(t, color="red", linestyle="--", linewidth=0.7, alpha=0.5)

        ax.axhline(dominant_val, color="green", linestyle=":", linewidth=0.9, alpha=0.4)

        unique_str = ", ".join(hex(v) for v in sorted(unique_vals)[:4])
        if len(unique_vals) > 4:
            unique_str += ", ..."

        ax.set_title(
            f"Byte{byte_num}: idle={hex(int(dominant_val))} ({dominant_pct:.0f}%), {trans_count} trans, vals=[{unique_str}]",
            fontsize=8
        )
        ax.grid(True, alpha=0.3)

    # Turn off unused axes
    for j in range(nbytes, len(axes)):
        axes[j].axis("off")

    # Set x-axis ticks: dash every 1s, number every 5s
    for ax in axes[:nbytes]:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=4)
        ax.tick_params(axis="x", which="major", length=7)

    fig.suptitle(f"CAN ID {hex(can_id)} - All Bytes", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def save_candidates_csv(candidates: List[TriggerCandidate], outpath: str):
    """Save candidates to CSV file."""
    rows = []
    for c in candidates:
        rows.append({
            "can_id_hex": hex(c.can_id),
            "can_id_dec": c.can_id,
            "byte_num": c.byte_num,
            "bit_num": c.bit_num,
            "analysis_type": c.analysis_type,
            "dominant_value_hex": hex(c.dominant_value),
            "dominant_pct": round(c.dominant_pct, 1),
            "active_values_hex": " ".join(hex(v) for v in c.active_values),
            "num_unique": len(c.unique_values),
            "state_change_count": c.state_change_count,
            "event_count": c.event_count,
            "median_event_gap": (None if math.isnan(c.median_event_gap) else round(c.median_event_gap, 3)),
            "trigger_match_rate": round(c.trigger_match_rate, 2) if c.trigger_match_rate > 0 else None,
            "score": round(c.score, 1),
        })

    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"\nSaved candidates to: {outpath}")


# =========================
# Main
# =========================
def parse_hex_arg(val: str) -> int:
    """Parse a hex string argument (e.g., '0x412', '412') to int."""
    val = val.strip().lower()
    if val.startswith("0x"):
        return int(val, 16)
    return int(val, 16)


def main():
    parser = argparse.ArgumentParser(
        description="Find candidate LKA trigger CAN IDs and bytes from CAN bus data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trigger_finder.py --csv data.csv
  python trigger_finder.py --csv data.csv --min-events 2 --max-events 4 --min-unique 4 --max-unique 6
  python trigger_finder.py --csv data.csv --details 5 --plot 5
  python trigger_finder.py --csv data.csv --force-plot 0x412
        """,
    )
    parser.add_argument("--csv", required=True, help="Path to CAN CSV log file")
    parser.add_argument("--min-events", type=int, default=2,
                        help="Minimum number of state changes (any value change) (default: 2)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of state changes allowed (optional; e.g. 4)")
    parser.add_argument("--min-unique", type=int, default=2,
                        help="Minimum unique values per byte (default: 2; use 4-5 for multi-state triggers)")
    parser.add_argument("--max-unique", type=int, default=None,
                        help="Maximum unique values per byte (default: auto -- 15 for CAN 2.0, 256 for CAN FD)")
    parser.add_argument("--min-dominant-pct", type=float, default=50.0,
                        help="Minimum percentage for dominant/idle value (default: 50)")
    parser.add_argument("--edge-seconds", type=float, default=None,
                    help="Exclude candidates where ALL trigger events occur within the first OR last N seconds (default: auto -- 5%% of test duration). Set 0 to disable.")
    parser.add_argument("--intertrigger-gap", type=float, default=None,
                        help="If set, require the time between consecutive trigger events to be ~this value (seconds). Example: --intertrigger-gap 3")
    parser.add_argument("--intertrigger-tol", type=float, default=0.5,
                        help="Tolerance for --intertrigger-gap in seconds (default: 0.5).")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top candidates to show (default: 20)")
    parser.add_argument("--details", type=int, default=5,
                        help="Number of candidates to show detailed analysis for (default: 5)")
    parser.add_argument("--plot", type=int, default=5,
                        help="Number of top candidates to plot (default: 5)")
    parser.add_argument("--force-plot", type=str, nargs="+", default=None,
                        help="Force plot specific CAN IDs (hex, e.g., 0x412 or '0x412 0xD5')")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots (default: trigger_finder_results/<csv_name>)")
    parser.add_argument("--outcsv", type=str, default=None,
                        help="Path to save candidates CSV (optional)")
    parser.add_argument("--trigger-times", type=float, nargs="+", default=None,
                        help="Known trigger timestamps in seconds (e.g., --trigger-times 19 37 55). "
                             "Only candidates with state changes near these times will be kept.")
    parser.add_argument("--trigger-window", type=float, default=3.0,
                        help="Window (±seconds) around each --trigger-times to look for state changes (default: 3.0)")
    parser.add_argument("--expected-bytes", type=int, default=None,
                        help="If set, forces Byte1..ByteN to exist (useful for 64-byte logs).")
    parser.add_argument("--no-bit-analysis", action="store_true", default=False,
                        help="Disable bit-level trigger analysis (faster but may miss bit-encoded triggers)")
    parser.add_argument("--min-trigger-match", type=float, default=1.0,
                        help="Minimum fraction of --trigger-times that must match state changes (default: 1.0 = all must match)")

    args = parser.parse_args()

    # Load data
    df = load_and_clean_csv(args.csv, expected_n_bytes=args.expected_bytes)

    # Find candidates
    if args.trigger_times:
        print(f"Known trigger times: {args.trigger_times}")
        print(f"Trigger window: ±{args.trigger_window}s")
        print()

    candidates = find_trigger_candidates(
        df=df,
        min_events=args.min_events,
        max_events=args.max_events,
        min_unique=args.min_unique,
        max_unique=args.max_unique,
        min_dominant_pct=args.min_dominant_pct,
        edge_seconds=args.edge_seconds,
        intertrigger_gap=args.intertrigger_gap,
        intertrigger_tol=args.intertrigger_tol,
        top_n=args.top,
        known_trigger_times=args.trigger_times,
        trigger_window=args.trigger_window,
        enable_bit_analysis=not args.no_bit_analysis,
        min_trigger_match_rate=args.min_trigger_match,
    )

    if not candidates:
        print("No trigger candidates found with current settings.")
        print("Try lowering --min-events or --min-dominant-pct")
        return

    print(f"\n{'='*60}")
    print(f"TOP {len(candidates)} TRIGGER CANDIDATES")
    print(f"{'='*60}")
    for i, c in enumerate(candidates):
        print(f"{i+1:2d}. {c}")

    print(f"\n\nDETAILED ANALYSIS (top {min(args.details, len(candidates))}):")
    for c in candidates[:args.details]:
        print_candidate_details(df, c)

    # Setup output directory for plots
    if args.outdir:
        plot_outdir = args.outdir
    else:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        plot_outdir = os.path.join("trigger_finder_results", base)

    os.makedirs(plot_outdir, exist_ok=True)
    clear_directory(plot_outdir)

    # Plot top candidates
    if args.plot > 0:
        print(f"\n\nPLOTTING top {min(args.plot, len(candidates))} candidates to: {plot_outdir}")
        for i, c in enumerate(candidates[:args.plot]):
            outpath = os.path.join(plot_outdir, f"candidate_{i+1:02d}_{c.can_id:03x}_byte{c.byte_num}.png")
            plot_trigger_candidate(df, c, outpath=outpath)
            print(f"  Saved: {outpath}")

    # Force plot specific CAN IDs (all bytes)
    if args.force_plot:
        print(f"\n\nFORCE PLOTTING specified CAN IDs:")
        for can_id_str in args.force_plot:
            can_id = parse_hex_arg(can_id_str)
            print(f"\n  Plotting all bytes for {hex(can_id)}...")
            outpath = os.path.join(plot_outdir, f"force_{can_id:03x}_all_bytes.png")
            plot_can_id_all_bytes(df, can_id, outpath=outpath)
            print(f"    Saved: {outpath}")

    # Save to CSV
    if args.outcsv:
        save_candidates_csv(candidates, args.outcsv)
    else:
        csv_out = os.path.join(plot_outdir, "trigger_candidates.csv")
        save_candidates_csv(candidates, csv_out)


if __name__ == "__main__":
    main()
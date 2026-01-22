# on_event_sampling.py
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


# Default list (you can override when calling functions)
DEFAULT_TARGET_IDS = [0x126, 0x220]

BYTE_COLS = [f"Byte{i}" for i in range(1, 9)]


# =========================
# Basic helpers
# =========================
def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    """
    Convert unsigned 16-bit values into signed int16 values.
    """
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


def parse_byte_pair(byte_pair_str: str) -> int:
    """
    "(Byte1,Byte2)" -> 0
    "(Byte3,Byte4)" -> 2
    """
    b1 = int(byte_pair_str.split("Byte")[1].split(",")[0])  # 1-based
    return b1 - 1  # convert to 0-based offset


def compute_u16_series_from_group(g: pd.DataFrame, offset: int, endian: str) -> np.ndarray:
    """
    Vectorized u16 extraction from Byte[offset] and Byte[offset+1].
    Returns numpy array length == len(g), with NaNs preserved.
    """
    b0 = g[BYTE_COLS[offset]].to_numpy(dtype=np.float64)
    b1 = g[BYTE_COLS[offset + 1]].to_numpy(dtype=np.float64)

    out = np.full(len(g), np.nan, dtype=np.float64)

    mask = (~np.isnan(b0)) & (~np.isnan(b1))
    if not np.any(mask):
        return out

    b0i = (b0[mask].astype(np.int64) & 0xFF)
    b1i = (b1[mask].astype(np.int64) & 0xFF)

    if endian == "little":
        out[mask] = (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 8) | b1i

    return out.astype(np.float64)


def u16_series_for_candidate(df: pd.DataFrame, can_id_dec: int, byte_pair_str: str, endian: str) -> Optional[pd.DataFrame]:
    """
    Returns Timestamp + unsigned u16 for a CAN ID decode config.
    """
    offset = parse_byte_pair(byte_pair_str)

    g = df[df["CAN_ID"] == int(can_id_dec)].copy().sort_values("Timestamp")
    if g.empty:
        return None

    u16_vals = compute_u16_series_from_group(g, offset, endian)
    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "u16": u16_vals}).dropna()

    return out.reset_index(drop=True) if len(out) > 10 else None


# =========================
# Sampling functions
# =========================
def signed_i16_series_for_id(df: pd.DataFrame, can_id_dec: int, byte_pair_str: str, endian: str) -> Optional[pd.DataFrame]:
    """
    Returns Timestamp + signed i16 series for a chosen CAN decode config.
    """
    s = u16_series_for_candidate(df, int(can_id_dec), byte_pair_str, endian)
    if s is None or len(s) < 10:
        return None

    s = s.copy()
    s["i16"] = u16_to_i16(s["u16"].values)
    return s[["Timestamp", "i16"]].dropna().reset_index(drop=True)


def exact_or_nearest_i16_at_times(series_df: pd.DataFrame, times: List[float]):
    """
    For each time t:
      - exact_i16: value ONLY if there is a sample with Timestamp == t
      - nearest_i16: nearest sample value
      - dt: abs(time - nearest_sample_time)
    """
    ts = series_df["Timestamp"].to_numpy(dtype=np.float64)
    ys = series_df["i16"].to_numpy(dtype=np.float64)

    exact_vals = []
    nearest_vals = []
    dts = []

    for t in times:
        # Exact match
        exact_mask = (ts == t)
        if np.any(exact_mask):
            exact_vals.append(float(ys[exact_mask][0]))
        else:
            exact_vals.append(np.nan)

        # Nearest match always
        idx = int(np.argmin(np.abs(ts - t)))
        nearest_vals.append(float(ys[idx]))
        dts.append(float(abs(ts[idx] - t)))

    return exact_vals, nearest_vals, dts


def build_on_event_sample_table(
    df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    on_events: List[float],
    target_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build a table of signed i16 samples for target CAN IDs at each ON trigger event.
    Uses RAW data only:
      - always tries to decode each CAN ID even if not a ranked candidate
      - auto-picks best (byte_pair, endian) based on smallest median dt

    Output columns per CAN ID:
      - 0x126_i16_exact
      - 0x126_i16_nearest
      - 0x126_dt_sec
      - 0x126_choice
    """
    if target_ids is None:
        target_ids = DEFAULT_TARGET_IDS

    results = pd.DataFrame({"event_time": on_events})

    # Sanity check: nearest Byte7 value near each ON event
    tb = trigger_df.sort_values("Timestamp").reset_index(drop=True)
    tbt = tb["Timestamp"].values
    tb7 = tb["Byte7_state"].values

    trig_vals = []
    for et in on_events:
        idx = int(np.argmin(np.abs(tbt - et)))
        trig_vals.append(int(tb7[idx]))
    results["trigger_byte7_nearest"] = trig_vals

    # For each target CAN ID, decode using RAW df
    for cid in target_ids:
        # quick check: CAN exists?
        n_rows = int((df["CAN_ID"] == int(cid)).sum())
        if n_rows == 0:
            print(f"[WARN] {hex(cid)} not present in this test file (0 rows).")
            results[f"{hex(cid)}_i16_exact"] = np.nan
            results[f"{hex(cid)}_i16_nearest"] = np.nan
            results[f"{hex(cid)}_dt_sec"] = np.nan
            results[f"{hex(cid)}_choice"] = "MISSING"
            continue

        best_choice = None
        best_exact = None
        best_nearest = None
        best_dts = None
        best_score = np.inf

        # Try every bytepair + endian
        for offset in range(0, 7):
            byte_pair = f"(Byte{offset+1},Byte{offset+2})"

            for endian in ["little", "big"]:
                series_df = signed_i16_series_for_id(df, cid, byte_pair, endian)
                if series_df is None or len(series_df) < 30:
                    continue

                exact_vals, nearest_vals, dts = exact_or_nearest_i16_at_times(series_df, on_events)

                dts_arr = np.asarray(dts, dtype=float)
                dts_arr = dts_arr[np.isfinite(dts_arr)]
                if len(dts_arr) == 0:
                    continue

                score = float(np.median(dts_arr))

                if score < best_score:
                    best_score = score
                    best_choice = (byte_pair, endian)
                    best_exact = exact_vals
                    best_nearest = nearest_vals
                    best_dts = dts

        if best_choice is None:
            print(f"[WARN] {hex(cid)} present, but could not decode any bytepair/endian cleanly.")
            results[f"{hex(cid)}_i16_exact"] = np.nan
            results[f"{hex(cid)}_i16_nearest"] = np.nan
            results[f"{hex(cid)}_dt_sec"] = np.nan
            results[f"{hex(cid)}_choice"] = "NO_DECODE"
            continue

        bp, en = best_choice
        results[f"{hex(cid)}_i16_exact"] = best_exact
        results[f"{hex(cid)}_i16_nearest"] = best_nearest
        results[f"{hex(cid)}_dt_sec"] = best_dts
        results[f"{hex(cid)}_choice"] = f"{bp} {en}"

        print(f"[OK] {hex(cid)} decoded using {bp} {en} (median dt={best_score:.4f}s)")

    return results
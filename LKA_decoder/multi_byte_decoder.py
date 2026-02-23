# multi_byte_explore.py
"""
Multi-byte exploration module for CAN decoding.

Goal:
- Explore different decode formats for specific CAN IDs (ex: 0x126, 0x220)
- Use ON trigger times (from 0x275 Byte7 == 0x04) to score which decode looks "active" after ON
- Supports:
    1) 2-byte decode (u16 / i16)
    2) 3-byte decode (u24 / i24)
    3) 2-byte bitfield decode (mask/shift, signed option)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import pandas as pd


import re


# -------------------------
# Byte column helpers
# -------------------------
def _get_byte_cols(df: pd.DataFrame) -> List[str]:
    """Get byte columns from df.attrs or discover from column names."""
    byte_cols = df.attrs.get("byte_cols")
    if not byte_cols:
        byte_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"^Byte\d+$", c)]
        byte_cols = sorted(byte_cols, key=lambda c: int(re.match(r"^Byte(\d+)$", c).group(1)))
    return byte_cols


# =========================
# Decode helpers
# =========================
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Safely convert float array with NaNs into uint8-like int array (still int64)."""
    return (arr.astype(np.int64) & 0xFF)


def decode_u16_from_bytes(b0: np.ndarray, b1: np.ndarray, endian: str) -> np.ndarray:
    """
    Decode unsigned 16-bit from two byte arrays b0,b1.
    endian:
      - little => value = (b1<<8) | b0
      - big    => value = (b0<<8) | b1
    Returns float array with NaNs where either byte is NaN.
    """
    out = np.full(len(b0), np.nan, dtype=np.float64)

    mask = (~np.isnan(b0)) & (~np.isnan(b1))
    if not np.any(mask):
        return out

    b0i = _to_uint8(b0[mask])
    b1i = _to_uint8(b1[mask])

    if endian == "little":
        out[mask] = (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 8) | b1i

    return out


def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    """Convert unsigned 16-bit values into signed i16 (two's complement)."""
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


def decode_u24_from_bytes(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, endian: str) -> np.ndarray:
    """
    Decode unsigned 24-bit from three byte arrays b0,b1,b2.
    We will treat the 3 bytes as a packed integer.

    endian:
      - little => value = (b2<<16) | (b1<<8) | b0
      - big    => value = (b0<<16) | (b1<<8) | b2

    Returns float array with NaNs where any required byte is NaN.
    """
    out = np.full(len(b0), np.nan, dtype=np.float64)

    mask = (~np.isnan(b0)) & (~np.isnan(b1)) & (~np.isnan(b2))
    if not np.any(mask):
        return out

    b0i = _to_uint8(b0[mask])
    b1i = _to_uint8(b1[mask])
    b2i = _to_uint8(b2[mask])

    if endian == "little":
        out[mask] = (b2i << 16) | (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 16) | (b1i << 8) | b2i

    return out


def u24_to_i24(u24_vals: np.ndarray) -> np.ndarray:
    """
    Convert unsigned 24-bit values to signed i24 (two's complement).
    i24 range: [-2^23, 2^23-1]
    """
    u = np.asarray(u24_vals, dtype=np.float64)
    sign_bit = 1 << 23
    full = 1 << 24
    return np.where(u >= sign_bit, u - full, u)


def sign_extend(value: int, bit_len: int) -> int:
    """
    Sign-extend an integer 'value' that is encoded using 'bit_len' bits.
    Example: value=0b111111111111 (12-bit) => -1 if signed
    """
    if bit_len <= 0:
        return value
    sign_bit = 1 << (bit_len - 1)
    mask = (1 << bit_len) - 1
    v = value & mask
    return (v ^ sign_bit) - sign_bit


def decode_bitfield_from_u16(u16_val: int, bit_shift: int, bit_len: int, signed: bool) -> int:
    """
    Extract a bitfield from a u16 integer:
      - shift right by bit_shift
      - mask out bit_len bits
      - optionally sign-extend
    """
    raw = (u16_val >> bit_shift) & ((1 << bit_len) - 1)
    if signed:
        return sign_extend(raw, bit_len)
    return raw


# =========================
# Series builders
# =========================
def build_u16_series(df: pd.DataFrame, can_id: int, start_byte: int, endian: str) -> Optional[pd.DataFrame]:
    """
    start_byte is 1-based: 1..N-1
    uses Byte[start_byte] + Byte[start_byte+1]
    returns DataFrame: Timestamp, u16
    """
    byte_cols = _get_byte_cols(df)
    g = df[df["CAN_ID"] == int(can_id)].copy().sort_values("Timestamp")
    if g.empty:
        return None

    offset = int(start_byte) - 1
    if offset < 0 or offset >= len(byte_cols) - 1:
        return None

    b0 = g[byte_cols[offset]].to_numpy(dtype=np.float64)
    b1 = g[byte_cols[offset + 1]].to_numpy(dtype=np.float64)

    u16 = decode_u16_from_bytes(b0, b1, endian=endian)
    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "u16": u16}).dropna()
    return out if len(out) >= 30 else None


def build_i16_series(df: pd.DataFrame, can_id: int, start_byte: int, endian: str) -> Optional[pd.DataFrame]:
    s = build_u16_series(df, can_id, start_byte, endian)
    if s is None:
        return None
    s = s.copy()
    s["i16"] = u16_to_i16(s["u16"].values)
    return s[["Timestamp", "i16"]].dropna().reset_index(drop=True)


def build_u24_series(df: pd.DataFrame, can_id: int, start_byte: int, endian: str) -> Optional[pd.DataFrame]:
    """
    start_byte is 1-based: 1..N-2
    uses Byte[start_byte] + Byte[start_byte+1] + Byte[start_byte+2]
    returns DataFrame: Timestamp, u24
    """
    byte_cols = _get_byte_cols(df)
    g = df[df["CAN_ID"] == int(can_id)].copy().sort_values("Timestamp")
    if g.empty:
        return None

    offset = int(start_byte) - 1
    if offset < 0 or offset >= len(byte_cols) - 2:
        return None

    b0 = g[byte_cols[offset]].to_numpy(dtype=np.float64)
    b1 = g[byte_cols[offset + 1]].to_numpy(dtype=np.float64)
    b2 = g[byte_cols[offset + 2]].to_numpy(dtype=np.float64)

    u24 = decode_u24_from_bytes(b0, b1, b2, endian=endian)
    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "u24": u24}).dropna()
    return out if len(out) >= 30 else None


def build_i24_series(df: pd.DataFrame, can_id: int, start_byte: int, endian: str) -> Optional[pd.DataFrame]:
    s = build_u24_series(df, can_id, start_byte, endian)
    if s is None:
        return None
    s = s.copy()
    s["i24"] = u24_to_i24(s["u24"].values)
    return s[["Timestamp", "i24"]].dropna().reset_index(drop=True)


def build_bitfield_series_from_u16(
    df: pd.DataFrame,
    can_id: int,
    start_byte: int,
    endian: str,
    bit_shift: int,
    bit_len: int,
    signed: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Decode a bitfield from a u16 stream.
    returns: Timestamp, val
    """
    s = build_u16_series(df, can_id, start_byte, endian)
    if s is None:
        return None

    u = s["u16"].to_numpy(dtype=np.float64)
    # Convert u16 float -> int safely
    u_int = np.asarray(np.round(u), dtype=np.int64)

    vals = []
    for x in u_int:
        if np.isnan(x):
            vals.append(np.nan)
        else:
            vals.append(float(decode_bitfield_from_u16(int(x), bit_shift, bit_len, signed=signed)))

    out = pd.DataFrame({"Timestamp": s["Timestamp"].values, "val": np.asarray(vals, dtype=np.float64)}).dropna()
    return out if len(out) >= 30 else None


# =========================
# Scoring helpers
# =========================
def window_segment(series_df: pd.DataFrame, start_t: float, end_t: float) -> Optional[np.ndarray]:
    """
    Returns numpy values from series_df in [start_t,end_t]
    series_df must have Timestamp and ONE value column.
    """
    if series_df is None or len(series_df) == 0:
        return None

    value_col = [c for c in series_df.columns if c != "Timestamp"]
    if len(value_col) != 1:
        return None

    col = value_col[0]
    seg = series_df[(series_df["Timestamp"] >= start_t) & (series_df["Timestamp"] <= end_t)]
    if len(seg) < 10:
        return None

    v = seg[col].to_numpy(dtype=np.float64)
    if len(v) < 10:
        return None
    return v


def segment_metrics(v: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Robust "activity" metrics used to score liveliness.
    Works for any numeric signal vector.
    """
    if v is None or len(v) < 15:
        return None

    p5 = np.nanpercentile(v, 5)
    p95 = np.nanpercentile(v, 95)
    rng = float(p95 - p5)

    d1 = np.diff(v)
    if len(d1) < 10:
        return None

    change_frac = float(np.mean(d1 != 0))
    big_jump_frac = float(np.mean(np.abs(d1) > 12000))

    return {
        "range_p95_p5": rng,
        "change_frac": change_frac,
        "big_jump_frac": big_jump_frac,
    }


def activity_score(metrics: Optional[Dict[str, float]]) -> float:
    if metrics is None:
        return 0.0
    return float(metrics["range_p95_p5"] * metrics["change_frac"])


def consistency_factor(values: List[float]) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return 1.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean <= 1e-9:
        return 0.0
    cv = std / mean
    return float(1.0 / (1.0 + cv))


def score_series_around_events(
    series_df: pd.DataFrame,
    on_events: List[float],
    pre_window: float = 8.0,
    post_window: float = 8.0,
) -> Optional[Dict[str, float]]:
    """
    Score ONE decoded series based on how active it becomes AFTER ON events.

    Returns dict with:
      - post_activity_median
      - pre_activity_median
      - gain
      - consistency
      - final_score
      - post_bigjump_median
      - num_events_used
    """
    if series_df is None or len(series_df) < 30:
        return None

    eps = 1e-6
    post_acts = []
    pre_acts = []
    post_bigjumps = []

    used = 0
    for et in on_events:
        pre_v = window_segment(series_df, et - pre_window, et)
        post_v = window_segment(series_df, et, et + post_window)
        if pre_v is None or post_v is None:
            continue

        used += 1
        pre_m = segment_metrics(pre_v)
        post_m = segment_metrics(post_v)

        pre_acts.append(activity_score(pre_m))
        post_acts.append(activity_score(post_m))
        post_bigjumps.append(post_m["big_jump_frac"] if post_m else 0.0)

    if used == 0:
        return None

    post_med = float(np.median(post_acts))
    pre_med = float(np.median(pre_acts))
    gain = float(min((post_med + eps) / (pre_med + eps), 50.0))
    cons = consistency_factor(post_acts)
    post_bigjump_med = float(np.median(post_bigjumps)) if len(post_bigjumps) else 0.0

    final = post_med * gain * cons

    return {
        "post_activity_median": post_med,
        "pre_activity_median": pre_med,
        "gain": gain,
        "consistency": cons,
        "final_score": float(final),
        "post_bigjump_median": post_bigjump_med,
        "num_events_used": int(used),
    }


# =========================
# Candidate generators
# =========================
def generate_2byte_decode_configs(n_bytes: int = 8) -> List[Dict[str, Any]]:
    cfgs = []
    for start_byte in range(1, n_bytes):  # Byte1..Byte(N-1)
        for endian in ["little", "big"]:
            cfgs.append({"kind": "u16/i16", "start_byte": start_byte, "endian": endian})
    return cfgs


def generate_3byte_decode_configs(n_bytes: int = 8) -> List[Dict[str, Any]]:
    cfgs = []
    for start_byte in range(1, n_bytes - 1):  # Byte1..Byte(N-2)
        for endian in ["little", "big"]:
            cfgs.append({"kind": "u24/i24", "start_byte": start_byte, "endian": endian})
    return cfgs


def generate_bitfield_decode_configs(
    n_bytes: int = 8,
    bit_lens: List[int] = [8, 10, 12, 13, 14, 15],
    bit_shifts: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    signed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a bunch of bitfield configs from u16.
    Defaults focus on reasonable field sizes for "distance-ish" signals.
    """
    cfgs = []
    for start_byte in range(1, n_bytes):
        for endian in ["little", "big"]:
            for bit_len in bit_lens:
                for bit_shift in bit_shifts:
                    if bit_shift + bit_len > 16:
                        continue
                    cfgs.append(
                        {
                            "kind": "bitfield",
                            "start_byte": start_byte,
                            "endian": endian,
                            "bit_len": bit_len,
                            "bit_shift": bit_shift,
                            "signed": signed,
                        }
                    )
    return cfgs

def decode_series_from_row(df: pd.DataFrame, row: pd.Series) -> Optional[pd.DataFrame]:
    """
    Re-decode a signal using one row from explore_decode_space_for_* output.
    Returns DataFrame with columns: ["Timestamp", "unsigned", "signed"]
    """
    kind = row["kind"]
    can_id = int(row["can_id_dec"])
    start_byte = int(row["start_byte"])
    endian = row["endian"]

    if kind == "u16/i16":
        # Get both u16 and i16
        s_u16 = build_u16_series(df, can_id, start_byte, endian)
        if s_u16 is None:
            return None
        s_i16 = s_u16.copy()
        s_i16["signed"] = u16_to_i16(s_u16["u16"].values)
        return pd.DataFrame({
            "Timestamp": s_u16["Timestamp"].values,
            "unsigned": s_u16["u16"].values,
            "signed": s_i16["signed"].values
        })

    if kind == "u24/i24":
        # Get both u24 and i24
        s_u24 = build_u24_series(df, can_id, start_byte, endian)
        if s_u24 is None:
            return None
        s_i24 = s_u24.copy()
        s_i24["signed"] = u24_to_i24(s_u24["u24"].values)
        return pd.DataFrame({
            "Timestamp": s_u24["Timestamp"].values,
            "unsigned": s_u24["u24"].values,
            "signed": s_i24["signed"].values
        })

    if kind == "bitfield":
        bit_shift = int(row["bit_shift"])
        bit_len = int(row["bit_len"])
        
        # Get signed version
        s_signed = build_bitfield_series_from_u16(
            df=df,
            can_id=can_id,
            start_byte=start_byte,
            endian=endian,
            bit_shift=bit_shift,
            bit_len=bit_len,
            signed=True,
        )
        
        # Get unsigned version
        s_unsigned = build_bitfield_series_from_u16(
            df=df,
            can_id=can_id,
            start_byte=start_byte,
            endian=endian,
            bit_shift=bit_shift,
            bit_len=bit_len,
            signed=False,
        )
        
        if s_signed is None or s_unsigned is None:
            return None
            
        return pd.DataFrame({
            "Timestamp": s_signed["Timestamp"].values,
            "unsigned": s_unsigned["val"].values,
            "signed": s_signed["val"].values
        })

    return None

def describe_decode(row: pd.Series) -> str:
    if row["kind"] == "u16/i16":
        return f"i16 Byte{row['start_byte']} {row['endian']}"

    if row["kind"] == "u24/i24":
        return f"i24 Byte{row['start_byte']} {row['endian']}"

    if row["kind"] == "bitfield":
        return (
            f"bitfield Byte{row['start_byte']} "
            f"shift={row['bit_shift']} len={row['bit_len']} "
            f"{'signed' if row['signed'] else 'unsigned'} "
            f"{row['endian']}"
        )

    return "unknown"


# =========================
# Exploration Runner
# =========================
def explore_decode_space_for_id(
    df: pd.DataFrame,
    can_id: int,
    on_events: List[float],
    pre_window: float = 8.0,
    post_window: float = 8.0,
    include_2byte: bool = True,
    include_3byte: bool = True,
    include_bitfields: bool = True,
    bitfield_signed: bool = True,
) -> pd.DataFrame:
    """
    Run decoding exploration for ONE CAN ID.

    Output:
    ranked DataFrame with decoding configuration + scores.
    """
    if can_id is None:
        raise ValueError("can_id is required")
    if len(on_events) == 0:
        raise ValueError("on_events is empty")

    # quick existence check
    n_rows = int((df["CAN_ID"] == int(can_id)).sum())
    if n_rows == 0:
        return pd.DataFrame()

    byte_cols = _get_byte_cols(df)
    n_bytes = len(byte_cols)

    configs: List[Dict[str, Any]] = []

    if include_2byte:
        configs.extend(generate_2byte_decode_configs(n_bytes=n_bytes))
    if include_3byte:
        configs.extend(generate_3byte_decode_configs(n_bytes=n_bytes))
    if include_bitfields:
        configs.extend(generate_bitfield_decode_configs(n_bytes=n_bytes, signed=bitfield_signed))

    results = []

    for cfg in configs:
        kind = cfg["kind"]
        start_byte = cfg["start_byte"]
        endian = cfg["endian"]

        # Build the series based on the candidate type
        series_df = None
        value_col_name = None

        if kind == "u16/i16":
            # We'll score using signed i16 version by default
            series_df = build_i16_series(df, can_id, start_byte, endian)
            value_col_name = "i16"

        elif kind == "u24/i24":
            series_df = build_i24_series(df, can_id, start_byte, endian)
            value_col_name = "i24"

        elif kind == "bitfield":
            series_df = build_bitfield_series_from_u16(
                df=df,
                can_id=can_id,
                start_byte=start_byte,
                endian=endian,
                bit_shift=cfg["bit_shift"],
                bit_len=cfg["bit_len"],
                signed=cfg["signed"],
            )
            value_col_name = "val"

        if series_df is None or len(series_df) < 30:
            continue

        score = score_series_around_events(
            series_df=series_df,
            on_events=on_events,
            pre_window=pre_window,
            post_window=post_window,
        )
        if score is None:
            continue
        
        # ============================================================
        # NOISE FILTERS
        # ============================================================
        
        # Filter 1: Pre-activity too high (noisy even before trigger)
        # If signal is already super active before trigger, it's probably just noise/counter
        if score["pre_activity_median"] > 10000: 
            continue
        
        # Filter 2: Check change fraction from segment_metrics
        # Calculate it manually from the series
        value_col_name = [c for c in series_df.columns if c != "Timestamp"][0]
        vals = series_df[value_col_name].values
        if len(vals) > 10:
            changes = np.diff(vals)
            change_frac = float(np.mean(changes != 0))
            
            # If 95%+ of samples change, it's probably a counter or high-freq noise
            if change_frac > 0.75:
                continue
        
        # Filter 3: Gain must be somewhat meaningful
        if score["gain"] < 2.0:  
            continue
        

        row = {
            "can_id_hex": hex(int(can_id)),
            "can_id_dec": int(can_id),
            "kind": kind,
            "start_byte": int(start_byte),
            "endian": endian,
            "value_col": value_col_name,
            **cfg,  # include bitfield fields too
            **score,
        }

        results.append(row)

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    return out


def explore_decode_space_for_targets(
    df: pd.DataFrame,
    on_events: List[float],
    target_ids: List[int],
    pre_window: float = 8.0,
    post_window: float = 8.0,
) -> pd.DataFrame:
    """
    Convenience wrapper: run exploration for multiple CAN IDs
    and return ONE combined ranked table.
    """
    all_rows = []
    for cid in target_ids:
        ranked = explore_decode_space_for_id(
            df=df,
            can_id=int(cid),
            on_events=on_events,
            pre_window=pre_window,
            post_window=post_window,
            include_2byte=True,
            include_3byte=True,
            include_bitfields=True,
        )
        if not ranked.empty:
            all_rows.append(ranked)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    return out
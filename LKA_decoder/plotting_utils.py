# plotting_utils.py
"""
Plotting utilities for CAN decoding + trigger overlay.

Supports:
- Classic candidate plot: u16 (red) + i16 (gold) + trigger overlay (dark blue)
- Multi-byte explore plot:
    - i16 (from u16)
    - i24 (from u24)
    - bitfield decode from u16

Also supports:
- optional smoothing (rolling mean or EMA)
- full test plotting with slight x-padding
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multi_byte_decoder import build_u16_series, build_i16_series, build_u24_series, build_i24_series, build_bitfield_series_from_u16, u16_to_i16


# -------------------------
# Colors / style
# -------------------------
COLOR_U16 = "red"
COLOR_I16 = "gold"
COLOR_SIGNAL = "gold"     # used for i24 / bitfield (single-line)
COLOR_TRIGGER = "darkblue"


# -------------------------
# Small helpers
# -------------------------
def smooth_array(
    y: np.ndarray,
    method: str = "rolling",
    window: int = 25,
    ema_span: int = 25
) -> np.ndarray:
    """
    Smooth a numeric array:
      - rolling mean (window samples)
      - EMA (span samples)
    """
    y = np.asarray(y, dtype=np.float64)

    if len(y) < 5:
        return y

    s = pd.Series(y)

    if method == "rolling":
        return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    if method == "ema":
        return s.ewm(span=ema_span, adjust=False).mean().to_numpy()

    return y


def compute_y_limits_from_data(
    arrays: List[np.ndarray],
    y_percentile_clip: Tuple[float, float] = (1, 99),
    y_pad_frac: float = 0.08
) -> Tuple[float, float]:
    """
    Find y-limits based on percentile clip across multiple arrays,
    with headroom padding.
    """
    combined = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            combined.append(arr)

    if len(combined) == 0:
        return (-1.0, 1.0)

    combined = np.concatenate(combined)

    if len(combined) > 20:
        lo, hi = np.nanpercentile(combined, y_percentile_clip)
        y_min, y_max = float(lo), float(hi)
    else:
        y_min, y_max = float(np.nanmin(combined)), float(np.nanmax(combined))

    # padding
    y_rng = max(1e-9, (y_max - y_min))
    y_min -= y_pad_frac * y_rng
    y_max += y_pad_frac * y_rng

    return y_min, y_max


def compute_x_limits_full_test(t: np.ndarray, pad_frac: float = 0.02) -> Tuple[float, float]:
    """
    Full test range but with a little x padding so signals have breathing room.
    """
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    x_pad = pad_frac * (tmax - tmin)
    return tmin - x_pad, tmax + x_pad


def trigger_series_from_df(
    trigger_bytes_df: pd.DataFrame,
    prefer_trigger: str = "Byte7",
    on_value: int = 0x04
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (tt, trig_vals, trig_label)
    where trig_vals is Byte7_state or Byte2_state or ON mask.
    """
    tb = trigger_bytes_df.sort_values("Timestamp").copy()
    tt = tb["Timestamp"].to_numpy(dtype=np.float64)

    if prefer_trigger == "Byte7":
        trig = tb["Byte7_state"].to_numpy(dtype=np.float64)
        trig_label = "0x275 Byte7"
    elif prefer_trigger == "Byte2":
        trig = tb["Byte2_state"].to_numpy(dtype=np.float64)
        trig_label = "0x275 Byte2"
    else:
        trig = (
            (tb["Byte7_state"].to_numpy(dtype=np.float64) == float(on_value))
            | (tb["Byte2_state"].to_numpy(dtype=np.float64) == float(on_value))
        ).astype(int)
        trig_label = "0x275 ON mask"

    return tt, trig, trig_label


# ======================================================================
# 1) Classic candidate plot (u16 red + i16 gold) + trigger overlay
# ======================================================================
def plot_candidate_u16_i16_with_trigger_overlay(
    t: np.ndarray,
    u16_vals: np.ndarray,
    i16_vals: np.ndarray,
    trigger_bytes_df: pd.DataFrame,
    title: str,
    prefer_trigger: str = "Byte7",
    on_value: int = 0x04,
    fig_size: Tuple[int, int] = (16, 7),
    y_percentile_clip: Tuple[float, float] = (1, 99),
    y_pad_frac: float = 0.08,
    smooth: bool = False,
    smooth_method: str = "rolling",
    smooth_window: int = 25,
    outpath: Optional[str] = None,
):
    """
    Plot u16 (red) + i16 (gold) and trigger overlay (dark blue).
    This assumes you already decoded the signal (arrays passed in).
    """

    t = np.asarray(t, dtype=np.float64)
    u16_vals = np.asarray(u16_vals, dtype=np.float64)
    i16_vals = np.asarray(i16_vals, dtype=np.float64)

    # smoothing
    if smooth:
        u16_vals = smooth_array(u16_vals, method=smooth_method, window=smooth_window, ema_span=smooth_window)
        i16_vals = smooth_array(i16_vals, method=smooth_method, window=smooth_window, ema_span=smooth_window)

    # full x range + small padding
    tmin, tmax = compute_x_limits_full_test(t, pad_frac=0.02)

    # y limits from both
    y_min, y_max = compute_y_limits_from_data([u16_vals, i16_vals], y_percentile_clip, y_pad_frac)

    # trigger series
    tt, trig, trig_label = trigger_series_from_df(trigger_bytes_df, prefer_trigger, on_value)
    z2 = (tt >= tmin) & (tt <= tmax)

    fig, ax1 = plt.subplots(figsize=fig_size)

    ax1.plot(t, u16_vals, color=COLOR_U16, linewidth=1.0, label="Unsigned u16")
    ax1.plot(t, i16_vals, color=COLOR_I16, linewidth=1.0, label="Signed i16")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Candidate value")
    ax1.grid(True, linewidth=0.5)
    ax1.set_xlim(tmin, tmax)
    ax1.set_ylim(y_min, y_max)

    ax2 = ax1.twinx()
    ax2.step(tt[z2], trig[z2], where="post", color=COLOR_TRIGGER, linewidth=1.2, label=trig_label)
    ax2.set_ylabel(trig_label)

    smooth_tag = f"SMOOTH({smooth_method}, N={smooth_window})" if smooth else "RAW"
    ax1.set_title(f"{title}   [{smooth_tag}]")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ======================================================================
# 2) Multi-byte explore plot (i16 OR i24 OR bitfield) + trigger overlay
# ======================================================================
def plot_single_signal_with_trigger_overlay(
    t: np.ndarray,
    y: np.ndarray,
    trigger_bytes_df: pd.DataFrame,
    title: str,
    prefer_trigger: str = "Byte7",
    on_value: int = 0x04,
    fig_size: Tuple[int, int] = (16, 7),
    y_percentile_clip: Tuple[float, float] = (1, 99),
    y_pad_frac: float = 0.08,
    smooth: bool = False,
    smooth_method: str = "rolling",
    smooth_window: int = 25,
    outpath: Optional[str] = None,
):
    """
    Plot ONE signal (gold) + trigger overlay.
    Used for i24 decode and bitfield decode.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if smooth:
        y = smooth_array(y, method=smooth_method, window=smooth_window, ema_span=smooth_window)

    tmin, tmax = compute_x_limits_full_test(t, pad_frac=0.02)
    y_min, y_max = compute_y_limits_from_data([y], y_percentile_clip, y_pad_frac)

    tt, trig, trig_label = trigger_series_from_df(trigger_bytes_df, prefer_trigger, on_value)
    z2 = (tt >= tmin) & (tt <= tmax)

    fig, ax1 = plt.subplots(figsize=fig_size)

    ax1.plot(t, y, color=COLOR_SIGNAL, linewidth=1.0, label="Decoded signal")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal value")
    ax1.grid(True, linewidth=0.5)
    ax1.set_xlim(tmin, tmax)
    ax1.set_ylim(y_min, y_max)

    ax2 = ax1.twinx()
    ax2.step(tt[z2], trig[z2], where="post", color=COLOR_TRIGGER, linewidth=1.2, label=trig_label)
    ax2.set_ylabel(trig_label)

    smooth_tag = f"SMOOTH({smooth_method}, N={smooth_window})" if smooth else "RAW"
    ax1.set_title(f"{title}   [{smooth_tag}]")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ======================================================================
# 3) Convenience wrapper to plot from an explore row (from multi_byte_explore)
# ======================================================================
def plot_from_explore_row(
    df: pd.DataFrame,
    explore_row: pd.Series,
    trigger_bytes_df: pd.DataFrame,
    prefer_trigger: str = "Byte7",
    on_value: int = 0x04,
    fig_size: Tuple[int, int] = (16, 7),
    smooth: bool = False,
    smooth_method: str = "rolling",
    smooth_window: int = 25,
    outpath: Optional[str] = None,
):
    """
    This plots a row from multi_byte_explore.explore_decode_space_for_id() output.
    It decodes the series again based on row config.

    Expected explore_row fields:
      - can_id_dec
      - kind: "u16/i16", "u24/i24", "bitfield"
      - start_byte
      - endian
      - for bitfield: bit_len, bit_shift, signed
    """
    can_id = int(explore_row["can_id_dec"])
    kind = str(explore_row["kind"])
    start_byte = int(explore_row["start_byte"])
    endian = str(explore_row["endian"])

    can_hex = hex(can_id)

    # ---- u16/i16 ----
    if kind == "u16/i16":
        s_u16 = build_u16_series(df, can_id, start_byte, endian)
        if s_u16 is None:
            print(f"[WARN] Could not decode {can_hex} u16 at Byte{start_byte} {endian}")
            return

        t = s_u16["Timestamp"].to_numpy(dtype=np.float64)
        u16_vals = s_u16["u16"].to_numpy(dtype=np.float64)
        i16_vals = u16_to_i16(u16_vals)

        title = f"{can_hex}  u16/i16  start=Byte{start_byte}  {endian}"
        plot_candidate_u16_i16_with_trigger_overlay(
            t=t,
            u16_vals=u16_vals,
            i16_vals=i16_vals,
            trigger_bytes_df=trigger_bytes_df,
            title=title,
            prefer_trigger=prefer_trigger,
            on_value=on_value,
            fig_size=fig_size,
            smooth=smooth,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            outpath=outpath,
        )
        return

    # ---- u24/i24 ----
    if kind == "u24/i24":
        s_i24 = build_i24_series(df, can_id, start_byte, endian)
        if s_i24 is None:
            print(f"[WARN] Could not decode {can_hex} i24 at Byte{start_byte} {endian}")
            return

        t = s_i24["Timestamp"].to_numpy(dtype=np.float64)
        y = s_i24["i24"].to_numpy(dtype=np.float64)

        title = f"{can_hex}  i24  start=Byte{start_byte}  {endian}"
        plot_single_signal_with_trigger_overlay(
            t=t,
            y=y,
            trigger_bytes_df=trigger_bytes_df,
            title=title,
            prefer_trigger=prefer_trigger,
            on_value=on_value,
            fig_size=fig_size,
            smooth=smooth,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            outpath=outpath,
        )
        return

    # ---- bitfield ----
    if kind == "bitfield":
        bit_len = int(explore_row["bit_len"])
        bit_shift = int(explore_row["bit_shift"])
        signed = bool(explore_row.get("signed", True))

        s_bf = build_bitfield_series_from_u16(
            df=df,
            can_id=can_id,
            start_byte=start_byte,
            endian=endian,
            bit_shift=bit_shift,
            bit_len=bit_len,
            signed=signed,
        )
        if s_bf is None:
            print(f"[WARN] Could not decode {can_hex} bitfield at Byte{start_byte} {endian} shift={bit_shift} len={bit_len}")
            return

        t = s_bf["Timestamp"].to_numpy(dtype=np.float64)
        y = s_bf["val"].to_numpy(dtype=np.float64)

        sign_tag = "signed" if signed else "unsigned"
        title = f"{can_hex}  bitfield({sign_tag})  start=Byte{start_byte} {endian}  shift={bit_shift} len={bit_len}"
        plot_single_signal_with_trigger_overlay(
            t=t,
            y=y,
            trigger_bytes_df=trigger_bytes_df,
            title=title,
            prefer_trigger=prefer_trigger,
            on_value=on_value,
            fig_size=fig_size,
            smooth=smooth,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            outpath=outpath,
        )
        return

    print(f"[WARN] Unknown kind='{kind}' for {can_hex}")

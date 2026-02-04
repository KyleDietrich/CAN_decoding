# plotting_utils.py
"""
Plotting utilities for CAN decoding + trigger overlay.

Supports:
- Parameterized trigger support (any CAN ID, any byte, multiple trigger values)
- Classic candidate plot: u16 (red) + i16 (gold) + trigger overlay (dark blue)
- Trigger state plot with ON event markers
- Optional smoothing (rolling mean or EMA)
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Colors / style
# -------------------------
COLOR_U16 = "red"
COLOR_I16 = "gold"
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

def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    """Convert unsigned 16-bit values to signed 16-bit."""
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


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

# ======================================================================
# Trigger state plot (whole test)
# ======================================================================
def plot_trigger_states_whole_test(
    trigger_df: pd.DataFrame,
    on_events: List[float],
    lka_id: int,
    trigger_byte: int,
    trigger_values: List[int],
    outpath: Optional[str] = None,
    fig_size: Tuple[int, int] = (16, 3),
):
    """
    Plot trigger byte state over time with ON event markers.
    
    Args:
        trigger_df: DataFrame with 'Timestamp' and 'trigger_byte_state' columns
        on_events: List of timestamps where trigger ON events occurred
        lka_id: CAN ID used for trigger
        trigger_byte: Which byte (1-8) contains the trigger
        trigger_values: List of values that indicate trigger ON
        outpath: Path to save figure (if None, displays instead)
        fig_size: Figure size tuple
    """
    t = trigger_df["Timestamp"].values
    trigger_state = trigger_df["trigger_byte_state"].values

    fig, ax = plt.subplots(figsize=fig_size)

    ax.step(t, trigger_state, where="post", linewidth=1.2, label=f"{hex(lka_id)} Byte{trigger_byte}")

    for et in on_events:
        ax.axvline(et, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trigger State (byte value)")
    
    values_str = ", ".join(hex(v) for v in trigger_values)
    ax.set_title(f"LKA Trigger States from {hex(lka_id)} Byte{trigger_byte}, ON events (-> {values_str}) marked red")
    
    ax.grid(True, linewidth=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()

        
# ======================================================================
# Candidate plot with trigger overlay (u16 red + i16 gold)
# ======================================================================
def plot_candidate_with_trigger_overlay(
    t: np.ndarray,
    u16_vals: np.ndarray,
    i16_vals: np.ndarray,
    trigger_df: pd.DataFrame,
    lka_id: int,
    trigger_byte: int,
    trigger_values: List[int],
    title: str,
    zoom_mode: str = "full",
    pad_seconds: float = 10.0,
    y_percentile_clip: Tuple[float, float] = (1, 99),
    fig_size: Tuple[int, int] = (16, 7),
    y_pad_frac: float = 0.08,
    smooth: bool = False,
    smooth_method: str = "rolling",
    smooth_window: int = 25,
    outpath: Optional[str] = None,
):
    """
    Plot u16 (red) + i16 (gold) candidate signal with trigger overlay (dark blue).
    
    Args:
        t: Timestamps for candidate signal
        u16_vals: Unsigned 16-bit decoded values
        i16_vals: Signed 16-bit decoded values
        trigger_df: DataFrame with 'Timestamp' and 'trigger_byte_state' columns
        lka_id: CAN ID used for trigger
        trigger_byte: Which byte (1-8) contains the trigger
        trigger_values: List of values that indicate trigger ON
        title: Plot title
        zoom_mode: "full" for entire test, "auto" to zoom around trigger events
        pad_seconds: Padding around trigger events when zoom_mode="auto"
        y_percentile_clip: Percentile range for y-axis limits
        fig_size: Figure size tuple
        y_pad_frac: Fractional padding for y-axis headroom
        smooth: Whether to apply smoothing
        smooth_method: "rolling" or "ema"
        smooth_window: Window size for smoothing
        outpath: Path to save figure (if None, displays instead)
    """
    t = np.asarray(t, dtype=np.float64)
    u16_vals = np.asarray(u16_vals, dtype=np.float64)
    i16_vals = np.asarray(i16_vals, dtype=np.float64)

    # Trigger series
    tb = trigger_df.sort_values("Timestamp").copy()
    tt = tb["Timestamp"].values
    trig = tb["trigger_byte_state"].values
    trig_label = f"{hex(lka_id)} Byte{trigger_byte}"

    # ON mask for multiple trigger values
    on_mask = np.isin(tb["trigger_byte_state"].values, trigger_values)

    # Decide plot window
    if zoom_mode == "auto":
        on_times = tt[on_mask]
        if len(on_times) > 0:
            tmin = float(np.min(on_times) - pad_seconds)
            tmax = float(np.max(on_times) + pad_seconds)
        else:
            tmin = float(np.nanmin(t))
            tmax = float(np.nanmax(t))
    else:
        tmin = float(np.nanmin(t))
        tmax = float(np.nanmax(t))
        x_pad = 0.02 * (tmax - tmin)
        tmin -= x_pad
        tmax += x_pad

    # Zoom masks
    z1 = (t >= tmin) & (t <= tmax)
    z2 = (tt >= tmin) & (tt <= tmax)

    t_zoom = t[z1]
    u16_zoom = u16_vals[z1]
    i16_zoom = i16_vals[z1]

    # Optional smoothing
    if smooth:
        u16_zoom = smooth_array(u16_zoom, method=smooth_method, window=smooth_window, ema_span=smooth_window)
        i16_zoom = smooth_array(i16_zoom, method=smooth_method, window=smooth_window, ema_span=smooth_window)

    if len(t_zoom) < 5:
        print(f"Skipping plot: not enough points in window.")
        return

    # Y-limits based on both series combined
    y_min, y_max = compute_y_limits_from_data([u16_zoom, i16_zoom], y_percentile_clip, y_pad_frac)

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=fig_size)

    ax1.plot(t_zoom, u16_zoom, color=COLOR_U16, linewidth=1.0, label="Candidate unsigned u16")
    ax1.plot(t_zoom, i16_zoom, color=COLOR_I16, linewidth=1.0, label="Candidate signed i16")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Candidate value")
    ax1.grid(True, linewidth=0.5)
    ax1.set_xlim(tmin, tmax)
    ax1.set_ylim(y_min, y_max)

    # Trigger overlay (right axis) dark blue
    ax2 = ax1.twinx()
    ax2.step(tt[z2], trig[z2], where="post", color=COLOR_TRIGGER, linewidth=1.2, label=trig_label)
    ax2.set_ylabel(trig_label)

    # Title with smooth tag
    smooth_tag = f"SMOOTH({smooth_method}, N={smooth_window})" if smooth else "RAW"
    ax1.set_title(f"{title}   [{smooth_tag}]")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()

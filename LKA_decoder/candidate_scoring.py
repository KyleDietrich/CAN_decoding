# candidate_scoring.py
from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import pandas as pd

BYTE_COLS = [f"Byte{i}" for i in range(1, 9)]


# =========================
# Decode helpers 
# =========================
def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


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


# =========================
# Candidate scoring
# =========================
def window_segment(series_df: pd.DataFrame, start_t: float, end_t: float) -> Optional[pd.DataFrame]:
    seg = series_df[(series_df["Timestamp"] >= start_t) & (series_df["Timestamp"] <= end_t)].copy()
    if len(seg) < 10:
        return None
    return seg


def segment_metrics_i16(seg: pd.DataFrame) -> Optional[Dict[str, float]]:
    v = seg["i16"].values
    if len(v) < 20:
        return None

    p5 = np.nanpercentile(v, 5)
    p95 = np.nanpercentile(v, 95)
    rng = float(p95 - p5)

    d1 = np.diff(v)
    if len(d1) < 10:
        return None

    change_frac = float(np.mean(d1 != 0))
    big_jump_frac = float(np.mean(np.abs(d1) > 12000))  # relaxed threshold
    unique_count = int(pd.Series(v).nunique())

    return {
        "range_p95_p5": rng,
        "change_frac": change_frac,
        "big_jump_frac": big_jump_frac,
        "unique_count": unique_count,
    }


def activity_score(m: Optional[Dict[str, float]]) -> float:
    """
    Stable measure of 'how alive' the signal is.
    """
    if m is None:
        return 0.0
    return float(m["range_p95_p5"] * m["change_frac"])


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


def get_i16_series_for_candidate(
    df: pd.DataFrame,
    can_id_dec: int,
    offset: int,
    endian: str
) -> Optional[pd.DataFrame]:
    """
    Decode candidate as signed i16 from a CAN ID + byte offset + endian.
    """
    g = df[df["CAN_ID"] == int(can_id_dec)].copy().sort_values("Timestamp")
    if g.empty or len(g) < 80:
        return None

    u16 = compute_u16_series_from_group(g, offset, endian)
    i16 = u16_to_i16(u16)

    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "i16": i16}).dropna()
    if len(out) < 80:
        return None

    return out


def build_candidates_multi_event_relaxed(
    df: pd.DataFrame,
    on_events: List[float],
    exclude_id: Optional[int] = None,
    pre_window: float = 20.0,
    post_window: float = 20.0,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Rank CAN_ID + bytepair + endian candidates based on activity AFTER trigger ON events.
    
    Args:
        df: Full CAN dataframe
        on_events: List of trigger event timestamps
        exclude_id: CAN ID to exclude from search (None to include all IDs)
        pre_window: Seconds before trigger to analyze
        post_window: Seconds after trigger to analyze
        top_n: Number of top candidates to return
    """
    if len(on_events) == 0:
        raise ValueError("No ON events found (trigger ON list is empty).")

    results = []
    eps = 1e-6

    for can_id, g in df.groupby("CAN_ID"):
        # Skip excluded ID if specified
        if exclude_id is not None and int(can_id) == int(exclude_id):
            continue
        if len(g) < 150:
            continue

        for offset in range(0, 7):
            for endian in ["little", "big"]:

                s = get_i16_series_for_candidate(df, int(can_id), offset, endian)
                if s is None:
                    continue

                post_acts, pre_acts, post_bigjumps = [], [], []
                valid_events = 0

                for et in on_events:
                    pre_seg = window_segment(s, et - pre_window, et)
                    post_seg = window_segment(s, et, et + post_window)

                    if pre_seg is None or post_seg is None:
                        continue

                    valid_events += 1

                    pre_m = segment_metrics_i16(pre_seg)
                    post_m = segment_metrics_i16(post_seg)

                    pre_acts.append(activity_score(pre_m))
                    post_acts.append(activity_score(post_m))
                    post_bigjumps.append(post_m["big_jump_frac"] if post_m else 0.0)

                if valid_events == 0:
                    continue

                post_med = float(np.median(post_acts))
                pre_med = float(np.median(pre_acts))

                # relaxed: only require *some* post activity
                if post_med < 100:
                    continue

                gain = (post_med + eps) / (pre_med + eps)
                gain = float(min(gain, 50.0))

                # relaxed gain threshold
                if gain < 1.05:
                    continue

                cons = consistency_factor(post_acts)

                # relaxed big-jump reject (avoid obvious counters)
                post_bigjump_med = float(np.median(post_bigjumps)) if len(post_bigjumps) else 0.0
                if post_bigjump_med > 0.10:
                    continue

                final = post_med * gain * cons

                results.append({
                    "can_id_hex": hex(int(can_id)),
                    "can_id_dec": int(can_id),
                    "byte_pair": f"(Byte{offset+1},Byte{offset+2})",
                    "endian": endian,
                    "final_score": final,
                    "post_activity_median": post_med,
                    "pre_activity_median": pre_med,
                    "gain_capped": gain,
                    "consistency": cons,
                    "num_events_used": valid_events,
                })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
    return out


def dedupe_by_can_id(candidates_df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    """
    Keep only the best bytepair/endian per CAN ID.
    """
    if candidates_df is None or len(candidates_df) == 0:
        return candidates_df

    tmp = candidates_df.copy()
    tmp["can_id_dec"] = tmp["can_id_dec"].astype(int)
    tmp = tmp.sort_values(score_col, ascending=False)
    tmp = tmp.drop_duplicates(subset=["can_id_dec"], keep="first").reset_index(drop=True)
    return tmp
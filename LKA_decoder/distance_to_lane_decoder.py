import argparse
import os
import re
import shutil
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from candidate_scoring import build_candidates_multi_event_relaxed, dedupe_by_can_id
from on_event_sampling import build_on_event_sample_table
from multi_byte_decoder import explore_decode_space_for_targets
from plotting_utils import plot_from_explore_row

# =========================
# Config
# =========================
LKA_ID = 0x275
TRIGGER_ON_VALUE = 0x04

BYTE_COLS = [f"Byte{i}" for i in range(1, 9)]


# =========================
# Helpers: parse + conversions
# =========================
def hex_to_int(x):
    """
    Converts hex strings like '275', '30A', 'FF' into int.
    Returns NaN for invalid values (ex: 'ErrorFrame', blanks).
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "" or s.lower() == "errorframe":
        return np.nan

    s = s.replace("0x", "").replace("0X", "")
    valid = all(ch in "0123456789abcdefABCDEF" for ch in s)
    if not valid:
        return np.nan

    try:
        return int(s, 16)
    except Exception:
        return np.nan


def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


def compute_u16_series_from_group(g: pd.DataFrame, offset: int, endian: str) -> np.ndarray:
    """
    Vectorized u16 extraction from Byte[offset] and Byte[offset+1] columns.
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
# Load + clean CSV
# =========================
def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Count ErrorFrame rows
    if "Identifier" in df.columns:
        errorframe_count = (df["Identifier"].astype(str).str.strip().str.lower() == "errorframe").sum()
    else:
        errorframe_count = 0

    print(f"Loaded: {csv_path}")
    print(f"Rows (raw): {len(df)}")
    print(f"ErrorFrame rows found: {errorframe_count}")

    # Parse CAN ID
    df["CAN_ID"] = df["Identifier"].apply(hex_to_int)

    # Parse bytes
    for c in BYTE_COLS:
        if c in df.columns:
            df[c] = df[c].apply(hex_to_int).astype(float)
        else:
            df[c] = np.nan

    # Parse timestamp
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")

    # Clean
    df = df.dropna(subset=["Timestamp", "CAN_ID"]).sort_values("Timestamp").reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Unique CAN IDs: {df['CAN_ID'].nunique()}")

    return df


# =========================
# Trigger extraction: 0x275 Byte2 + Byte7
# =========================
def compute_trigger_bytes(df, lka_id=0x275):
    lka = df[df["CAN_ID"] == int(lka_id)].copy().sort_values("Timestamp")
    if lka.empty:
        raise ValueError(f"No rows found for LKA CAN ID = {hex(lka_id)}")

    out = pd.DataFrame({
        "Timestamp": lka["Timestamp"].values,
        "Byte2_state": pd.to_numeric(lka["Byte2"], errors="coerce"),
        "Byte7_state": pd.to_numeric(lka["Byte7"], errors="coerce"),
    }).dropna()

    out["Byte2_state"] = out["Byte2_state"].astype(int)
    out["Byte7_state"] = out["Byte7_state"].astype(int)

    return out


def detect_on_events(trigger_df: pd.DataFrame, on_value: int = TRIGGER_ON_VALUE) -> List[float]:
    """
    ON event = Byte7 transitions into 0x04.
    """
    s = trigger_df["Byte7_state"].values
    t = trigger_df["Timestamp"].values

    prev = np.roll(s, 1)
    prev[0] = s[0]

    on_mask = (s == on_value) & (prev != on_value)
    return t[on_mask].astype(float).tolist()


# =========================
# Plot triggers (whole test)
# =========================
def plot_trigger_states_whole_test(trigger_df: pd.DataFrame, on_events: List[float], outpath: str):
    t = trigger_df["Timestamp"].values
    b2 = trigger_df["Byte2_state"].values
    b7 = trigger_df["Byte7_state"].values

    fig, ax = plt.subplots(figsize=(16, 3))

    ax.step(t, b2, where="post", linewidth=1.2, label="0x275 Byte2")
    ax.step(t, b7, where="post", linewidth=1.2, label="0x275 Byte7")

    for et in on_events:
        ax.axvline(et, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trigger State (byte value)")
    ax.set_title("LKA Trigger States from 0x275 (Byte2 + Byte7), ON events marked red")
    ax.grid(True, linewidth=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def u16_series_for_candidate(df, can_id_dec, byte_pair_str, endian):
    b1 = int(byte_pair_str.split("Byte")[1].split(",")[0])  # 1-based
    offset = b1 - 1

    g = df[df["CAN_ID"] == int(can_id_dec)].copy().sort_values("Timestamp")
    if g.empty:
        return None

    u16 = compute_u16_series_from_group(g, offset, endian)
    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "u16": u16}).dropna()
    return out if len(out) > 10 else None

def smooth_array(y: np.ndarray, method: str = "rolling", window: int = 25, ema_span: int = 25) -> np.ndarray:
    """
    Smooth a numpy array using:
      - rolling mean (window in samples)
      - EMA (span in samples)
    """
    y = np.asarray(y, dtype=np.float64)

    if len(y) < 5:
        return y

    s = pd.Series(y)

    if method == "rolling":
        return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    if method == "ema":
        return s.ewm(span=ema_span, adjust=False).mean().to_numpy()

    return y  # fallback (no smoothing)

# =========================
# Plot candidates (whole test)
# unsigned u16 red, signed i16 yellow
# =========================
def candidate_series_u16(df: pd.DataFrame, can_id_dec: int, offset: int, endian: str) -> Optional[pd.DataFrame]:
    g = df[df["CAN_ID"] == int(can_id_dec)].copy().sort_values("Timestamp")
    if g.empty:
        return None

    u16 = compute_u16_series_from_group(g, offset, endian)
    out = pd.DataFrame({"Timestamp": g["Timestamp"].values, "u16": u16}).dropna()
    if len(out) < 50:
        return None
    return out

def pick_best_row_for_can_id(cands_df: pd.DataFrame, target_can_id: int) -> Optional[pd.Series]:
    """
    Returns the highest-ranked candidate row (bytepair/endian) for a given CAN ID.
    If CAN ID is not in candidates, returns None.
    """
    if cands_df is None or len(cands_df) == 0:
        return None

    sub = cands_df[cands_df["can_id_dec"].astype(int) == int(target_can_id)]
    if sub.empty:
        return None

    # already ranked, so top row is best
    return sub.iloc[0]


def plot_candidate_with_trigger_overlay(
    df,
    cand_row,
    trigger_bytes_df,
    zoom_mode="full",          # "full" or "auto"
    pad_seconds=10,
    prefer_trigger="Byte7",    # "Byte7" or "Byte2" or "either"
    on_value=0x04,
    y_percentile_clip=(1, 99),
    fig_size=(16, 7),          # taller plot
    y_pad_frac=0.08,           # extra headroom (8%)
    smooth=False,
    smooth_method="rolling",   # "rolling" or "ema"
    smooth_window=25,
    outpath: Optional[str] = None
):
    can_hex = cand_row["can_id_hex"]
    can_id = int(cand_row["can_id_dec"])
    byte_pair = cand_row["byte_pair"]
    endian = cand_row["endian"]

    # Candidate signal (u16 raw)
    s = u16_series_for_candidate(df, can_id, byte_pair, endian)
    t = s["Timestamp"].values
    u16_vals = s["u16"].values

    # Compute both interpretations
    i16_vals = u16_to_i16(u16_vals)

    # Trigger series (Byte7/Byte2)
    tb = trigger_bytes_df.sort_values("Timestamp").copy()
    tt = tb["Timestamp"].values

    if prefer_trigger == "Byte7":
        trig = tb["Byte7_state"].values
        trig_label = "0x275 Byte7"
        on_mask = (tb["Byte7_state"].values == on_value)
    elif prefer_trigger == "Byte2":
        trig = tb["Byte2_state"].values
        trig_label = "0x275 Byte2"
        on_mask = (tb["Byte2_state"].values == on_value)
    else:
        trig = ((tb["Byte7_state"].values == on_value) | (tb["Byte2_state"].values == on_value)).astype(int)
        trig_label = "0x275 ON mask"
        on_mask = (trig == 1)

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

        x_pad = 0.02 * (tmax - tmin)   # 2% padding on both sides
        tmin -= x_pad
        tmax += x_pad


    # Zoom masks
    z1 = (t >= tmin) & (t <= tmax)
    z2 = (tt >= tmin) & (tt <= tmax)

    t_zoom = t[z1]
    u16_zoom = u16_vals[z1]
    i16_zoom = i16_vals[z1]

    # Optional smoothing (apply AFTER zoom so window sizing is consistent)
    if smooth:
        u16_zoom = smooth_array(u16_zoom, method=smooth_method, window=smooth_window, ema_span=smooth_window)
        i16_zoom = smooth_array(i16_zoom, method=smooth_method, window=smooth_window, ema_span=smooth_window)


    if len(t_zoom) < 5:
        print(f"Skipping {can_hex} {byte_pair} {endian}: not enough points in window.")
        return

    # Y-limits based on both series combined (prevents clipping)
    combined = np.concatenate([u16_zoom[np.isfinite(u16_zoom)], i16_zoom[np.isfinite(i16_zoom)]])
    if len(combined) > 20:
        lo, hi = np.nanpercentile(combined, y_percentile_clip)
        y_min, y_max = float(lo), float(hi)
    else:
        y_min = float(np.nanmin(combined))
        y_max = float(np.nanmax(combined))

    # Add headroom padding so it doesn't slam into the top/bottom
    y_rng = max(1e-9, (y_max - y_min))
    y_min = y_min - y_pad_frac * y_rng
    y_max = y_max + y_pad_frac * y_rng

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=fig_size)

    # Unsigned (red) + signed (yellow) on same axis
    ax1.plot(t_zoom, u16_zoom, color="red", linewidth=1.0, label="Candidate unsigned u16")
    ax1.plot(t_zoom, i16_zoom, color="gold", linewidth=1.0, label="Candidate signed i16")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Candidate value")
    ax1.grid(True, linewidth=0.5)
    ax1.set_xlim(tmin, tmax)
    ax1.set_ylim(y_min, y_max)

    # Trigger overlay (right axis) dark blue
    ax2 = ax1.twinx()
    ax2.step(tt[z2], trig[z2], where="post", color="darkblue", linewidth=1.2, label=trig_label)
    ax2.set_ylabel(trig_label)

    # Title
    smooth_tag = f"SMOOTH({smooth_method}, N={smooth_window})" if smooth else "RAW"
    ax1.set_title(f"{can_hex}  {byte_pair}  {endian}  (u16 + i16)  + Trigger Overlay   [{smooth_tag}]")


    # Legends (left + right)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()
        

def infer_test_folder_name(csv_path: str) -> str:
    """
    Tries to infer a folder name like 'Test_4' from the CSV filename.
    If it can't find one, it falls back to the base filename (sanitized).
    """
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Match patterns like:
    # "Ram 4500 Test 4", "RAM_Test_4", "test4", "Test-04"
    m = re.search(r"(test)\s*[_\-]?\s*(\d+)", base, flags=re.IGNORECASE)
    if m:
        return f"Test_{int(m.group(2))}"

    # fallback: sanitize filename into folder-safe name
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_")
    return safe if safe else "unknown_test"


def prepare_clean_output_folder(root_outdir: str, csv_path: str) -> str:
    """
    Creates lka_results/<test_folder>/ and clears it if it already has contents.
    Returns the full path to the test folder.
    """
    test_folder = infer_test_folder_name(csv_path)
    outdir = os.path.join(root_outdir, test_folder)

    os.makedirs(outdir, exist_ok=True)

    # Clear contents (but keep the folder)
    for name in os.listdir(outdir):
        p = os.path.join(outdir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            print(f"Warning: could not remove {p}: {e}")

    return outdir


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Find distance-like CAN candidates using LKA trigger 0x275 Byte7.")
    parser.add_argument("--csv", required=True, help="Path to the CSV log file")
    parser.add_argument("--outdir", default="lka_results", help="Output directory for plots/results")
    parser.add_argument("--topn", type=int, default=25, help="Number of top candidates to keep")
    parser.add_argument("--pre", type=float, default=8.0, help="Pre-window seconds for scoring")
    parser.add_argument("--post", type=float, default=8.0, help="Post-window seconds for scoring")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Create/clean a dedicated folder for this test inside args.outdir
    test_outdir = prepare_clean_output_folder(args.outdir, args.csv)
    print(f"Results folder: {test_outdir}")

    # Load
    df = load_and_clean_csv(args.csv)

    # Trigger states + ON events
    trigger_df = compute_trigger_bytes(df, lka_id=LKA_ID)
    on_events = detect_on_events(trigger_df, on_value=TRIGGER_ON_VALUE)

    print(f"Found {len(on_events)} ON events (Byte7 -> {hex(TRIGGER_ON_VALUE)})")

    # Trigger plot (whole test)
    trigger_plot_path = os.path.join(test_outdir, "trigger_states_whole_test.png")
    plot_trigger_states_whole_test(trigger_df, on_events, trigger_plot_path)
    print(f"Saved trigger plot: {trigger_plot_path}")

    # Candidate scoring (relaxed)
    cands = build_candidates_multi_event_relaxed(
        df=df,
        on_events=on_events,
        exclude_id=LKA_ID,
        pre_window=args.pre,
        post_window=args.post,
        top_n=args.topn,
    )

    if cands.empty:
        print("No candidates found with current scoring settings.")
        return

    # Save table
    csv_out = os.path.join(test_outdir, "top_candidates.csv")
    cands.to_csv(csv_out, index=False)
    print(f"Saved candidates table: {csv_out}")

    # Dedup by CAN ID (keep best bytepair/endian per ID)
    cands_unique = dedupe_by_can_id(cands, score_col="final_score")
    print(f"Unique CAN IDs in top list: {len(cands_unique)}")

    # --------------------------
    # Grab signed i16 values at each ON event for 0x126 and 0x220
    # --------------------------
    sample_table = build_on_event_sample_table(
        df=df,
        trigger_df=trigger_df,
        on_events=on_events,
        target_ids=[0x126, 0x220]
    )

    sample_csv = os.path.join(test_outdir, "on_event_signed_samples_126_220.csv")
    sample_table.to_csv(sample_csv, index=False)
    print(f"Saved ON-event signed sample table: {sample_csv}")
    print("\nSigned values at ON events:")
    print(sample_table)


    # --------------------------
    # ALWAYS plot a specific CAN ID (even if it’s not in top candidates)
    # --------------------------
    FORCE_CAN_ID = 0x126

    best_row = pick_best_row_for_can_id(cands, FORCE_CAN_ID)

    if best_row is None:
        print("0x126 not found in ranked candidates — plotting fallback Byte1-Byte2 little.")
        best_row = pd.Series({
            "can_id_hex": hex(FORCE_CAN_ID),
            "can_id_dec": FORCE_CAN_ID,
            "byte_pair": "(Byte1,Byte2)",
            "endian": "little",
            "final_score": -1,
        })

    force_outpath = os.path.join(test_outdir, f"FORCED_{FORCE_CAN_ID:03x}_{best_row['byte_pair']}_{best_row['endian']}.png")
    plot_candidate_with_trigger_overlay(
        df=df,
        cand_row=best_row,
        trigger_bytes_df=trigger_df,
        zoom_mode="full",
        outpath=force_outpath
    )
    print(f"Saved forced 0x126 plot: {force_outpath}")

    # Save smoothed version too
    forced_outpath_smooth = os.path.join(test_outdir, f"FORCED_{FORCE_CAN_ID:03x}_{best_row['byte_pair']}_{best_row['endian']}_smoothed.png")

    plot_candidate_with_trigger_overlay(
        df=df,
        cand_row=best_row,
        trigger_bytes_df=trigger_df,
        zoom_mode="full",
        outpath=forced_outpath_smooth,
        smooth=True,
        smooth_method="rolling",   # or "ema"
        smooth_window=25
    )


    # Plot each candidate across whole test
    for i in range(len(cands_unique)): 
        row = cands_unique.iloc[i]
        can_hex = row["can_id_hex"].replace("0x", "")
        byte_pair = row["byte_pair"].replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
        endian = row["endian"]

        outpath = os.path.join(test_outdir, f"cand_{i:02d}_{can_hex}_{byte_pair}_{endian}.png")
        plot_candidate_with_trigger_overlay(
            df=df,
            cand_row=row,
            trigger_bytes_df=trigger_df,
            zoom_mode="full",
            outpath=outpath,
            smooth=False
        )

        # Save smoothed version too
        outpath_smooth = os.path.join(test_outdir, f"cand_{i:02d}_{can_hex}_{byte_pair}_{endian}_smoothed.png")

        plot_candidate_with_trigger_overlay(
            df=df,
            cand_row=row,
            trigger_bytes_df=trigger_df,
            zoom_mode="full",
            outpath=outpath_smooth,
            smooth=True,
            smooth_method="rolling",   # or "ema"
            smooth_window=25
        )

    print(f"Saved {len(cands_unique)} candidate plots to: {test_outdir}")

    ranked = explore_decode_space_for_targets(
        df=df,
        on_events=on_events,
        target_ids=[0x126, 0x220],
        pre_window=args.pre,
        post_window=args.post,
    )

    # --------------------------
    # Save + plot the explored decode configs
    # - Save exploration CSV inside a subfolder within the test folder
    # - Plot TOP 15 explored options for BOTH 0x126 and 0x220
    # --------------------------
    explore_outdir = os.path.join(test_outdir, "explore_decode_space")
    os.makedirs(explore_outdir, exist_ok=True)

    explore_csv = os.path.join(explore_outdir, "multi_byte_ranked_126_220.csv")
    ranked.to_csv(explore_csv, index=False)
    print(f"Saved multi-byte exploration ranked table: {explore_csv}")
    print(ranked.head(20))

    for cid in [0x126, 0x220]:
        top_df = ranked[ranked["can_id_dec"].astype(int) == int(cid)].head(15).reset_index(drop=True)
        if top_df.empty:
            print(f"[WARN] No explore results for {hex(cid)}")
            continue

        for rank in range(len(top_df)):
            row = top_df.iloc[rank]

            out_raw = os.path.join(explore_outdir, f"EXPLORE_{cid:03x}_top{rank+1:02d}_raw.png")
            out_smooth = os.path.join(explore_outdir, f"EXPLORE_{cid:03x}_top{rank+1:02d}_smoothed.png")

            plot_from_explore_row(
                df=df,
                explore_row=row,
                trigger_bytes_df=trigger_df,
                outpath=out_raw,
                smooth=False,
            )

            plot_from_explore_row(
                df=df,
                explore_row=row,
                trigger_bytes_df=trigger_df,
                outpath=out_smooth,
                smooth=True,
                smooth_method="rolling",
                smooth_window=25,
            )
        

if __name__ == "__main__":
    main()
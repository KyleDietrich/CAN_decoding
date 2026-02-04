#!/usr/bin/env python3
"""
trigger_finder.py

Automatically find candidate LKA trigger CAN IDs and bytes from CAN bus data.

Looks for signals that:
- Have a dominant "idle" value most of the time
- Transition to one or more distinct "active" values occasionally
- Have a relatively small number of unique values (state-based, not continuous)
- Show clear transition events (not noisy/continuous signals)

Usage:
    python trigger_finder.py --csv data.csv
    python trigger_finder.py --csv data.csv --min-events 2 --max-unique 10
"""

import argparse
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BYTE_COLS = [f"Byte{i}" for i in range(1, 9)]


# =========================
# Helpers
# =========================
def hex_to_int(x):
    """Convert hex string to int, return NaN for invalid values."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "errorframe":
        return np.nan
    s = s.replace("0x", "").replace("0X", "")
    if not all(ch in "0123456789abcdefABCDEF" for ch in s):
        return np.nan
    try:
        return int(s, 16)
    except Exception:
        return np.nan


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """Load and clean CAN CSV data."""
    df = pd.read_csv(csv_path)
    
    print(f"Loaded: {csv_path}")
    print(f"Rows (raw): {len(df)}")
    
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
    print(f"Time range: {df['Timestamp'].min():.2f}s - {df['Timestamp'].max():.2f}s")
    print()
    
    return df


@dataclass
class TriggerCandidate:
    """Stores analysis results for a potential trigger signal."""
    can_id: int
    byte_num: int  # 1-8
    unique_values: List[int]
    dominant_value: int
    dominant_pct: float
    active_values: List[int]
    transition_count: int
    score: float
    
    def __str__(self):
        active_hex = [hex(v) for v in self.active_values]
        return (
            f"CAN {hex(self.can_id)} Byte{self.byte_num}: "
            f"idle={hex(self.dominant_value)} ({self.dominant_pct:.1f}%), "
            f"active={active_hex}, "
            f"transitions={self.transition_count}, "
            f"score={self.score:.1f}"
        )


def analyze_byte_for_triggers(
    timestamps: np.ndarray,
    values: np.ndarray,
    can_id: int,
    byte_num: int,
    min_events: int = 2,
    max_unique: int = 15,
    min_dominant_pct: float = 50.0,
) -> Optional[TriggerCandidate]:
    """
    Analyze a single byte stream to see if it looks like a trigger signal.
    
    Args:
        timestamps: Array of timestamps
        values: Array of byte values
        can_id: CAN ID for reporting
        byte_num: Byte number (1-8) for reporting
        min_events: Minimum number of transition events required
        max_unique: Maximum unique values (filters out continuous signals)
        min_dominant_pct: Minimum percentage for dominant/idle value
    
    Returns:
        TriggerCandidate if this looks like a trigger, None otherwise
    """
    # Remove NaN values
    mask = ~np.isnan(values)
    values = values[mask].astype(int)
    timestamps = timestamps[mask]
    
    if len(values) < 100:
        return None
    
    # Get unique values and their counts
    unique_vals, counts = np.unique(values, return_counts=True)
    
    # Filter: too many unique values = probably continuous signal
    if len(unique_vals) > max_unique:
        return None
    
    # Filter: only 1 unique value = no transitions at all
    if len(unique_vals) < 2:
        return None
    
    # Find dominant (idle) value
    dominant_idx = np.argmax(counts)
    dominant_value = unique_vals[dominant_idx]
    dominant_count = counts[dominant_idx]
    dominant_pct = 100.0 * dominant_count / len(values)
    
    # Filter: must have a clear dominant value
    if dominant_pct < min_dominant_pct:
        return None
    
    # Active values = everything except dominant
    active_values = [int(v) for v in unique_vals if v != dominant_value]
    
    # Count transitions INTO active values (from non-active)
    prev_values = np.roll(values, 1)
    prev_values[0] = values[0]
    
    # Transition = current is active AND previous was not active (or was different active)
    is_active = np.isin(values, active_values)
    was_not_active = ~np.isin(prev_values, active_values)
    transitions = np.sum(is_active & was_not_active)
    
    # Filter: need minimum number of transition events
    if transitions < min_events:
        return None
    
    # Score the candidate
    # Higher score = better trigger candidate
    # - More transitions (up to a point) is good
    # - Higher dominant percentage is good
    # - Fewer unique values is good (cleaner state machine)
    score = (
        min(transitions, 20) * 10  # Cap transition contribution
        + dominant_pct * 0.5       # Reward clear idle state
        - len(unique_vals) * 2     # Penalty for too many states
    )
    
    return TriggerCandidate(
        can_id=can_id,
        byte_num=byte_num,
        unique_values=[int(v) for v in unique_vals],
        dominant_value=int(dominant_value),
        dominant_pct=dominant_pct,
        active_values=active_values,
        transition_count=int(transitions),
        score=score,
    )


def find_trigger_candidates(
    df: pd.DataFrame,
    min_events: int = 2,
    max_unique: int = 15,
    min_dominant_pct: float = 50.0,
    top_n: int = 20,
) -> List[TriggerCandidate]:
    """
    Scan all CAN IDs and bytes to find potential trigger signals.
    
    Args:
        df: Cleaned CAN dataframe
        min_events: Minimum transition events required
        max_unique: Maximum unique values allowed
        min_dominant_pct: Minimum percentage for dominant value
        top_n: Number of top candidates to return
    
    Returns:
        List of TriggerCandidate objects, sorted by score descending
    """
    candidates = []
    
    can_ids = df["CAN_ID"].unique()
    print(f"Analyzing {len(can_ids)} CAN IDs x 8 bytes = {len(can_ids) * 8} signals...")
    
    for can_id in can_ids:
        can_id = int(can_id)
        g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
        
        if len(g) < 100:
            continue
        
        timestamps = g["Timestamp"].values
        
        for byte_num in range(1, 9):
            byte_col = f"Byte{byte_num}"
            values = g[byte_col].values
            
            candidate = analyze_byte_for_triggers(
                timestamps=timestamps,
                values=values,
                can_id=can_id,
                byte_num=byte_num,
                min_events=min_events,
                max_unique=max_unique,
                min_dominant_pct=min_dominant_pct,
            )
            
            if candidate is not None:
                candidates.append(candidate)
    
    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    
    return candidates[:top_n]


def print_candidate_details(df: pd.DataFrame, candidate: TriggerCandidate):
    """Print detailed analysis of a trigger candidate."""
    can_id = candidate.can_id
    byte_num = candidate.byte_num
    byte_col = f"Byte{byte_num}"
    
    g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
    values = g[byte_col].values
    timestamps = g["Timestamp"].values
    
    # Remove NaN
    mask = ~np.isnan(values)
    values = values[mask].astype(int)
    timestamps = timestamps[mask]
    
    print(f"\n{'='*60}")
    print(f"CAN ID: {hex(can_id)} | Byte: {byte_num}")
    print(f"{'='*60}")
    
    # Value distribution
    print(f"\nValue Distribution:")
    unique_vals, counts = np.unique(values, return_counts=True)
    for val, cnt in sorted(zip(unique_vals, counts), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(values)
        marker = " <-- IDLE" if val == candidate.dominant_value else (" <-- ACTIVE" if val in candidate.active_values else "")
        print(f"  {hex(val):>6}: {cnt:6d} ({pct:5.1f}%){marker}")
    
    # Find transition times
    print(f"\nTransition Events (into active states):")
    prev_values = np.roll(values, 1)
    prev_values[0] = values[0]
    
    is_active = np.isin(values, candidate.active_values)
    was_not_active = ~np.isin(prev_values, candidate.active_values)
    transition_mask = is_active & was_not_active
    
    transition_times = timestamps[transition_mask]
    transition_vals = values[transition_mask]
    
    for i, (t, v) in enumerate(zip(transition_times[:15], transition_vals[:15])):
        print(f"  {i+1:2d}. t={t:8.3f}s -> {hex(v)}")
    
    if len(transition_times) > 15:
        print(f"  ... and {len(transition_times) - 15} more")
    
    print(f"\nSuggested usage:")
    if len(candidate.active_values) == 1:
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
    Shows the byte value with idle and active regions highlighted.
    """
    can_id = candidate.can_id
    byte_num = candidate.byte_num
    byte_col = f"Byte{byte_num}"
    
    g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
    values = g[byte_col].values
    timestamps = g["Timestamp"].values
    
    # Remove NaN
    mask = ~np.isnan(values)
    values = values[mask].astype(int)
    timestamps = timestamps[mask]
    
    if len(values) < 10:
        print(f"Not enough data to plot {hex(can_id)} Byte{byte_num}")
        return
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the signal
    ax.step(timestamps, values, where="post", linewidth=1.2, color="blue", label=f"{hex(can_id)} Byte{byte_num}")
    
    # Mark transition events with vertical lines
    prev_values = np.roll(values, 1)
    prev_values[0] = values[0]
    is_active = np.isin(values, candidate.active_values)
    was_not_active = ~np.isin(prev_values, candidate.active_values)
    transition_mask = is_active & was_not_active
    transition_times = timestamps[transition_mask]
    
    for t in transition_times:
        ax.axvline(t, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    
    # Add horizontal lines for idle and active values
    ax.axhline(candidate.dominant_value, color="green", linestyle=":", linewidth=1.0, alpha=0.5, label=f"Idle ({hex(candidate.dominant_value)})")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Byte Value")
    
    active_hex = ", ".join(hex(v) for v in candidate.active_values)
    ax.set_title(f"{hex(can_id)} Byte{byte_num} | Idle={hex(candidate.dominant_value)} ({candidate.dominant_pct:.1f}%) | Active={active_hex} | {candidate.transition_count} transitions")
    
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
    fig_size: Tuple[int, int] = (16, 12),
):
    """
    Plot all 8 bytes of a CAN ID in a single figure with subplots.
    Useful for exploring a specific CAN ID to understand its structure.
    """
    g = df[df["CAN_ID"] == can_id].sort_values("Timestamp")
    
    if len(g) < 10:
        print(f"Not enough data to plot {hex(can_id)}")
        return
    
    timestamps = g["Timestamp"].values
    
    fig, axes = plt.subplots(4, 2, figsize=fig_size, sharex=True)
    axes = axes.flatten()
    
    for byte_num in range(1, 9):
        ax = axes[byte_num - 1]
        byte_col = f"Byte{byte_num}"
        values = g[byte_col].values
        
        # Handle NaN
        mask = ~np.isnan(values)
        plot_timestamps = timestamps[mask]
        plot_values = values[mask].astype(int)
        
        if len(plot_values) < 5:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Byte{byte_num}")
            continue
        
        # Get value distribution
        unique_vals, counts = np.unique(plot_values, return_counts=True)
        dominant_val = unique_vals[np.argmax(counts)]
        dominant_pct = 100 * np.max(counts) / len(plot_values)
        
        # Plot
        ax.step(plot_timestamps, plot_values, where="post", linewidth=1.0, color="blue")
        
        # Mark transitions from dominant value
        prev = np.roll(plot_values, 1)
        prev[0] = dominant_val
        trans_mask = (prev == dominant_val) & (plot_values != dominant_val)
        trans_count = np.sum(trans_mask)
        
        for t in plot_timestamps[trans_mask]:
            ax.axvline(t, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        
        # Horizontal line for dominant
        ax.axhline(dominant_val, color="green", linestyle=":", linewidth=1.0, alpha=0.5)
        
        ax.set_ylabel(f"Byte{byte_num}")
        ax.grid(True, alpha=0.3)
        
        # Title with stats
        unique_str = ", ".join(hex(v) for v in sorted(unique_vals)[:5])
        if len(unique_vals) > 5:
            unique_str += "..."
        ax.set_title(f"Byte{byte_num}: idle={hex(dominant_val)} ({dominant_pct:.0f}%), {trans_count} trans, vals=[{unique_str}]", fontsize=9)
    
    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    
    fig.suptitle(f"CAN ID {hex(can_id)} - All Bytes", fontsize=12, fontweight='bold')
    fig.tight_layout()
    
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
            "dominant_value_hex": hex(c.dominant_value),
            "dominant_pct": round(c.dominant_pct, 1),
            "active_values_hex": " ".join(hex(v) for v in c.active_values),
            "num_unique": len(c.unique_values),
            "transition_count": c.transition_count,
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
    else:
        return int(val, 16)


def main():
    parser = argparse.ArgumentParser(
        description="Find candidate LKA trigger CAN IDs and bytes from CAN bus data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trigger_finder.py --csv data.csv
  python trigger_finder.py --csv data.csv --min-events 3 --top 30
  python trigger_finder.py --csv data.csv --details 5 --plot 5
  python trigger_finder.py --csv data.csv --force-plot 0x412
        """
    )
    parser.add_argument("--csv", required=True, help="Path to CAN CSV log file")
    parser.add_argument("--min-events", type=int, default=2, 
                        help="Minimum number of transition events (default: 2)")
    parser.add_argument("--max-unique", type=int, default=15,
                        help="Maximum unique values per byte (default: 15)")
    parser.add_argument("--min-dominant-pct", type=float, default=50.0,
                        help="Minimum percentage for dominant/idle value (default: 50)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top candidates to show (default: 20)")
    parser.add_argument("--details", type=int, default=5,
                        help="Number of candidates to show detailed analysis for (default: 5)")
    parser.add_argument("--plot", type=int, default=5,
                        help="Number of top candidates to plot (default: 5)")
    parser.add_argument("--force-plot", type=str, nargs='+', default=None,
                        help="Force plot specific CAN IDs (hex, e.g., 0x412 or '0x412 0xD5')")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots (default: trigger_finder_results/<csv_name>)")
    parser.add_argument("--outcsv", type=str, default=None,
                        help="Path to save candidates CSV (optional)")
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_clean_csv(args.csv)
    
    # Find candidates
    candidates = find_trigger_candidates(
        df=df,
        min_events=args.min_events,
        max_unique=args.max_unique,
        min_dominant_pct=args.min_dominant_pct,
        top_n=args.top,
    )
    
    if not candidates:
        print("No trigger candidates found with current settings.")
        print("Try lowering --min-events or --min-dominant-pct")
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TOP {len(candidates)} TRIGGER CANDIDATES")
    print(f"{'='*60}")
    for i, c in enumerate(candidates):
        print(f"{i+1:2d}. {c}")
    
    # Print details for top N
    print(f"\n\nDETAILED ANALYSIS (top {args.details}):")
    for c in candidates[:args.details]:
        print_candidate_details(df, c)
    
    # Setup output directory for plots
    if args.outdir:
        plot_outdir = args.outdir
    else:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        plot_outdir = os.path.join("trigger_finder_results", base)
    
    os.makedirs(plot_outdir, exist_ok=True)
    
    # Plot top candidates
    if args.plot > 0:
        print(f"\n\nPLOTTING top {min(args.plot, len(candidates))} candidates to: {plot_outdir}")
        for i, c in enumerate(candidates[:args.plot]):
            outpath = os.path.join(
                plot_outdir, 
                f"candidate_{i+1:02d}_{c.can_id:03x}_byte{c.byte_num}.png"
            )
            plot_trigger_candidate(df, c, outpath=outpath)
            print(f"  Saved: {outpath}")
    
    # Force plot specific CAN IDs (all 8 bytes)
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

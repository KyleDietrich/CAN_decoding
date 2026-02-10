# nearby_id_explorer.py
"""
Exhaustively decode and plot all possible formats for CAN IDs near a target ID.

Usage:
    from nearby_id_explorer import explore_nearby_ids
    
    explore_nearby_ids(
        df=df,
        target_id=0x412,
        range_offset=5,
        trigger_df=trigger_df,
        lka_id=0x412,
        trigger_byte=1,
        trigger_values=[0x22],
        outdir="lka_results/Test_10/nearby_exploration"
    )
"""

from typing import List, Optional, Dict, Any
import os
import numpy as np
import pandas as pd
from plotting_utils import plot_candidate_with_trigger_overlay

BYTE_COLS = [f"Byte{i}" for i in range(1, 9)]


# =========================
# 4-byte decode functions (NEW)
# =========================
def decode_u32_from_bytes(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, endian: str) -> np.ndarray:
    """
    Decode unsigned 32-bit from four byte arrays.
    endian:
      - little => value = (b3<<24) | (b2<<16) | (b1<<8) | b0
      - big    => value = (b0<<24) | (b1<<16) | (b2<<8) | b3
    """
    out = np.full(len(b0), np.nan, dtype=np.float64)

    mask = (~np.isnan(b0)) & (~np.isnan(b1)) & (~np.isnan(b2)) & (~np.isnan(b3))
    if not np.any(mask):
        return out

    b0i = (b0[mask].astype(np.int64) & 0xFF)
    b1i = (b1[mask].astype(np.int64) & 0xFF)
    b2i = (b2[mask].astype(np.int64) & 0xFF)
    b3i = (b3[mask].astype(np.int64) & 0xFF)

    if endian == "little":
        out[mask] = (b3i << 24) | (b2i << 16) | (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 24) | (b1i << 16) | (b2i << 8) | b3i

    return out


def u32_to_i32(u32_vals: np.ndarray) -> np.ndarray:
    """Convert unsigned 32-bit to signed i32 (two's complement)."""
    u = np.asarray(u32_vals, dtype=np.float64)
    sign_bit = 1 << 31
    full = 1 << 32
    return np.where(u >= sign_bit, u - full, u)


# =========================
# Helper decode functions (reused from multi_byte_decoder)
# =========================
def decode_u16_from_bytes(b0: np.ndarray, b1: np.ndarray, endian: str) -> np.ndarray:
    out = np.full(len(b0), np.nan, dtype=np.float64)
    mask = (~np.isnan(b0)) & (~np.isnan(b1))
    if not np.any(mask):
        return out

    b0i = (b0[mask].astype(np.int64) & 0xFF)
    b1i = (b1[mask].astype(np.int64) & 0xFF)

    if endian == "little":
        out[mask] = (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 8) | b1i

    return out


def u16_to_i16(u16_vals: np.ndarray) -> np.ndarray:
    u = np.asarray(u16_vals, dtype=np.float64)
    return np.where(u >= 32768, u - 65536, u)


def decode_u24_from_bytes(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, endian: str) -> np.ndarray:
    out = np.full(len(b0), np.nan, dtype=np.float64)
    mask = (~np.isnan(b0)) & (~np.isnan(b1)) & (~np.isnan(b2))
    if not np.any(mask):
        return out

    b0i = (b0[mask].astype(np.int64) & 0xFF)
    b1i = (b1[mask].astype(np.int64) & 0xFF)
    b2i = (b2[mask].astype(np.int64) & 0xFF)

    if endian == "little":
        out[mask] = (b2i << 16) | (b1i << 8) | b0i
    else:
        out[mask] = (b0i << 16) | (b1i << 8) | b2i

    return out


def u24_to_i24(u24_vals: np.ndarray) -> np.ndarray:
    u = np.asarray(u24_vals, dtype=np.float64)
    sign_bit = 1 << 23
    full = 1 << 24
    return np.where(u >= sign_bit, u - full, u)


def sign_extend(value: int, bit_len: int) -> int:
    if bit_len <= 0:
        return value
    sign_bit = 1 << (bit_len - 1)
    mask = (1 << bit_len) - 1
    v = value & mask
    return (v ^ sign_bit) - sign_bit


def decode_bitfield_from_u16(u16_val: int, bit_shift: int, bit_len: int, signed: bool) -> int:
    raw = (u16_val >> bit_shift) & ((1 << bit_len) - 1)
    if signed:
        return sign_extend(raw, bit_len)
    return raw


# =========================
# Series builders
# =========================
def build_series_for_config(df: pd.DataFrame, can_id: int, cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Build a DataFrame with [Timestamp, unsigned, signed] for any decode config.
    """
    kind = cfg["kind"]
    start_byte = cfg["start_byte"]
    endian = cfg["endian"]

    g = df[df["CAN_ID"] == int(can_id)].copy().sort_values("Timestamp")
    if g.empty or len(g) < 30:
        return None

    if kind == "u16/i16":
        offset = int(start_byte) - 1
        if offset < 0 or offset > 6:
            return None

        b0 = g[BYTE_COLS[offset]].to_numpy(dtype=np.float64)
        b1 = g[BYTE_COLS[offset + 1]].to_numpy(dtype=np.float64)

        u16 = decode_u16_from_bytes(b0, b1, endian)
        i16 = u16_to_i16(u16)

        out = pd.DataFrame({
            "Timestamp": g["Timestamp"].values,
            "unsigned": u16,
            "signed": i16
        }).dropna()
        return out if len(out) >= 30 else None

    elif kind == "u24/i24":
        offset = int(start_byte) - 1
        if offset < 0 or offset > 5:
            return None

        b0 = g[BYTE_COLS[offset]].to_numpy(dtype=np.float64)
        b1 = g[BYTE_COLS[offset + 1]].to_numpy(dtype=np.float64)
        b2 = g[BYTE_COLS[offset + 2]].to_numpy(dtype=np.float64)

        u24 = decode_u24_from_bytes(b0, b1, b2, endian)
        i24 = u24_to_i24(u24)

        out = pd.DataFrame({
            "Timestamp": g["Timestamp"].values,
            "unsigned": u24,
            "signed": i24
        }).dropna()
        return out if len(out) >= 30 else None

    elif kind == "u32/i32":
        offset = int(start_byte) - 1
        if offset < 0 or offset > 4:
            return None

        b0 = g[BYTE_COLS[offset]].to_numpy(dtype=np.float64)
        b1 = g[BYTE_COLS[offset + 1]].to_numpy(dtype=np.float64)
        b2 = g[BYTE_COLS[offset + 2]].to_numpy(dtype=np.float64)
        b3 = g[BYTE_COLS[offset + 3]].to_numpy(dtype=np.float64)

        u32 = decode_u32_from_bytes(b0, b1, b2, b3, endian)
        i32 = u32_to_i32(u32)

        out = pd.DataFrame({
            "Timestamp": g["Timestamp"].values,
            "unsigned": u32,
            "signed": i32
        }).dropna()
        return out if len(out) >= 30 else None

    elif kind == "bitfield":
        # Bitfield from u16
        offset = int(start_byte) - 1
        if offset < 0 or offset > 6:
            return None

        b0 = g[BYTE_COLS[offset]].to_numpy(dtype=np.float64)
        b1 = g[BYTE_COLS[offset + 1]].to_numpy(dtype=np.float64)

        u16 = decode_u16_from_bytes(b0, b1, endian)
        
        # Decode unsigned
        unsigned_vals = []
        for val in u16:
            if np.isnan(val):
                unsigned_vals.append(np.nan)
            else:
                unsigned_vals.append(float(decode_bitfield_from_u16(int(val), cfg["bit_shift"], cfg["bit_len"], signed=False)))
        
        # Decode signed
        signed_vals = []
        for val in u16:
            if np.isnan(val):
                signed_vals.append(np.nan)
            else:
                signed_vals.append(float(decode_bitfield_from_u16(int(val), cfg["bit_shift"], cfg["bit_len"], signed=True)))

        out = pd.DataFrame({
            "Timestamp": g["Timestamp"].values,
            "unsigned": np.array(unsigned_vals),
            "signed": np.array(signed_vals)
        }).dropna()
        return out if len(out) >= 30 else None

    elif kind == "single_byte":
        byte_num = int(start_byte)
        byte_col = BYTE_COLS[byte_num - 1]
        
        byte_vals = g[byte_col].to_numpy(dtype=np.float64)
        i8_vals = np.where(byte_vals >= 128, byte_vals - 256, byte_vals)

        out = pd.DataFrame({
            "Timestamp": g["Timestamp"].values,
            "unsigned": byte_vals,
            "signed": i8_vals
        }).dropna()
        return out if len(out) >= 30 else None

    return None


# =========================
# Config generators
# =========================
def generate_all_configs() -> List[Dict[str, Any]]:
    """Generate ALL possible decode configs: single-byte, 2-byte, 3-byte, 4-byte, bitfields."""
    configs = []

    # # Single bytes (Byte1 through Byte8)
    # for byte_num in range(1, 9):
    #     configs.append({
    #         "kind": "single_byte",
    #         "start_byte": byte_num,
    #         "endian": "N/A",
    #         "label": f"Byte{byte_num}"
    #     })

    # # 2-byte (Byte1-2 through Byte7-8)
    # for start_byte in range(1, 8):
    #     for endian in ["little", "big"]:
    #         configs.append({
    #             "kind": "u16/i16",
    #             "start_byte": start_byte,
    #             "endian": endian,
    #             "label": f"u16_Byte{start_byte}_{endian}"
    #         })

    # # 3-byte (Byte1-3 through Byte6-8)
    # for start_byte in range(1, 7):
    #     for endian in ["little", "big"]:
    #         configs.append({
    #             "kind": "u24/i24",
    #             "start_byte": start_byte,
    #             "endian": endian,
    #             "label": f"u24_Byte{start_byte}_{endian}"
    #         })

    # # 4-byte (Byte1-4 through Byte5-8)
    # for start_byte in range(1, 6):
    #     for endian in ["little", "big"]:
    #         configs.append({
    #             "kind": "u32/i32",
    #             "start_byte": start_byte,
    #             "endian": endian,
    #             "label": f"u32_Byte{start_byte}_{endian}"
    #         })

    # Bitfields (reasonable subset)
    for start_byte in range(1, 8):
        for endian in ["little", "big"]:
            for bit_len in [8, 10, 12, 14]:
                for bit_shift in [0, 2, 4, 6]:
                    if bit_shift + bit_len > 16:
                        continue
                    configs.append({
                        "kind": "bitfield",
                        "start_byte": start_byte,
                        "endian": endian,
                        "bit_len": bit_len,
                        "bit_shift": bit_shift,
                        "label": f"bitfield_Byte{start_byte}_{endian}_len{bit_len}_shift{bit_shift}"
                    })

    return configs


# =========================
# Main exploration function
# =========================
def explore_nearby_ids(
    df: pd.DataFrame,
    target_id: int,
    range_offset: int,
    trigger_df: pd.DataFrame,
    lka_id: int,
    trigger_byte: int,
    trigger_values: List[int],
    outdir: str,
    smooth: bool = False,
):
    """
    Exhaustively decode and plot all formats for CAN IDs near target_id.
    
    Args:
        df: Full CAN dataframe
        target_id: Center CAN ID (e.g., 0x412)
        range_offset: How many existing IDs up/down to explore (e.g., 5)
        trigger_df: Trigger state DataFrame
        lka_id: LKA CAN ID for trigger overlay
        trigger_byte: Trigger byte number
        trigger_values: List of trigger values
        outdir: Output directory for plots
        smooth: Whether to smooth plots
    """
    os.makedirs(outdir, exist_ok=True)

    # Get all unique CAN IDs that exist in the data, sorted
    all_ids = sorted(df["CAN_ID"].unique())
    all_ids = [int(x) for x in all_ids if not np.isnan(x)]
    
    print(f"\nTotal unique CAN IDs in data: {len(all_ids)}")
    
    # Find target_id position in sorted list
    if target_id not in all_ids:
        print(f"Warning: Target ID {hex(target_id)} not found in data!")
        print(f"Available IDs near target: {[hex(x) for x in all_ids if abs(x - target_id) < 20]}")
        return
    
    target_idx = all_ids.index(target_id)
    
    # Select range_offset IDs below and above
    start_idx = max(0, target_idx - range_offset)
    end_idx = min(len(all_ids), target_idx + range_offset + 1)
    
    target_ids = all_ids[start_idx:end_idx]
    
    print(f"\nExploring {len(target_ids)} nearest CAN IDs around {hex(target_id)}:")
    print(f"  IDs: {[hex(x) for x in target_ids]}")
    print(f"  Target position: index {target_idx} (±{range_offset} neighbors)")

    # Generate all decode configs
    all_configs = generate_all_configs()
    print(f"  Total decode configs per ID: {len(all_configs)}")
    print(f"  Total plots to generate: {len(target_ids) * len(all_configs)}")

    plot_count = 0
    for can_id in target_ids:
        can_hex = hex(can_id).replace("0x", "")
        
        # Check if this CAN ID has enough data
        if (df["CAN_ID"] == can_id).sum() < 30:
            print(f"Skipping {hex(can_id)}: insufficient data")
            continue

        for cfg in all_configs:
            s = build_series_for_config(df, can_id, cfg)
            if s is None or len(s) < 30:
                continue

            # Generate filename
            filename = f"{can_hex}_{cfg['label']}.png"
            outpath = os.path.join(outdir, filename)

            # Plot
            try:
                plot_candidate_with_trigger_overlay(
                    t=s["Timestamp"].values,
                    u16_vals=s["unsigned"].values,
                    i16_vals=s["signed"].values,
                    trigger_df=trigger_df,
                    lka_id=lka_id,
                    trigger_byte=trigger_byte,
                    trigger_values=trigger_values,
                    title=f"{hex(can_id)} {cfg['label']}",
                    zoom_mode="full",
                    outpath=outpath,
                    smooth=smooth,
                )
                plot_count += 1
            except Exception as e:
                print(f"  Error plotting {filename}: {e}")

    print(f"\nGenerated {plot_count} plots in: {outdir}")

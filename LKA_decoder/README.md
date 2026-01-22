# LKA Distance-to-Lane CAN Decoder (RAM 4500)

This project is a Python-based workflow for **finding and visualizing the CAN signal that represents “distance to lane”** using a known LKA trigger message as a reference. 

**ONLY TESTED ON A RAM 4500**

We use CAN ID **0x275** (LKA trigger states) to detect when Lane Keep Assist becomes active, then search all other CAN messages for signals that become “alive” after that trigger.

The goal is to narrow down candidate CAN IDs / byte pairs that behave like a distance-to-lane measurement signal.

---

## What This Script Does

Given a CAN log CSV, the script:

### 1) Loads and cleans CAN data

- Reads the CSV exported from your CAN tool
- Converts:
  - CAN Identifier hex → integer CAN ID
  - Byte1–Byte8 hex → numeric
  - Timestamp → numeric float
- Removes invalid rows (missing timestamp / invalid CAN_ID)

### 2) Extracts the LKA trigger states (CAN ID `0x275`)

From ID **0x275** it pulls:

- `Byte2_state`
- `Byte7_state`

These bytes appear to represent LKA trigger/command states.

### 3) Detects ON events

An “ON event” is defined as:

- `Byte7_state` transitioning into **0x04**

These ON timestamps become our alignment reference.

### 4) Scores candidate signals across the full CAN bus

For each CAN ID (excluding `0x275`), the script tests:

Every 2-byte window (Byte1–Byte2 ... Byte7–Byte8)  
Both byte orders:

- **little endian**
- **big endian**

Each candidate is scored by comparing activity:

- **before** trigger ON
- **after** trigger ON

A good candidate is:

- significantly more active after ON
- reasonably consistent across multiple ON events
- not a pure counter / reset-like jump signal

### 5) Generates clear plots for candidates

For each top candidate, the script produces one plot showing:

- Candidate interpretation as **unsigned u16** (red)
- Candidate interpretation as **signed i16** (yellow)
- Trigger signal overlay from **0x275 Byte7** (dark blue) on a 2nd axis

Plots cover **the entire test duration** (full timeline).

### 6) Always plots CAN ID `0x126`

Even if it does not rank in the top candidate list, the script forces a plot for **CAN ID 0x126**, since testing suggests this ID may contain the distance-to-lane signal.

---

## Repository Contents

Typical structure:

```plaintext
truck_CAN_decode/
├── distance_to_lane_decoder.py
├── README.md
└── lka_results/
    ├── Test_4/
    │   ├── trigger_states_whole_test.png
    │   ├── top_candidates.csv
    │   ├── cand_00_...
    │   ├── cand_01_...
    │   └── FORCED_126_...
    └── Test_5/
        └── ...
```

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

Recommended: run inside a venv

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib
```

---

## Running the Script

Example:

```bash
python distance_to_lane_decoder.py --csv "/path/to/Ram 4500 Test 4.csv"
```

Optional arguments:

```bash
python distance_to_lane_decoder.py \
  --csv "/path/to/Test 4.csv" \
  --outdir "lka_results" \
  --topn 25 \
  --pre 10 \
  --post 10
```

### CLI Arguments

| Argument | Description | Default |
| --------- | ------------- | --------- |
| `--csv` | Path to CAN log CSV | **required** |
| `--outdir` | Root folder for output plots/results | `lka_results` |
| `--topn` | Number of top candidates to keep | `25` |
| `--pre` | Pre-trigger window (seconds) for scoring | `10.0` |
| `--post` | Post-trigger window (seconds) for scoring | `10.0` |

---

## Output Files

Each run creates a per-test folder under `lka_results/`.

If the folder already exists, it is cleared and replaced with the new output.

Inside the test folder you will find:

### `trigger_states_whole_test.png`

A step plot of:

- `0x275 Byte2`
- `0x275 Byte7`

With ON events marked.

### `top_candidates.csv`

The ranked candidate table (after scoring) including:

- CAN ID
- byte pair
- endianness
- activity score after ON
- activity score before ON
- gain ratio
- consistency factor
- final score

### `cand_XX_*.png`

Candidate plots for the ranked list (deduped to one best row per CAN ID).

### `FORCED_126_*.png`

A forced plot for CAN ID **0x126** (best scoring byte-pair/endian if available, otherwise fallback).

---

## How Candidate Scoring Works (High Level)

For each candidate (CAN ID + byte pair + endian):

1. Extract the candidate as an `int16` signal
2. For each ON event:
   - collect pre-window and post-window segments
   - compute an “activity score”:
     - `activity = range_p95_p5 * change_fraction`
3. Compute:
   - `post_activity_median`
   - `pre_activity_median`
   - `gain = post / pre` (capped)
   - `consistency` across events
4. Final score:
   - `final = post_median * gain * consistency`

This tends to surface signals that “wake up” right when LKA activates.

---

## Notes / Current Findings

Across testing:

- CAN ID **0x220** appeared often in early scoring runs
- CAN ID **0x102** appears in some tests
- CAN ID **0x126** has emerged as a strong candidate and visually resembles expected distance behavior in multiple trials

Because of that, 0x126 is always plotted for every test.

---

## Project Context

This script is built for CAN decoding experiments on a RAM 4500, using LKA trigger state CAN messages to reverse-engineer the distance-to-lane signal.

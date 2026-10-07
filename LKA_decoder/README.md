# LKA Decoder

Python scripts for reverse-engineering Lane Keep Assist (LKA) signals from CAN logs.

The workflow has two steps:

1. **Find the trigger** – `trigger_finder.py` searches the log for the CAN ID / byte that changes when LKA activates.
2. **Find the distance-to-lane signal** – `distance_to_lane_decoder.py` uses that trigger to find signals that "wake up" right after LKA turns on.

---

## Setup

```bash
pip install numpy pandas matplotlib
```

### Input CSV

The scripts expect a CAN log exported to CSV with:

- a CAN ID column (`id`, `Identifier`, or `ID`)
- a time column (`Time` or `Timestamp`, in seconds)
- data byte columns (`Data0`–`Data63` or `Byte1`–`Byte64`), as hex values

Both classic CAN (8 bytes) and CAN FD (up to 64 bytes) logs work.

---

## Step 1: Find the Trigger

```bash
python trigger_finder.py --csv "path/to/test.csv"
```

It looks for bytes (and single bits) that sit at one idle value most of the time and change only a few times, which is how an LKA status signal usually behaves.

Useful options:

| Argument | What it does |
| --- | --- |
| `--trigger-times 19 37 55` | Only keep candidates that change near these known times (seconds) |
| `--min-events` / `--max-events` | Limit how many times the value can change |
| `--min-unique` / `--max-unique` | Limit how many different values the byte can have |
| `--force-plot 0x412` | Plot all bytes of a specific CAN ID |
| `--plot 5` | Number of top candidates to plot |

Run `python trigger_finder.py -h` for the full list.

**Output:** `trigger_finder_results/<csv_name>/` with candidate plots and `trigger_candidates.csv`.

---

## Step 2: Find the Distance-to-Lane Signal

```bash
python distance_to_lane_decoder.py --csv "path/to/test.csv" \
  --lka-can-id 0x275 --trigger-byte 7 --trigger-value 0x04
```

| Argument | Description | Default |
| --- | --- | --- |
| `--csv` | Path to the CAN log | **required** |
| `--lka-can-id` | CAN ID of the trigger (hex) | **required** |
| `--trigger-byte` | Byte number (1–64) holding the trigger | **required** |
| `--trigger-value` | Value(s) that mean "LKA on" (hex, can list several) | **required** |
| `--outdir` | Output folder | `lka_results` |
| `--topn` | Number of candidates to keep | `25` |
| `--pre` | Seconds before each trigger to score | `8` |
| `--post` | Seconds after each trigger to score | `8` |

### How it works

1. Finds every time the trigger byte switches **into** one of the trigger values (an "ON event").
2. For every CAN ID, it tries every pair of neighboring bytes, both little and big endian.
3. Each pair is scored on how much more active it is **after** ON than **before**, and how consistent that is across all ON events. Counter-like signals are filtered out.
4. The best byte pair for each CAN ID is plotted over the whole test.

### Output

Results go to `lka_results/<Test_N>/` (the folder is cleared on each run):

- `trigger_states_whole_test.png` – the trigger byte with ON events marked
- `top_candidates.csv` – ranked candidates with their scores
- `cand_XX_<id>_<bytes>_<endian>.png` – candidate as unsigned (red) and signed (gold), with the trigger overlaid (blue)
- `cand_XX_..._smoothed.png` – same plot with a rolling-average filter

---

## Files

| File | Purpose |
| --- | --- |
| `trigger_finder.py` | Step 1: find the LKA trigger ID/byte |
| `distance_to_lane_decoder.py` | Step 2: score and plot distance-to-lane candidates |
| `candidate_scoring.py` | Scoring logic used by step 2 |
| `plotting_utils.py` | Shared plotting functions |
| `multi_byte_decoder.py` | Optional deeper search (3-byte values and bitfields) |
| `nearby_id_explorer.py` | Optional: plots every decode format for CAN IDs near a target ID |
| `on_event_sampling.py` | Helpers for sampling signal values at ON events |

The deep decode and nearby-ID searches are commented out in `distance_to_lane_decoder.py` by default. Uncomment those blocks in `main()` to run them.

---


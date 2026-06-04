# ============================================================
# config.py — Central configuration for BrainMRI FL benchmark
# ============================================================

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Kaggle ────────────────────────────────────────────────────────────────────
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.environ.get("KAGGLE_API_KEY",  "")

def check_kaggle():
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise EnvironmentError(
            "Kaggle credentials not set.\n"
            "  export KAGGLE_USERNAME='your_username'\n"
            "  export KAGGLE_API_KEY='your_key'\n"
        )

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / "data_brain"
RAW_DIR   = DATA_DIR / "raw"
NPY_DIR   = DATA_DIR / "npy"
TEST_DIR  = DATA_DIR / "test"
GS_DIR    = DATA_DIR / "gs"

for _d in [RAW_DIR, NPY_DIR, TEST_DIR, GS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Classes ───────────────────────────────────────────────────────────────────
# 4-class: glioma=0, meningioma=1, notumor=2, pituitary=3
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE    = 224

# ── FL parameters ─────────────────────────────────────────────────────────────
N_CLIENTS       = int(os.environ.get("N_CLIENTS",    "3"))
N_ROUNDS        = int(os.environ.get("N_ROUNDS",     "10"))
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", "3"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",   "32"))
LR              = float(os.environ.get("LR",         "0.0001"))
TASK_TIMEOUT    = 7200
FREEZE_BACKBONE = os.environ.get("FREEZE_BACKBONE", "1") == "1"
FREEZE_ROUNDS   = 999  # freeze backbone entire run — do NOT put in .env

# ── Repeat config ─────────────────────────────────────────────────────────────
# N_REPEATS       = int(os.environ.get("N_REPEATS",       "3"))
N_REPEATS       = int(os.environ.get("N_REPEATS",       "1")) # Quick smoke test
BASE_SPLIT_SEED = int(os.environ.get("BASE_SPLIT_SEED", "42"))

# ── GS settings ───────────────────────────────────────────────────────────────
GS_ITER_COUNT = int(os.environ.get("GS_ITER_COUNT", "50"))

# (run_name, use_gs, maskP) — order: baseline first, then GS high→low
RUNS = [
    ("baseline", False, 0.0),
    ("gs_50",    True,  0.5),
    ("gs_20",    True,  0.2),
    ("gs_0",     True,  0.0),
]

# ── Per-site sample distribution ──────────────────────────────────────────────
# Dataset: ~1321 glioma, ~1339 meningioma, ~1595 notumor, ~1457 pituitary
# (Training + Testing folders pooled — verify with convert.py output)
#
# Non-IID design: each site emphasizes a different tumor class
#   site_1: glioma-heavy
#   site_2: meningioma-heavy
#   site_3: pituitary-heavy
#   test:   balanced 100 per class = 400 total
#
# Per-repeat total usage: ~3 × (sum of site rows) + test
# With 3 repeats and different seeds, keep per-repeat total well below
# dataset size to avoid exhausting any class.
#
# PLACEHOLDER — finalize after running data/convert.py and checking
# actual class counts printed during conversion.
SITE_DISTRIBUTIONS = [
    # (site_idx, glioma, meningioma, notumor, pituitary)
    (1, 500, 500, 500, 500),   # balanced
    (2, 500, 500, 500, 500),   # balanced
    (3, 500, 500, 500, 500),   # balanced
]
# Test: balanced
TEST_N_PER_CLASS = 150   # 150 × 4 = 600 test images

# ── Output paths ──────────────────────────────────────────────────────────────
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = ROOT / "plots"
WS_DIR      = ROOT / "runs" / "workspaces"
RUN_STATE   = ROOT / "runs" / "state.json"

for _d in [CKPT_DIR, RESULTS_DIR, PLOTS_DIR, WS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

def ckpt_dir(run: str) -> Path:
    d = CKPT_DIR / run
    d.mkdir(parents=True, exist_ok=True)
    return d

def results_path(run: str) -> Path:
    return RESULTS_DIR / f"{run}.json"

def gs_out_dir(run: str) -> Path:
    d = GS_DIR / run
    d.mkdir(parents=True, exist_ok=True)
    return d

def ws_dir(run: str) -> Path:
    d = WS_DIR / run
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Summary ───────────────────────────────────────────────────────────────────
def print_config():
    print()
    print("  BrainMRI FL Benchmark — Config")
    print("  " + "─" * 40)
    print(f"  Root         : {ROOT}")
    print(f"  Classes      : {CLASSES}")
    print(f"  Clients      : {N_CLIENTS}")
    print(f"  Rounds       : {N_ROUNDS}")
    print(f"  Local epochs : {LOCAL_EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  GS iters     : {GS_ITER_COUNT}")
    print(f"  Freeze rounds: {FREEZE_ROUNDS} (backbone frozen entire run)")
    print()
    print("  Site distributions (placeholder — verify after convert):")
    for site_idx, *counts in SITE_DISTRIBUTIONS:
        total = sum(counts)
        parts = " ".join(f"{CLASSES[i]}={counts[i]}" for i in range(NUM_CLASSES))
        print(f"    site_{site_idx}: {parts}  total={total}")
    test_total = TEST_N_PER_CLASS * NUM_CLASSES
    print(f"    test   : {TEST_N_PER_CLASS} per class  total={test_total}")
    print()
    print("  Runs:")
    for name, use_gs, maskP in RUNS:
        gs_str = f"GS maskP={maskP}" if use_gs else "baseline (no GS)"
        print(f"    {name:<10} : {gs_str}")
    print("  " + "─" * 40)
    print()

if __name__ == "__main__":
    print_config()
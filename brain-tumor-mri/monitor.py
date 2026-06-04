# ============================================================
# monitor.py — Live terminal dashboard for BrainMRI FL benchmark
# Usage: python monitor.py
# ============================================================

import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
import zoneinfo
from config import RUNS, N_ROUNDS, N_REPEATS, LOCAL_EPOCHS, CLASSES

TZ       = zoneinfo.ZoneInfo("America/New_York")
ROOT     = Path(__file__).resolve().parent
WS_DIR   = ROOT / "runs" / "workspaces"
LOG      = ROOT / "runs" / "benchmark.log"
STATE    = ROOT / "runs" / "state.json"

BASE_RUNS = [r[0] for r in RUNS]
RUN_NAMES = [
    f"rep{rep}_{run}"
    for rep in range(1, N_REPEATS + 1)
    for run in BASE_RUNS
]
SITES = [f"site-{i+1}" for i in range(3)]

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
GRAY   = "\033[90m"
WHITE  = "\033[30m"
PURPLE = "\033[35m"

def clr(text, color):
    return f"{color}{text}{RESET}"

def clear_screen():
    os.system("clear")

def load_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {}

def get_latest_metrics(run_name: str) -> dict:
    p = ROOT / "results" / f"{run_name}.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        rounds = data.get("rounds", [])
        return rounds[-1] if rounds else {}
    except Exception:
        return {}

def get_client_status(run_name: str, site: str) -> dict:
    log_path = WS_DIR / f"train_{run_name}" / site / "log.txt"
    if not log_path.exists():
        return {"round": 0, "epoch": 0, "loss": None, "status": "waiting"}
    try:
        lines = log_path.read_text().splitlines()
    except Exception:
        return {"round": 0, "epoch": 0, "loss": None, "status": "waiting"}

    result = {"round": 0, "epoch": 0, "loss": None, "status": "waiting"}
    for line in reversed(lines):
        m = re.search(r"round=(\d+)\s+epoch=(\d+)/(\d+)\s+loss=([\d.]+)", line)
        if m:
            result = {
                "round" : int(m.group(1)),
                "epoch" : int(m.group(2)),
                "loss"  : float(m.group(4)),
                "status": "training",
            }
            break
        m2 = re.search(r"round=(\d+)\s+complete", line)
        if m2:
            result = {
                "round" : int(m2.group(1)),
                "status": "round_complete",
                "epoch" : 0, "loss": None,
            }
            break
    return result

def get_gs_status(run_name: str) -> str:
    log = WS_DIR / f"gs_{run_name}" / "server" / "log.txt"
    if not log.exists():
        return "pending"
    try:
        content = log.read_text()
        if "All clients completed successfully" in content:
            return "complete"
        if "ERROR" in content or "failed" in content.lower():
            return "failed"
        if "Initializing BroadcastAndProcess" in content:
            return "running"
    except Exception:
        pass
    return "pending"


def get_gs_site_status(run_name: str, site_name: str) -> tuple:
    """Returns (done: bool, size_gb: float)"""
    if site_name == "test":
        out = ROOT / "data_brain" / "gs" / run_name / "test" / "images.npy"
    else:
        i   = site_name.split("-")[1]
        out = ROOT / "data_brain" / "gs" / run_name / f"site_{i}" / "images.npy"
    if out.exists():
        return True, out.stat().st_size / 1024**3
    return False, 0.0

def get_benchmark_tail(n=6) -> list:
    if not LOG.exists():
        return []
    try:
        lines = LOG.read_text().splitlines()
        filtered = [
            l for l in lines
            if l.strip() and not any(skip in l for skip in [
                "transaction info", "downloaded to all",
                "UserWarning", "pip uninstall", "cupy-cuda",
                "warn(f", "────────", "══════", "─────",
                "Root", "Data", "Clients", "Rounds",
                "Batch", "Image", "GS iter",
            ])
        ]
        return filtered[-n:]
    except Exception:
        return []


def _get_run_names():
    state_path = ROOT / "runs" / "state.json"
    try:
        with open(state_path) as f:
            state = json.load(f)
        active_reps = set()
        for key in state:
            m = re.match(r"rep(\d+)", key)
            if m:
                active_reps.add(int(m.group(1)))
        if not active_reps:
            active_reps = {1}
    except Exception:
        active_reps = {1}
    return [
        f"rep{rep}_{run}"
        for rep in sorted(active_reps)
        for run in BASE_RUNS
    ]

def render():
    clear_screen()
    state   = load_state()
    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    width   = 72

    print(clr("═" * width, BLUE))
    print(clr("  BrainMRI FL Benchmark Monitor".center(width), BOLD + BLUE))
    print(clr(f"  {now_str}".center(width), GRAY))
    print(clr("═" * width, BLUE))
    print()

    # ── Runs overview ─────────────────────────────────────────────────────────
    print(clr("  Runs", BOLD + WHITE))
    print(clr("  " + "─" * (width-2), GRAY))

    for run in _get_run_names():
        s       = state.get(run, {})
        status  = s.get("status", "pending")
        metrics = get_latest_metrics(run)
        rounds  = metrics.get("round", s.get("last_round", 0))

        if status == "complete":
            icon  = clr("✅", GREEN)
            color = GREEN
        elif status == "running":
            icon  = clr("🔄", YELLOW)
            color = YELLOW
        elif status == "failed":
            icon  = clr("❌", RED)
            color = RED
        else:
            icon  = clr("⏳", GRAY)
            color = GRAY

        if metrics:
            metrics_str = (
                f"acc={metrics.get('accuracy',0):.3f}  "
                f"auc={metrics.get('auc_roc',0):.3f}  "
                f"f1={metrics.get('f1',0):.3f}"
            )
        else:
            metrics_str = "no metrics yet"

        print(
            f"  {icon}  "
            f"{clr(run.ljust(18), color + BOLD)}  "
            f"{clr(f'round {rounds:02d}/{N_ROUNDS}', BLUE)}  "
            f"{clr(metrics_str, BLUE)}"
        )
    print()

    # ── Latest global metrics ─────────────────────────────────────────────────
    active_run = next(
        (r for r in RUN_NAMES
         if state.get(r, {}).get("status") == "running"),
        None
    )
    latest_run = active_run or next(
        (r for r in reversed(RUN_NAMES) if get_latest_metrics(r)),
        None
    )

    if latest_run:
        metrics = get_latest_metrics(latest_run)
        if metrics:
            print(clr("  Latest Global Test Metrics", BOLD + WHITE))
            print(clr("  " + "─" * (width-2), GRAY))
            cm     = metrics.get("conf_matrix", [])
            cm_str = ""
            if cm and len(cm) == 4:
                cm_str = "CM " + str([[cm[i][j] for j in range(4)]
                                       for i in range(4)])
            print(
                f"  {clr(latest_run, BOLD + CYAN)}  "
                f"round {metrics.get('round',0):02d}/{N_ROUNDS}  "
                f"acc={metrics.get('accuracy',0):.3f}  "
                f"auc={metrics.get('auc_roc',0):.3f}  "
                f"f1={metrics.get('f1',0):.3f}  "
                f"loss={metrics.get('loss',0):.3f}"
            )
            if cm_str:
                print(f"  {clr(cm_str[:width-2], GRAY)}")
            print()

        # ── Client health ─────────────────────────────────────────────────────
        clients = metrics.get("clients", []) if metrics else []
        if clients:
            print(clr("  Client Health", BOLD + WHITE))
            print(clr("  " + "─" * (width-2), GRAY))
            print(
                f"  {'client':<8} {'n':>6} {'mode':<28} "
                f"{'loss':>7} {'acc':>7} {'auc':>7} {'f1':>7}"
            )
            for c in clients:
                lm = c.get("local_metrics", {})
                print(
                    f"  {c.get('client_id','?'):<8} "
                    f"{c.get('n_samples',0):>6,} "
                    f"{c.get('train_mode',''):<28} "
                    f"{c.get('avg_loss',0):>7.3f} "
                    f"{lm.get('accuracy',0):>7.3f} "
                    f"{lm.get('auc_roc',0):>7.3f} "
                    f"{lm.get('f1',0):>7.3f}"
                )
            print()

    # ── Active run detail ─────────────────────────────────────────────────────
    if active_run:
        gs_status = get_gs_status(active_run)
        print(clr(f"  Active: {active_run}", BOLD))
        print(clr("  " + "─" * (width-2), GRAY))

        if gs_status == "running":
            print(clr("  🔄 GS Transform running...", YELLOW))
            for site_name in ["test"] + SITES:
                done, size = get_gs_site_status(active_run, site_name)
                label = clr(site_name.ljust(8), BOLD)
                if done:
                    print(f"  {label}  {clr(f'done ({size:.1f}GB)', GREEN)}")
                else:
                    print(f"  {label}  {clr('transforming...', YELLOW)}")
        else:
            for site in SITES:
                cs = get_client_status(active_run, site)
                r  = cs["round"]
                e  = cs["epoch"]
                l  = cs["loss"]

                r_filled = int(r / max(N_ROUNDS, 1) * 20)
                r_bar    = "█" * r_filled + "░" * (20 - r_filled)

                if cs["status"] == "training":
                    e_filled = int(e / max(LOCAL_EPOCHS, 1) * 6)
                    e_bar    = "█" * e_filled + "░" * (6 - e_filled)
                    loss_str = clr(f"loss={l:.4f}", BLUE) if l else ""
                    print(
                        f"  {clr(site, BOLD)}  "
                        f"r{clr(str(r).zfill(2), YELLOW)}/{N_ROUNDS} "
                        f"|{clr(r_bar, GREEN)}|  "
                        f"ep{e}/{LOCAL_EPOCHS}"
                        f"|{clr(e_bar, YELLOW)}|  "
                        f"{loss_str}"
                    )
                elif cs["status"] == "round_complete":
                    print(
                        f"  {clr(site, BOLD)}  "
                        f"r{clr(str(r).zfill(2), GREEN)}/{N_ROUNDS} "
                        f"|{clr(r_bar, GREEN)}|  "
                        f"{clr('waiting for aggregation...', YELLOW)}"
                    )
                else:
                    print(
                        f"  {clr(site, BOLD)}  "
                        f"r00/{N_ROUNDS} |{'░'*20}|  "
                        f"{clr('waiting...', GRAY)}"
                    )
        print()

    # ── Log tail ──────────────────────────────────────────────────────────────
    print(clr("  Recent Activity", BOLD + WHITE))
    print(clr("  " + "─" * (width-2), GRAY))
    for line in get_benchmark_tail(6):
        line = line.strip()
        if "ERROR" in line or "❌" in line:
            print(f"  {clr(line[:width+20], RED)}")
        elif "✅" in line or "complete" in line.lower():
            print(f"  {clr(line[:width+20], GREEN)}")
        elif "Round" in line or "round" in line:
            print(f"  {clr(line[:width+20], BLUE)}")
        else:
            print(f"  {clr(line[:width+20], GRAY)}")

    print()
    print(clr("  " + "─" * (width-2), GRAY))
    print(clr("  Refreshing every 5s — Ctrl+C to exit", GRAY))

if __name__ == "__main__":
    try:
        while True:
            render()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")
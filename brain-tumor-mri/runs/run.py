# ============================================================
# runs/run.py
# Main orchestrator for BrainMRI FL benchmark.
# ============================================================

import sys
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RUNS, N_CLIENTS, N_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LR, IMG_SIZE, GS_ITER_COUNT, TASK_TIMEOUT,
    NPY_DIR, TEST_DIR, GS_DIR,
    ckpt_dir, results_path, gs_out_dir, ws_dir,
    RUN_STATE, ROOT as BENCHMARK_ROOT, print_config,
    N_REPEATS, BASE_SPLIT_SEED,
)

NVFLARE_BIN = Path(sys.executable).parent / "nvflare"
PYTHON_BIN  = sys.executable
REPO_ROOT   = BENCHMARK_ROOT


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if RUN_STATE.exists():
        with open(RUN_STATE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    RUN_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_STATE, "w") as f:
        json.dump(state, f, indent=2)


def get_run_status(state: dict, run_name: str) -> dict:
    return state.get(run_name, {
        "status"    : "pending",
        "last_round": 0,
        "start_time": None,
        "end_time"  : None,
    })


def mark_run(state, run_name, status, last_round=0):
    if run_name not in state:
        state[run_name] = {}
    state[run_name]["status"]     = status
    state[run_name]["last_round"] = last_round
    if status == "running" and not state[run_name].get("start_time"):
        state[run_name]["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if status in ("complete", "failed"):
        state[run_name]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)


# ── Split ─────────────────────────────────────────────────────────────────────

def run_split_for_repeat(repeat_idx: int) -> bool:
    split_seed = BASE_SPLIT_SEED + repeat_idx
    print()
    print(f"  Regenerating split for repeat {repeat_idx+1} "
          f"(SPLIT_SEED={split_seed})...")

    result = subprocess.run(
        [PYTHON_BIN, str(BENCHMARK_ROOT / "data" / "split.py")],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT),
             "SPLIT_SEED": str(split_seed)},
    )

    if result.returncode != 0:
        print(f"  Split failed for repeat {repeat_idx+1}")
        return False
    print(f"  Split complete for repeat {repeat_idx+1}")
    return True


# ── GS transform ──────────────────────────────────────────────────────────────

def build_gs_job(run_name: str, maskP: float) -> Path:
    job_dir = BENCHMARK_ROOT / "runs" / f"job_gs_{run_name}"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)

    # test set + all training sites get transformed
    gs_clients = [
        ("test", TEST_DIR / "images.npy",
         gs_out_dir(run_name) / "test" / "images.npy")
    ]
    for i in range(N_CLIENTS):
        gs_clients.append((
            f"site-{i+1}",
            NPY_DIR / f"site_{i+1}" / "images.npy",
            gs_out_dir(run_name) / f"site_{i+1}" / "images.npy",
        ))

    n_gs_clients = len(gs_clients)  # N_CLIENTS + 1 (test)

    server_cfg_dir = job_dir / "app_server" / "config"
    server_cfg_dir.mkdir(parents=True)
    server_cfg = {
        "format_version"     : 2,
        "task_data_filters"  : [],
        "task_result_filters": [],
        "components": [{
            "id"  : "gs_processor",
            "path": "gs_1ch.controller.gs_controller.GS1chResponseProcessor",
            "args": {"override_params_path": None}
        }],
        "workflows": [{
            "id"  : "gs_workflow",
            "path": "nvflare.app_common.workflows.broadcast_and_process.BroadcastAndProcess",
            "args": {
                "processor"                   : "gs_processor",
                "task_name"                   : "gs_transform",
                "min_responses_required"      : 0,
                "wait_time_after_min_received": 10,
                "timeout"                     : TASK_TIMEOUT,
            }
        }]
    }
    with open(server_cfg_dir / "config_fed_server.json", "w") as f:
        json.dump(server_cfg, f, indent=2)

    server_custom = job_dir / "app_server" / "custom"
    server_custom.mkdir(parents=True)
    _copy_gs1ch(server_custom)

    for site_name, in_path, out_path in gs_clients:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        app_name = f"app_{site_name}" if site_name == "test" else f"app_{site_name}"

        client_cfg_dir = job_dir / app_name / "config"
        client_cfg_dir.mkdir(parents=True)
        client_custom = job_dir / app_name / "custom"
        client_custom.mkdir(parents=True)
        _copy_gs1ch(client_custom)

        client_cfg = {
            "format_version"     : 2,
            "task_data_filters"  : [],
            "task_result_filters": [],
            "executors": [{
                "tasks": ["gs_transform"],
                "executor": {
                    "path": "gs_1ch.executor.gs_executor.GS1chExecutor",
                    "args": {
                        "input_path"      : str(in_path),
                        "output_path"     : str(out_path),
                        "iter_count"      : GS_ITER_COUNT,
                        "maskP"           : maskP,
                        "auto_chunk"      : True,
                        "verbose"         : True,
                        "time_budget_warn": 300,
                        "time_budget_slow": 1800,
                    }
                }
            }]
        }
        with open(client_cfg_dir / "config_fed_client.json", "w") as f:
            json.dump(client_cfg, f, indent=2)

    deploy_map = {"app_server": ["server"]}
    for site_name, _, _ in gs_clients:
        deploy_map[f"app_{site_name}"] = [site_name]
    with open(job_dir / "meta.json", "w") as f:
        json.dump({
            "name"        : f"gs_{run_name}",
            "resource_spec": {},
            "deploy_map"  : deploy_map,
            "min_clients" : n_gs_clients,
        }, f, indent=2)

    return job_dir, gs_clients


def run_gs_transform(run_name: str, maskP: float) -> bool:
    # check all outputs exist including test
    all_done = (gs_out_dir(run_name) / "test" / "images.npy").exists() and all(
        (gs_out_dir(run_name) / f"site_{i+1}" / "images.npy").exists()
        for i in range(N_CLIENTS)
    )
    if all_done:
        print(f"  GS already done for {run_name} -- skipping.")
        return True

    print(f"  Running GS transform for {run_name} (maskP={maskP})...")

    workspace   = ws_dir(f"gs_{run_name}")
    job_dir, gs_clients = build_gs_job(run_name, maskP)
    n_gs_clients = len(gs_clients)
    all_site_names = [s for s, _, _ in gs_clients]

    preflight = f"""
import sys
from pathlib import Path
sys.path.insert(0, '{REPO_ROOT}')
from gs_1ch.core.diagnostic import run_diagnostics
policy = {{'on_first_run': 'auto', 'on_gpu_change': 'auto', 'on_oom': 'abort_job', 'verbose': True}}
for site_name in {all_site_names}:
    d = Path('{workspace}') / site_name / 'local' / 'gs_1ch_diagnostic.txt'
    d.parent.mkdir(parents=True, exist_ok=True)
    run_diagnostics(diag_path=d, policy=policy)
    print(f'Diagnostic ready: {{site_name}}')
"""
    result = subprocess.run(
        [PYTHON_BIN, "-c", preflight],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True, capture_output=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"  GS pre-flight failed for {run_name}")
        return False

    clients_str = ",".join(all_site_names)
    cmd = [
        str(NVFLARE_BIN), "simulator", str(job_dir),
        "--workspace", str(workspace),
        "--n_clients", str(n_gs_clients),
        "--clients",   clients_str,
        "--threads",   "1",
    ]
    result = subprocess.run(
        cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    success = (
        result.returncode == 0
        and (gs_out_dir(run_name) / "test" / "images.npy").exists()
        and all(
            (gs_out_dir(run_name) / f"site_{i+1}" / "images.npy").exists()
            for i in range(N_CLIENTS)
        )
    )
    print(f"  {'GS complete' if success else 'GS FAILED'} for {run_name}")
    return success


# ── FL training ───────────────────────────────────────────────────────────────

def build_train_job(run_name: str, use_gs: bool, start_round: int) -> Path:
    job_dir = BENCHMARK_ROOT / "runs" / f"job_train_{run_name}"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)

    ckpt = ckpt_dir(run_name)
    res  = results_path(run_name)

    # test set matches training domain
    if use_gs:
        test_images = str(gs_out_dir(run_name) / "test" / "images.npy")
    else:
        test_images = str(TEST_DIR / "images.npy")
    test_labels = str(TEST_DIR / "labels.npy")  # labels never change

    server_cfg_dir = job_dir / "app_server" / "config"
    server_cfg_dir.mkdir(parents=True)
    server_cfg = {
        "format_version"     : 2,
        "task_data_filters"  : [],
        "task_result_filters": [],
        "workflows": [{
            "id"  : "fl_trainer",
            "path": "train.controller.FLController",
            "args": {
                "run_name"        : run_name,
                "test_images_path": test_images,
                "test_labels_path": test_labels,
                "ckpt_dir"        : str(ckpt),
                "results_path"    : str(res),
                "n_rounds"        : N_ROUNDS,
                "task_timeout"    : TASK_TIMEOUT,
                "start_round"     : start_round,
            }
        }]
    }
    with open(server_cfg_dir / "config_fed_server.json", "w") as f:
        json.dump(server_cfg, f, indent=2)

    server_custom = job_dir / "app_server" / "custom"
    server_custom.mkdir(parents=True)
    _copy_train_scripts(server_custom)

    for i in range(N_CLIENTS):
        site = f"site-{i+1}"
        if use_gs:
            images_path = gs_out_dir(run_name) / f"site_{i+1}" / "images.npy"
        else:
            images_path = NPY_DIR / f"site_{i+1}" / "images.npy"
        labels_path = NPY_DIR / f"site_{i+1}" / "labels.npy"

        client_cfg_dir = job_dir / f"app_{site}" / "config"
        client_cfg_dir.mkdir(parents=True)
        client_custom = job_dir / f"app_{site}" / "custom"
        client_custom.mkdir(parents=True)
        _copy_train_scripts(client_custom)

        client_cfg = {
            "format_version"     : 2,
            "task_data_filters"  : [],
            "task_result_filters": [],
            "executors": [{
                "tasks": ["train"],
                "executor": {
                    "path": "train.executor.FLExecutor",
                    "args": {
                        "images_path" : str(images_path),
                        "labels_path" : str(labels_path),
                        "batch_size"  : BATCH_SIZE,
                        "local_epochs": LOCAL_EPOCHS,
                        "lr"          : LR,
                    }
                }
            }]
        }
        with open(client_cfg_dir / "config_fed_client.json", "w") as f:
            json.dump(client_cfg, f, indent=2)

    deploy_map = {"app_server": ["server"]}
    for i in range(N_CLIENTS):
        deploy_map[f"app_site-{i+1}"] = [f"site-{i+1}"]
    with open(job_dir / "meta.json", "w") as f:
        json.dump({
            "name"        : f"train_{run_name}",
            "resource_spec": {},
            "deploy_map"  : deploy_map,
            "min_clients" : N_CLIENTS,
        }, f, indent=2)

    return job_dir


def run_training(run_name: str, use_gs: bool, start_round: int) -> bool:
    print(f"  Running FL training for {run_name} "
          f"(start_round={start_round})...")
    workspace = ws_dir(f"train_{run_name}")
    job_dir   = build_train_job(run_name, use_gs, start_round)
    clients   = ",".join([f"site-{i+1}" for i in range(N_CLIENTS)])
    cmd = [
        str(NVFLARE_BIN), "simulator", str(job_dir),
        "--workspace", str(workspace),
        "--n_clients", str(N_CLIENTS),
        "--clients",   clients,
        "--threads",   str(N_CLIENTS),
    ]
    result = subprocess.run(
        cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    success = result.returncode == 0 and results_path(run_name).exists()
    print(f"  {'Training complete' if success else 'Training FAILED'} "
          f"for {run_name}")
    return success


# ── Copy helpers ──────────────────────────────────────────────────────────────

def _copy_gs1ch(dst: Path):
    try:
        import gs_1ch
        src = Path(gs_1ch.__file__).parent
    except ImportError:
        raise ImportError("gs_1ch not installed.")
    gs_dst = dst / "gs_1ch"
    if not gs_dst.exists():
        shutil.copytree(src, gs_dst)


def _copy_train_scripts(dst: Path):
    for script in ["executor.py", "controller.py", "metrics.py"]:
        shutil.copy(BENCHMARK_ROOT / "train" / script, dst / script)
    model_dst = dst / "model"
    model_dst.mkdir(exist_ok=True)
    shutil.copy(
        BENCHMARK_ROOT / "model" / "efficientnet.py",
        model_dst / "efficientnet.py"
    )
    (model_dst / "__init__.py").touch()
    shutil.copy(BENCHMARK_ROOT / "config.py", dst / "config.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",    type=str,  default=None)
    parser.add_argument("--repeat", type=int,  default=None)
    parser.add_argument("--reset",  action="store_true")
    args = parser.parse_args()

    print_config()

    if args.reset:
        if RUN_STATE.exists():
            RUN_STATE.unlink()
        for d in [
            BENCHMARK_ROOT / "checkpoints",
            BENCHMARK_ROOT / "results",
            BENCHMARK_ROOT / "runs" / "workspaces",
            BENCHMARK_ROOT / "data_brain" / "gs",
        ]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        for p in (BENCHMARK_ROOT / "runs").glob("job_train_*"):
            shutil.rmtree(p)
        for p in (BENCHMARK_ROOT / "runs").glob("job_gs_*"):
            shutil.rmtree(p)
        print("  Reset complete.")
        print()

    state = load_state()

    runs_to_execute = RUNS
    if args.run:
        runs_to_execute = [r for r in RUNS if r[0] == args.run]
        if not runs_to_execute:
            print(f"  Unknown run: {args.run}")
            sys.exit(1)

    repeat_indices = list(range(N_REPEATS))
    if args.repeat is not None:
        if args.repeat < 1 or args.repeat > N_REPEATS:
            print(f"  Invalid repeat: {args.repeat}")
            sys.exit(1)
        repeat_indices = [args.repeat - 1]

    total_start = time.time()

    for repeat_idx in repeat_indices:
        repeat_num = repeat_idx + 1
        print()
        print(f"  {'#'*60}")
        print(f"  REPEAT {repeat_num}/{N_REPEATS}")
        print(f"  {'#'*60}")

        repeat_key    = f"repeat_{repeat_num}"
        repeat_status = state.get(repeat_key, {}).get("split_status", "pending")

        if repeat_status != "complete":
            ok = run_split_for_repeat(repeat_idx)
            if not ok:
                state.setdefault(repeat_key, {})["split_status"] = "failed"
                save_state(state)
                continue
            state.setdefault(repeat_key, {})["split_status"] = "complete"
            save_state(state)
        else:
            print(f"  Split already complete for repeat {repeat_num}.")

        for base_run_name, use_gs, maskP in runs_to_execute:
            run_name = f"rep{repeat_num}_{base_run_name}"
            status   = get_run_status(state, run_name)

            print()
            print(f"  {'='*50}")
            print(f"  Run: {run_name}  GS={use_gs}  maskP={maskP}")
            print(f"  {'='*50}")

            if status["status"] == "complete":
                print("  Already complete -- skipping.")
                continue

            mark_run(state, run_name, "running")

            if use_gs:
                ok = run_gs_transform(run_name, maskP)
                if not ok:
                    mark_run(state, run_name, "failed", status["last_round"])
                    continue

            start_round = status["last_round"]
            ok = run_training(run_name, use_gs, start_round)

            if not ok:
                rp = results_path(run_name)
                last = 0
                if rp.exists():
                    with open(rp) as f:
                        saved = json.load(f)
                    last = len(saved.get("rounds", []))
                mark_run(state, run_name, "failed", last)
                continue

            mark_run(state, run_name, "complete", N_ROUNDS)
            print(f"  {run_name} complete.")

    elapsed = time.time() - total_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins  = divmod(mins, 60)
    time_str   = (f"{hrs}h {mins}m {secs}s" if hrs > 0
                  else f"{mins}m {secs}s" if mins > 0
                  else f"{secs}s")

    state = load_state()
    expected = [
        f"rep{r+1}_{run_name}"
        for r in range(N_REPEATS)
        for run_name, _, _ in RUNS
    ]
    n_complete = sum(1 for n in expected
                     if state.get(n, {}).get("status") == "complete")
    n_failed   = sum(1 for n in expected
                     if state.get(n, {}).get("status") == "failed")

    print()
    print(f"  {'='*50}")
    print("  Benchmark Summary")
    print(f"  {'-'*50}")
    for name in expected:
        s    = state.get(name, {}).get("status", "pending")
        icon = "✅" if s == "complete" else "❌" if s == "failed" else "⏭"
        print(f"  {icon}  {name:<20} : {s}")
    print(f"  {'-'*50}")
    print(f"  {n_complete}/{len(expected)} complete | "
          f"{n_failed} failed | {time_str}")
    print(f"  {'='*50}")
    print()

    sys.exit(0 if n_complete == len(expected) else 1)


if __name__ == "__main__":
    main()
# ============================================================
# train/controller.py
# NVFlare Controller for federated EfficientNetB0 training.
# ============================================================

import sys
import json
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import ReturnCode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_ROUNDS, TASK_TIMEOUT, NUM_CLASSES, BATCH_SIZE
from model.efficientnet import EfficientNetB0, MedImageDataset, get_weights, set_weights, weighted_fedavg
from train.metrics import evaluate, print_round_metrics

TASK_TRAIN = "train"


class FLController(Controller):

    def __init__(
        self,
        run_name         : str,
        test_images_path : str,
        test_labels_path : str,
        ckpt_dir         : str,
        results_path     : str,
        n_rounds         : int = N_ROUNDS,
        task_timeout     : int = TASK_TIMEOUT,
        start_round      : int = 0,
    ):
        super().__init__()
        self.run_name         = run_name
        self.test_images_path = Path(test_images_path)
        self.test_labels_path = Path(test_labels_path)
        self.ckpt_dir         = Path(ckpt_dir)
        self.results_path     = Path(results_path)
        self.n_rounds         = n_rounds
        self.task_timeout     = task_timeout
        self.start_round      = start_round

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.global_model = EfficientNetB0(
            num_classes=NUM_CLASSES, pretrained=True
        ).to(self.device)

        self.round_results         = []
        self._round_client_results = []

    def start_controller(self, fl_ctx: FLContext):
        self.logger.info(
            f"[{self.run_name}] Controller started -- "
            f"rounds={self.n_rounds} start={self.start_round} "
            f"device={self.device}"
        )
        if self.start_round > 0:
            ckpt = self.ckpt_dir / f"round_{self.start_round:02d}.pt"
            if ckpt.exists():
                self.global_model.load_state_dict(
                    torch.load(str(ckpt), map_location=self.device)
                )
                self.logger.info(f"[{self.run_name}] Resumed from {ckpt}")
            if self.results_path.exists():
                with open(self.results_path) as f:
                    saved = json.load(f)
                self.round_results = saved.get("rounds", [])

    def stop_controller(self, fl_ctx: FLContext):
        self.logger.info(f"[{self.run_name}] Controller stopped.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        for round_idx in range(self.start_round, self.n_rounds):
            if abort_signal.triggered:
                return

            self.logger.info(
                f"[{self.run_name}] -- Round "
                f"{round_idx+1}/{self.n_rounds} --"
            )

            task_data                   = Shareable()
            task_data["global_weights"] = get_weights(self.global_model)
            task_data["current_round"]  = round_idx

            task = Task(
                name               = TASK_TRAIN,
                data               = task_data,
                timeout            = self.task_timeout,
                result_received_cb = self._process_client_result,
            )
            self._round_client_results = []

            self.broadcast_and_wait(
                task                         = task,
                fl_ctx                       = fl_ctx,
                abort_signal                 = abort_signal,
                min_responses                = 0,
                wait_time_after_min_received = 30,
            )

            if abort_signal.triggered:
                return

            if not self._round_client_results:
                self.logger.error(f"[{self.run_name}] No client results.")
                self.system_panic("No client results.", fl_ctx)
                return

            # ── FedAvg ────────────────────────────────────────────────────────
            weight_list   = [r["updated_weights"]
                             for r in self._round_client_results]
            sample_counts = [r["n_samples"]
                             for r in self._round_client_results]
            avg_weights   = weighted_fedavg(weight_list, sample_counts)
            set_weights(self.global_model, avg_weights)

            # ── Global eval ───────────────────────────────────────────────────
            metrics = self._evaluate()

            # ── Per-client summary ────────────────────────────────────────────
            total = sum(sample_counts)
            client_summaries = [
                {
                    "client_id"    : r["client_id"],
                    "n_samples"    : r["n_samples"],
                    "avg_loss"     : round(r["avg_loss"], 6),
                    "weight_pct"   : round(r["n_samples"] / total * 100, 1),
                    "train_mode"   : r.get("train_mode", ""),
                    "n_trainable"  : r.get("n_trainable", 0),
                    "local_metrics": r.get("local_metrics", {}),
                }
                for r in self._round_client_results
            ]

            round_result = {
                "round"         : round_idx + 1,
                "n_clients"     : len(self._round_client_results),
                "total_samples" : total,
                "avg_train_loss": float(np.mean(
                    [r["avg_loss"] for r in self._round_client_results]
                )),
                "clients"       : client_summaries,
                **metrics,
            }
            self.round_results.append(round_result)
            print_round_metrics(
                self.run_name, round_idx,
                self.n_rounds, round_result
            )

            for cs in client_summaries:
                lm = cs.get("local_metrics", {})
                self.logger.info(
                    f"[{self.run_name}]   {cs['client_id']}: "
                    f"samples={cs['n_samples']:,} "
                    f"loss={cs['avg_loss']:.4f} "
                    f"mode={cs['train_mode']} "
                    f"local_acc={lm.get('accuracy',0):.4f} "
                    f"local_auc={lm.get('auc_roc',0):.4f} "
                    f"local_f1={lm.get('f1',0):.4f}"
                )

            self._save_checkpoint(round_idx + 1)
            self._save_results()

        self._save_results(final=True)
        self.logger.info(f"[{self.run_name}] All rounds complete.")

    def _process_client_result(self, client_task, fl_ctx: FLContext):
        response = client_task.result
        client   = client_task.client
        if response is None:
            self.logger.warning(
                f"[{self.run_name}] None response from {client.name}"
            )
            return
        try:
            self._round_client_results.append({
                "client_id"      : response.get("client_id", client.name),
                "updated_weights": response["updated_weights"],
                "n_samples"      : response["n_samples"],
                "avg_loss"       : response["avg_loss"],
                "train_mode"     : response.get("train_mode", ""),
                "n_trainable"    : response.get("n_trainable", 0),
                "local_metrics"  : response.get("local_metrics", {}),
            })
        except Exception as e:
            self.logger.error(
                f"[{self.run_name}] Failed to unpack result "
                f"from {client.name}: {e}"
            )

    def _evaluate(self) -> dict:
        if not self.test_images_path.exists():
            return {"accuracy": 0.0, "auc_roc": 0.0,
                    "f1": 0.0, "loss": 0.0,
                    "conf_matrix": [], "n_samples": 0}
        try:
            dataset = MedImageDataset(
                str(self.test_images_path),
                str(self.test_labels_path)
            )
            loader  = DataLoader(
                dataset, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=2, pin_memory=True
            )
            return evaluate(self.global_model, loader, self.device)
        except Exception as e:
            self.logger.error(
                f"[{self.run_name}] Evaluation failed: {e}"
            )
            return {"accuracy": 0.0, "auc_roc": 0.0,
                    "f1": 0.0, "loss": 0.0,
                    "conf_matrix": [], "n_samples": 0}

    def _save_checkpoint(self, round_num: int):
        ckpt = self.ckpt_dir / f"round_{round_num:02d}.pt"
        torch.save(self.global_model.state_dict(), str(ckpt))

    def _save_results(self, final: bool = False):
        data = {
            "run_name"     : self.run_name,
            "n_rounds"     : self.n_rounds,
            "rounds"       : self.round_results,
            "complete"     : final,
            "final_metrics": self.round_results[-1]
                             if self.round_results else {},
        }
        with open(self.results_path, "w") as f:
            json.dump(data, f, indent=2)

    def process_result_of_unknown_task(
        self, client, task_name, client_task_id, result, fl_ctx
    ):
        pass
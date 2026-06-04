# ============================================================
# train/executor.py
# NVFlare Executor for local EfficientNetB0 training.
# ============================================================

import sys
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import ReturnCode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BATCH_SIZE, LR, LOCAL_EPOCHS, FREEZE_ROUNDS, FREEZE_BACKBONE, NUM_CLASSES, CLASSES
from model.efficientnet import EfficientNetB0, OCTDataset, get_weights, set_weights
from train.metrics import train_one_epoch, evaluate

TASK_TRAIN = "train"


class FLExecutor(Executor):

    def __init__(
        self,
        images_path  : str,
        labels_path  : str,
        batch_size   : int   = BATCH_SIZE,
        local_epochs : int   = LOCAL_EPOCHS,
        lr           : float = LR,
        class_weights: list  = None,
    ):
        super().__init__()
        self.images_path  = Path(images_path)
        self.labels_path  = Path(labels_path)
        self.batch_size   = batch_size
        self.local_epochs = local_epochs
        self.lr           = lr

        # ── Auto class weights ────────────────────────────────────────────────
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            labels = np.load(self.labels_path)
            n_total = len(labels)
            weights = []
            for i in range(NUM_CLASSES):
                n_cls = (labels == i).sum()
                weights.append(
                    n_total / (NUM_CLASSES * n_cls) if n_cls > 0 else 1.0
                )
            self.class_weights = weights
            print(
                f"[FLExecutor] Auto class weights: "
                + " ".join(
                    f"{CLASSES[i]}={weights[i]:.3f}"
                    for i in range(NUM_CLASSES)
                )
            )

        if not self.images_path.exists():
            raise FileNotFoundError(
                f"[FLExecutor] images_path not found: {self.images_path}"
            )
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"[FLExecutor] labels_path not found: {self.labels_path}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = EfficientNetB0(
            num_classes=NUM_CLASSES, pretrained=True
        ).to(self.device)

        dataset           = OCTDataset(
            str(self.images_path), str(self.labels_path)
        )
        self.train_loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )

    def execute(
        self,
        task_name    : str,
        shareable    : Shareable,
        fl_ctx       : FLContext,
        abort_signal : Signal,
    ) -> Shareable:

        client_id = fl_ctx.get_identity_name()
        logger    = self.logger

        if task_name != TASK_TRAIN:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        current_round  = shareable.get("current_round", 0)
        global_weights = shareable.get("global_weights", None)

        if global_weights is not None:
            try:
                set_weights(self.model, global_weights)
            except Exception as e:
                logger.error(f"[FLExecutor] Failed to load weights: {e}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # ── Freeze policy ─────────────────────────────────────────────────────
        if FREEZE_BACKBONE and current_round < FREEZE_ROUNDS:
            self.model.freeze_backbone()
            train_mode = "frozen_backbone_fc_only"
        else:
            self.model.unfreeze_last_blocks()
            train_mode = "fine_tune_last_blocks"

        n_trainable = self.model.count_parameters()

        logger.info(
            f"[FLExecutor] {client_id} round={current_round+1} "
            f"mode={train_mode} trainable={n_trainable:,}"
        )

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )

        t_start = time.time()
        losses  = []

        for epoch in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            loss = train_one_epoch(
                self.model, self.train_loader,
                optimizer, self.device,
                class_weights=self.class_weights,
                train_mode=train_mode,
            )
            losses.append(loss)
            logger.info(
                f"[FLExecutor] {client_id} round={current_round+1} "
                f"epoch={epoch+1}/{self.local_epochs} loss={loss:.4f}"
            )

        elapsed = time.time() - t_start

        local_metrics = evaluate(
            self.model, self.train_loader, self.device
        )

        reply                    = make_reply(ReturnCode.OK)
        reply["updated_weights"] = get_weights(self.model)
        reply["n_samples"]       = len(self.train_loader.dataset)
        reply["avg_loss"]        = float(np.mean(losses))
        reply["client_id"]       = client_id
        reply["current_round"]   = current_round
        reply["elapsed_sec"]     = elapsed
        reply["train_mode"]      = train_mode
        reply["n_trainable"]     = int(n_trainable)
        reply["local_metrics"]   = local_metrics

        logger.info(
            f"[FLExecutor] {client_id} round={current_round+1} "
            f"complete -- loss={np.mean(losses):.4f} "
            f"samples={len(self.train_loader.dataset)} "
            f"time={elapsed:.1f}s"
        )
        return reply

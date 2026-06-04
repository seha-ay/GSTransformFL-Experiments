# ============================================================
# train/metrics.py
# Evaluation and training for 4-class OCT classification.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
)


def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    all_labels = []
    all_preds  = []
    all_probs  = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    accuracy = float((all_preds == all_labels).mean())
    loss_avg = total_loss / max(n_batches, 1)

    try:
        auc = float(roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        ))
    except ValueError:
        auc = 0.0

    f1 = float(f1_score(
        all_labels, all_preds,
        average="macro", zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "accuracy"   : round(accuracy, 6),
        "auc_roc"    : round(auc,      6),
        "f1"         : round(f1,       6),
        "loss"       : round(loss_avg, 6),
        "conf_matrix": cm,
        "n_samples"  : int(len(all_labels)),
    }


def train_one_epoch(
    model,
    loader       : DataLoader,
    optimizer,
    device       : torch.device,
    class_weights: list = None,
    train_mode   : str  = "",
) -> float:
    model.train()

    if train_mode == "frozen_backbone_fc_only":
        model.model.train()
        model.model.features.eval()
        model.model.classifier.train()

    elif train_mode == "fine_tune_last_blocks":
        model.model.train()
        model.model.features.eval()
        model.model.features[6].train()
        model.model.features[7].train()
        model.model.features[8].train()

    if class_weights is not None:
        w         = torch.tensor(
            class_weights, dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches  = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def print_round_metrics(
    run_name  : str,
    round_idx : int,
    n_rounds  : int,
    metrics   : dict,
):
    print(
        f"  [{run_name}] Round {round_idx+1:02d}/{n_rounds} -- "
        f"acc={metrics['accuracy']:.4f} | "
        f"auc={metrics['auc_roc']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"loss={metrics['loss']:.4f}"
    )

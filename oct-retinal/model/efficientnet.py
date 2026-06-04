# ============================================================
# model/efficientnet.py
# EfficientNet-B0 for 4-class OCT classification.
# Adapted for grayscale input (1 channel) and federated
# weight exchange.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from pathlib import Path


# ── Model ─────────────────────────────────────────────────────────────────────

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 adapted for single-channel input and
    multi-class classification.

    Changes from standard EfficientNet-B0:
      - First conv: 3->1 input channels (grayscale, weights averaged)
      - Classifier head: 1280 -> 256 -> ReLU -> Dropout(0.4) -> num_classes
      - Pretrained ImageNet weights optionally loaded and adapted
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()

        weights  = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # ── Adapt first conv for single-channel input ─────────────────────────
        orig_conv = backbone.features[0][0]
        new_conv  = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size = orig_conv.kernel_size,
            stride      = orig_conv.stride,
            padding     = orig_conv.padding,
            bias        = False,
        )
        if pretrained:
            new_conv.weight.data = orig_conv.weight.data.mean(
                dim=1, keepdim=True
            )
        backbone.features[0][0] = new_conv

        # ── Replace classifier head ───────────────────────────────────────────
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.model.train()
        self.model.features.eval()

    def unfreeze_last_blocks(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze last 3 MBConv blocks + classifier
        for block in [
            self.model.features[6],
            self.model.features[7],
            self.model.features[8],
            self.model.classifier,
        ]:
            for param in block.parameters():
                param.requires_grad = True
        self.model.train()
        self.model.features.eval()
        self.model.features[6].train()
        self.model.features[7].train()
        self.model.features[8].train()

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True


# ── Dataset ───────────────────────────────────────────────────────────────────

class OCTDataset(Dataset):
    """
    Loads pre-processed OCT images and labels from .npy files.
    Expects images shape (N, H, W) float32 range [0,1].
    Adds channel dim: (H, W) -> (1, H, W).
    """

    def __init__(self, images_path: str, labels_path: str):
        self.images = np.load(images_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)

        assert self.images.ndim == 3, \
            f"Expected (N, H, W), got {self.images.shape}"
        assert len(self.images) == len(self.labels), \
            "Images and labels length mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = torch.from_numpy(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ── Weight utilities ──────────────────────────────────────────────────────────

def get_weights(model: nn.Module) -> dict:
    return {k: v.cpu().numpy().copy()
            for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: dict):
    state = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    model.load_state_dict(state)


def weighted_fedavg(weight_list: list, sample_counts: list) -> dict:
    total = sum(sample_counts)
    avg   = {}
    for key in weight_list[0].keys():
        weighted = [w[key] * (n / total)
                    for w, n in zip(weight_list, sample_counts)]
        result   = np.sum(weighted, axis=0)
        avg[key] = np.atleast_1d(result) if np.ndim(result) == 0 \
                   else result
    return avg


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print()
    print("  EfficientNetB0 -- self-test")
    print("  " + "-" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")

    model = EfficientNetB0(num_classes=4, pretrained=False).to(device)
    print(f"  Parameters : {model.count_parameters():,}")

    x   = torch.randn(4, 1, 224, 224).to(device)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"
    print(f"  Forward    : {tuple(x.shape)} -> {tuple(out.shape)} OK")

    model.freeze_backbone()
    n = model.count_parameters()
    print(f"  Frozen     : {n:,} trainable params OK")

    w = get_weights(model)
    set_weights(model, w)
    print(f"  Weights    : round-trip OK")

    avg = weighted_fedavg([w, w], [100, 200])
    for k in w:
        assert np.allclose(avg[k], w[k]), f"FedAvg mismatch on {k}"
    print(f"  FedAvg     : OK")

    print()
    print("  EfficientNetB0 self-test passed.")
    print()

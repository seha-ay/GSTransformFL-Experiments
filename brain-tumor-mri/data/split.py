# ============================================================
# data/split.py
# Stratified random split for Brain MRI FL benchmark.
# 4-class: glioma=0, meningioma=1, notumor=2, pituitary=3
#
# No patient IDs in this dataset — image-level stratified
# random split with fixed seed per repeat.
#
# Usage:
#   python data/split.py                    # seed=42
#   SPLIT_SEED=43 python data/split.py      # seed=43
# ============================================================

import sys
import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    NPY_DIR, TEST_DIR, N_CLIENTS,
    CLASSES, CLASS2IDX, IMG_SIZE,
    SITE_DISTRIBUTIONS, TEST_N_PER_CLASS,
    print_config,
)

RANDOM_SEED = int(os.environ.get("SPLIT_SEED", "42"))


def load_data():
    images_path   = NPY_DIR / "images.npy"
    metadata_path = NPY_DIR / "metadata.csv"
    if not images_path.exists():
        raise FileNotFoundError("Run data/convert.py first.")
    print("  Loading metadata...")
    meta   = pd.read_csv(metadata_path)
    print("  Memory-mapping images...")
    images = np.load(images_path, mmap_mode="r")
    print(f"  {len(images):,} images total")
    for cls in CLASSES:
        n = (meta.class_name == cls).sum()
        print(f"    {cls:<12}: {n:>5,} images")
    print()
    return images, meta


def clean_existing():
    for f in [NPY_DIR / "split_info.json"]:
        if f.exists():
            f.unlink()
    for site_idx, *_ in SITE_DISTRIBUTIONS:
        d = NPY_DIR / f"site_{site_idx}"
        if d.exists():
            shutil.rmtree(d)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def allocate_indices(meta, split_targets, rng):
    """
    Stratified random allocation — no patient awareness needed.
    For each class, shuffle image indices then assign to splits
    in order: test first, then sites.
    Returns dict: split_name -> list of metadata row indices.
    """
    split_indices = {name: [] for name in split_targets}

    for cls in CLASSES:
        cls_idx  = CLASS2IDX[cls]
        cls_rows = meta[meta["label"] == cls_idx].index.tolist()
        cls_rows = list(cls_rows)
        rng.shuffle(cls_rows)

        # Calculate total needed across all splits for this class
        total_needed = sum(
            cfg["targets"][cls_idx] for cfg in split_targets.values()
        )
        if total_needed > len(cls_rows):
            raise ValueError(
                f"Class '{cls}': need {total_needed} images but only "
                f"{len(cls_rows)} available. "
                f"Reduce SITE_DISTRIBUTIONS or TEST_N_PER_CLASS."
            )

        cursor = 0
        # Assign test first, then sites in order
        ordered = ["test"] + [
            f"site_{i}" for i, *_ in SITE_DISTRIBUTIONS
        ]
        for split_name in ordered:
            if split_name not in split_targets:
                continue
            n = split_targets[split_name]["targets"][cls_idx]
            split_indices[split_name].extend(cls_rows[cursor:cursor + n])
            cursor += n

    return split_indices


def save_split(images, meta, indices, out_dir, split_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    indices      = list(indices)
    n            = len(indices)
    split_labels = meta.loc[indices, "label"].values.astype(np.int64)
    split_meta   = meta.loc[indices].reset_index(drop=True)

    img_path = out_dir / "images.npy"
    img_mm   = np.lib.format.open_memmap(
        str(img_path), mode="w+", dtype=np.float32,
        shape=(n, IMG_SIZE, IMG_SIZE),
    )
    np_idx = np.array(indices)
    for start in range(0, n, 1000):
        end               = min(start + 1000, n)
        img_mm[start:end] = images[np_idx[start:end]]
        img_mm.flush()
    del img_mm

    np.save(out_dir / "labels.npy", split_labels)
    split_meta.to_csv(out_dir / "meta.csv", index=False)

    size_gb = img_path.stat().st_size / 1024**3
    counts  = {cls: int((split_labels == i).sum())
               for i, cls in enumerate(CLASSES)}
    print(
        f"  {split_name:<10}: {n:>5,} images | "
        + " ".join(f"{cls}={counts[cls]}" for cls in CLASSES)
        + f" | {size_gb:.2f}GB"
    )
    return {
        "n_images": n,
        "counts"  : counts,
        "path"    : str(out_dir),
    }


if __name__ == "__main__":
    print_config()
    print(f"  SPLIT_SEED = {RANDOM_SEED}")
    print()

    print("  Cleaning existing splits...")
    clean_existing()

    print("  [1/3] Loading data...")
    images, meta = load_data()

    print("  [2/3] Allocating indices (stratified random)...")
    rng = np.random.default_rng(RANDOM_SEED)

    split_targets = {
        "test": {
            "out_dir" : TEST_DIR,
            "targets" : [TEST_N_PER_CLASS] * 4,
        }
    }
    for site_idx, *counts in SITE_DISTRIBUTIONS:
        split_targets[f"site_{site_idx}"] = {
            "out_dir" : NPY_DIR / f"site_{site_idx}",
            "targets" : counts,
        }

    split_indices = allocate_indices(meta, split_targets, rng)

    print()
    print("  [3/3] Saving splits...")
    split_info = {
        "n_clients"  : N_CLIENTS,
        "classes"    : CLASSES,
        "split_seed" : RANDOM_SEED,
        "splits"     : {},
    }

    for name in ["test"] + [f"site_{i}" for i, *_ in SITE_DISTRIBUTIONS]:
        cfg  = split_targets[name]
        info = save_split(
            images, meta,
            split_indices[name],
            cfg["out_dir"], name,
        )
        split_info["splits"][name] = info

    print()
    with open(NPY_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("  Split info saved.")
    print()
    print("  Next: python runs/run.py")
    print()
# ============================================================
# data/split.py
# Patient-aware non-IID split for OCT FL benchmark.
# 4-class: CNV=0, DME=1, DRUSEN=2, NORMAL=3
#
# Usage:
#   python data/split.py
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
    print(f"  {len(images):,} images | {meta.patient_id.nunique():,} patients")
    for cls in CLASSES:
        n = (meta.class_name == cls).sum()
        p = meta[meta.class_name == cls]["patient_id"].nunique()
        print(f"    {cls:<8}: {n:>6,} images | {p:>4,} patients")
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


def allocate_patients(meta, split_targets, rng):
    """
    Patient-safe allocation.
    For each class, assign patients to splits to meet image targets.
    Returns dict: split_name -> list of (meta_idx) selected.
    """
    split_indices = {name: [] for name in split_targets}
    used_patients = set()

    for cls in CLASSES:
        cls_idx  = CLASS2IDX[cls]
        cls_meta = meta[meta["label"] == cls_idx].copy()

        # Build patient table for this class
        pt = (
            cls_meta.groupby("patient_id")
            .size()
            .reset_index(name="n_images")
            .sort_values("n_images", ascending=False)
        )
        pt = pt[~pt["patient_id"].isin(used_patients)]
        patients = pt["patient_id"].tolist()
        rng.shuffle(patients)

        for split_name, cfg in split_targets.items():
            target    = cfg["targets"][cls_idx]
            collected = 0
            assigned  = []

            for pid in patients:
                if pid in used_patients:
                    continue
                pid_imgs = cls_meta[
                    cls_meta["patient_id"] == pid
                ].index.tolist()
                assigned.append((pid, pid_imgs))
                collected += len(pid_imgs)
                if collected >= target:
                    break

            # Sample exactly target images
            all_imgs = []
            for pid, imgs in assigned:
                all_imgs.extend(imgs)
                used_patients.add(pid)

            rng.shuffle(all_imgs)
            selected = all_imgs[:target]

            if len(selected) < target:
                raise ValueError(
                    f"{split_name} {cls}: need {target}, got {len(selected)}"
                )

            split_indices[split_name].extend(selected)

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


def verify_no_overlap(split_info):
    all_meta = []
    for name, info in split_info["splits"].items():
        df        = pd.read_csv(Path(info["path"]) / "meta.csv")
        df["split"] = name
        all_meta.append(df)
    combined = pd.concat(all_meta, ignore_index=True)
    bad = (
        combined.groupby("patient_id")["split"]
        .nunique()
        .reset_index()
    )
    bad = bad[bad["split"] > 1]
    if len(bad) > 0:
        raise AssertionError(
            f"Patient overlap in {len(bad)} patients"
        )
    print("  No patient overlap verified.")


if __name__ == "__main__":
    print_config()

    print("  Cleaning existing splits...")
    clean_existing()

    print("  [1/3] Loading data...")
    images, meta = load_data()

    print("  [2/3] Allocating patients...")
    rng = np.random.default_rng(RANDOM_SEED)

    split_targets = {
        "test": {
            "out_dir" : TEST_DIR,
            "targets" : [TEST_N_PER_CLASS] * 4,
        }
    }
    for site_idx, cnv, dme, drusen, normal in SITE_DISTRIBUTIONS:
        split_targets[f"site_{site_idx}"] = {
            "out_dir" : NPY_DIR / f"site_{site_idx}",
            "targets" : [cnv, dme, drusen, normal],
        }

    split_indices = allocate_patients(meta, split_targets, rng)

    print()
    print("  [3/3] Saving splits...")
    split_info = {
        "n_clients" : N_CLIENTS,
        "classes"   : CLASSES,
        "splits"    : {},
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
    verify_no_overlap(split_info)

    with open(NPY_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("  Split info saved.")
    print()
    print("  Next: python runs/run.py")
    print()

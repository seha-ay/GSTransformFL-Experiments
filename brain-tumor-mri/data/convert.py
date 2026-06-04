# ============================================================
# data/convert.py
# Downloads Brain Tumor MRI dataset from Kaggle (if needed),
# converts JPEG images to float32 .npy arrays, then removes
# raw files.
#
# Usage:
#   python data/convert.py
# ============================================================

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, NPY_DIR, IMG_SIZE, CLASSES, CLASS2IDX, check_kaggle

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
# Dataset unpacks to two top-level folders: Training/ and Testing/
TRAINING_DIR   = RAW_DIR / "Training"
TESTING_DIR    = RAW_DIR / "Testing"


# ── Download ──────────────────────────────────────────────────────────────────

def download():
    images_path = NPY_DIR / "images.npy"
    meta_path   = NPY_DIR / "metadata.csv"
    if images_path.exists() and meta_path.exists():
        print("  NPY already converted — skipping download.")
        return

    if TRAINING_DIR.exists() and any(TRAINING_DIR.rglob("*.jpg")):
        print("  Raw images already present — skipping download.")
        return

    check_kaggle()
    from config import KAGGLE_USERNAME, KAGGLE_KEY
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

    print("  Downloading Brain Tumor MRI dataset from Kaggle...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path  = str(RAW_DIR),
            unzip = True,
            quiet = False,
        )
        print(f"  Downloaded and extracted to {RAW_DIR}")
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}") from e


# ── Scan ──────────────────────────────────────────────────────────────────────

def find_all_images() -> pd.DataFrame:
    """
    Pool all images from Training/ and Testing/ subfolders.
    No patient IDs in this dataset — image-level records only.
    Filename format: e.g. Te-gl_0010.jpg, Tr-pi_0001.jpg
    """
    records = []
    for split_dir, split_name in [(TRAINING_DIR, "train"), (TESTING_DIR, "test")]:
        if not split_dir.exists():
            print(f"  WARNING: {split_dir} not found")
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  WARNING: {cls_dir} not found")
                continue
            for p in sorted(cls_dir.glob("*.jpg")):
                if p.name.startswith("._"):
                    continue
                records.append({
                    "image_file" : str(p),
                    "label"      : CLASS2IDX[cls],
                    "class_name" : cls,
                    "orig_split" : split_name,
                })

    df = pd.DataFrame(records)
    print(f"  Found {len(df):,} images total")
    for cls in CLASSES:
        n = (df.class_name == cls).sum()
        print(f"    {cls:<12}: {n:>5,} images")
    print(f"  (Training: {(df.orig_split=='train').sum():,}  "
          f"Testing: {(df.orig_split=='test').sum():,})")
    return df


# ── Convert ───────────────────────────────────────────────────────────────────

def convert(df: pd.DataFrame):
    images_path   = NPY_DIR / "images.npy"
    metadata_path = NPY_DIR / "metadata.csv"

    if images_path.exists() and metadata_path.exists():
        print("  Already converted — skipping.")
        return

    NPY_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    print(f"  Converting {n:,} images to float32 NPY ({IMG_SIZE}×{IMG_SIZE})...")

    img_mm = np.lib.format.open_memmap(
        str(images_path),
        mode  = "w+",
        dtype = np.float32,
        shape = (n, IMG_SIZE, IMG_SIZE),
    )

    out_idx = 0
    CHUNK   = 2000

    for start in tqdm(range(0, n, CHUNK), desc="  Converting"):
        end   = min(start + CHUNK, n)
        chunk = df.iloc[start:end]
        for _, row in chunk.iterrows():
            try:
                img = Image.open(row["image_file"]).convert("L")
                if img.size != (IMG_SIZE, IMG_SIZE):
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img_mm[out_idx] = np.array(img, dtype=np.float32) / 255.0
                out_idx += 1
            except Exception as e:
                print(f"  WARNING: skipped {row['image_file']}: {e}")
        img_mm.flush()

    final_n = out_idx
    if final_n < n:
        final = np.array(img_mm[:final_n])
        del img_mm
        np.save(images_path, final)
        del final
        df = df.iloc[:final_n].reset_index(drop=True)
    else:
        del img_mm

    df.iloc[:final_n].reset_index(drop=True).to_csv(metadata_path, index=False)

    size_gb = images_path.stat().st_size / 1024**3
    print(f"  Saved {final_n:,} images — {size_gb:.2f} GB")
    print(f"  Metadata: {metadata_path}")
    print()
    print("  Final class counts:")
    meta = pd.read_csv(metadata_path)
    for cls in CLASSES:
        n_cls = (meta.class_name == cls).sum()
        print(f"    {cls:<12}: {n_cls:>5,}")
    print()
    print("  *** Use these counts to finalize SITE_DISTRIBUTIONS in config.py ***")


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_raw():
    print("  Cleaning up raw JPEG files to free disk space...")
    for folder_name in ["Training", "Testing", "__MACOSX"]:
        d = RAW_DIR / folder_name
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed raw/{folder_name}")
    print("  Cleanup complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  Brain Tumor MRI — Download + Convert")
    print("  " + "─" * 40)
    print()

    print("  [1/3] Downloading...")
    download()
    print()

    print("  [2/3] Scanning images...")
    df = find_all_images()
    print()

    print("  [3/3] Converting...")
    convert(df)
    print()

    cleanup_raw()
    print()
    print("  Done. Next: update SITE_DISTRIBUTIONS in config.py,")
    print("  then run: python data/split.py")
    print()
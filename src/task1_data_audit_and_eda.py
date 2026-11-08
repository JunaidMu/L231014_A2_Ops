"""
Task 1.2 & 1.3: Dataset Download (via Kaggle API), Dataset Audit, and Visual EDA
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)
"""

import hashlib
import os
import shutil
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError

# Set random seed for reproducible visual sampling
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dataset() -> Path:
    """
    Downloads the Kaggle dataset 'cashbowman/ai-generated-images-vs-real-images'
    using kagglehub API if not already present in data/raw.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Check if already populated with images
    existing_files = [
        p for p in RAW_DATA_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]
    if len(existing_files) > 10:
        print(f"[INFO] Dataset already found in {RAW_DATA_DIR} ({len(existing_files)} files).")
        return RAW_DATA_DIR

    print("[INFO] Downloading dataset via Kaggle API (kagglehub)...")
    cache_path = Path(kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images"))
    print(f"[INFO] Downloaded to cache: {cache_path}")

    # Copy folders into data/raw so DVC (Task 1.4) can track data/raw directly
    print(f"[INFO] Copying dataset into {RAW_DATA_DIR} for project & DVC tracking...")
    for item in cache_path.iterdir():
        dest = RAW_DATA_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print("[INFO] Dataset ready in data/raw!")
    return RAW_DATA_DIR


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 content hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_dataset_audit(data_dir: Path) -> pd.DataFrame:
    """
    Task 1.2: Walks the dataset and builds a table with, for every image:
    class, file size, width, height, colour mode, format, and content hash (SHA-256).
    Detects unreadable/corrupt files and exact duplicates.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print(f"\n[INFO] Walking dataset directory: {data_dir} ...")
    all_files = sorted([p for p in data_dir.rglob("*") if p.is_file() and not p.name.startswith(".")])

    for file_path in all_files:
        # The label of an image is the folder it sits in
        label = file_path.parent.name
        file_size_bytes = file_path.stat().st_size
        file_size_kb = file_size_bytes / 1024.0
        sha256_hash = compute_sha256(file_path)

        is_corrupt = False
        width, height, color_mode, img_format = np.nan, np.nan, None, None

        try:
            # Verify integrity first
            with Image.open(file_path) as img:
                img.verify()
            # Re-open to read dimensions and metadata cleanly
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                color_mode = img.mode
                img_format = img.format
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            is_corrupt = True
            print(f"  [CORRUPT/UNREADABLE] {file_path.relative_to(PROJECT_ROOT)} -> {exc}")

        records.append({
            "filepath": str(file_path.relative_to(PROJECT_ROOT)),
            "class": label,
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_kb, 2),
            "width": width,
            "height": height,
            "color_mode": color_mode,
            "format": img_format,
            "sha256": sha256_hash,
            "is_corrupt": is_corrupt,
        })

    df = pd.DataFrame(records)
    # Mark duplicates (all occurrences after the first matching SHA-256 hash)
    df["is_duplicate"] = df.duplicated(subset=["sha256"], keep="first")

    audit_csv_path = REPORTS_DIR / "dataset_audit.csv"
    df.to_csv(audit_csv_path, index=False)
    print(f"[INFO] Full per-image audit table saved to: {audit_csv_path}")

    # Summary Statistics for Task 1.2 Report Table
    valid_df = df[~df["is_corrupt"]].copy()

    summary_rows = []
    for cls_name, group in df.groupby("class"):
        valid_group = group[~group["is_corrupt"]]
        summary_rows.append({
            "Class": cls_name,
            "Total Files": len(group),
            "Valid Images": len(valid_group),
            "Corrupt Files": int(group["is_corrupt"].sum()),
            "Exact Duplicates (within/across)": int(group["is_duplicate"].sum()),
            "Width (Min / Median / Max)": (
                f"{int(valid_group['width'].min())} / "
                f"{int(valid_group['width'].median())} / "
                f"{int(valid_group['width'].max())}"
            ),
            "Height (Min / Median / Max)": (
                f"{int(valid_group['height'].min())} / "
                f"{int(valid_group['height'].median())} / "
                f"{int(valid_group['height'].max())}"
            ),
            "File Size KB (Min / Median / Max)": (
                f"{valid_group['file_size_kb'].min():.1f} / "
                f"{valid_group['file_size_kb'].median():.1f} / "
                f"{valid_group['file_size_kb'].max():.1f}"
            ),
        })

    # Overall row
    summary_rows.append({
        "Class": "ALL (Total)",
        "Total Files": len(df),
        "Valid Images": len(valid_df),
        "Corrupt Files": int(df["is_corrupt"].sum()),
        "Exact Duplicates (within/across)": int(df["is_duplicate"].sum()),
        "Width (Min / Median / Max)": (
            f"{int(valid_df['width'].min())} / "
            f"{int(valid_df['width'].median())} / "
            f"{int(valid_df['width'].max())}"
        ),
        "Height (Min / Median / Max)": (
            f"{int(valid_df['height'].min())} / "
            f"{int(valid_df['height'].median())} / "
            f"{int(valid_df['height'].max())}"
        ),
        "File Size KB (Min / Median / Max)": (
            f"{valid_df['file_size_kb'].min():.1f} / "
            f"{valid_df['file_size_kb'].median():.1f} / "
            f"{valid_df['file_size_kb'].max():.1f}"
        ),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = REPORTS_DIR / "dataset_audit_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 95)
    print("TASK 1.2: DATASET AUDIT SUMMARY TABLE (Junaid Muhammad | L231014_A2_Ops)")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("-" * 95)
    print(f"Colour Modes Found : {valid_df['color_mode'].value_counts().to_dict()}")
    print(f"Image Formats Found: {valid_df['format'].value_counts().to_dict()}")
    print(f"Total Corrupt Files: {int(df['is_corrupt'].sum())}")
    print(f"Total Exact Duplicates (by SHA-256): {int(df['is_duplicate'].sum())}")
    print("=" * 95 + "\n")

    return df


def run_visual_eda(df: pd.DataFrame) -> None:
    """
    Task 1.3: Generates a grid of sample images per class and histograms of
    width, height, and file size per class.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    valid_df = df[(~df["is_corrupt"]) & (~df["is_duplicate"])].copy()
    classes = sorted(valid_df["class"].unique())

    # 1. Sample Image Grid per Class (2 rows x 5 columns)
    n_samples = 5
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(15, 3.2 * len(classes)))
    if len(classes) == 1:
        axes = [axes]

    for row_idx, cls_name in enumerate(classes):
        cls_subset = valid_df[valid_df["class"] == cls_name]
        sampled = cls_subset.sample(n=min(n_samples, len(cls_subset)), random_state=42)

        for col_idx, (_, row) in enumerate(sampled.iterrows()):
            ax = axes[row_idx][col_idx]
            img_path = PROJECT_ROOT / row["filepath"]
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                ax.imshow(img_rgb)
            ax.set_title(
                f"{cls_name}\n{int(row['width'])}x{int(row['height'])} ({row['color_mode']})",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle(
        "Task 1.3: Sample Images per Class (L231014_A2_Ops - VisionGuard)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    grid_path = FIGURES_DIR / "task1_3_sample_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved sample image grid to: {grid_path}")

    # 2. Histograms of Width, Height, and File Size (KB) per Class
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("width", "Image Width (pixels)", axes[0]),
        ("height", "Image Height (pixels)", axes[1]),
        ("file_size_kb", "File Size (KB)", axes[2]),
    ]

    for col, label, ax in metrics:
        sns.histplot(
            data=valid_df,
            x=col,
            hue="class",
            kde=True,
            bins=30,
            element="step",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Distribution of {label} per Class", fontsize=12, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

    fig.suptitle(
        "Task 1.3: Histograms of Width, Height, and File Size per Class (L231014_A2_Ops)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    hist_path = FIGURES_DIR / "task1_3_histograms.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved histograms figure to: {hist_path}")


if __name__ == "__main__":
    dataset_dir = ensure_dataset()
    audit_df = run_dataset_audit(dataset_dir)
    run_visual_eda(audit_df)

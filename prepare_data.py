"""
prepare_data.py
===============
Scans the raw wildlife trail-camera images, extracts species labels from
filenames, and organises them into train / validation / test splits.
Also copies a sample of unlabeled images into data/raw/images for the
"classify unseen data" step.

Usage:
    python prepare_data.py
"""

import os
import random
import shutil
from pathlib import Path

# ────────────────────────────── configuration ──────────────────────────────

SEED = 42
random.seed(SEED)

# Source folder with all raw images
SOURCE_DIR = Path(r"C:\Users\User\OneDrive\COMPUTER VISION IMAGES\Images")

# Destination inside the project
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
UNSEEN_DIR = PROJECT_ROOT / "data" / "raw" / "images"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

# Species keywords to search for in filenames (lowercase)
# We only keep classes with enough images for meaningful training.
SPECIES_KEYWORDS = {
    "deer": "deer",
    "elk": "elk",
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Number of unlabeled images to copy as unseen data
UNSEEN_SAMPLE_SIZE = 30

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────── helper functions ─────────────────────────────

def find_species(filename: str) -> str | None:
    """Return the species label if the filename contains a known keyword."""
    name_lower = filename.lower()
    for keyword, label in SPECIES_KEYWORDS.items():
        if keyword in name_lower:
            return label
    return None


def collect_images(source: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Walk the source tree and bucket images by species label.

    Returns:
        labeled: dict mapping species name → list of image paths
        unlabeled: list of image paths with no species keyword
    """
    labeled: dict[str, list[Path]] = {v: [] for v in SPECIES_KEYWORDS.values()}
    unlabeled: list[Path] = []

    for dirpath, _, filenames in os.walk(source):
        for fname in filenames:
            if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            full_path = Path(dirpath) / fname
            species = find_species(fname)
            if species:
                labeled[species].append(full_path)
            else:
                unlabeled.append(full_path)

    return labeled, unlabeled


def split_list(items: list, train_r: float, val_r: float):
    """Shuffle and split a list into train / val / test."""
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def copy_images(image_paths: list[Path], dest_dir: Path):
    """Copy a list of images into dest_dir, handling name collisions."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    for src in image_paths:
        name = src.name
        # handle duplicates by prepending a counter
        if name in seen_names:
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while name in seen_names:
                name = f"{stem}_{counter}{suffix}"
                counter += 1
        seen_names.add(name)
        shutil.copy2(src, dest_dir / name)


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    print("Scanning images in", SOURCE_DIR)
    labeled, unlabeled = collect_images(SOURCE_DIR)

    print("\n── Labeled image counts ──")
    for species, paths in labeled.items():
        print(f"  {species}: {len(paths)}")
    print(f"  unlabeled: {len(unlabeled)}")

    # Clear old processed data
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    if UNSEEN_DIR.exists():
        shutil.rmtree(UNSEEN_DIR)

    # Split and copy labeled images
    print("\n── Splitting and copying ──")
    for species, paths in labeled.items():
        if len(paths) < 5:
            print(f"  SKIP {species} (only {len(paths)} images)")
            continue

        train, val, test = split_list(paths, TRAIN_RATIO, VAL_RATIO)
        print(f"  {species}: train={len(train)}, val={len(val)}, test={len(test)}")

        copy_images(train, TRAIN_DIR / species)
        copy_images(val, VAL_DIR / species)
        copy_images(test, TEST_DIR / species)

    # Copy unseen images
    print(f"\n── Copying {UNSEEN_SAMPLE_SIZE} unseen images ──")
    sample = random.sample(unlabeled, min(UNSEEN_SAMPLE_SIZE, len(unlabeled)))
    copy_images(sample, UNSEEN_DIR)
    print(f"  Copied {len(sample)} unseen images to {UNSEEN_DIR}")

    # Summary
    print("\n── Final folder structure ──")
    for split_name, split_dir in [("train", TRAIN_DIR), ("validation", VAL_DIR), ("test", TEST_DIR)]:
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len([f for f in class_dir.iterdir() if f.is_file()])
                print(f"  {split_name}/{class_dir.name}: {count}")

    print("\nDone! Data is ready for the notebook.")


if __name__ == "__main__":
    main()

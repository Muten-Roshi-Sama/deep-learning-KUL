"""
Step 1 — Check the environment, look at CIFAR-10, and prepare the split.

1. Checks that the required software modules are loaded (see
   REQUIRED_MODULES below), e.g.:

       module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl

   and that the dataset (CIFAR10_PATH in common.py) and the pretrained
   ResNet18 weights (RESNET18_WEIGHTS) are readable. Nothing is downloaded.

2. Explores the data: array shapes, pixel range, class balance, and a grid
   of example images per class. Compare with Fashion-MNIST in lab 1: the
   images are small (32x32), but now have 3 colour channels.

3. Splits the official training set into train and validation, like lab 1:

  - train:      45,000 images of the official training set
  - validation: the other VAL_SIZE (5,000) training images, chosen at random
                with the same class balance (used to compare the models)
  - final test: the official test set, 10,000 images (used once, at the very
                end, in 04_compare_and_evaluate.py)

   and computes the mean and std of each colour channel over the training
   images only (no leakage), used to normalize the images for the CNN
   trained from scratch. The images themselves are not copied: only the
   indices of the split and the statistics are saved.

Output: processed/split.npz, plots/examples_per_class.png
Next step: 02_train_scratch_cnn.py
"""

import os
import sys

REQUIRED_MODULES = [
    "PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl",
]
N_EXAMPLES = 8  # example images per class in the plot


def check_modules_loaded() -> bool:
    """Check that the required modules are loaded (Lmod lists them in $LOADEDMODULES)."""
    loaded = [m for m in os.environ.get("LOADEDMODULES", "").split(":") if m]
    missing = [m for m in REQUIRED_MODULES if m not in loaded]

    if missing:
        print("Error: required module(s) not loaded:")
        for m in missing:
            print(f" - {m}")
        print("\nLoad them first, e.g.:")
        for m in missing:
            print(f"    module load {m}")
        return False

    print("OK: Required modules loaded:")
    for m in REQUIRED_MODULES:
        print(f" - {m}")
    print()
    return True


def check_access(path) -> bool:
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        return False
    if not os.access(path, os.R_OK):
        print(f"Error: Permission denied for '{path}'.")
        return False
    print(f"OK: {path} is readable")
    return True


def plot_examples(images, labels, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from common import LABELS, SEED

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(LABELS), N_EXAMPLES, figsize=(N_EXAMPLES * 1.1, len(LABELS) * 1.2))
    for c, label in enumerate(LABELS):
        examples = rng.choice(np.flatnonzero(labels == c), N_EXAMPLES, replace=False)
        for ax, i in zip(axes[c], examples):
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[c, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("CIFAR-10 training images (32 x 32 x 3)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not check_modules_loaded():
        sys.exit(1)

    # Imported only after the module check: common.py needs numpy/torch,
    # which are only available once the modules are loaded.
    import numpy as np
    from sklearn.model_selection import train_test_split

    from common import (
        CIFAR10_PATH,
        LABELS,
        PLOTS_DIR,
        RESNET18_WEIGHTS,
        SEED,
        VAL_SIZE,
        ensure_dirs,
        load_raw_cifar10,
        save_split,
    )

    if not (check_access(CIFAR10_PATH) and check_access(RESNET18_WEIGHTS)):
        sys.exit(1)
    ensure_dirs()

    # --- Explore ---------------------------------------------------------
    official_train_x, official_train_y, test_x, test_y = load_raw_cifar10()
    print(f"\nTraining images: {official_train_x.shape}, {official_train_x.dtype}")
    print(f"Test images:     {test_x.shape}, {test_x.dtype}")
    print("  shape = (images, height, width, colour channels)")
    print(f"Pixel values: min {official_train_x.min()}, max {official_train_x.max()}")

    print("\nTraining images per class:")
    counts = np.bincount(official_train_y, minlength=len(LABELS))
    for label, count in zip(LABELS, counts):
        print(f"  {label:10s} {count}")

    out_path = PLOTS_DIR / "examples_per_class.png"
    plot_examples(official_train_x, official_train_y, out_path)
    print(f"Saved {out_path}")

    # --- Split and normalization statistics -------------------------------
    train_idx, val_idx = train_test_split(
        np.arange(len(official_train_y)),
        test_size=VAL_SIZE, random_state=SEED, stratify=official_train_y,
    )
    print(f"\nTrain:      {len(train_idx):6d} images")
    print(f"Validation: {len(val_idx):6d} images")
    print(f"Final test: {len(test_y):6d} images (official test set)")

    # Per colour channel, over all training pixels, on the 0-1 scale.
    train_pixels = official_train_x[train_idx].reshape(-1, 3) / 255.0
    mean = train_pixels.mean(axis=0)
    std = train_pixels.std(axis=0)
    print("\nChannel statistics (R, G, B), on the 0-1 scale:")
    print(f"  CIFAR-10 train: mean {np.round(mean, 3)}, std {np.round(std, 3)}")
    print("  The CNN trained from scratch (step 02) normalizes with these; the pretrained")
    print("  ResNet18 (step 03) with the ImageNet statistics it was trained with.")

    save_split(train_idx, val_idx, mean, std)
    print("\nSaved the split and channel statistics to processed/split.npz")


if __name__ == "__main__":
    main()

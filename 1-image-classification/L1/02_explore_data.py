"""
Step 2 — Explore the raw data: shapes, pixel values, class balance, examples.

Look at the data before you model it. You should leave this step knowing:
  - how many images you have, and that each is a 28x28 grid flattened to 784 numbers
  - that pixels are integers 0-255 (not yet standardised)
  - that the 10 classes are balanced (same count each) — so accuracy is a fair metric
  - what the garments actually look like (saved plot)

Requires: 01_check_dataset_access.py
Output: plots/example_images.png
Next step: 03_plot_pca.py
"""

import matplotlib.pyplot as plt
import numpy as np

from common import IMAGE_SHAPE, LABELS, PLOTS_DIR, ensure_dirs, load_raw_fashion_mnist


def main():
    ensure_dirs()
    # Official Fashion-MNIST split: 60,000 train images, 10,000 test images.
    # train_x shape is (60000, 784): one flattened image per row.
    # train_y shape is (60000,): one integer label 0-9 per image.
    # train_x, train_y, test_x, test_y = load_raw_fashion_mnist()

    # print(f"Train X: {train_x.shape}  |  Train y: {train_y.shape}")
    # print(f"Test  X: {test_x.shape}  |  Test  y: {test_y.shape}")
    # print(f"Each image is {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]} pixels, flattened to {train_x.shape[1]} features.")
    # print(f"Pixel values: min {train_x.min():.0f}, max {train_x.max():.0f}, mean {train_x.mean():.1f}")

    train_x, train_y, test_x, test_y = load_raw_fashion_mnist()

    print("=" * 60)
    print("SHAPES")
    print("=" * 60)
    print(f"Train X: {train_x.shape}  |  Train y: {train_y.shape}")
    print(f"Test  X: {test_x.shape}  |  Test  y: {test_y.shape}")
    print(f"Each image is {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]} pixels, "
        f"flattened to {train_x.shape[1]} features.")
    print(f"Pixel values: min {train_x.min():.0f}, max {train_x.max():.0f}, "
        f"mean {train_x.mean():.1f}")

    print()
    print("=" * 60)
    print("DTYPES")
    print("=" * 60)
    print(f"train_x dtype: {train_x.dtype}")
    print(f"train_y dtype: {train_y.dtype}")

    print()
    print("=" * 60)
    print("NDIM / SIZE")
    print("=" * 60)
    print(f"train_x.ndim = {train_x.ndim}   (2D: samples × features)")
    print(f"train_y.ndim = {train_y.ndim}   (1D: one label per sample)")
    print(f"train_x.size = {train_x.size:,}   (60000 × 784 = {60000*784:,})")
    print(f"train_y.size = {train_y.size:,}")

    print()
    print("=" * 60)
    print("ONE IMAGE (first sample)")
    print("=" * 60)
    print(f"train_x[0].shape = {train_x[0].shape}")   # (784,)
    print(f"train_x[0][:10]  = {train_x[0][:10]}")    # first 10 pixel values
    print(f"train_x[0].min() = {train_x[0].min()}, "
        f"max() = {train_x[0].max()}, mean() = {train_x[0].mean():.1f}")

    print()
    print("=" * 60)
    print("ONE LABEL (first sample)")
    print("=" * 60)
    print(f"train_y[0] = {train_y[0]}   (integer class index)")
    print(f"train_y[:20] = {train_y[:20]}")
    print(f"Unique labels: {np.unique(train_y)}")
    print(f"Number of classes: {len(np.unique(train_y))}")

    print()
    print("=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)
    labels, counts = np.unique(train_y, return_counts=True)
    for lbl, cnt in zip(labels, counts):
        print(f"  Class {lbl}: {cnt} samples ({100*cnt/len(train_y):.1f}%)")

    print()
    print("=" * 60)
    print("RESHAPING DEMO")
    print("=" * 60)
    img_flat = train_x[0]                 # (784,)
    img_2d   = img_flat.reshape(28, 28)   # (28, 28)
    print(f"Flat shape: {img_flat.shape}")
    print(f"Reshaped:   {img_2d.shape}")
    print(f"Reshaped first row (28 pixels):\n{img_2d[0]}")





    # np.unique(..., return_counts=True) finds every distinct label in train_y
    # and how many times it appears. unique is e.g. [0, 1, ..., 9]; counts is
    # the matching tally (6000 each on Fashion-MNIST — perfectly balanced).
    # If one class dominated, a dummy "always predict that class" model would
    # already look accurate, and accuracy would be a misleading metric.
    print("\nClass distribution (train):")
    unique, counts = np.unique(train_y, return_counts=True)
    for label_idx, count in zip(unique.astype(int), counts):
        print(f"  {LABELS[label_idx]:12s}: {count:5d} ({100 * count / len(train_y):.1f}%)")

    # One row of example images per class.
    # train_y == class_idx is a boolean mask of length 60,000: True where the
    # label is this class. Indexing train_x with that mask keeps only those
    # rows. [:n_examples] then takes the first 8 of them. So this is
    # "the first 8 training images whose label is class_idx".
    n_examples = 8
    fig, axes = plt.subplots(len(LABELS), n_examples, figsize=(n_examples, len(LABELS) * 1.1))
    for class_idx, label in enumerate(LABELS):
        examples = train_x[train_y == class_idx][:n_examples]
        for j, image in enumerate(examples):
            ax = axes[class_idx, j]
            # The MLP sees 784 numbers; we reshape back to 28x28 only to draw.
            ax.imshow(image.reshape(IMAGE_SHAPE), cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[class_idx, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Fashion-MNIST examples (training set)")
    plt.tight_layout()

    out_path = PLOTS_DIR / "example_images.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


# Python executes the whole file on `import` *and* on `python this_file.py`.
# The if-guard means main() runs only when you launch this file as a program,
# not when another script does `from 02_explore_data import ...`.
# Every numbered script in this lab is a standalone program: that is why each
# one ends with this pair of lines.
if __name__ == "__main__":
    main()

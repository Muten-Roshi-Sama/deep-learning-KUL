"""
Step 4 — Preprocess: split off a validation set and standardize.

  - train:      50,000 images of the official training set
  - validation: the other VAL_SIZE (10,000) training images, chosen at random
                with the same class balance (used to choose settings)
  - final test: the official test set, 10,000 images (used once, at the very
                end, in 16_train_final_and_evaluate.py)

The scaler is fit on the training images only, then applied to validation
and test (no leakage). It standardizes each pixel to mean 0 / std 1 over the
training set. Both the raw pixels (0-255) and the standardized features are
saved, so 10_experiment_preprocessing.py can compare training on each.

Requires: 01_check_dataset_access.py
Output: processed/*.npy ({train,val,test}_x_{raw,scaled}, {train,val,test}_y)
Next step: 05_inspect_dataloaders.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import LABELS, SEED, VAL_SIZE, load_raw_fashion_mnist, save_processed


def print_class_balance(name, y):
    """Print how many images of each class sit in a split.

    After a stratified split, train / val / test should all look like the
    official data: about 10% per class. If one split were 90% trousers, the
    validation score would not mean what you think it means.
    """
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass balance ({name}, n={len(y)}):")
    for label_idx, count in zip(unique.astype(int), counts):
        print(f"  {LABELS[label_idx]:12s}: {count:5d} ({100 * count / len(y):.1f}%)")


def main():
    
      # 1. Load the data
      official_train_x, official_train_y, test_x, test_y = load_raw_fashion_mnist()


      # 2. Split into training and validation
      train_x, val_x, train_y, val_y = train_test_split(
            official_train_x, official_train_y, 
            train_size=0.8, 
            test_size=0.2,
            stratify=official_train_y,
            random_state=0
      )

      # 3. Verify shapes
      print(f"Train:      {len(train_y):6d} images")
      print(f"Validation: {len(val_y):6d} images")
      print(f"Final test: {len(test_y):6d} images (official test set)")
      print_class_balance("train", train_y)
      print_class_balance("validation", val_y)
      print_class_balance("test", test_y)

      # 4. Scale the features
      scaler = StandardScaler()
      
      scaler.fit(train_x)    # learn the mean and std from training only

      train_x_scaled = scaler.transform(train_x) # use scaler to normalize 
      val_x_scaled = scaler.transform(val_x)
      test_x_scaled = scaler.transform(test_x)     
            
      # Note: possible to fit and transform in one line
      # train_x_scaled = scaler.fit_transform(train_x) 
      
      # Note : to avoid data leakage, do not scale using the validation set !

      

      # 5. Print
      centre = 14 * 28 + 14
      print("\nPixel stats before scaling (train):")
      print(f"  all pixels: min {train_x.min():.0f}, max {train_x.max():.0f}, "
            f"mean {train_x.mean():.1f}, std {train_x.std():.1f}")
      print(f"  centre pixel (index {centre}): mean {train_x[:, centre].mean():.1f}, "
            f"std {train_x[:, centre].std():.1f}")
      print("Pixel stats after scaling (train):")
      print(f"  all pixels: min {train_x_scaled.min():.1f}, max {train_x_scaled.max():.1f}, "
            f"mean {train_x_scaled.mean():.2f}, std {train_x_scaled.std():.2f}")
      print(f"  centre pixel (index {centre}): mean {train_x_scaled[:, centre].mean():.2f}, "
            f"std {train_x_scaled[:, centre].std():.2f}")

      save_processed(
            train_x, train_x_scaled, train_y,
            val_x, val_x_scaled, val_y,
            test_x, test_x_scaled, test_y,
      )
      print("\nSaved raw and standardized arrays to processed/")


      # ---------------- Original code ----------------

    # official_train_* is the 60,000-image official training set.
    # test_* is the official 10,000-image test set — it is NOT split further.
#     official_train_x, official_train_y, test_x, test_y = load_raw_fashion_mnist()

    # ----------------------------------------------------------------------
    # TODO 1a — split the official training set into train + validation.
    #
    #   train_test_split is sklearn's standard way to cut a dataset in two.
    #
    #   Arguments that matter here:
    #     test_size=VAL_SIZE  — how many samples go to the SECOND returned pair
    #                           (10_000 → val gets 10k, train keeps 50k).
    #                           sklearn still *names* that part "test" in the
    #                           docs; here it is our VALIDATION set.
    #     random_state=SEED   — same random draw every time you run this.
    #     stratify=y          — keep the same class proportions in both parts.
    #                           Without it, a lucky draw could dump most shirts
    #                           into validation.
    #
    #   Return order is always:
    #       X_train, X_heldout, y_train, y_heldout
    #
    #   Hint: train_test_split(features, labels, test_size=..., random_state=...,
    #                          stratify=...)
    #
    #   The real test set is already loaded above as test_x / test_y. It must
    #   stay untouched until step 16.
    # ----------------------------------------------------------------------
#     train_x, val_x, train_y, val_y = train_test_split(
#       official_train_x, official_train_y, 
#       train_size=0.8, 
#       test_size=0.2, 
#       random_state=0
#       )

#     print(f"Train:      {len(train_y):6d} images")
#     print(f"Validation: {len(val_y):6d} images")
#     print(f"Final test: {len(test_y):6d} images (official test set)")
#     print_class_balance("train", train_y)
#     print_class_balance("validation", val_y)
#     print_class_balance("test", test_y)

    # ----------------------------------------------------------------------
    # TODO 1b — standardize the features (mean 0, std 1 per pixel).
    #
    #   StandardScaler learns one mean and one std PER FEATURE (here: per pixel).
    #   The scaler has to LEARN those numbers from one split and then apply
    #   the same numbers to the others. Which split should it learn from, and
    #   why would using all three be wrong? (data leakage — theory recap.)
    #
    #   Hint: one call to .fit_transform(), two calls to .transform().
    #   Keep the result as float32: .astype(np.float32)
    #
    #   Note this bug does not crash. Getting it wrong still trains a model
    #   and still prints plausible numbers — they are just optimistic.
    # ----------------------------------------------------------------------
#     scaler = StandardScaler()
#     scaler.fit(train_x)
#     train_x_scaled = train_x.transform()
#     val_x_scaled = ...
#     test_x_scaled = ...

    # Images are stored flattened in row-major order: pixel index = row*28 + col.
    # Row 14, column 14 is the middle of the 28x28 grid, so
    #     14 * 28 + 14 = 406
    # is the centre pixel. Border pixels are often constantly 0 (white
    # background); after scaling they stay 0. The centre actually contains
    # ink, so it is the pixel where "mean 0, std 1" should be visible.
#     centre = 14 * 28 + 14
#     print("\nPixel stats before scaling (train):")
#     print(f"  all pixels: min {train_x.min():.0f}, max {train_x.max():.0f}, "
#           f"mean {train_x.mean():.1f}, std {train_x.std():.1f}")
#     print(f"  centre pixel (index {centre}): mean {train_x[:, centre].mean():.1f}, "
#           f"std {train_x[:, centre].std():.1f}")
#     print("Pixel stats after scaling (train):")
#     print(f"  all pixels: min {train_x_scaled.min():.1f}, max {train_x_scaled.max():.1f}, "
#           f"mean {train_x_scaled.mean():.2f}, std {train_x_scaled.std():.2f}")
#     print(f"  centre pixel (index {centre}): mean {train_x_scaled[:, centre].mean():.2f}, "
#           f"std {train_x_scaled[:, centre].std():.2f}")

#     save_processed(
#         train_x, train_x_scaled, train_y,
#         val_x, val_x_scaled, val_y,
#         test_x, test_x_scaled, test_y,
#     )
#     print("\nSaved raw and standardized arrays to processed/")


if __name__ == "__main__":
    main()

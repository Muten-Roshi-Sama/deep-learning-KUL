"""
Step 2 — Build LeNet-5 for RGB images and train it from scratch.

Look at the LeNet-5 figure from the slides. Your job is to work out BY HAND
the parameters of every layer, and fill them in below (replace each `None`).
Use the figure for the sizes (32x32 input, 6@28x28, 6@14x14, ...), but note:
our images have 3 colour channels (RGB), not 1.

Hints for the Conv2d(in_channels, out_channels, kernel_size) layers:
  - in_channels:  how many channels does the layer receive?
  - out_channels: how many feature maps does the figure show after it?
  - kernel_size:  stride 1, no padding, so out_size = in_size - kernel_size + 1.
                  Which kernel turns 32x32 into 28x28?
  - MaxPool2d(kernel_size) divides the size by its kernel size (28 -> 14).
  - Linear(in_features, out_features): after Flatten, in_features is
    channels * height * width of the last feature map.

When you run the script it prints the shape after every layer, so you can
check your numbers against the figure. A wrong number crashes at that layer.

Two stages:
  1. DEVICE = "cpu": runs on a tiny subset for 2 epochs. Only checks that the
     model is correct. Nothing is saved.
  2. When it works, set DEVICE = "cuda" (run on a GPU node): full training.

Requires: 01_prepare_data.py
Output (on cuda): artifacts/models/scratch_cnn.pt, artifacts/histories/scratch_cnn.json,
                  plots/scratch_cnn_curves.png
Next step: 03_train_pretrained.py (first complete mini_resnet.py)
"""

import time

import torch
import torch.nn as nn
import torchvision.transforms as T

from common import (
    EPOCHS, LABELS, count_parameters, load_datasets, load_split, make_loaders,
    plot_comparison, quick_subsets, save_model, set_seed, summarize, train_model,
)

DEVICE = "cuda"  # TODO: first run on the CPU. When everything works, change to "cuda"
LR = 1e-3


def show_shapes(model):
    """Print the shape of one image after every layer."""
    x = torch.zeros(1, 3, 32, 32)  # a batch of one RGB image
    print(f"{'input':10s} -> {tuple(x.shape[1:])}")
    for layer in model:
        x = layer(x)
        print(f"{type(layer).__name__:10s} -> {tuple(x.shape[1:])}")


def main():
    set_seed()
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print('DEVICE = "cuda" but there is no GPU in your compute instance: defaulting to CPU (full training will be slow).')
        device = "cpu"
    print(f"Running on {device.upper()}" + (" (quick check: small subset, 2 epochs, nothing saved)" if DEVICE == "cpu" else ""))

    # The channel mean/std of our training images, computed in step 01.
    *_, mean, std = load_split()

    # TODO: fill in every `None` (by hand, from the LeNet-5 figure)
    model = nn.Sequential(
        T.Normalize(mean, std),
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10),
    )
    show_shapes(model)

    

    # QUESTION: compute by hand the number of parameters of the first Conv2d
    # (weights + biases) and of the first Linear layer. Do they add up with this?
    print(f"Trainable parameters: {count_parameters(model):,}")

    train_ds, val_ds, test_ds = load_datasets()
    epochs = EPOCHS
    if DEVICE == "cpu":  # quick check: a small subset and only 2 epochs
        train_ds, val_ds = quick_subsets(train_ds, val_ds)
        epochs = 2
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, test_ds)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.time()
    history = train_model(model, train_loader, val_loader, optimizer, torch.device(device),
                          epochs=epochs, restore_best=True, desc="scratch_cnn")
    history["train_time_s"] = time.time() - start
    print(f"\nTraining took {history['train_time_s']:.0f} s. {summarize(history)}")

    if DEVICE == "cpu":
        print('\nCPU check done. If there were no errors and the numbers look right, '
              'set DEVICE = "cuda" at the top of this file and run it on a GPU node.')
    else:
        save_model(model, history, "scratch_cnn")
        print(plot_comparison({"scratch_cnn": history}, "LeNet-5 from scratch", "scratch_cnn_curves.png"))


if __name__ == "__main__":
    main()

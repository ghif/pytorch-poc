import os
from time import process_time
import timeit

from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2

import models as M
import trainer as T

from torch.utils.tensorboard import SummaryWriter

# Constants
DATADIR = "/Users/mghifary/Work/Code/AI/data"
MODEL_DIR = "models"
MODEL_SUFFIX = "convnet-checkmps"
BATCH_SIZE = 128
EPOCHS = 1

# set tensorboard "log_dir" to "logs"
writer = SummaryWriter(f"logs/fashion-mnist_{MODEL_SUFFIX}")

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # v2.RandAugment(),
    ]
)

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Download training data from open datasets.
train_data = datasets.FashionMNIST(
    root=DATADIR,
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.FashionMNIST(
    root=DATADIR,
    train=False,
    download=True,
    transform=inference_transform,
)

# Create data loaders
train_dataloader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE,
    shuffle=False,
)

for X, y in train_dataloader:
    [_, c, dx1, dx2] = X.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

num_classes = len(torch.unique(train_data.train_labels))

dataiter = iter(train_dataloader)
images, labels = next(dataiter)

for device in ["cpu", "mps"]:
    # Initialize model
    # model = M.MLP(c, dx1, dx2, 512, num_classes)
    model = M.ConvNet(c, dx1, dx2, num_classes=num_classes)
    print(model)
    print(f"\n Measuring performance on \"{device}\" device")


    print(f"Check training time ...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    checkpoint_dir = os.path.join(MODEL_DIR, "fashion-mnist")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"{MODEL_SUFFIX}.pth")

    T.fit(
        model, 
        train_dataloader, 
        test_dataloader, 
        loss_fn, 
        optimizer, 
        n_epochs=EPOCHS, 
        checkpoint_dir=checkpoint_dir,
        model_name=MODEL_SUFFIX, 
        writer=writer,
        device=device
    )
    
    print("Done!")

    start_t = process_time()
    with torch.no_grad():
        model = model.to(device)
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

    elapsed_time_pred = process_time() - start_t
    print(f"Inference elapsed time ({device}): {elapsed_time_pred} s")
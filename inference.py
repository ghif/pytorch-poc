import os
from time import process_time

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import trainer as T

import models as M

# Constants
BATCH_SIZE = 10000

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DATADIR = "/Users/mghifary/Work/Code/AI/data"
DATASET = "cifar10"
MODEL_DIR = "models"
# MODEL_SUFFIX = "tinyresnetv2-randaug-exp1"
# MODEL_SUFFIX = "resnet18-randaug-exp1"
MODEL_SUFFIX = "convnet-exp1"
# MODEL_SUFFIX = "convnet-randaug-exp1"
# MODEL_SUFFIX = "resnet18-randaug-exp2"
# MODEL_SUFFIX = "plainnet18-randaug-exp2"

# Load data
if DATASET == "fashion_mnist":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.FashionMNIST(
        root=DATADIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.FashionMNIST(
        root=DATADIR,
        train=False,
        download=True,
        transform=transform,
    )
elif DATASET == "cifar10":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_data = datasets.CIFAR10(
        root=DATADIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.CIFAR10(
        root=DATADIR,
        train=False,
        download=True,
        transform=transform,
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

print(f"Infer {DATASET} dataset with {device} device")
for X, y in test_dataloader:
    [_, c, dx1, dx2] = X.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

# Load model
checkpoint_dir = os.path.join(MODEL_DIR, DATASET)
checkpoint_path = os.path.join(checkpoint_dir, f"{MODEL_SUFFIX}.pth")

# model = M.TinyResnetV2(c, M.ResidualBlock, num_classes=10)
# model = M.TinyResnet(c, M.ResidualBlock, num_classes=10)
# model = M.ResNet(c, 18, M.ResidualBlock, num_classes=10)
model = M.ConvNet(c, dx1, dx2, num_classes=10)
# model = M.PlainNet(c, 18, M.PlainBlock, num_classes=10)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

train_loss, train_accuracy, _ = T.evaluate(model, train_dataloader, device=device)
test_loss, test_accuracy, _ = T.evaluate(model, test_dataloader, device=device)

print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
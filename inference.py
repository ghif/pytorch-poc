import os
from time import process_time

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

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
# MODEL_SUFFIX = "convnet-exp1"
# MODEL_SUFFIX = "convnet-randaug-exp1"
# MODEL_SUFFIX = "resnet18-randaug-exp1"
MODEL_SUFFIX = "plainnet18-randaug-exp2"

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
# model = M.ConvNet(c, dx1, dx2, num_classes=10)
model = M.PlainNet(c, 18, M.PlainBlock, num_classes=10)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)
model.eval()

# Predict 1 batch of test data
print(f"Using {device} device")

model_size = M.check_model_size(model)
print(f"Model size: {model_size:.4f} MB")

with torch.no_grad():
    dataiter = iter(test_dataloader)
    X_test, y_test = next(dataiter)
    X_test, y_test = X_test.to(device), y_test.to(device)

    start_t = process_time()
    test_pred = model(X_test)
    elapsed_time = process_time() - start_t
    y_test_pred = torch.argmax(test_pred, axis=1)

n_correct = torch.sum(y_test == y_test_pred)
n_samples = len(y_test)
acc = n_correct / n_samples
print(f"Accuracy: {acc.item():.4f} (Correct prediction: {n_correct} out of {n_samples}, Inference time: {elapsed_time:.4f}")
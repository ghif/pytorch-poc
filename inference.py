import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import models as M

# Constants
c = 1
dx1 = 28
dx2 = 28
BATCH_SIZE = 128

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
DATADIR = "/Users/mghifary/Work/Code/AI/data"
MODEL_DIR = "models"

# Load model
checkpoint_dir = os.path.join(MODEL_DIR, "fashion-mnist_mlp")
checkpoint_path = os.path.join(checkpoint_dir, "model.pth")

model = M.NeuralNetwork(c, dx1, dx2)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)
model.eval()

# Load data
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

# Predict 1 batch of test data
with torch.no_grad():
    dataiter = iter(test_dataloader)
    X_test, y_test = next(dataiter)
    X_test, y_test = X_test.to(device), y_test.to(device)

    test_pred = model(X_test)
    y_test_pred = torch.argmax(test_pred, axis=1)

acc = torch.sum(y_test == y_test_pred) / len(y_test)
print(f"Accuracy: {acc.item():.4f}")
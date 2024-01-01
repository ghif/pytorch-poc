import os
from time import process_time

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import models as M

# Constants
c = 1
dx1 = 28
dx2 = 28
BATCH_SIZE = 10000

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DATADIR = "/Users/mghifary/Work/Code/AI/data"
MODEL_DIR = "models"
MODEL_SUFFIX = "tinyresnetv2-randaug-exp1"


# Load model
checkpoint_dir = os.path.join(MODEL_DIR, "fashion-mnist")
checkpoint_path = os.path.join(checkpoint_dir, "tinyresnetv2-randaug-exp1.pth")

model = M.TinyResnetV2(c, M.ResidualBlock, num_classes=10)
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
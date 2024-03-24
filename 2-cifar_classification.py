import os
from time import process_time
import numpy as np

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
DATASET = "cifar10"
MODEL_DIR = "models"
# MODEL_SUFFIX = "convnet-randaug-exp2"
# MODEL_SUFFIX = "tinyresnetv2-randaug-exp1"
# MODEL_SUFFIX = "plainnet18-randaug-exp2"
# MODEL_SUFFIX = "convnet-randaug-exp1"
# MODEL_SUFFIX = "resnet18-randaug-exp2"
# MODEL_SUFFIX = "resnet18-gelu-randaug-exp1"
MODEL_SUFFIX = "convnet-gelu-randaug-exp1"
BATCH_SIZE = 128
EPOCHS = 100

# set tensorboard "log_dir" to "logs"
writer = SummaryWriter(f"logs/{DATASET}_{MODEL_SUFFIX}")

train_transform = transforms.Compose(
    [
        v2.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Download training data from open datasets.
train_data = datasets.CIFAR10(
    root=DATADIR,
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.CIFAR10(
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

num_classes = len(train_data.classes)

dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# Initialize model
# model = M.MLP(c, dx1, dx2, 512, num_classes)
model = M.ConvNet(
    c, 
    dx1, 
    dx2, 
    num_classes=num_classes,
    with_gelu=True
)
# model = M.TinyResnetV2(c, M.ResidualBlock, num_classes=num_classes)
# model = M.TinyResnet(c, M.ResidualBlock, num_classes=num_classes)
# model = M.PlainNet(c, 18, M.PlainBlock, num_classes=num_classes)
# model = M.ResNet(
#     c, 
#     18, 
#     M.ResidualBlock, 
#     num_classes=num_classes,
#     with_gelu=True
# )

print(model)

device = "mps"
print(f"Using device {device}")

print(f"\n Measuring performance on \"{device}\" device")
print(f"Check training time ...")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
checkpoint_dir = os.path.join(MODEL_DIR, DATASET)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


history = T.fit(
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

elapsed_time_train = np.sum(history["train_times"])
print(f"Training elapsed time ({device}): {elapsed_time_train:.4f} s")

start_t = process_time()
with torch.no_grad():
    model = model.to(device)
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

elapsed_time_pred = process_time() - start_t
print(f"Inference elapsed time ({device}): {elapsed_time_pred:.4f} s")
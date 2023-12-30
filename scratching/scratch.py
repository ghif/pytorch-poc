import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.transforms import v2

import models as M
import trainer as T

from torch.utils.tensorboard import SummaryWriter

from plot_lib import set_default

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Initialize model
c = 1
dx1 = 28
dx2 = 28
num_classes = 10
# model = M.NeuralNetwork(c, dx1, dx2, num_classes, with_bn=True)
# model = M.ResNet(1, 18, M.ResidualBlock, num_classes=num_classes)
# model = M.TinyResnet(c, M.ResidualBlock, num_classes=num_classes)
# model = M.TinyResnetV2(c, M.ResidualBlock, num_classes=num_classes)
model = M.ConvNet(c, dx1, dx2, num_classes=num_classes)
# model = M.MLP(c, dx1, dx2, 512, num_classes)
model = model.to(device)
print(model)
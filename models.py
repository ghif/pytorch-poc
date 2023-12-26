import torch
from torch import nn
from torch import Tensor
from typing import Type

class NeuralNetwork(nn.Module):
    def __init__(self, ch, dx1, dx2, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ch * dx1 * dx2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

def make_residual_layer(
    block: Type[ResidualBlock],
    in_channels: int,
    out_channels: int,
    blocks: int,
    expansion: int = 1,
    stride: int = 1
) -> nn.Sequential:
    """
    This method is used to build the `layer1` to `layer4` of the ResNet.
    Each layer can contain multiple 'ResidualBlock'.

    Args:
        block (Type[ResidualBlock]): The type of ResidualBlock to use.
        out_channels (int): The number of output channels of the first ResidualBlock.
        blocks (int): The number of ResidualBlock to stack together.
        stride (int, optional): The stride of the first ResidualBlock. Defaults to 1.
    """
    downsample = None
    if stride != 1:
        """
        This should pass from `layer2` to `layer4` or 
        when building ResNets50 and above. Section 3.3 of the paper
        Deep Residual Learning for Image Recognition
        (https://arxiv.org/pdf/1512.03385v1.pdf).
        """
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels*expansion,
                kernel_size=1,
                stride=stride,
                bias=False 
            ),
            nn.BatchNorm2d(out_channels * expansion),
        )
    layers = []
    layers.append(
        block(
            in_channels, out_channels, stride, expansion, downsample
        )
    )
    in_channels = out_channels * expansion
    for i in range(1, blocks):
        layers.append(block(
            in_channels,
            out_channels,
            expansion=expansion
        ))
    return nn.Sequential(*layers)

class TinyResnet(nn.Module):
    def __init__(
        self,
        img_channel: int,
        block: Type[ResidualBlock],
        num_classes: int = 10
    ) -> None:
        super().__init__()
        expansion = 1
        in_channels = 64

        self.conv1 = nn.Conv2d(
            img_channel,
            in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_residual_layer(block, in_channels, 64, 2, expansion=expansion)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * expansion, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits

        
    
class ResNet(nn.Module):
    def __init__(
        self, 
        img_channel: int, 
        num_layers: int,
        block: Type[ResidualBlock],
        num_classes: int = 10
    ) -> None:
        super().__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `ResidualBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNet (18 to 152) contain a Conv2d => BN => ReLU for the first layers
        # Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            img_channel,
            self.in_channels,
            kernel_size=7, 
            stride=2, 
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
        self, 
        block: Type[ResidualBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # The spatial dimension of the final layer's feature map should be (7, 7) for all ResNets.
        # print("Dimension of the final layer's feature map: ", x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits
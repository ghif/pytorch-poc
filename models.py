import torch
from torch import nn
from torch import Tensor
from typing import Type

class MLP(nn.Module):
    def __init__(self, ch, dx1, dx2, d_hid, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ch * dx1 * dx2, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),

            nn.Linear(d_hid, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),

            nn.Linear(d_hid, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),

            nn.Linear(d_hid, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class NeuralNetwork(nn.Module):
    def __init__(self, ch, dx1, dx2, num_classes, with_bn=False):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        if with_bn:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(ch * dx1 * dx2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        else:
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

class ConvNet(nn.Module):
    def __init__(self, ch, dx1, dx2, num_classes, with_gelu=False):
        super(ConvNet, self).__init__()

        if with_gelu:
            act_func = nn.GELU()
        else:
            act_func = nn.ReLU()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act_func,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (dx1 // 4) * (dx2 // 4), 128),
            act_func,
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        logits = self.convnet(x)
        return logits
    
class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        with_gelu: bool = False
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

        self.with_gelu = with_gelu
        if with_gelu:
            self.gelu = nn.GELU()
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
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
        if self.with_gelu:
            out = self.gelu(out)
        else:
            out = self.bn1(out)
            out = self.relu(out)
        out = self.conv2(out)

        if not self.with_gelu:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
    
class PlainBlock(nn.Module):
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
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # if self.downsample is not None:
        #     identity = self.downsample(x)
        out = self.relu(out)
        return  out

def make_residual_layer(
    block: Type[ResidualBlock],
    in_channels: int,
    out_channels: int,
    blocks: int,
    expansion: int = 1,
    stride: int = 1,
    with_gelu: bool = False
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
            in_channels, out_channels, stride, expansion, downsample, with_gelu=with_gelu
        )
    )
    in_channels = out_channels * expansion
    for i in range(1, blocks):
        layers.append(block(
            in_channels,
            out_channels,
            expansion=expansion,
            with_gelu=with_gelu
        ))
    return nn.Sequential(*layers)

def make_plain_layer(
    block: Type[PlainBlock],
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

class TinyResnetV2(nn.Module):
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
            kernel_size=3,
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
        num_classes: int = 10,
        with_gelu: bool = False
    ) -> None:
        super().__init__()

        self.with_gelu = with_gelu
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

        if with_gelu:
            self.gelu = nn.GELU()

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = self._make_layer(block, 64, layers[0])        
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer1 = make_residual_layer(block, self.in_channels, 64, layers[0], expansion=self.expansion, with_gelu=with_gelu)
        self.layer2 = make_residual_layer(block, 64, 128, layers[1], stride=2, expansion=self.expansion, with_gelu=with_gelu)
        self.layer3 = make_residual_layer(block, 128, 256, layers[2], stride=2, expansion=self.expansion, with_gelu=with_gelu)
        self.layer4 = make_residual_layer(block, 256, 512, layers[3], stride=2, expansion=self.expansion, with_gelu=with_gelu)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    # def _make_layer(
    #     self, 
    #     block: Type[ResidualBlock],
    #     out_channels: int,
    #     blocks: int,
    #     stride: int = 1
    # ) -> nn.Sequential:
    #     downsample = None
    #     if stride != 1:
    #         """
    #         This should pass from `layer2` to `layer4` or 
    #         when building ResNets50 and above. Section 3.3 of the paper
    #         Deep Residual Learning for Image Recognition
    #         (https://arxiv.org/pdf/1512.03385v1.pdf).
    #         """
    #         downsample = nn.Sequential(
    #             nn.Conv2d(
    #                 self.in_channels, 
    #                 out_channels*self.expansion,
    #                 kernel_size=1,
    #                 stride=stride,
    #                 bias=False 
    #             ),
    #             nn.BatchNorm2d(out_channels * self.expansion),
    #         )
    #     layers = []
    #     layers.append(
    #         block(
    #             self.in_channels, out_channels, stride, self.expansion, downsample
    #         )
    #     )
    #     self.in_channels = out_channels * self.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(
    #             self.in_channels,
    #             out_channels,
    #             expansion=self.expansion
    #         ))
    #     return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        if self.with_gelu:
            x = self.gelu(x)
        else:
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
    
class PlainNet(nn.Module):
    def __init__(
        self, 
        img_channel: int, 
        num_layers: int,
        block: Type[PlainBlock],
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
        block: Type[PlainBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
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
    
def check_model_size(model: nn.Module):
    """
    This method is used to check the size of a model.

    Args:
        model (nn.Module): The model to check.

    Returns: 
        size_all_mb (float): The size of the model in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
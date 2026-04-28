import torch
import torch.nn as nn
import math


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super().__init__()
        D = int(math.floor(planes * (base_width / 64.0))) * cardinality
        # 1x1 reduction
        self.conv1 = nn.Conv2d(inplanes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        # 3x3 grouped conv
        self.conv2 = nn.Conv2d(
            D,
            D,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(D)
        # 1x1 expansion
        self.conv3 = nn.Conv2d(D, planes * ResNeXtBottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * ResNeXtBottleneck.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, layers, cardinality, base_width, num_classes=1000, zero_init_residual=False, for_cifar=False):
        """
        - layers:        e.g. [3,4,6,3] for ResNeXt-50
        - cardinality:   e.g. 32
        - base_width:    e.g. 4  (so 32×4d)
        - num_classes:   1000 for ImageNet, or 10/100 for CIFAR
        - zero_init_residual: zero-init last BN in each block if True
        - for_cifar:     if True, use CIFAR‐style stem (3x3 conv stride=1, no maxpool)
        """
        super().__init__()
        self.inplanes = 64

        if for_cifar:
            # CIFAR stem: 3×3 conv, stride=1, padding=1 (no maxpool)
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = None
        else:
            # ImageNet stem: 7×7 conv stride=2 + BN + ReLU + 3×3 maxpool stride=2
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build 4 ResNeXt stages
        self.layer1 = self._make_layer(planes=64, blocks=layers[0], cardinality=cardinality, base_width=base_width, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=layers[1], cardinality=cardinality, base_width=base_width, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=layers[2], cardinality=cardinality, base_width=base_width, stride=2)
        
        if len(layers) > 3 and layers[3] != 0:
            self.layer4 = self._make_layer(planes=512, blocks=layers[3], cardinality=cardinality, base_width=base_width, stride=2)
        else:
            self.layer4 = None
        
        last_planes = 512 if self.layer4 is not None else 256
        self.fc = nn.Linear(last_planes * ResNeXtBottleneck.expansion, num_classes)    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Optionally zero‐init each block’s final BN so the residual branch starts as zero
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, cardinality, base_width, stride=1):
        """
        planes:     the “base channel” for this stage
        blocks:     how many ResNeXtBottleneck units
        stride:     stride for the first block in this stage
        Returns: nn.Sequential of length=blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * ResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * ResNeXtBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * ResNeXtBottleneck.expansion),
            )

        layers = []
        # First block in stage (may downsample / change channels)
        layers.append(
            ResNeXtBottleneck(
                inplanes=self.inplanes,
                planes=planes,
                cardinality=cardinality,
                base_width=base_width,
                stride=stride,
                downsample=downsample
            )
        )
        self.inplanes = planes * ResNeXtBottleneck.expansion
        # Remaining blocks (stride=1, no downsample)
        for _ in range(1, blocks):
            layers.append(
                ResNeXtBottleneck(
                    inplanes=self.inplanes,
                    planes=planes,
                    cardinality=cardinality,
                    base_width=base_width,
                    stride=1,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        # ResNeXt layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)      # → [batch, 512*expansion, 1, 1]
        x = torch.flatten(x, 1)  # → [batch, 512*expansion]
        x = self.fc(x)           # → [batch, num_classes]
        return x


def resnext50_32x4d(pretrained=False, num_classes=1000, for_cifar=False):
    """
    Builds ResNeXt-50 32×4d:
      - layers = [3,4,6,3]
      - cardinality = 32
      - base_width = 4
      - expansion = 4 (in ResNeXtBottleneck)
      - if pretrained=True, loads official ImageNet weights (1000 classes)
      - if for_cifar=True, uses a CIFAR‐style stem and must set num_classes=10 or 100 manually.
    """
    model = ResNeXt(layers=[3, 4, 6, 3],
                    cardinality=32,
                    base_width=4,
                    num_classes=num_classes,
                    zero_init_residual=False,
                    for_cifar=for_cifar)
    if pretrained and not for_cifar:
        # Official PyTorch‐hosted weights for ResNeXt-50 32×4d
        state_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
            progress=True
        )
        model.load_state_dict(state_dict)
    return model

def resnext29_8x64d(num_classes=10):
    """
    ResNeXt-29 8x64d as used in CIFAR-10/100 paper
    - 29 total layers = (3+1)*3 blocks + stem + FC
    - 3 stages, 3 bottleneck blocks each
    - cardinality = 8, base_width = 64
    """
    return ResNeXt(
        layers=[3, 3, 3],            # 9 bottlenecks = 29 layers
        cardinality=8,
        base_width=64,
        num_classes=num_classes,
        zero_init_residual=False,
        for_cifar=True              # Use CIFAR-style stem
    )

def resnext29_16x64d(num_classes=10):
    """
    ResNeXt-29 8x64d as used in CIFAR-10/100 paper
    - 29 total layers = (3+1)*3 blocks + stem + FC
    - 3 stages, 3 bottleneck blocks each
    - cardinality = 8, base_width = 64
    """
    return ResNeXt(
        layers=[3, 3, 3],            # 9 bottlenecks = 29 layers
        cardinality=16,
        base_width=64,
        num_classes=num_classes,
        zero_init_residual=False,
        for_cifar=True              # Use CIFAR-style stem
    )

def resnext101_32x4d(pretrained=False, num_classes=1000, for_cifar=False):
    """
    Builds ResNeXt-101 32×8d:
      - layers = [3, 4, 23, 3]
      - cardinality = 32
      - base_width = 4
    """
    model = ResNeXt(layers=[3, 4, 23, 3],
                    cardinality=32,
                    base_width=4,
                    num_classes=num_classes,
                    zero_init_residual=False,
                    for_cifar=for_cifar)
    if pretrained and not for_cifar:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 
            progress=True
        )
        model.load_state_dict(state_dict)
    return model


# ===========================
# Example usage / sanity check
# ===========================
if __name__ == "__main__":
    # 1) Build an ImageNet ResNeXt-50 (32×4d) with random weights
    model_imagenet = resnext50_32x4d(pretrained=False, num_classes=1000, for_cifar=False)
    print("ResNeXt-50 32x4d (ImageNet) →", model_imagenet)

    # Test forward on a dummy ImageNet-sized batch
    dummy = torch.randn(2, 3, 224, 224)
    out_imagenet = model_imagenet(dummy)
    print("ImageNet output shape:", out_imagenet.shape)  # expected: [2, 1000]

    # 2) Build a CIFAR-10 ResNeXt-50 (32×4d):
    #    - Stem is a 3×3 conv stride=1
    #    - Final FC outputs 10 classes
    model_cifar10 = resnext50_32x4d(pretrained=False, num_classes=10, for_cifar=True)
    print("ResNeXt-50 32x4d (CIFAR-10) →", model_cifar10)

    # Test forward on a dummy CIFAR batch
    dummy_cifar = torch.randn(2, 3, 32, 32)
    out_cifar = model_cifar10(dummy_cifar)
    print("CIFAR-10 output shape:", out_cifar.shape)  # expected: [2, 10]

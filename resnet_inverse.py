from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3x3_transpose(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        output_padding=1 if stride == 2 else 0,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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

        return out


class Bottleneck_Up(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(planes * self.expansion, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_transpose(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)
            # print(identity.shape)
            # print(out.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck_Up]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_Up) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck_Up]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
#%%
def _make_layers(inplanes, outplanes, layernum, stride=2, expansion=4):
    upsampler = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
        conv1x1(inplanes, outplanes, stride=1),
        nn.BatchNorm2d(outplanes),
    )
    layers = []
    for i in range(layernum):
        if i == layernum - 1:
            layer = Bottleneck_Up(outplanes, inplanes // expansion, stride=stride, upsample=upsampler)
        else:
            layer = Bottleneck_Up(inplanes, inplanes // expansion, stride=1)
        layers.append(layer)
    return nn.Sequential(*layers)


class ResNetInverse(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        block: Type[Union[BasicBlock, Bottleneck_Up]] = Bottleneck_Up,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        to_rgb_layer=False,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if to_rgb_layer:
            self.layer0 = _make_layers(64, 64, 1, stride=2)
            self.to_rgb = nn.Sequential(
                norm_layer(64,), # affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)
            )
        else:
            self.layer0 = _make_layers(64, 3, 1, stride=2)
            self.to_rgb = nn.Identity()
        self.layer1 = _make_layers(256, 64, layers[0], stride=2)
        self.layer2 = _make_layers(512, 256, layers[1], stride=2, )
        self.layer3 = _make_layers(1024, 512, layers[2], stride=2, )
        self.layer4 = _make_layers(2048, 1024, layers[3], stride=2, )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_Up) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def forward(self, x):
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        # x = self.bn1(x)
        # x = self.deconv1(x)
        x = self.layer0(x)
        # print(x.shape)
        x = self.to_rgb(x)
        return x


# define a resnet wrapper that stops at avg_pool layer, not fc layer
class ResNetWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x_vec = self.resnet.avgpool(x)
        x_vec = torch.flatten(x_vec, 1)
        return x_vec, x


if __name__ =="__main__":
    #%%
    from torch.nn import functional as F
    # upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    upsampler4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv1x1(2048, 1024, stride=1),
        nn.BatchNorm2d(1024),
    )
    layer4 = Bottleneck_Up(1024, 512, upsample=upsampler4, stride=2)  # layer4
    upsampler3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv1x1(1024, 512, stride=1),
        nn.BatchNorm2d(512),
    )
    layer3 = Bottleneck_Up(512, 256, upsample=upsampler3, stride=2)  # layer3
    upsampler2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv1x1(512, 256, stride=1),
        nn.BatchNorm2d(256),
    )
    layer2 = Bottleneck_Up(256, 128, upsample=upsampler2, stride=2)  # layer2
    upsampler1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv1x1(256, 64, stride=1),
        nn.BatchNorm2d(64),
    )
    layer1 = Bottleneck_Up(64, 64, upsample=upsampler1, stride=2)  # layer1
    upsampler0 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv1x1(64, 3, stride=1),
        nn.BatchNorm2d(3),
    )
    layer0 = Bottleneck_Up(3, 16, upsample=upsampler0, stride=2)  # layer1
    # layer1.eval()
    # layer2.eval()
    # layer3.eval()
    # layer4.eval()
    bn1 = nn.BatchNorm2d(64)
    deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
    #%%
    res_inv = ResNetInverse([3, 4, 6, 3])
    #%%
    x = torch.randn(1, 2048, 7, 7)
    # res_inv.eval()

    res_inv(x)
    #%%
    x = torch.randn(1, 2048, 7, 7)
    x = layer4(x)
    print(x.shape)
    x = layer3(x)
    print(x.shape)
    x = layer2(x)
    print(x.shape)
    x = layer1(x)
    print(x.shape)
    x = layer0(x)
    print(x.shape)
    # x = bn1(x)
    # x = deconv1(x)
    # print(x.shape)

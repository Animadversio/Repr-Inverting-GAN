from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from resnet_inverse import conv3x3, conv1x1
#%%
# conditional batch norm class
# use additional input to modulate the batch norm
class ConditionalBatchNorm2d(nn.Module):
    """ Conditional Batch Normalization
    https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    """
    def __init__(self, num_features, num_modul_dim, gamma_init_scale=0.02):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Linear(num_modul_dim, num_features * 2, bias=True)
        self.embed.bias.data.zero_()
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].zero_()  # Initialise scale at 0
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        # out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        out = (1 + gamma.view(-1, self.num_features, 1, 1)) * out + beta.view(-1, self.num_features, 1, 1)
        return out

#%%
class Bottleneck_UpAtk_Modulated(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        cond_dim: int,
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
        if not stride == 1:
            self.conv_upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        else:
            self.conv_upsample = nn.Identity()
        self.conv2 = conv3x3(width, width, 1, groups, dilation)
        self.bn2 = ConditionalBatchNorm2d(width, cond_dim)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        self.cond_dim = cond_dim

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_upsample(out)
        out = self.conv2(out)
        out = self.bn2(out, y)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Sequential_Modulated(nn.Sequential):

    def forward(self, x, y):
        for module in self:
            if isinstance(module, (Bottleneck_UpAtk_Modulated, ConditionalBatchNorm2d)):
                x = module(x, y)
            else:
                x = module(x)
        return x


def _make_layers_modulated(inplanes, outplanes, cond_dim, layernum, stride=2, expansion=4,
                 blockClass=Bottleneck_UpAtk_Modulated):
    upsampler = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
        conv1x1(inplanes, outplanes, stride=1),
        nn.BatchNorm2d(outplanes),
    )
    layers = []
    for i in range(layernum):
        if i == layernum - 1:
            layer = blockClass(outplanes, inplanes // expansion, cond_dim, stride=stride, upsample=upsampler)
        else:
            layer = blockClass(inplanes, inplanes // expansion, cond_dim, stride=1)
        layers.append(layer)
    return Sequential_Modulated(*layers)


class ModulatedResNetInverse(nn.Module):

    def __init__(
            self,
            layers: List[int],
            cond_dim: int = 2048,
            zero_init_residual: bool = False,
            blockClass: Type[Union[Bottleneck_UpAtk_Modulated]] = Bottleneck_UpAtk_Modulated,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            leaky_relu_rgb=True,
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
        self.layer0 = _make_layers_modulated(64, 64, cond_dim, 1, stride=2, blockClass=blockClass)
        self.to_rgb = Sequential_Modulated(
            ConditionalBatchNorm2d(64, cond_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True) if leaky_relu_rgb else nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )
        self.layer1 = _make_layers_modulated(256, 64, cond_dim, layers[0], stride=2, blockClass=blockClass)
        self.layer2 = _make_layers_modulated(512, 256, cond_dim, layers[1], stride=2, blockClass=blockClass)
        self.layer3 = _make_layers_modulated(1024, 512, cond_dim, layers[2], stride=2, blockClass=blockClass)
        self.layer4 = _make_layers_modulated(2048, 1024, cond_dim, layers[3], stride=2, blockClass=blockClass)
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
                if isinstance(m, Bottleneck_UpAtk_Modulated) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def forward(self, x, cond=None):
        if cond is None:
            cond = x.mean(dim=[2, 3], keepdim=False)
        x = self.layer4(x, cond)
        x = self.layer3(x, cond)
        x = self.layer2(x, cond)
        x = self.layer1(x, cond)
        x = self.layer0(x, cond)
        x = self.to_rgb(x, cond)
        # x = self.bn1(x)
        # x = self.deconv1(x)
        return x

#%%
resnet_inv = ModulatedResNetInverse([3, 4, 6, 3], cond_dim=2048, leaky_relu_rgb=True)
#%%
device = "cpu"
resnet_inv = resnet_inv.to(device)
resnet_inv.eval()
out = resnet_inv(torch.randn(1, 2048, 7, 7))
#%%
out.mean().backward()
#%%

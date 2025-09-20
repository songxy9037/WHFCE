import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .do_conv_pytorch import DOConv2d
from torchvision.ops import DeformConv2d
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    return DOConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return DOConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 stride=1, dilation=1, groups=1, bias=False):
        super(DeformConv, self).__init__()

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=True
        )

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)  # BatchNorm 层
        self.relu = nn.ReLU(inplace=True)       # ReLU 激活

        # 初始化偏移量为零
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # identity = x
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        # out = self.bn(out)  # 加 BatchNorm
        # # out = self.relu(out)  # 加 ReLU
        # out += identity
        # out = self.relu(out)
        return out

class MultiScaleDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 3], groups=1):
        super(MultiScaleDeformConv, self).__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must have the same length"

        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            padding = ((kernel_size - 1) // 2) * dilation
            branch = nn.Sequential(
                DeformConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                    dilation=dilation,
                    groups=groups,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        self.fusion = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            out = branch(x)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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

    def forward(self, x):
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

class DeformBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(DeformBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 替换第一层 3x3 卷积为 DeformConv
        self.conv1 = DeformConv(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # 替换第二层 3x3 卷积为 DeformConv
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DeformBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 保持 1x1 卷积不变
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        # 替换 3x3 卷积为 DeformConv
        self.conv2 = DeformConv(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = norm_layer(planes)

        # 保持 1x1 卷积不变
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 第一层 1x1 卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层 DeformConv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三层 1x1 卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeformResNet34(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super(DeformResNet34, self).__init__()

        # 使用标准的 ResNet34 初始化
        self.model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

        # 替换 layer2, layer3, layer4 的模块为 DeformBasicBlock
        self.model.layer2 = self._make_deform_layer(self.model.layer2)
        self.model.layer3 = self._make_deform_layer(self.model.layer3)
        self.model.layer4 = self._make_deform_layer(self.model.layer4)

        if pretrained:
            # 加载预训练权重
            state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet34-b627a593.pth', progress=progress)
            model_dict = self.model.state_dict()
            # 过滤掉不匹配的参数
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.size() == model_dict[k].size()}
            # 更新模型的 state_dict
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def _make_deform_layer(self, layer):
        deform_layers = []
        for i in range(len(layer)):
            block = layer[i]
            deform_block = DeformBasicBlock(
                block.conv1.in_channels,
                block.conv2.out_channels,
                stride=block.stride,
                downsample=block.downsample,
                norm_layer=self.model._norm_layer
            )
            deform_layers.append(deform_block)
        return nn.Sequential(*deform_layers)

    def forward(self, x):
        return self.model(x)

class DeformResNet50(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super(DeformResNet50, self).__init__()

        # 使用标准的 ResNet-50 初始化
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

        # 替换 layer2, layer3, layer4 的模块为 DeformBottleneck
        self.model.layer2 = self._make_deform_layer(self.model.layer2)
        self.model.layer3 = self._make_deform_layer(self.model.layer3)
        self.model.layer4 = self._make_deform_layer(self.model.layer4)

        if pretrained:
            # 加载预训练权重
            state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=progress)
            model_dict = self.model.state_dict()
            # 过滤掉不匹配的参数
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # 更新模型的 state_dict
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def _make_deform_layer(self, layer):
        deform_layers = []
        for i in range(len(layer)):
            block = layer[i]
            deform_block = DeformBottleneck(
                block.conv1.in_channels,
                block.conv3.out_channels // block.expansion,
                stride=block.stride,
                downsample=block.downsample,
                groups=block.groups,
                base_width=block.base_width,
                dilation=block.dilation,
                norm_layer=self.model._norm_layer
            )
            deform_layers.append(deform_block)
        return nn.Sequential(*deform_layers)

    def forward(self, x):
        return self.model(x)

class DeformResNet(nn.Module):
    def __init__(self, block, layers, pretrained=False, progress=True, **kwargs):
        super(DeformResNet, self).__init__()

        # 使用标准的 ResNet 初始化
        self.model = ResNet(block, layers, **kwargs)

        # 替换 layer2, layer3, layer4 的模块为 DeformBlock
        self.model.layer2 = self._make_deform_layer(self.model.layer2, block)
        self.model.layer3 = self._make_deform_layer(self.model.layer3, block)
        self.model.layer4 = self._make_deform_layer(self.model.layer4, block)

        if pretrained:
            # 根据层数选择预训练权重的 URL
            if layers == [2, 2, 2, 2]:
                url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
            elif layers == [3, 4, 6, 3] and block == BasicBlock:
                url = 'https://download.pytorch.org/models/resnet34-b627a593.pth'
            elif layers == [3, 4, 6, 3] and block == Bottleneck:
                url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            else:
                url = None

            if url:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=progress)
                model_dict = self.model.state_dict()
                # 过滤掉不匹配的参数
                pretrained_dict = {k: v for k, v in state_dict.items()
                                   if k in model_dict and v.size() == model_dict[k].size()}
                # 更新模型的 state_dict
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

    def _make_deform_layer(self, layer, block):
        deform_layers = []
        for i in range(len(layer)):
            original_block = layer[i]
            if isinstance(block(), BasicBlock):
                deform_block = DeformBasicBlock(
                    original_block.conv1.in_channels,
                    original_block.conv2.out_channels,
                    stride=original_block.stride,
                    downsample=original_block.downsample,
                    norm_layer=self.model._norm_layer
                )
            elif isinstance(block(), Bottleneck):
                deform_block = DeformBottleneck(
                    original_block.conv1.in_channels,
                    original_block.conv3.out_channels // original_block.expansion,
                    stride=original_block.stride,
                    downsample=original_block.downsample,
                    groups=original_block.groups,
                    base_width=original_block.base_width,
                    dilation=original_block.dilation,
                    norm_layer=self.model._norm_layer
                )
            deform_layers.append(deform_block)
        return nn.Sequential(*deform_layers)

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        # self.conv1_6 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=1, padding=3,
        #                        bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # # 多尺度可变形卷积模块

        # self.deformconv2 = deformable_LKA_Attention(d_model=64)
        # self.deformconv3 = deformable_LKA_Attention(d_model=128)
        # self.deformconv4 = deformable_LKA_Attention(d_model=256)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        c1 = self.layer1(x)

        c2 = self.layer2(c1)
        # c2 = self.deformconv2(c2)

        c3 = self.layer3(c2)
        # c3 = self.deformconv3(c3)

        c4 = self.layer4(c3)
        # c4 = self.deformconv4(c4)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return c1, c2, c3, c4

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def deform_resnet18(pretrained=False, progress=True, **kwargs):
    return DeformResNet(DeformBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def deform_resnet34(pretrained=False, progress=True, **kwargs):
    return DeformResNet(DeformBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def deform_resnet50(pretrained=False, progress=True, **kwargs):
    return DeformResNet(DeformBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
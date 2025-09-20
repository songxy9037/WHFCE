from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models import do_conv_pytorch as doconv
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse  # 确保已安装 pytorch_wavelets 库

def convbn(in_planes, out_planes, kernel_size, stride, padding, dilation):
    """
    DO-Conv 无痛涨点：使用 over-parameterized 卷积层提高 CNN 性能。
    对于输入特征，先使用权重进行 depthwise 卷积，对输出结果进行传统卷积。
    """
    return nn.Sequential(
        doconv.DOConv2d(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if dilation == 1 else dilation,
            dilation=dilation,
            bias=False
        ),
        nn.BatchNorm2d(out_planes)
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, padding=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            convbn(in_planes, planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True)
        )
        self.conv2 = convbn(planes, planes, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用卷积核大小为 7 来捕捉更广泛的空间上下文
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv2(combined))
        return attention

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class HL_LH_CatAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(HL_LH_CatAttention, self).__init__()
        mip = max(8, inp // reduction)

        # 1x1卷积处理拼接后的特征
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 分别为 HL 和 LH 特征的注意力设置卷积层
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, hl, lh):
        identity_hl = hl
        identity_lh = lh

        n, c, h, w = hl.size()

        # 分别对 水平 和 垂直 进行池化
        pool_hl = nn.AdaptiveAvgPool2d((None, 1))(hl)  # HL：压缩宽度维度
        pool_lh = nn.AdaptiveAvgPool2d((1, None))(lh)  # LH：压缩高度维度
        # pool_lh = pool_lh.permute(0, 1, 3, 2)

        # reverse HL # 分别对 HL 和 LH 进行池化
        # pool_lh = nn.AdaptiveAvgPool2d((None, 1))(lh)  # HL：压缩宽度维度
        # pool_hl = nn.AdaptiveAvgPool2d((1, None))(hl)  # LH：压缩高度维度

        # 对 HL 和 LH 分别进行相同的卷积和注意力操作（共享权重）
        y_hl = self.conv1(pool_hl)
        y_hl = self.bn1(y_hl)
        y_hl = self.act(y_hl)

        y_lh = self.conv1(pool_lh)
        y_lh = self.bn1(y_lh)
        y_lh = self.act(y_lh)

        # 对 HL 和 LH 分别计算注意力
        a_hl = self.conv_h(y_hl).sigmoid()
        a_lh = self.conv_w(y_lh).sigmoid()  # 注意 LH 的维度需要调整回来

        # 应用注意力到 HL 和 LH 上
        out_hl = identity_hl * a_lh + identity_hl
        out_lh = identity_lh * a_hl + identity_lh

        return out_hl, out_lh
class SpatialAttentionWithResidual(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionWithResidual, self).__init__()
        # 使用卷积核大小为7来捕捉更广泛的空间上下文
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入进行平均池化和最大池化操作，保持通道维度
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 将平均池化和最大池化的结果在通道维度上拼接
        combined = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积和激活函数得到空间注意力图
        attention = self.sigmoid(self.conv(combined))

        # 将输入特征与空间注意力图相乘，增强显著区域
        out = x * attention

        # 使用残差连接，叠加原始特征
        out = out + x
        return out

class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, wave='haar', downsample=True, use_ca_attention=False, use_sa_attention=False):
        super(WaveBlock, self).__init__()
        self.downsample = downsample
        self.use_ca_attention = use_ca_attention
        self.use_sa_attention = use_sa_attention
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)  # 小波分解
        self.idwt = DWTInverse(mode='zero', wave=wave)       # 逆小波变换

        # 定义四个分支的卷积层
        self.layerLL = self._make_layer(BasicBlock, in_channels, num_blocks)
        self.layerH = self._make_layer(BasicBlock, in_channels, num_blocks)
        # self.layerLH = self._make_layer(BasicBlock, in_channels, num_blocks)
        # self.layerHL = self._make_layer(BasicBlock, in_channels, num_blocks)
        # self.layerHH = self._make_layer(BasicBlock, in_channels, num_blocks)

        if self.use_ca_attention:
            print('use_ca')
            self.hl_lh_attention = HL_LH_CatAttention(in_channels, in_channels)  # HL 和 LH 注意力模块
        if self.use_sa_attention:
            print('use_sa')
            self.sa_attention = SpatialAttentionWithResidual()
        self.conv_hl = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 处理LL
        self.conv_lh = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 处理LL
        # 空间注意力机制
        self.sa = SpatialAttention(in_channels * 4, in_channels)

        if self.downsample:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.convfin = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def _make_layer(self, block, planes, num_blocks):
        layers = [block(planes, planes) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        LL, Yh = self.dwt(x)
        # 提取小波系数
        LH, HL, HH = Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yh[0][:, :, 2, :, :]

        # 分别通过各自的卷积层
        LL = self.layerLL(LL)
        LH = self.layerH(LH)
        HL = self.layerH(HL)
        HH = self.layerH(HH)

        if self.use_ca_attention:
            # 对 HL 和 LH 进行单独的通道注意力处理
            hl_out, lh_out = self.hl_lh_attention(HL, LH)
        else:
            # 若不使用注意力机制，直接返回原始HL和LH
            hl_out, lh_out = HL, LH

        if self.use_sa_attention:
            # 将 hl_out 和 lh_out 相加，生成组合特征
            hl_lh_combined = hl_out + lh_out
            # 对 hl_lh_combined 应用空间注意力
            hl_lh_attention = self.sa_attention(hl_lh_combined)
            # 将生成的注意力加回到 hl_out 和 lh_out 中，作为残差
            hl_out = hl_out + hl_lh_attention
            lh_out = lh_out + hl_lh_attention
            hl_out = self.conv_hl(hl_out)
            lh_out = self.conv_lh(lh_out)

        # 拼接并通过注意力机制
        big_f = torch.cat([LL, lh_out, hl_out, HH], dim=1)

        big_f_a = self.sa(big_f)

        # 注意力增强
        LH = LH + LH * big_f_a
        HL = HL + HL * big_f_a
        HH = HH + HH * big_f_a

        # 重构小波系数
        H = torch.stack([LH, HL, HH], dim=2)
        out = self.idwt((LL, [H]))

        # 下采样或卷积
        if self.downsample:
            out = self.down(out)
        else:
            out = self.convfin(out)

        return out

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.firstconv = nn.Sequential(
            convbn(3, 64, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=3)
        self.layer2 = WaveBlock(64, 128, num_blocks=4)
        self.layer3 = WaveBlock(128, 256, num_blocks=6)
        self.layer4 = WaveBlock(256, 512, num_blocks=3)

    def _make_layer(self, block, planes, num_blocks):
        layers = [block(planes, planes) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x1 = self.layer1(x)  # 输出尺寸与输入相同
        x2 = self.layer2(x1) # 输出尺寸减半
        x3 = self.layer3(x2) # 输出尺寸再次减半
        x4 = self.layer4(x3) # 输出尺寸再次减半
        return x1, x2, x3, x4


class WaveletAttentionModel(nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar', use_ca_attention=True, use_sa_attention=True, export_ll=False):
        super(WaveletAttentionModel, self).__init__()
        self.use_ca_attention = use_ca_attention  # 控制是否使用通道注意力机制
        self.use_sa_attention = use_sa_attention  # 控制是否使用空间注意力机制
        self.export_ll = export_ll  # 控制是否输出 ll_out

        # 小波分解和逆变换
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)
        self.idwt = DWTInverse(mode='zero', wave=wave)

        # 定义高频和低频卷积以及注意力模块
        if self.use_ca_attention:
            self.hl_lh_attention = HL_LH_CatAttention(in_channels, in_channels)  # HL 和 LH 通道注意力模块
        self.conv_hh = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 处理 HH 子带
        self.conv_ll = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 LL 子带
        self.conv_hl = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 HL 子带
        self.conv_lh = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 LH 子带
        self.conv_high_freq = nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1)  # 处理拼接后的高频特征

        # 定义空间注意力
        self.sa = SpatialAttentionWithResidual()

    def forward(self, x):
        # 小波分解，获取低频 LL 和高频子带（LH、HL、HH）
        LL, Yh = self.dwt(x)

        # 提取 LH, HL, HH 子带
        LH = Yh[0][:, :, 0, :, :]
        HL = Yh[0][:, :, 1, :, :]
        HH = Yh[0][:, :, 2, :, :]

        # 通道注意力处理
        if self.use_ca_attention:
            hl_out, lh_out = self.hl_lh_attention(HL, LH)
        else:
            hl_out, lh_out = HL, LH

        # 空间注意力处理
        if self.use_sa_attention:
            hl_lh_combined = hl_out + lh_out
            hl_lh_attention = self.sa(hl_lh_combined)
            hl_out = hl_out + hl_lh_attention
            lh_out = lh_out + hl_lh_attention

            # 对 HL 和 LH 子带进行卷积
            hl_out = self.conv_hl(hl_out)
            lh_out = self.conv_lh(lh_out)

        # 对 HH 子带进行卷积
        hh_out = self.conv_hh(HH)

        ll_out = self.conv_ll(LL)

        # 重新组合高频子带
        combined_high = torch.stack([hl_out, lh_out, hh_out], dim=2)

        # 逆小波变换
        output = self.idwt((ll_out, [combined_high]))

        # 根据 export_ll 决定是否输出 ll_out
        if self.export_ll:
            return output, ll_out
        else:
            return output

class LHDACT_ex(nn.Module):

    def __init__(self):
        super(LHDACT_ex, self).__init__()
        self.inplanes = 64
        inplanes1 = 64
        k_size = 3
        self.firstconv = nn.Sequential(convbn(3, 64, 7, 1, 3, 1),
                                       nn.ReLU(inplace=True),

    )

        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 4, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 6, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, 2, 1, 1)

        self.waveatten2 = WaveletAttentionModel(64, 64)
        self.waveatten3 = WaveletAttentionModel(128, 128)
        self.waveatten4 = WaveletAttentionModel(256, 256)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        raw_fea = self.firstconv(x)


        fea1 = self.layer1(raw_fea)

        fea2 = self.layer2(fea1)
        fea2_wave = self.waveatten2(fea2)

        fea3 = self.layer3(fea2_wave)
        fea3_wave = self.waveatten3(fea3)

        fea4 = self.layer4(fea3_wave)
        fea4_wave = self.waveatten4(fea4)


        return fea1, fea2_wave, fea3_wave, fea4_wave

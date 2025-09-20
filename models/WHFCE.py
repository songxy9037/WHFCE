import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models import do_conv_pytorch as doconv
from .resnet import resnet34
import torch.nn.functional as F

##################bone###########
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    '''
    DO-Conv无痛涨点：使用over-parameterized卷积层提高CNN性能
    对于输入特征，先使用权重进行depthwise卷积，对输出结果进行权重为的传统卷积，
    '''
    return nn.Sequential(doconv.DOConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

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


class WaveX(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(WaveX, self).__init__()
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

        # 分别对 HL 和 LH 进行池化
        pool_hl = nn.AdaptiveAvgPool2d((None, 1))(hl)  # HL：压缩宽度维度
        pool_lh = nn.AdaptiveAvgPool2d((1, None))(lh)  # LH：压缩高度维度
        # pool_lh = pool_lh.permute(0, 1, 3, 2)

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

class Direction_Guide(nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar', wavex=True, export_ll=True, export_hf=False, only_ll=False):
        super(Direction_Guide, self).__init__()
        self.wavex = wavex  # 控制是否使用通道注意力机制
        self.export_ll = export_ll  # 控制是否输出 ll_out
        self.export_hf = export_hf
        self.only_ll = only_ll
        # 小波分解和逆变换
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)
        self.idwt = DWTInverse(mode='zero', wave=wave)
        if self.export_hf:
            self.conv_hf = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理高频
        # 定义高频和低频卷积以及注意力模块
        if self.wavex:
            self.wavex = WaveX(in_channels, in_channels)  # HL 和 LH 通道注意力模块

        self.conv_hh = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 处理 HH 子带
        self.conv_ll = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 LL 子带
        self.conv_hl = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 HL 子带
        self.conv_lh = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 处理 LH 子带
        self.conv_high_freq = nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1)  # 处理拼接后的高频特征



    def forward(self, x):
        # 小波分解，获取低频 LL 和高频子带（LH、HL、HH）
        LL, Yh = self.dwt(x)
        ll_out = self.conv_ll(LL)
        if self.export_hf:
            LH = Yh[0][:, :, 0, :, :]
            HL = Yh[0][:, :, 1, :, :]
            HH = Yh[0][:, :, 2, :, :]
            if self.export_hf:
                HF = LH + HL + HH
                hf_out = self.conv_hf(HF)
        if self.only_ll is False:
            if self.wavex:
                hl_out, lh_out = self.wavex(HL, LH)
            else:
                hl_out, lh_out = HL, LH


            hh_out = self.conv_hh(HH)
            combined_high = torch.stack([hl_out, lh_out, hh_out], dim=2)
            output = self.idwt((ll_out, [combined_high]))

        if self.export_ll and not self.export_hf:
            return output, ll_out
        elif self.export_hf and not self.export_ll:
            return output, hf_out
        elif self.only_ll and self.export_hf:
            return ll_out, hf_out
        elif self.export_hf and self.export_ll:
            return output, ll_out, hf_out
        else:
            return output

class HighFeatureFusion(nn.Module):
    def __init__(self):
        super(HighFeatureFusion, self).__init__()

        # 特征下采样模块
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

    def forward(self,H_A_2, H_B_2, H_A_3, H_B_3, H_A_4, H_B_4, H_A_5, H_B_5):
        # 特征提取和逐层融合
        H_A_3 = self.down1(H_A_2) + H_A_3  # 下采样并与上一层融合

        H_B_3 = self.down1(H_B_2) + H_B_3  # 下采样并与上一层融合

        H_A_4 = self.down2(H_A_3) + H_A_4  # 下采样并与上一层融合

        H_B_4 = self.down2(H_B_3) + H_B_4  # 下采样并与上一层融合


        H_A_5 = self.down3(H_A_4) + H_A_5  # 下采样并与上一层融合

        H_B_5 = self.down3(H_B_4) + H_B_5  # 下采样并与上一层融合

        encoder_high_features = {
            'layer1': (H_A_2, H_B_2),
            'layer2': (H_A_3, H_B_3),
            'layer3': (H_A_4, H_B_4),
            'layer4': (H_A_5, H_B_5)
        }
        return encoder_high_features


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = convbn(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              pad=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Diff3DConvModule(nn.Module):
    def __init__(self, in_channels_X, in_channels, out_channels, kernel_size=(3, 3, 3),groups=1):
        super(Diff3DConvModule, self).__init__()

        # 2D卷积层，用于处理每组拼接后的特征
        # 每组特征的输入通道数可能不同，因此需要独立的卷积层

        self.conv2d_group1 = nn.Sequential(
            nn.Conv2d(in_channels_X, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2d_group2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2d_group3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3d = nn.Conv3d(
            out_channels,  # 三组，每组经过2D卷积后通道数为in_channels
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            groups=groups,
            bias=False
        )
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 3D卷积用于将深度维度从3降至1
        self.conv_reduce = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3,1,1),
            stride=(3,1,1),  # 步幅为3，在深度维度上滑动3步
            padding=(0,0,0),  # 无填充，以确保输出深度维度为1
            groups=groups,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm3d(out_channels)
        self.relu_reduce = nn.ReLU(inplace=True)

        # 残差连接，如果输入通道数和输出通道数不同，使用1x1x1卷积调整
        if in_channels * 3 != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, X_A, X_B, ll_A, ll_B, layer0, layer1):
        """
        前向传播函数
        输入:
            X_A, X_B: 两组输入特征，形状为 [batch, C_X, H, W]
            ll_A, ll_B: 两组低层输入特征，形状为 [batch, C_ll, H, W]
            layer0, layer1: 两组高层输入特征，形状为 [batch, C_layer, H, W]
        输出:
            2D特征图，形状为 [batch, out_channels, H, W]
        """

        # Step 1: 拼接每组特征并通过独立的2D卷积层处理
        # diff_group1: [batch, out_channels, H, W]
        diff_group1 = torch.cat((X_A, X_B), dim=1)  # 拼接 X_A 和 X_B
        diff_group1 = self.conv2d_group1(diff_group1)


        # diff_group2: [batch, out_channels, H, W]
        diff_group2 = torch.cat((ll_A, ll_B), dim=1)  # 拼接 ll_A 和 ll_B
        diff_group2 = self.conv2d_group2(diff_group2)


        # diff_group3: [batch, out_channels, H, W]
        diff_group3 = torch.cat((layer0, layer1), dim=1)  # 拼接 layer0 和 layer1
        diff_group3 = self.conv2d_group3(diff_group3)


        # Step 2: 堆叠拼接后的特征到新的维度（深度维度）
        # stacked_diff: [batch, out_channels, 3, H, W]
        stacked_diff = torch.stack([diff_group1, diff_group2, diff_group3], dim=2)

        # Step 3: 通过3D卷积层
        # output: [batch, out_channels, D', H', W']
        output = self.conv3d(stacked_diff)
        output = self.bn3d(output)
        output = self.relu(output)
        # 5. 合并深度维度 D=3 到 D=1
        output = self.conv_reduce(output)  # [B, out_channels, 1, H, W]
        output = self.bn_reduce(output)
        output = self.relu_reduce(output).squeeze(2)

        # 6. 残差连接
        # residual = self.residual(stacked_diff)  # [B, out_channels, 3, H, W] 或已调整通道数
        # residual = self.conv_reduce(residual) if not isinstance(self.residual, nn.Identity) else residual.squeeze(2)
        # residual = residual.mean(dim=2, keepdim=True)  # [B, out_channels, 1, H, W] 或已合并
        # residual = residual.squeeze(2)  # [B, out_channels, H, W]
        #
        # # 7. 添加残差并应用激活函数
        # output += residual
        #
        # out = self.relu_reduce(output)

        return output  # [B, out_channels, H, W]

class DecoderConv(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, dropout_prob=0.5):
        """
        通用解码器模块，支持可调节的通道数参数。

        参数：
        - in_channels_list: 每层的输入通道数列表 [in_channels3, in_channels2, in_channels1]
        - out_channels_list: 每层的输出通道数列表 [out_channels3, out_channels2, out_channels1]
        - dropout_prob: Dropout 概率
        """
        super(DecoderConv, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 第三层解码器
        self.decoder3 = nn.Sequential(
            BasicConv2d(in_channels_list[0], in_channels_list[0], kernel_size=3, padding=1,
                        dilation=1),
            nn.Dropout(dropout_prob),
            TransBasicConv2d(in_channels_list[0], out_channels_list[0], kernel_size=2, stride=2, padding=0, dilation=1,
                             bias=False)
        )
        # self.AM2 = AlignModule(out_channels_list[0], in_channels_list[1])
        # 第二层解码器
        self.decoder2 = nn.Sequential(
            BasicConv2d(in_channels_list[1], in_channels_list[1], kernel_size=3, padding=1,
                        dilation=1),
            nn.Dropout(dropout_prob),
            TransBasicConv2d(in_channels_list[1], out_channels_list[1], kernel_size=2, stride=2, padding=0, dilation=1,
                             bias=False)
        )
        # self.AM1 = AlignModule(out_channels_list[1], in_channels_list[2])
        # 第一层解码器
        self.decoder1 = nn.Sequential(
            BasicConv2d(in_channels_list[2], out_channels_list[2], kernel_size=3, padding=1,
                        dilation=1),
        )

    def forward(self, x3, x2, x1):
        x3_up = self.decoder3(x3)
        x2_up = self.decoder2(torch.cat((x2, x3_up), dim=1))
        x1_up = self.decoder1(torch.cat((x1, x2_up), dim=1))
        return x1_up




class BaseNet(nn.Module):
    """
    Base
    """
    def __init__(self,):
        super(BaseNet, self).__init__()

        self.ResNet = resnet34(pretrained=False)
        self.decoder = DecoderConv([256,256,128],[128,64,128],0.5)

        self.conv4 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                      nn.ReLU(True))
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=128, out_channels=2)
        # 设置热力图保存目录
        self.heatmap_dir = 'heatmaps_base_img5'  # 可以根据需要修改为其他路径
        # if not os.path.exists(self.heatmap_dir):
        #     os.makedirs(self.heatmap_dir)

        # 初始化热力图批次计数器
        self.batch_counter = 0
        print(f"Heatmap batch counter initialized to {self.batch_counter}")

    def forward(self, x1,x2):

        heatfeature = []
        # 提取特征
        X_A_1, X_A_2, X_A_3, X_A_4 = self.ResNet(x1)
        X_B_1, X_B_2, X_B_3, X_B_4 = self.ResNet(x2)
        # # 保存编码器特征用于计算对比损失
        # encoder_features = {
        #     'layer2': (X_A_2, X_B_2),
        #     'layer3': (X_A_3, X_B_3),
        #     'layer4': (X_A_4, X_B_4)
        # }
        heatfeature.append(X_A_2)
        heatfeature.append(X_B_2)
        heatfeature.append(X_A_3)
        heatfeature.append(X_B_3)
        heatfeature.append(X_A_4)
        heatfeature.append(X_B_4)

        # 各层特征拼接和卷积
        diff_4 = torch.cat((X_A_4, X_B_4), dim=1)
        diff_3 = torch.cat((X_A_3, X_B_3), dim=1)
        diff_2 = torch.cat((X_A_2, X_B_2), dim=1)
        diff_4 = self.conv4(diff_4)
        diff_3 = self.conv3(diff_3)
        diff_2 = self.conv2(diff_2)

        # 解码器
        X2 = self.decoder(diff_4, diff_3, diff_2)

        # 上采样和分类输出
        X = self.upsamplex2(X2)
        output = self.classifier(X)
        # 生成并保存热力图
        self.generate_heatmaps(heatfeature)

        # 递增批次计数器
        self.batch_counter += 1
        print(f"Heatmap batch counter incremented to {self.batch_counter}")

        return output

    def generate_heatmaps(self, heatfeature):
        """
        生成并保存 heatfeature 列表中每个特征图的热力图。

        Args:
            heatfeature (list of torch.Tensor): 特征图列表。
        """
        for idx, feature in enumerate(heatfeature):
            # 假设 feature 的形状为 [batch_size, channels, height, width]
            batch_size, channels, height, width = feature.size()
            # 在通道维度上进行平均，得到二维特征图
            feature_mean = feature.mean(dim=1)  # 形状: [batch_size, height, width]
            for i in range(batch_size):
                try:
                    heatmap = feature_mean[i].detach().cpu().numpy()
                    # 归一化到 [0, 1]
                    heatmap_min = heatmap.min()
                    heatmap_max = heatmap.max()
                    if heatmap_max - heatmap_min > 1e-8:
                        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
                    else:
                        heatmap = heatmap - heatmap_min  # 避免除以零

                    plt.figure(figsize=(4, 4))
                    plt.axis('off')
                    plt.imshow(heatmap, cmap='jet')
                    plt.tight_layout(pad=0)

                    # 定义文件名，包含层级、样本编号和批次计数器
                    filename = f'heatmap_layer{idx + 1}_sample{i + 1}_batch{self.batch_counter}.png'
                    filepath = os.path.join(self.heatmap_dir, filename)
                    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    print(f"保存热力图: {filename}")
                except Exception as e:
                    print(f"生成热力图时出错，层级: {idx + 1}, 样本: {i + 1}, 错误: {e}")
                    continue



class WHFCE(nn.Module):

    def __init__(self,):
        super(WHFCE, self).__init__()

        self.ResNet = resnet34(pretrained=False)

        self.decoder = DecoderConv([256,256,128],[128,64,128],0.5)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=128, out_channels=2)
        self.conv4 = nn.Sequential(convbn(512+512, 256, 3, 1, 1, 1),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(convbn(256+256, 128, 3, 1, 1, 1),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(convbn(128+256, 64, 3, 1, 1, 1),
                                   nn.ReLU(True))
        self.WaveAtten1 = Direction_Guide(64,64,'haar', only_ll=True, export_hf=True)
        self.WaveAtten2 = Direction_Guide(64,64,'haar', wavex=True, export_ll=True, export_hf=True)
        self.WaveAtten3 = Direction_Guide(128, 128, 'haar', wavex=True,  export_ll=True, export_hf=True)
        self.WaveAtten4 = Direction_Guide(256, 256, 'haar', wavex=True, export_ll=False, export_hf=True)
        self.Highfuse = HighFeatureFusion()
        self.diff3d_4 = Diff3DConvModule(512,256, 256)
        self.diff3d_3 = Diff3DConvModule(256, 128, 128)
        self.diff3d_2 = Diff3DConvModule(128, 128, 64)
    def forward(self, x1,x2):
        '''1=256
        2=128
        3=64
        4=32
        5=16
        '''

        X_A_1, X_A_2, X_A_3, X_A_4 = self.ResNet(x1)    # torch.Size([ 64, 128, 128])torch.Size([ 128, 64, 64])torch.Size([ 256, 32, 32])
        X_B_1, X_B_2, X_B_3, X_B_4 = self.ResNet(x2)

        ll_A_2, hf_A_2 = self.WaveAtten1(X_A_1)
        ll_B_2, hf_B_2 = self.WaveAtten1(X_B_1)
        X_A_2, ll_A_3, hf_A_3 = self.WaveAtten2(X_A_2) # torch.Size([ 64, 128, 128])
        X_B_2, ll_B_3, hf_B_3 = self.WaveAtten2(X_B_2)
        X_A_3, ll_A_4, hf_A_4 = self.WaveAtten3(X_A_3) # torch.Size([ 128, 64, 64])
        X_B_3, ll_B_4, hf_B_4 = self.WaveAtten3(X_B_3)
        X_A_4, hf_A_5 = self.WaveAtten4(X_A_4)# torch.Size([ 256, 32, 32])
        X_B_4, hf_B_5 = self.WaveAtten4(X_B_4)
        encoder_high_features = self.Highfuse(hf_A_2, hf_B_2, hf_A_3, hf_B_3, hf_A_4, hf_B_4, hf_A_5, hf_B_5)
        # 各层分别对主特征和低频特征进行 cat 和卷积
        # diff_4 = torch.cat((X_A_4, X_B_4, ll_A_4, ll_B_4, encoder_high_features['layer3'][0], encoder_high_features['layer3'][1]), dim=1)
        # diff_3 = torch.cat((X_A_3, X_B_3, ll_A_3, ll_B_3, encoder_high_features['layer2'][0], encoder_high_features['layer2'][1]), dim=1)
        # diff_2 = torch.cat((X_A_2, X_B_2, ll_A_2, ll_B_2, encoder_high_features['layer1'][0], encoder_high_features['layer1'][1]), dim=1)
        # diff_4 = self.conv4(diff_4)
        # diff_3 = self.conv3(diff_3)
        # diff_2 = self.conv2(diff_2)
        diff_4 = self.diff3d_4(X_A_4, X_B_4, ll_A_4, ll_B_4, encoder_high_features['layer3'][0], encoder_high_features['layer3'][1])
        diff_3 = self.diff3d_3(X_A_3, X_B_3, ll_A_3, ll_B_3, encoder_high_features['layer2'][0], encoder_high_features['layer2'][1])
        diff_2 = self.diff3d_2(X_A_2, X_B_2, ll_A_2, ll_B_2, encoder_high_features['layer1'][0], encoder_high_features['layer1'][1])
        # 将各层的主特征和低频特征传入解码器
        X2 = self.decoder(
            diff_4,
            diff_3,
            diff_2)
        # 上采样和分类输出
        X = self.upsamplex2(X2)
        output = self.classifier(X)
        return output, encoder_high_features
# import os
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
# class WaveXNet(nn.Module):
#     """
#     CrossAtten2
#     """
#     def __init__(self, use_ca_attention=True, use_sa_attention=False):
#         super(WaveXNet, self).__init__()
#
#         self.ResNet = resnet34(pretrained=False)
#
#         self.decoder = DecoderConv([256,256,128],[128,64,128],0.5)
#
#         self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.classifier = TwoLayerConv2d(in_channels=128, out_channels=2)
#         self.conv4 = nn.Sequential(convbn(512+512, 256, 3, 1, 1, 1),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(convbn(256+256, 128, 3, 1, 1, 1),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(convbn(128+256, 64, 3, 1, 1, 1),
#                                    nn.ReLU(True))
#         self.WaveAtten1 = WaveletAttentionModel(64,64,'haar', only_ll=True, export_hf=True)
#         self.WaveAtten2 = WaveletAttentionModel(64,64,'haar', use_ca_attention=use_ca_attention, use_sa_attention=use_sa_attention,
#                                                 export_ll=True, export_hf=True)
#         self.WaveAtten3 = WaveletAttentionModel(128, 128, 'haar', use_ca_attention=use_ca_attention, use_sa_attention=use_sa_attention,
#                                                 export_ll=True, export_hf=True)
#         self.WaveAtten4 = WaveletAttentionModel(256, 256, 'haar', use_ca_attention=use_ca_attention, use_sa_attention=use_sa_attention,
#                                                 export_ll=False, export_hf=True)
#         self.Highfuse = HighFeatureFusion()
#         self.diff3d_4 = Diff3DConvModule(512,256, 256)
#         self.diff3d_3 = Diff3DConvModule(256, 128, 128)
#         self.diff3d_2 = Diff3DConvModule(128, 128, 64)
#
#         # 设置热力图保存目录
#         self.heatmap_dir = 'heatmaps'  # 可以根据需要修改为其他路径
#         if not os.path.exists(self.heatmap_dir):
#             os.makedirs(self.heatmap_dir)
#         # 初始化热力图批次计数器
#         self.batch_counter = 0
#         print(f"Heatmap batch counter initialized to {self.batch_counter}")
#
#     def forward(self, x1, x2):
#         '''1=256
#         2=128
#         3=64
#         4=32
#         5=16
#         '''
#         heatfeature = []
#         X_A_1, X_A_2, X_A_3, X_A_4 = self.ResNet(x1)    # e.g., torch.Size([64, 128, 128])
#         X_B_1, X_B_2, X_B_3, X_B_4 = self.ResNet(x2)
#         heatfeature.append(X_A_2)
#         heatfeature.append(X_B_2)
#         heatfeature.append(X_A_3)
#         heatfeature.append(X_B_3)
#         heatfeature.append(X_A_4)
#         heatfeature.append(X_B_4)
#
#         ll_A_2, hf_A_2 = self.WaveAtten1(X_A_1)
#         ll_B_2, hf_B_2 = self.WaveAtten1(X_B_1)
#         X_A_2, ll_A_3, hf_A_3 = self.WaveAtten2(X_A_2) # e.g., torch.Size([64, 128, 128])
#         X_B_2, ll_B_3, hf_B_3 = self.WaveAtten2(X_B_2)
#         X_A_3, ll_A_4, hf_A_4 = self.WaveAtten3(X_A_3) # e.g., torch.Size([128, 64, 64])
#         X_B_3, ll_B_4, hf_B_4 = self.WaveAtten3(X_B_3)
#         X_A_4, hf_A_5 = self.WaveAtten4(X_A_4)       # e.g., torch.Size([256, 32, 32])
#         X_B_4, hf_B_5 = self.WaveAtten4(X_B_4)
#         encoder_high_features = self.Highfuse(hf_A_2, hf_B_2, hf_A_3, hf_B_3, hf_A_4, hf_B_4, hf_A_5, hf_B_5)
#         heatfeature.append(X_A_2)
#         heatfeature.append(X_B_2)
#         heatfeature.append(X_A_3)
#         heatfeature.append(X_B_3)
#         heatfeature.append(X_A_4)
#         heatfeature.append(X_B_4)
#
#         # 处理差异特征
#         diff_4 = self.diff3d_4(X_A_4, X_B_4, ll_A_4, ll_B_4, encoder_high_features['layer3'][0], encoder_high_features['layer3'][1])
#         diff_3 = self.diff3d_3(X_A_3, X_B_3, ll_A_3, ll_B_3, encoder_high_features['layer2'][0], encoder_high_features['layer2'][1])
#         diff_2 = self.diff3d_2(X_A_2, X_B_2, ll_A_2, ll_B_2, encoder_high_features['layer1'][0], encoder_high_features['layer1'][1])
#
#         # 解码
#         X2 = self.decoder(diff_4, diff_3, diff_2)
#         X = self.upsamplex2(X2)
#         output = self.classifier(X)
#
#         # 生成并保存热力图
#         self.generate_heatmaps(heatfeature)
#         # 递增批次计数器
#         self.batch_counter += 1
#         print(f"Heatmap batch counter incremented to {self.batch_counter}")
#         return output, encoder_high_features
#
#     def generate_heatmaps(self, heatfeature):
#         """
#         生成并保存 heatfeature 列表中每个特征图的热力图。
#
#         Args:
#             heatfeature (list of torch.Tensor): 特征图列表。
#         """
#         for idx, feature in enumerate(heatfeature):
#             # 假设 feature 的形状为 [batch_size, channels, height, width]
#             batch_size, channels, height, width = feature.size()
#             # 在通道维度上进行平均，得到二维特征图
#             feature_mean = feature.mean(dim=1)  # 形状: [batch_size, height, width]
#             for i in range(batch_size):
#                 try:
#                     heatmap = feature_mean[i].detach().cpu().numpy()
#                     # 归一化到 [0, 1]
#                     heatmap_min = heatmap.min()
#                     heatmap_max = heatmap.max()
#                     if heatmap_max - heatmap_min > 1e-8:
#                         heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
#                     else:
#                         heatmap = heatmap - heatmap_min  # 避免除以零
#
#                     plt.figure(figsize=(4, 4))
#                     plt.axis('off')
#                     plt.imshow(heatmap, cmap='jet')
#                     plt.tight_layout(pad=0)
#
#                     # 定义文件名，包含层级、样本编号和批次计数器
#                     filename = f'heatmap_layer{idx + 1}_sample{i + 1}_batch{self.batch_counter}.png'
#                     filepath = os.path.join(self.heatmap_dir, filename)
#                     plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
#                     plt.close()
#
#                     print(f"保存热力图: {filename}")
#                 except Exception as e:
#                     print(f"生成热力图时出错，层级: {idx + 1}, 样本: {i + 1}, 错误: {e}")
#                     continue
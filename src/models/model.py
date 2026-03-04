import torch
import torch.nn as nn
import functools
import numpy as np
from timm.models.layers import trunc_normal_
from .utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)
from MinkowskiOps import (
    to_sparse,
)


class BasicConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0, bias=True):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(ch_in, ch_out,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,bias=bias)
        self.Norm = nn.BatchNorm2d(ch_out)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.Norm(x)
        x = self.ReLU(x)
        return x

class ResConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResConvBlock, self).__init__()
        self.formatLayer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(ch_out, ch_out*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ch_out*2, ch_out*4, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.pixelShuffle = nn.PixelShuffle(2)
        self.maxpooling = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.formatLayer(x)
        out = self.ConvBlock(residual)
        out = self.pixelShuffle(out)
        out = self.maxpooling(out)
        return self.sigmoid(out + residual)

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConvBlock, self).__init__()
        self.Conv = BasicConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self,x,size_demo):
        output_size = size_demo.size()[2:] # Compatible with irregular inputs
        x = nn.functional.interpolate(x,size=output_size,mode='bilinear', align_corners=True)
        x = self.Conv(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, ngf=64):
        super(ResUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv_1 = ResConvBlock(ch_in=img_ch, ch_out=ngf)  # [B, ngf, H, W]
        self.Conv_2 = ResConvBlock(ch_in=ngf, ch_out=2 * ngf)  # [B, 2 * ngf, H / 2, W / 2]
        self.Conv_3 = ResConvBlock(ch_in=2 * ngf, ch_out=4 * ngf)  # [B, 4 * ngf, H / 4, W / 4]
        self.Conv_4 = ResConvBlock(ch_in=4 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 8, W / 8]
        self.Conv_5 = ResConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 16, W / 16]
        self.Conv_6 = ResConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 32, W / 32]
        self.Conv_7 = ResConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 64, W / 64]
        self.Conv_8 = ResConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up_8_l = UpConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 128, W / 128]
        self.Up_conv_8_l = ResConvBlock(ch_in=16 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up_7_l = UpConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 64, W / 64]
        self.Up_conv_7_l = ResConvBlock(ch_in=16 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 64, W / 64]

        self.Up_6_l = UpConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 32, W / 32]
        self.Up_conv_6_l = ResConvBlock(ch_in=16 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 32, W / 32]

        self.Up_5_l = UpConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 16, W / 16]
        self.Up_conv_5_l = ResConvBlock(ch_in=16 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 16, W / 16]

        self.Up_4_l = UpConvBlock(ch_in=8 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 8, W / 8]
        self.Up_conv_4_l = ResConvBlock(ch_in=16 * ngf, ch_out=8 * ngf)  # [B, 8 * ngf, H / 8, W / 8]

        self.Up_3_l = UpConvBlock(ch_in=8 * ngf, ch_out=4 * ngf)  # [B, 4 * ngf, H / 4, W / 4]
        self.Up_conv_3_l = ResConvBlock(ch_in=8 * ngf, ch_out=4 * ngf)  # [B, 4 * ngf, H / 4, W / 4]

        self.Up_2_l = UpConvBlock(ch_in=4 * ngf, ch_out=2 * ngf)  # [B, 2 * ngf, H / 2, W / 2]
        self.Up_conv_2_l = ResConvBlock(ch_in=4 * ngf, ch_out=2 * ngf)  # [B, 2 * ngf, H / 2, W / 2]

        self.Up_1_l = UpConvBlock(ch_in=2 * ngf, ch_out=ngf)  # [B, ngf, H, W]
        self.Up_conv_1_l = ResConvBlock(ch_in=2 * ngf, ch_out=ngf)  # [B, ngf, H, W]

        self.Out_l = nn.Conv2d(ngf, output_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        # encoding path
        x1 = self.Conv_1(x)  # [B, ngf, H, W]

        x2 = self.Maxpool(x1)  # [B, ngf, H / 2, W / 2]
        x2 = self.Conv_2(x2)  # [B, 2 * ngf, H / 2, W / 2]

        x3 = self.Maxpool(x2)  # [B, 2 * ngf, H / 4, W / 4]
        x3 = self.Conv_3(x3)  # [B, 4 * ngf, H / 4, W / 4]

        x4 = self.Maxpool(x3)  # [B, 4 * ngf, H / 8, W / 8]
        x4 = self.Conv_4(x4)  # [B, 8 * ngf, H / 8, W / 8]

        x5 = self.Maxpool(x4)  # [B, 8 * ngf, H / 16, W / 16]
        x5 = self.Conv_5(x5)  # [B, 8 * ngf, H / 16, W / 16]

        x6 = self.Maxpool(x5)  # [B, 8 * ngf, H / 32, W / 32]
        x6 = self.Conv_6(x6)  # [B, 8 * ngf, H / 32, W / 32]

        x7 = self.Maxpool(x6)  # [B, 8 * ngf, H / 64, W / 64]
        x7 = self.Conv_7(x7)  # [B, 8 * ngf, H / 64, W / 64]

        x8 = self.Maxpool(x7)  # [B, 8 * ngf, H / 128, W / 128]
        x8 = self.Conv_8(x8)  # [B, 8 * ngf, H / 128, W / 128]

        x9 = self.Maxpool(x8)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d8_l = self.Up_8_l(x9, x8)  # [B, 8 * ngf, H / 128, W / 128]
        d8_l = torch.cat((x8, d8_l), dim=1)  # [B, 16 * ngf, H / 128, W / 128]
        d8_l = self.Up_conv_8_l(d8_l)  # [B, 8 * ngf, H / 128, W / 128]

        d7_l = self.Up_7_l(d8_l, x7)  # [B, 8 * ngf, H / 64, W / 64]
        d7_l = torch.cat((x7, d7_l), dim=1)  # [B, 16 * ngf, H / 64, W / 64]
        d7_l = self.Up_conv_7_l(d7_l)  # [B, 8 * ngf, H / 64, W / 64]

        d6_l = self.Up_6_l(d7_l, x6)  # [B, 8 * ngf, H / 32, W / 32]
        d6_l = torch.cat((x6, d6_l), dim=1)  # [B, 16 * ngf, H / 32, W / 32]
        d6_l = self.Up_conv_6_l(d6_l)  # [B, 8 * ngf, H / 32, W / 32]

        d5_l = self.Up_5_l(d6_l, x5)  # [B, 8 * ngf, H / 16, W / 16]
        d5_l = torch.cat((x5, d5_l), dim=1)  # [B, 16 * ngf, H / 16, W / 16]
        d5_l = self.Up_conv_5_l(d5_l)  # [B, 8 * ngf, H / 16, W / 16]

        d4_l = self.Up_4_l(d5_l, x4)  # [B, 8 * ngf, H / 8, W / 8]
        d4_l = torch.cat((x4, d4_l), dim=1)  # [B, 16 * ngf, H / 8, W / 8]
        d4_l = self.Up_conv_4_l(d4_l)  # [B, 8 * ngf, H / 8, W / 8]

        d3_l = self.Up_3_l(d4_l, x3)  # [B, 4 * ngf, H / 4, W / 4]
        d3_l = torch.cat((x3, d3_l), dim=1)  # [B, 8 * ngf, H / 4, W / 4]
        d3_l = self.Up_conv_3_l(d3_l)  # [B, 4 * ngf, H / 4, W / 4]

        d2_l = self.Up_2_l(d3_l, x2)  # [B, 2 * ngf, H / 2, W / 2]
        d2_l = torch.cat((x2, d2_l), dim=1)  # [B, 4 * ngf, H / 2, W / 2]
        d2_l = self.Up_conv_2_l(d2_l)  # [B, 2 * ngf, H / 2, W / 2]

        d1_l = self.Up_1_l(d2_l, x1)  # [B, ngf, H, W]
        d1_l = torch.cat((x1, d1_l), dim=1)  # [B, 2 * ngf, H, W]
        d1_l = self.Up_conv_1_l(d1_l)  # [B, ngf, H, W]

        out_l = self.Out_l(d1_l)


        return self.sigmoid(out_l)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print("The parameters have been initialized as the kaiming distribution.")


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, x,y):
        # 假定image_tensor的形状为(batch_size, channels, height, width)
        # 其中channels为3，对应RGB三个颜色通道

        # 计算R和G通道的差异
        rg_diff_x = (x[:, 0, :, :] - x[:, 1, :, :])
        rg_diff_y = (y[:, 0, :, :] - y[:, 1, :, :])
        # 计算R和B通道的差异
        rb_diff_x = (x[:, 0, :, :] - x[:, 2, :, :])
        rb_diff_y = (y[:, 0, :, :] - y[:, 1, :, :])
        # 计算G和B通道的差异
        gb_diff_x = (x[:, 1, :, :] - x[:, 2, :, :])
        gb_diff_y = (y[:, 0, :, :] - y[:, 1, :, :])

        # 将三个差异求和并取平均作为损失
        loss =self.mseloss(rg_diff_x,rg_diff_y)+self.mseloss(rb_diff_x,rb_diff_y)+self.mseloss(gb_diff_x,gb_diff_y)

        return loss / 3

class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, x):
        # 假定image_tensor的形状为(batch_size, channels, height, width)
        # 其中channels为3，对应RGB三个颜色通道

        # 计算R和G通道的差异
        rg_diff = torch.abs(x[:, 0, :, :] - x[:, 1, :, :])
        # 计算R和B通道的差异
        rb_diff = torch.abs(x[:, 0, :, :] - x[:, 2, :, :])
        # 计算G和B通道的差异
        gb_diff = torch.abs(x[:, 1, :, :] - x[:, 2, :, :])

        # 将三个差异求和并取平均作为损失
        loss = torch.mean(rg_diff) + torch.mean(rb_diff) +torch.mean(gb_diff)


        return loss / 3




if __name__ == '__main__':

    model = ResUNet(img_ch=1, output_ch=1,ngf=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = torch.randn((16, 1, 256, 256)).to(device).type(torch.float32)
    left,right = model(inputs)
    print("Output shape:", left.shape,right.shape)


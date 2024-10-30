import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.ops import DeformConv2d # 引入可变形卷积

import numpy as  np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_deform=False):
        super(ResidualBlock, self).__init__()
        
        # 定义偏移量生成层
        self.use_deform = use_deform
        if use_deform:
            self.offset_conv1 = nn.Conv2d(in_channels, 18, kernel_size=3, stride=stride, padding=1)
            self.conv1 = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.offset_conv2 = nn.Conv2d(out_channels, 18, kernel_size=3, stride=1, padding=1)
            self.conv2 = DeformConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        identity = x
        if self.use_deform:
            # 计算偏移量并应用到可变形卷积
            offset1 = self.offset_conv1(x)
            out = self.conv1(x, offset1)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        
        if self.use_deform:
            offset2 = self.offset_conv2(out)
            out = self.conv2(out, offset2)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(2, 256, 256)):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size
        self.channel_mult = 64

        # Input layer
        self.initial = nn.Sequential(
            nn.Conv2d(2, self.channel_mult, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 编码器部分，encoder3 和 encoder4 使用了可变形卷积
        self.encoder1 = self._make_layer(self.channel_mult, self.channel_mult, 2)
        self.encoder2 = self._make_layer(self.channel_mult, self.channel_mult*2, 2)
        self.encoder3 = self._make_layer(self.channel_mult*2, self.channel_mult*4, 2, use_deform=True)  # 使用可变形卷积
        self.encoder4 = self._make_layer(self.channel_mult*4, self.channel_mult*8, 2, use_deform=True)  # 使用可变形卷积

        # Pyramid Pooling Module
        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*8, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_mult*8, output_size)

    def _make_layer(self, in_channels, out_channels, blocks, use_deform=False):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride=2, use_deform=use_deform))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels, use_deform=use_deform))
        return nn.Sequential(*layers)

    def _residual_block(self, in_channels, out_channels, stride=1, use_deform=False):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample, use_deform)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = self.ppm(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [x4, x3, x2, x1]


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(2, 256, 256)):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 64
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.channel_mult*8*4*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 解码层
        self.decoder1 = self._up_block(self.channel_mult*8, self.channel_mult*4)
        self.decoder2 = self._up_block(self.channel_mult*4, self.channel_mult*2)
        self.decoder3 = self._up_block(self.channel_mult*2, self.channel_mult)
        self.decoder4 = self._up_block(self.channel_mult, self.channel_mult//2)
        self.decoder5 = self._up_block(self.channel_mult // 2, self.channel_mult // 2)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult//2, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.fusion1 = nn.Conv2d(self.channel_mult*12, self.channel_mult*4, 1)
        self.fusion2 = nn.Conv2d(self.channel_mult*6, self.channel_mult*2, 1)
        self.fusion3 = nn.Conv2d(self.channel_mult*3, self.channel_mult, 1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, encoder_features):
        x = self.fc(x)
        x = x.view(-1, self.channel_mult*8, 4, 4)

        x = self.decoder1(x)
        x = torch.cat([x, encoder_features[0]], 1)
        x = self.fusion1(x)
        
        x = self.decoder2(x)
        x = torch.cat([x, encoder_features[1]], 1)
        x = self.fusion2(x)
        
        x = self.decoder3(x)
        x = torch.cat([x, encoder_features[2]], 1)
        x = self.fusion3(x)
        
        x = self.decoder4(x)
        x = self.decoder5(x)
        
        x = self.final(x)
        return x

"""
# 测试代码
input_size = (2, 256, 256)
embedding_size = 128
batch_size = 4

# 创建模型
encoder = CNN_Encoder(embedding_size, input_size)
decoder = CNN_Decoder(embedding_size, input_size)

# 创建测试输入
x = torch.randn(batch_size, *input_size)

# 前向传播测试
encoded, features = encoder(x)
decoded = decoder(encoded, features)

# 打印各个张量的形状
print(f"Input shape: {x.shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Decoded shape: {decoded.shape}")
"""
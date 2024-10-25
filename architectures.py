import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.ops import DeformConv2d # 引入可变形卷积

import numpy as  np

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(2, 256, 256)):
        super(CNN_Encoder, self).__init__()
        
        self.input_size = input_size
        self.channel_mult = 16

        # 第一个卷积层 (2 -> 16 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.channel_mult*1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 保存conv1的输出作为第一个残差连接
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, kernel_size=1, stride=2),
            nn.BatchNorm2d(self.channel_mult*2)
        )
        
        # 第二个卷积层 (16 -> 32 channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 5, 2, 2),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 第三个可变形卷积层 (32 -> 64 channels)
        self.deform_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        
        # 调整第二个残差连接的维度 (32 -> 64 channels)
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, kernel_size=1, stride=2),
            nn.BatchNorm2d(self.channel_mult*4)
        )

        # 后续卷积层
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )

        # 金字塔池化模块
        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel_mult*16, self.channel_mult*16, 1),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Linear(self.channel_mult*16, output_size)

    def forward(self, x):
        # 第一个残差块
        x1 = self.conv1(x)  # (B, 16, 128, 128)
        res1 = x1  # 保存第一层输出
        
        x2 = self.conv2(x1)  # (B, 32, 64, 64)
        x2 = x2 + self.res_conv1(res1)  # 第一个残差连接
        
        # 第二个残差块
        res2 = x2  # 保存用于第二个残差连接
        x3 = self.deform_conv3(x2)  # (B, 64, 32, 32)
        x3 = x3 + self.res_conv3(res2)  # 第二个残差连接
        
        # 后续处理保持不变...
        x = self.conv4(x3)
        x = self.conv5(x)
        x = self.ppm(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(2, 256, 256)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 256
        self.input_width = 256
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 2

        # 计算全连接层输出维度
        self.fc_output_dim = self.channel_mult * 16 * (self.input_height // 32) * (self.input_width // 32)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.LayerNorm(self.fc_output_dim),
            nn.ELU(inplace=True)
        )

        # 反卷积层，逐步恢复图像尺寸
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult*16, self.channel_mult*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # 将输出限制在 [-1, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.channel_mult * 16, self.input_height // 32, self.input_width // 32)
        x = self.deconv(x)
        return x

"""
# 测试代码
batch_size = 32
latent_dim = 128  # 自定义潜在空间维度
x = torch.randn(batch_size, 2, 256, 256)
encoder = CNN_Encoder(output_size=latent_dim)
decoder = CNN_Decoder(embedding_size=latent_dim)

# 前向传播测试
encoded = encoder(x)
decoded = decoder(encoded)

print(f"Input shape: {x.shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Decoded shape: {decoded.shape}")
"""
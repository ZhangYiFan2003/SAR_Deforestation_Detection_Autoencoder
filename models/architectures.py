import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.ops import DeformConv2d # 引入可变形卷积

import numpy as  np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)  # Add Dropout after BatchNorm
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

        # Input layer: 2 -> 32 channels, size: 256x256 -> 128x128
        self.initial = nn.Sequential(
            nn.Conv2d(2, self.channel_mult, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # encoder1: 32 -> 32 channels, size: 128x128 -> 64x64
        self.encoder1 = self._make_layer(self.channel_mult, self.channel_mult, 2)  
        # encoder2: 32 -> 64 channels, size: 64x64 -> 32x32
        self.encoder2 = self._make_layer(self.channel_mult, self.channel_mult*2, 2)  
        # encoder3: 64 -> 128 channels, size: 32x32 -> 16x16
        self.encoder3 = self._make_layer(self.channel_mult*2, self.channel_mult*4, 2)  
        # encoder4: 128 -> 256 channels, size: 16x16 -> 8x8
        self.encoder4 = self._make_layer(self.channel_mult*4, self.channel_mult*8, 2)  

        # Pyramid Pooling Module (PPM)
        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*8, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_mult*8, output_size)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride=2))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _residual_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        # x: [batch, 2, 256, 256]
        x = self.initial(x)          # -> [batch, 32, 128, 128]
        x1 = self.encoder1(x)        # -> [batch, 32, 64, 64]
        x2 = self.encoder2(x1)       # -> [batch, 64, 32, 32]
        x3 = self.encoder3(x2)       # -> [batch, 128, 16, 16]
        x4 = self.encoder4(x3)       # -> [batch, 256, 8, 8]
        x = self.ppm(x4)             # -> [batch, 256, 1, 1]
        x = torch.flatten(x, 1)      # -> [batch, 256]
        x = self.fc(x)               # -> [batch, output_size]
        return x, [x4, x3, x2, x1]   # Return encoded features and feature maps

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(2, 256, 256)):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 64  
        
        # Fully connected layer: embedding_size -> 256*4*4
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.channel_mult*8*4*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # Decoder main layers
        self.decoder1 = self._up_block(self.channel_mult*8, self.channel_mult*4)  # 256 -> 128
        self.decoder2 = self._up_block(self.channel_mult*4, self.channel_mult*2)  # 128 -> 64
        self.decoder3 = self._up_block(self.channel_mult*2, self.channel_mult)    # 64 -> 32
        self.decoder4 = self._up_block(self.channel_mult, self.channel_mult//2)   # 32 -> 16
        self.decoder5 = self._up_block(self.channel_mult // 2, self.channel_mult // 2)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult//2, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Feature fusion layers
        self.fusion1 = nn.Conv2d(self.channel_mult*12, self.channel_mult*4, 1)  # 256+128=384 -> 128
        self.fusion2 = nn.Conv2d(self.channel_mult*6, self.channel_mult*2, 1)   # 128+64=192 -> 64
        self.fusion3 = nn.Conv2d(self.channel_mult*3, self.channel_mult, 1)     # 64+32=96 -> 32
        #self.fusion4 = nn.Conv2d(int(self.channel_mult*1.5), self.channel_mult//2, 1) # 32+16=48 -> 16

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x, encoder_features):
        # x: [batch, embedding_size]
        x = self.fc(x)
        x = x.view(-1, self.channel_mult*8, 4, 4)  # -> [batch, 256, 4, 4]
        
        # Upsampling and feature fusion
        x = self.decoder1(x)                        # -> [batch, 128, 8, 8]
        x = torch.cat([x, encoder_features[0]], 1)  # -> [batch, 384, 8, 8]
        x = self.fusion1(x)                         # -> [batch, 128, 8, 8]
        
        x = self.decoder2(x)                        # -> [batch, 64, 16, 16]
        x = torch.cat([x, encoder_features[1]], 1)  # -> [batch, 192, 16, 16]
        x = self.fusion2(x)                         # -> [batch, 64, 16, 16]
        
        x = self.decoder3(x)                        # -> [batch, 32, 32, 32]
        x = torch.cat([x, encoder_features[2]], 1)  # -> [batch, 96, 32, 32]
        x = self.fusion3(x)                         # -> [batch, 32, 32, 32]
        
        x = self.decoder4(x)                        # -> [batch, 16, 64, 64]
        #x = torch.cat([x, encoder_features[3]], 1)  # -> [batch, 48, 64, 64]
        #x = self.fusion4(x)                         # -> [batch, 16, 64, 64]
        
        x = self.decoder5(x)                       # -> [batch, 16, 128, 128]
        
        x = self.final(x)                           # -> [batch, 2, 256, 256]
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
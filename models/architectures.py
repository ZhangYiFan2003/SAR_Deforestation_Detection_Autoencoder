import torch
from torch import nn
from torch.nn import functional as F

#####################################################################################################################################################

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
        out = self.dropout(out)  # 在 BatchNorm 后添加 Dropout
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out

#####################################################################################################################################################


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,  kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        """
        输入:
            x: 输入特征图 (B, C, H, W)
        返回:
            out: 自注意力增强的特征图
            attention: 自注意力映射
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height)         # B, C', N
        proj_key    = self.key_conv(x).view(m_batchsize, -1, width*height)           # B, C', N
        energy      = torch.bmm(proj_query.permute(0, 2, 1), proj_key)               # B, N, N
        attention   = self.softmax(energy)                                           # B, N, N
        proj_value  = self.value_conv(x).view(m_batchsize, -1, width*height)         # B, C, N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                      # B, C, N
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        out = self.norm(out)
        return out

#####################################################################################################################################################

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(2, 256, 256)):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size
        self.channel_mult = 64  

        # 输入层: 2 -> 64 channels, 尺寸: 256x256 -> 128x128
        self.initial = nn.Sequential(
            nn.Conv2d(2, self.channel_mult, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # encoder1: 64 -> 64 channels, 尺寸: 128x128 -> 64x64
        self.encoder1 = self._make_layer(self.channel_mult, self.channel_mult, 2)  
        # encoder2: 64 -> 128 channels, 尺寸: 64x64 -> 32x32
        self.encoder2 = self._make_layer(self.channel_mult, self.channel_mult * 2, 2)  
        # encoder3: 128 -> 256 channels, 尺寸: 32x32 -> 16x16
        self.encoder3 = self._make_layer(self.channel_mult * 2, self.channel_mult * 4, 2)  
        # encoder4: 256 -> 512 channels, 尺寸: 16x16 -> 8x8
        self.encoder4 = self._make_layer(self.channel_mult * 4, self.channel_mult * 8, 2)  
        # encoder5: 512 -> 512 channels, 尺寸: 8x8 -> 4x4
        self.encoder5 = self._make_layer(self.channel_mult * 8, self.channel_mult * 8, 2)  

        # 在 encoder5 后添加自注意力层
        self.attention = SelfAttention(self.channel_mult * 8)

        # FPN 侧边卷积层，将通道数统一为256
        self.lateral_conv1 = nn.Conv2d(self.channel_mult*8, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(self.channel_mult*8, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(self.channel_mult*4, 256, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(self.channel_mult*2, 256, kernel_size=1)

        # FPN 输出卷积层
        self.fpn_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, output_size)

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
        x = self.initial(x)          # -> [batch, 64, 128, 128]
        x1 = self.encoder1(x)        # -> [batch, 64, 64, 64]
        x2 = self.encoder2(x1)       # -> [batch, 128, 32, 32]
        x3 = self.encoder3(x2)       # -> [batch, 256, 16, 16]
        x4 = self.encoder4(x3)       # -> [batch, 512, 8, 8]
        x5 = self.encoder5(x4)       # -> [batch, 512, 4, 4]

        # 自注意力层
        x5 = self.attention(x5)      # -> [batch, 512, 4, 4]

        # FPN自顶向下路径
        p5 = self.lateral_conv1(x5)  # -> [batch, 256, 4, 4]
        p4 = self.lateral_conv2(x4) + F.interpolate(p5, size=x4.shape[2:], mode='nearest')  # -> [batch, 256, 8, 8]
        p3 = self.lateral_conv3(x3) + F.interpolate(p4, size=x3.shape[2:], mode='nearest')  # -> [batch, 256, 16, 16]
        p2 = self.lateral_conv4(x2) + F.interpolate(p3, size=x2.shape[2:], mode='nearest')  # -> [batch, 256, 32, 32]

        # FPN输出卷积
        p5 = self.fpn_conv1(p5)  # -> [batch, 256, 4, 4]
        p4 = self.fpn_conv2(p4)  # -> [batch, 256, 8, 8]
        p3 = self.fpn_conv3(p3)  # -> [batch, 256, 16, 16]
        p2 = self.fpn_conv4(p2)  # -> [batch, 256, 32, 32]

        # 使用最高层特征进行分类
        x = self.avgpool(p5)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # 返回编码特征和 FPN 特征图
        return x, [p5, p4, p3, p2]

#####################################################################################################################################################

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(2, 256, 256)):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 64  

        # 全连接层: embedding_size -> 256*4*4
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256*4*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 解码器主要层
        self.decoder1 = self._up_block(256, 256)  # 256 -> 256
        self.decoder2 = self._up_block(256, 256)  # 256 -> 256
        self.decoder3 = self._up_block(256, 256)  # 256 -> 256
        self.decoder4 = self._up_block(256, 128)  # 256 -> 128
        self.decoder5 = self._up_block(128, 64)   # 128 -> 64
        self.decoder6 = self._up_block(64, 32)    # 64 -> 32

        # 在 decoder3 后添加自注意力层
        self.attention_decoder = SelfAttention(256)

        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x, fpn_features):
        # x: [batch, embedding_size]
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)  # -> [batch, 256, 4, 4]

        # 上采样和特征融合
        x = self.decoder1(x)                        # -> [batch, 256, 8, 8]
        x = x + fpn_features[1]                     # 与 FPN 的 p4 特征融合

        x = self.decoder2(x)                        # -> [batch, 256, 16, 16]
        x = x + fpn_features[2]                     # 与 FPN 的 p3 特征融合

        x = self.decoder3(x)                        # -> [batch, 256, 32, 32]

        # 在这里添加自注意力层
        x = self.attention_decoder(x)               # -> [batch, 256, 32, 32]

        x = self.decoder4(x)                        # -> [batch, 128, 64, 64]
        x = self.decoder5(x)                        # -> [batch, 64, 128, 128]
        x = self.decoder6(x)                        # -> [batch, 32, 256, 256]

        x = self.final(x)                           # -> [batch, 2, 256, 256]
        return x

#####################################################################################################################################################

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
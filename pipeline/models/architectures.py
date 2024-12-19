import torch
from torch import nn
from torch.nn import functional as F

#####################################################################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # Save input as identity for skip connection
        identity = x
        
        # First convolution + BatchNorm + LeakyReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        
        # Second convolution + BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection and apply activation
        out += identity
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out

#####################################################################################################################################################

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Channel dimension of input
        self.channel_in = in_dim
        
        # Query, key, and value convolutions
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,  kernel_size=1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.tensor(0.1))
        # Softmax for attention weights
        self.softmax  = nn.Softmax(dim=-1)
        # Normalize the output
        self.norm = nn.BatchNorm2d(in_dim)
    
    def forward(self, x):
        """
        Inputs:
            x: Input feature map (B, C, H, W)
        Outputs:
            out: Self-attention enhanced feature map
            attention: Attention weights
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

class Encoder(nn.Module):
    def __init__(self, output_size, input_size=(2, 256, 256)):
        super(Encoder, self).__init__()
        self.input_size = input_size
        # Base channel multiplier
        self.channel_mult = 64  
        
        # Initial convolutional layer: Downscales input size by 2
        self.initial = nn.Sequential(
            nn.Conv2d(2, self.channel_mult, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Series of residual blocks for encoding
        self.encoder1 = self._make_layer(self.channel_mult, self.channel_mult, 2)  
        self.encoder2 = self._make_layer(self.channel_mult, self.channel_mult * 2, 2)  
        self.encoder3 = self._make_layer(self.channel_mult * 2, self.channel_mult * 4, 2)  
        self.encoder4 = self._make_layer(self.channel_mult * 4, self.channel_mult * 8, 2)  
        self.encoder5 = self._make_layer(self.channel_mult * 8, self.channel_mult * 8, 2)  
        
        # Add self-attention layer to capture global context
        self.attention = SelfAttention(self.channel_mult * 8)
        
        # Lateral convolution layers for FPN (Feature Pyramid Network)
        self.lateral_conv1 = nn.Conv2d(self.channel_mult*8, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(self.channel_mult*8, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(self.channel_mult*4, 256, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(self.channel_mult*2, 256, kernel_size=1)
        
        # Output convolution layers for FPN
        self.fpn_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Global average pooling and fully connected layer for output
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, output_size)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        """
        Creates a layer with a specified number of residual blocks.
        """
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride=2))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        """
        Creates a single residual block with optional downsampling.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        x = self.initial(x)          # -> [batch, 64, 128, 128]
        x1 = self.encoder1(x)        # -> [batch, 64, 64, 64]
        x2 = self.encoder2(x1)       # -> [batch, 128, 32, 32]
        x3 = self.encoder3(x2)       # -> [batch, 256, 16, 16]
        x4 = self.encoder4(x3)       # -> [batch, 512, 8, 8]
        x5 = self.encoder5(x4)       # -> [batch, 512, 4, 4]
        
        # Apply self-attention on the final features
        x5 = self.attention(x5)      # -> [batch, 512, 4, 4]
        
        # FPN pathway: Top-down feature fusion
        p5 = self.lateral_conv1(x5)  # -> [batch, 256, 4, 4]
        p4 = self.lateral_conv2(x4) + F.interpolate(p5, size=x4.shape[2:], mode='nearest')  # -> [batch, 256, 8, 8]
        p3 = self.lateral_conv3(x3) + F.interpolate(p4, size=x3.shape[2:], mode='nearest')  # -> [batch, 256, 16, 16]
        p2 = self.lateral_conv4(x2) + F.interpolate(p3, size=x2.shape[2:], mode='nearest')  # -> [batch, 256, 32, 32]
        
        # FPN output convolutions
        p5 = self.fpn_conv1(p5)  # -> [batch, 256, 4, 4]
        p4 = self.fpn_conv2(p4)  # -> [batch, 256, 8, 8]
        p3 = self.fpn_conv3(p3)  # -> [batch, 256, 16, 16]
        p2 = self.fpn_conv4(p2)  # -> [batch, 256, 32, 32]
        
        # Classification using the highest-level feature map
        x = self.avgpool(p5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Return encoded features and FPN maps
        return x, [p5, p4, p3, p2]

#####################################################################################################################################################

class Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(2, 256, 256)):
        super(Decoder, self).__init__()
        self.input_dim = embedding_size
        # Base channel multiplier
        self.channel_mult = 64  
        
        # Fully connected layer to map embedding size to feature map size (256 x 4 x 4)
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256*4*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Main upsampling blocks for decoding
        self.decoder1 = self._up_block(256, 256)  
        self.decoder2 = self._up_block(256, 256)  
        self.decoder3 = self._up_block(256, 256)  
        self.decoder4 = self._up_block(256, 128)  
        self.decoder5 = self._up_block(128, 64)   
        self.decoder6 = self._up_block(64, 32)    
        
        # Self-attention layer after decoder3 to enhance feature representation
        self.attention_decoder = SelfAttention(256)
        
        # Final output layer: Convolution to 2 channels with Tanh activation
        self.final = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def _up_block(self, in_channels, out_channels):
        """
        Creates an upsampling block with ConvTranspose2d for increasing spatial resolution.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    
    def forward(self, x, fpn_features):
        """
        Forward pass of the decoder.
        """
        # Fully connected layer to reshape latent vector to initial feature map
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)  # -> [batch, 256, 4, 4]
        
        # Upsampling and feature fusion with FPN outputs
        x = self.decoder1(x)                        # -> [batch, 256, 8, 8]
        x = x + fpn_features[1]                     # Fuse with FPN feature p4
        
        x = self.decoder2(x)                        # -> [batch, 256, 16, 16]
        x = x + fpn_features[2]                     # Fuse with FPN feature p3
        
        x = self.decoder3(x)                        # -> [batch, 256, 32, 32]
        
        # Self-attention layer for refining feature representation
        x = self.attention_decoder(x)               # -> [batch, 256, 32, 32]
        
        x = self.decoder4(x)                        # -> [batch, 128, 64, 64]
        x = self.decoder5(x)                        # -> [batch, 64, 128, 128]
        x = self.decoder6(x)                        # -> [batch, 32, 256, 256]
        
        x = self.final(x)                           # -> [batch, 2, 256, 256]
        return x

#####################################################################################################################################################

"""
# test code
input_size = (2, 256, 256)
embedding_size = 128
batch_size = 4

encoder = Encoder(embedding_size, input_size)
decoder = Decoder(embedding_size, input_size)

x = torch.randn(batch_size, *input_size)

encoded, fpn_features = encoder(x)

print("FPN Feature Shapes:")
for i, feature in enumerate(fpn_features):
    print(f"p{i+2}: {feature.shape}")

decoded_input = torch.randn(batch_size, embedding_size)  
fpn_shapes_match = True

x = decoder.fc(decoded_input)  
x = x.view(-1, 256, 4, 4)      
print(f"Decoder Initial Shape: {x.shape}")

x = decoder.decoder1(x)        # -> [batch, 256, 8, 8]
print(f"Decoder Stage 1 Shape: {x.shape}, FPN Feature p4 Shape: {fpn_features[1].shape}")
if x.shape != fpn_features[1].shape:
    print("Shape mismatch at Decoder Stage 1 with FPN Feature p4!")
    fpn_shapes_match = False
x = x + fpn_features[1]

x = decoder.decoder2(x)        # -> [batch, 256, 16, 16]
print(f"Decoder Stage 2 Shape: {x.shape}, FPN Feature p3 Shape: {fpn_features[2].shape}")
if x.shape != fpn_features[2].shape:
    print("Shape mismatch at Decoder Stage 2 with FPN Feature p3!")
    fpn_shapes_match = False
x = x + fpn_features[2]

x = decoder.decoder3(x)        # -> [batch, 256, 32, 32]
print(f"Decoder Stage 3 Shape: {x.shape}, FPN Feature p2 Shape: {fpn_features[3].shape}")
if x.shape != fpn_features[3].shape:
    print("Shape mismatch at Decoder Stage 3 with FPN Feature p2!")
    fpn_shapes_match = False

x = decoder.decoder4(x)        # -> [batch, 128, 64, 64]
print(f"Decoder Stage 4 Shape: {x.shape}")
x = decoder.decoder5(x)        # -> [batch, 64, 128, 128]
print(f"Decoder Stage 5 Shape: {x.shape}")
x = decoder.decoder6(x)        # -> [batch, 32, 256, 256]
print(f"Decoder Stage 6 Shape: {x.shape}")

x = decoder.final(x)           # -> [batch, 2, 256, 256]
print(f"Final Decoded Output Shape: {x.shape}")

if fpn_shapes_match:
    print("All FPN features match decoder input sizes correctly.")
else:
    print("Mismatch found between FPN features and decoder input sizes.")
"""
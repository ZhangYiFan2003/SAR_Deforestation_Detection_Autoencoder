import sys
import os

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add PlotNeuralNet root directory to Python path
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from pycore.tikzeng import *  # Import modules needed for plotting
from pycore.blocks import *

# Define your architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input layer
    to_input('input.png', name='input'),

    # Initial Convolution Layer
    to_ConvConvRelu(
        name='ccr_b0',
        s_filer=256,
        n_filer=(64, 64),
        offset="(0,0,0)",
        to="(input-east)",
        width=(2, 2),
        height=40,
        depth=40,
        caption='Initial Conv'
    ),
    to_connection('input', 'ccr_b0'),

    # Encoder Block 1
    to_ConvConvRelu(
        name='ccr_b1',
        s_filer=128,
        n_filer=(64, 64),
        offset="(1.5,0,0)",
        to="(ccr_b0-east)",
        width=(2, 2),
        height=32,
        depth=32,
        caption='Encoder Block 1'
    ),
    to_connection('ccr_b0', 'ccr_b1'),

    # Encoder Block 2
    to_ConvConvRelu(
        name='ccr_b2',
        s_filer=64,
        n_filer=(128, 128),
        offset="(1.5,0,0)",
        to="(ccr_b1-east)",
        width=(2.5, 2.5),
        height=25,
        depth=25,
        caption='Encoder Block 2'
    ),
    to_connection('ccr_b1', 'ccr_b2'),

    # Encoder Block 3
    to_ConvConvRelu(
        name='ccr_b3',
        s_filer=32,
        n_filer=(256, 256),
        offset="(1.5,0,0)",
        to="(ccr_b2-east)",
        width=(3, 3),
        height=16,
        depth=16,
        caption='Encoder Block 3'
    ),
    to_connection('ccr_b2', 'ccr_b3'),

    # Encoder Block 4
    to_ConvConvRelu(
        name='ccr_b4',
        s_filer=16,
        n_filer=(512, 512),
        offset="(1.5,0,0)",
        to="(ccr_b3-east)",
        width=(3.5, 3.5),
        height=12,
        depth=12,
        caption='Encoder Block 4'
    ),
    to_connection('ccr_b3', 'ccr_b4'),

    # Encoder Block 5
    to_ConvConvRelu(
        name='ccr_b5',
        s_filer=8,
        n_filer=(512, 512),
        offset="(1.5,0,0)",
        to="(ccr_b4-east)",
        width=(4, 4),
        height=8,
        depth=8,
        caption='Encoder Block 5'
    ),
    to_connection('ccr_b4', 'ccr_b5'),

    # Self-Attention Layer
    to_Conv(
        name='self_attention',
        s_filer=8,
        n_filer=512,
        offset="(2,0,0)",
        to="(ccr_b5-east)",
        width=10,
        height=8,
        depth=8,
        caption='Self-Attention'
    ),
    to_connection('ccr_b5', 'self_attention'),

    # Decoder Starts Here
    # Upsampling Block 1
    *block_Unconv(
        name="b6",
        botton="self_attention",
        top='end_b6',
        s_filer=8,
        n_filer=256,
        offset="(2.1,0,0)",
        size=(8, 8, 8),
        opacity=0.5
    ),
    to_skip(of='ccr_b5', to='ccr_res_b6', pos=1.25),

    # Upsampling Block 2
    *block_Unconv(
        name="b7",
        botton="end_b6",
        top='end_b7',
        s_filer=16,
        n_filer=256,
        offset="(2.1,0,0)",
        size=(12, 12, 8),
        opacity=0.5
    ),
    to_skip(of='ccr_b4', to='ccr_res_b7', pos=1.25),

    # Self-Attention in Decoder
    to_Conv(
        name='self_attention_dec',
        s_filer=32,
        n_filer=256,
        offset="(2,0,0)",
        to="(end_b7-east)",
        width=8,
        height=16,
        depth=16,
        caption='Self-Attention'
    ),
    to_connection('end_b7', 'self_attention_dec'),

    # Upsampling Block 3
    *block_Unconv(
        name="b8",
        botton="self_attention_dec",
        top='end_b8',
        s_filer=32,
        n_filer=256,
        offset="(2.1,0,0)",
        size=(16, 16, 8),
        opacity=0.5
    ),

    # Upsampling Block 4
    *block_Unconv(
        name="b9",
        botton="end_b8",
        top='end_b9',
        s_filer=64,
        n_filer=128,
        offset="(2.1,0,0)",
        size=(25, 25, 6),
        opacity=0.5
    ),

    # Upsampling Block 5
    *block_Unconv(
        name="b10",
        botton="end_b9",
        top='end_b10',
        s_filer=128,
        n_filer=64,
        offset="(2.1,0,0)",
        size=(32, 32, 4),
        opacity=0.5
    ),

    # Upsampling Block 6
    *block_Unconv(
        name="b11",
        botton="end_b10",
        top='end_b11',
        s_filer=256,
        n_filer=32,
        offset="(2.1,0,0)",
        size=(40, 40, 2),
        opacity=0.5
    ),

    # Output Layer
    to_ConvSoftMax(
        name='output',
        offset="(0.75,0,0)",
        to="(end_b11-east)",
        width=1,
        height=40,
        depth=40,
        caption='Output'
    ),
    to_connection('end_b11', 'output'),

    to_end()
]

# Generate the .tex file
def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()

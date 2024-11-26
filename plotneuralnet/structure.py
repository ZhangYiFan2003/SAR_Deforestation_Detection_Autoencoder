import sys
import os

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add PlotNeuralNet root directory to Python path
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from S10_Visuallize_model_structure.PlotNeuralNet.pycore.tikzeng import *  # Import modules needed for plotting
from S10_Visuallize_model_structure.PlotNeuralNet.pycore.blocks import *

# Define the updated architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input (Larger size layer to represent the original image size 256x256)
    to_input('../examples/fcn8s/cats.jpg', to='(-2,0,0)', name="input_image"),
    
    # Transition Layer to simulate the downsampling effect of Conv2D
    to_Conv(name='transition', s_filer=256, n_filer=2, offset="(0,0,0)", to="(input_image-east)",
            width=1, height=64, depth=64, caption="Input Image (256x256)"),

    # Initial Conv Layer (Downsampling to 128x128)
    to_Conv(name='initial', s_filer=128, n_filer=64, offset="(2,0,0)", to="(transition-east)",
            width=4, height=40, depth=40, caption="Conv2d(7x7, s=2)"),

    # Encoder Block 1
    *block_2ConvPool(name='b1', botton='initial', top='pool_b1', s_filer=64, n_filer=64,
                     offset="(2,0,0)", size=(32,32,3.5), opacity=0.5),

    # Encoder Block 2
    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', s_filer=32, n_filer=128,
                     offset="(2,0,0)", size=(25,25,4.5), opacity=0.5),

    # Encoder Block 3
    *block_2ConvPool(name='b3', botton='pool_b2', top='pool_b3', s_filer=16, n_filer=256,
                     offset="(2,0,0)", size=(16,16,5.5), opacity=0.5),

    # Encoder Block 4
    *block_2ConvPool(name='b4', botton='pool_b3', top='pool_b4', s_filer=8, n_filer=512,
                     offset="(2,0,0)", size=(8,8,6.5), opacity=0.5),

    # Encoder Block 5
    *block_2ConvPool(name='b5', botton='pool_b4', top='pool_b5', s_filer=4, n_filer=512,
                     offset="(2,0,0)", size=(4,4,6.5), opacity=0.5),

    # Self-Attention Layer (after Encoder Block 5)
    to_SoftMax(name='self_attention', s_filer=4, offset="(2.5,0,0)", to="(pool_b5-east)",
               width=1, height=4, depth=4, caption="Self-Attention"),

    # FPN (Feature Pyramid Network)
    to_Conv(name='fpn_p5', s_filer=4, n_filer=256, offset="(3,0,0)", to="(self_attention-east)",
            width=3, height=4, depth=4, caption="FPN P5"),
    to_Conv(name='fpn_p4', s_filer=8, n_filer=256, offset="(2.5,0,0)", to="(fpn_p5-east)",
            width=3, height=8, depth=8, caption="FPN P4"),
    to_Conv(name='fpn_p3', s_filer=16, n_filer=256, offset="(2.5,0,0)", to="(fpn_p4-east)",
            width=3, height=16, depth=16, caption="FPN P3"),
    to_Conv(name='fpn_p2', s_filer=32, n_filer=256, offset="(2.5,0,0)", to="(fpn_p3-east)",
            width=3, height=32, depth=32, caption="FPN P2"),

    # Decoder FC and Reshape
    to_Conv(name='fc_decoder', s_filer=1, n_filer=256, offset="(2.5,0,0)", to="(fpn_p2-east)",
            width=2, height=2, depth=2, caption="FC"),
    to_UnPool(name='reshape', offset="(1.0,0,0)", to="(fc_decoder-east)",
              width=1, height=4, depth=4, opacity=0.5, caption="Reshape"),

    # Decoder Block 1
    *block_Unconv(name='d1', botton='reshape', top='end_d1', s_filer=8, n_filer=256,
                  offset="(2.5,0,0)", size=(8,8,5.0), opacity=0.5),
    to_skip(of='fpn_p4', to='ccr_res_d1', pos=1.25),

    # Decoder Block 2
    *block_Unconv(name='d2', botton='end_d1', top='end_d2', s_filer=16, n_filer=256,
                  offset="(2.5,0,0)", size=(16,16,4.5), opacity=0.5),
    to_skip(of='fpn_p3', to='ccr_res_d2', pos=1.25),

    # Decoder Block 3
    *block_Unconv(name='d3', botton='end_d2', top='end_d3', s_filer=32, n_filer=256,
                  offset="(2.5,0,0)", size=(25,25,3.5), opacity=0.5),

    # Self-Attention Layer (after Decoder Block 3)
    to_SoftMax(name='decoder_attention', s_filer=32, offset="(1.5,0,0)", to="(end_d3-east)",
               width=1, height=25, depth=25, caption="Self-Attention"),

    # Decoder Block 4
    *block_Unconv(name='d4', botton='end_d3', top='end_d4', s_filer=64, n_filer=128,
                  offset="(2.5,0,0)", size=(32,32,2.5), opacity=0.5),

    # Decoder Block 5
    *block_Unconv(name='d5', botton='end_d4', top='end_d5', s_filer=128, n_filer=64,
                  offset="(2.5,0,0)", size=(40,40,2.0), opacity=0.5),

    # Decoder Block 6
    *block_Unconv(name='d6', botton='end_d5', top='end_d6', s_filer=256, n_filer=32,
                  offset="(2.5,0,0)", size=(64,64,1.5), opacity=0.5),

    # Final Output Layer
    to_Conv(name='final', s_filer=256, n_filer=2, offset="(2.5,0,0)", to="(end_d6-east)",
            width=1, height=64, depth=64, caption="Conv2d(3x3)"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()

import os
import numpy as np
from PIL import Image
import tifffile as tiff  # 使用 tifffile 库来保存多通道 tiff 文件

def split_and_fuse_images(vv_dir, vh_dir, output_dir, tile_size=256):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    vv_files = sorted([f for f in os.listdir(vv_dir) if f.endswith('.tif')])
    vh_files = sorted([f for f in os.listdir(vh_dir) if f.endswith('.tif')])

    assert len(vv_files) == len(vh_files), "VV 和 VH 文件数量不一致"

    for idx, (vv_file, vh_file) in enumerate(zip(vv_files, vh_files)):
        vv_image = Image.open(os.path.join(vv_dir, vv_file))
        vh_image = Image.open(os.path.join(vh_dir, vh_file))

        vv_image = np.array(vv_image)
        vh_image = np.array(vh_image)

        # 确保 VV 和 VH 图像大小一致
        assert vv_image.shape == vh_image.shape, f"{vv_file} 和 {vh_file} 的尺寸不匹配"

        # 计算可以整除的大小
        h, w = vv_image.shape
        new_h = (h // tile_size) * tile_size
        new_w = (w // tile_size) * tile_size

        vv_image = vv_image[:new_h, :new_w]
        vh_image = vh_image[:new_h, :new_w]

        # 切割并保存图像块
        for i in range(0, new_h, tile_size):
            for j in range(0, new_w, tile_size):
                vv_tile = vv_image[i:i+tile_size, j:j+tile_size]
                vh_tile = vh_image[i:i+tile_size, j:j+tile_size]

                # 将 VV 和 VH 融合为 2 通道的图像
                fused_image = np.stack([vv_tile, vh_tile], axis=0)  # 堆叠为 (C, H, W)

                # 保存为 2 通道的 TIFF 文件
                fused_filename = os.path.join(output_dir, f"{vv_file[:-4]}_{i}_{j}_fused.tif")
                tiff.imwrite(fused_filename, fused_image, photometric='minisblack')

        print(f"已处理并融合 {vv_file} 和 {vh_file}")

if __name__ == "__main__":
    vv_dir = '/home/yifan/Documents/data/forest/train/VV'
    vh_dir = '/home/yifan/Documents/data/forest/train/VH'
    output_dir = '/home/yifan/Documents/data/forest/train/processed'
    split_and_fuse_images(vv_dir, vh_dir, output_dir)

import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import numpy as np

def fuse_and_split_images(
    vv_dir,
    vh_dir,
    fused_dir,
    tiles_dir,
    tile_size=256,
    prefix_fused="fused",
    prefix_tile="tile"
):
    """
    融合 VV 和 VH 图像为多波段 GeoTIFF，并将融合后的大图切分为带有地理信息的完整瓦片（仅保存 256x256 瓦片）。
    
    :param vv_dir:       VV 图像所在目录
    :param vh_dir:       VH 图像所在目录
    :param fused_dir:    融合后图像的输出目录
    :param tiles_dir:    切分后瓦片的输出目录
    :param tile_size:    瓦片大小（像素），默认为 256
    :param prefix_fused: 融合后文件名前缀，默认为 "fused"
    :param prefix_tile:  切分后瓦片文件名前缀，默认为 "tile"
    """
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    
    # 获取 VV 和 VH 目录下的所有 TIFF 文件，假设文件名可以对应
    vv_files = sorted([f for f in os.listdir(vv_dir) if f.endswith('.tif')])
    vh_files = sorted([f for f in os.listdir(vh_dir) if f.endswith('.tif')])
    
    assert len(vv_files) == len(vh_files), "VV 和 VH 目录中的文件数量不匹配"
    
    for vv_file, vh_file in zip(vv_files, vh_files):
        vv_path = os.path.join(vv_dir, vv_file)
        vh_path = os.path.join(vh_dir, vh_file)
        
        # 融合后的文件名
        fused_filename = f"{os.path.splitext(vv_file)[0]}.tif"
        fused_path = os.path.join(fused_dir, fused_filename)
        
        # 检查是否已经融合过，避免重复
        if not os.path.exists(fused_path):
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                # 检查 CRS 和 Transform 是否一致
                if (vv_src.crs != vh_src.crs) or (vv_src.transform != vh_src.transform):
                    raise ValueError(f"文件 {vv_file} 和 {vh_file} 的 CRS 或 Transform 不一致")
                
                # 检查尺寸是否一致
                if (vv_src.width != vh_src.width) or (vv_src.height != vh_src.height):
                    raise ValueError(f"文件 {vv_file} 和 {vh_file} 的尺寸不一致")
                
                # 读取 VV 和 VH 数据
                vv_data = vv_src.read(1)  # 假设 VV 是单波段
                vh_data = vh_src.read(1)  # 假设 VH 是单波段
                
                # 堆叠为 2 波段
                fused_data = np.stack([vv_data, vh_data], axis=0)
                
                # 定义输出元数据
                fused_meta = vv_src.meta.copy()
                fused_meta.update({
                    'count': 2,  # 两个波段
                    'dtype': fused_data.dtype
                })
                
                # 写出融合后的多波段 GeoTIFF
                with rasterio.open(fused_path, 'w', **fused_meta) as dst:
                    dst.write(fused_data)
                
                print(f"已融合并保存: {fused_path}")
        else:
            print(f"已存在融合文件: {fused_path}")
        
        # 切分融合后的图像为瓦片
        with rasterio.open(fused_path) as fused_src:
            src_transform = fused_src.transform
            src_crs = fused_src.crs
            src_width = fused_src.width
            src_height = fused_src.height
            band_count = fused_src.count  # 应为 2
            
            # 计算完整瓦片的数量
            num_tiles_row = src_height // tile_size
            num_tiles_col = src_width // tile_size
            
            for tile_row in range(num_tiles_row):
                for tile_col in range(num_tiles_col):
                    row_off = tile_row * tile_size
                    col_off = tile_col * tile_size
                    
                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=tile_size,
                        height=tile_size
                    )
                    
                    # 计算该瓦片的仿射变换
                    tile_transform = window_transform(window, src_transform)
                    
                    # 读取数据
                    tile_data = fused_src.read(
                        indexes=list(range(1, band_count + 1)),
                        window=window
                    )
                    
                    # 定义瓦片文件名
                    tile_filename = f"{os.path.splitext(fused_filename)[0]}_{row_off}_{col_off}_fused.tif"
                    tile_path = os.path.join(tiles_dir, tile_filename)
                    
                    # 定义瓦片元数据
                    tile_meta = fused_src.meta.copy()
                    tile_meta.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': tile_transform
                    })
                    
                    # 写出瓦片
                    with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                        dst.write(tile_data)
                    
                    print(f"已切分并保存瓦片: {tile_path}")
    
    print("所有图像已成功融合和切分。")

# 示例调用
if __name__ == "__main__":
    vv_dir = '/home/yifan/Documents/data/forest/test/VV'
    vh_dir = '/home/yifan/Documents/data/forest/test/VH'
    fused_dir = '/home/yifan/Documents/data/forest/test/fused'
    tiles_dir = '/home/yifan/Documents/data/forest/test/processed'
    fuse_and_split_images(vv_dir, vh_dir, fused_dir, tiles_dir, tile_size=256, prefix_fused="fused", prefix_tile="tile")
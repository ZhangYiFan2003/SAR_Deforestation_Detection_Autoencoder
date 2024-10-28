import os
import numpy as np
import tifffile as tiff  # 用于读取多通道 tiff 文件

def check_and_remove_nan_images(directory):
    """
    遍历目录中的所有 tiff 文件，检查它们是否包含 NaN 值。如果包含，则删除这些文件。
    
    Args:
        directory (string): 包含 .tif 图像的目录。
    """
    # 获取目录中所有 .tif 文件
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    
    for file in tif_files:
        file_path = os.path.join(directory, file)
        
        try:
            # 读取 tiff 文件
            img = tiff.imread(file_path)
            
            # 检查是否包含 NaN 值
            if np.isnan(img).any():
                print(f"文件 {file} 包含 NaN 值，正在删除...")
                os.remove(file_path)  # 删除文件
            else:
                print(f"文件 {file} 没有 NaN 值，保留。")
                
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")

if __name__ == "__main__":
    # 修改为你存储处理后 .tif 文件的文件夹路径
    processed_directory = '/home/yifan/Documents/data/forest/train/processed'

    # 处理 processed 文件夹
    print("正在处理 processed 文件夹...")
    check_and_remove_nan_images(processed_directory)

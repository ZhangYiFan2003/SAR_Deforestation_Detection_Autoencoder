import os
import numpy as np
import tifffile as tiff

def remove_invalid_tif_files(tiff_dir):
    """
    读取目录中的所有 tiff 文件，检查它们的最大值和最小值，如果都为 0，则删除该文件。
    
    Args:
        tiff_dir (string): 存放 tiff 文件的目录。
    """
    # 获取所有 tiff 文件
    tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tif')]

    removed_files = 0

    for tiff_file in tiff_files:
        tiff_path = os.path.join(tiff_dir, tiff_file)
        
        try:
            # 读取 tiff 文件
            img = tiff.imread(tiff_path)
            
            # 检查最小值和最大值
            img_min = np.min(img)
            img_max = np.max(img)

            if img_min == 0 and img_max == 0:
                print(f"文件 {tiff_file} 的最大值和最小值都为 0，删除该文件。")
                os.remove(tiff_path)  # 删除文件
                removed_files += 1
            else:
                print(f"文件 {tiff_file} 最大值: {img_max}, 最小值: {img_min}")
        except Exception as e:
            print(f"读取文件 {tiff_file} 时出现错误: {e}")
    
    print(f"共删除 {removed_files} 个无效的 TIFF 文件。")

if __name__ == "__main__":
    # 需要处理的目录
    tiff_dir = '/home/yifan/Documents/data/forest/test/processed'  # 根据你保存的文件路径进行修改
    remove_invalid_tif_files(tiff_dir)

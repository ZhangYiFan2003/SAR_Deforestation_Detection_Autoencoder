import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
import tifffile as tiff
#from osgeo import gdal  # 用于读取多通道 tiff 文件
import argparse

class ProcessedForestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 预处理后图像的根目录，包含已处理好的 2 通道 tiff 图像。
            transform (callable, optional): 可选的图像变换。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有切割后的图像块文件（假设都是 2 通道的 .tif 文件）
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取预处理后的 2 通道图像
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # 使用 tifffile 读取多通道 tiff 图像
        combined_image = tiff.imread(img_path)  # 读取为 (H, W, C) 或 (C, H, W) 格式的图像
        
        # 检查数据的形状是否符合预期
        if combined_image.ndim == 2:
            # 如果图像只有两个维度，则认为是单通道，扩展为 (C, H, W) 形式
            combined_image = combined_image[np.newaxis, ...]  # 变成 (1, H, W)

        elif combined_image.ndim == 3:
            if combined_image.shape[-1] == 2:
                # 如果是 (H, W, C) 形式的图像，转置为 (C, H, W)
                combined_image = np.transpose(combined_image, (2, 0, 1))  # 变成 (C, H, W)

        # 确保图像是 (C, H, W) 形式并且有 2 个通道
        if combined_image.shape[0] != 2:
            raise ValueError(f"预期的通道数是 2，但得到了 {combined_image.shape[0]}")

        # 手动归一化到 [0, 1] 范围
        combined_image = (combined_image - combined_image.min()) / (combined_image.max() - combined_image.min())

        # 转换为 PyTorch Tensor
        combined_image = torch.from_numpy(combined_image).float()

        # 应用图像变换
        if self.transform:
            combined_image = self.transform(combined_image)

        return combined_image


class ProcessedForestDataLoader(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        # 定义数据增强转换
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),                     # 随机水平翻转
            #transforms.RandomVerticalFlip(),                       # 随机垂直翻转
            #transforms.RandomRotation(90),                         # 随机旋转90度倍数
        ])

        self.train_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/train/processed', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        self.test_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/test/processed', transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)

"""
def main():
    # 创建命令行参数
    parser = argparse.ArgumentParser(description='Test ProcessedForestDataset and DataLoader')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    args = parser.parse_args()

    # 设置是否使用 CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 实例化 ProcessedForestDataLoader
    data_loader = ProcessedForestDataLoader(args)

    # 测试训练数据加载器
    print("开始测试训练数据集加载器...")
    for batch_idx, data in enumerate(data_loader.train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  图像块的大小: {data.size()}")

        # 打印图像块的一些数值信息
        print(f"  图像的最小值: {data.min().item()}, 最大值: {data.max().item()}")

        # 只测试一小部分，避免输出太多内容
        if batch_idx == 1:  # 只打印前两个批次
            break

    # 测试测试数据加载器
    print("开始测试测试数据集加载器...")
    for batch_idx, data in enumerate(data_loader.test_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  图像块的大小: {data.size()}")

        # 打印图像块的一些数值信息
        print(f"  图像的最小值: {data.min().item()}, 最大值: {data.max().item()}")

        if batch_idx == 1:  # 只打印前两个批次
            break


if __name__ == "__main__":
    main()
    """
import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import tifffile as tiff

import torch
from torchvision.utils import save_image

from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')#16, 32, 64, ...
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='FOREST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--lr', type=float, default=5e-3, 
                    help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay for the optimizer')
parser.add_argument('--reconstruction_loss_type', type=str, default='MSE', choices=['MSE', 'SSIM', 'Hybrid'],
                    help='Type of reconstruction loss to use: MSE, SSIM, or Hybrid')
parser.add_argument('--beta', type=float, default=0.1, metavar='B',
                    help='Weight for KL divergence in the VAE loss function (default: 1.0)')  # 添加beta参数
parser.add_argument('--step_size', type=int, default=3, 
                    help='Step size for learning rate scheduler')
parser.add_argument('--gamma', type=float, default=0.3, 
                    help='Gamma for learning rate scheduler')
parser.add_argument('--patience', type=int, default=5, 
                    help='Patience for early stopping')
parser.add_argument('--delta', type=float, default=0.01, 
                    help='Minimum change to qualify as improvement for early stopping')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

vae = VAE(args)
ae = AE(args)
architectures = {'AE':  ae,
                 'VAE': vae}

print(args.model)
if __name__ == "__main__":
    
    try:
        os.stat(args.results_path)
    except :
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('---------------------------------------------------------')
        sys.exit()

    try:
        
        if args.cuda:
            print("Using GPU for training")
        else:
            print("Using CPU for training")
        
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            autoenc.test(epoch)
            
            # 保存模型权重
            save_path = os.path.join(args.results_path, f'{args.model}_epoch_{epoch}.pth')
            torch.save(autoenc.model.state_dict(), save_path)
            print(f'Model weights saved at {save_path}')
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")
    
    """
    with torch.no_grad():
        # 从测试集中获取一批图像
        images = next(iter(autoenc.test_loader))  # 直接获取图像数据
        if isinstance(images, tuple):  # 如果返回的是tuple，只取第一个元素（图像数据）
            images = images[0]
        images = images.to(autoenc.device)
        
        # 对于VAE模型，需要特别处理
        if args.model == 'VAE':
            recon_images, _, _ = autoenc.model(images)  # VAE返回重建图像、均值和方差
        else:
            recon_images = autoenc.model(images)  # AE只返回重建图像
        
        # 确保结果保存目录存在
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        
        # 保存原始图像
        for i, img in enumerate(images):
            img_np = img.cpu().numpy()
            save_path = os.path.join(args.results_path, f'original_image_{i}.tif')
            tiff.imwrite(save_path, img_np)
        
        # 保存重建图像
        for i, recon_img in enumerate(recon_images):
            recon_img_np = recon_img.cpu().numpy()
            save_path = os.path.join(args.results_path, f'reconstructed_image_{i}.tif')
            tiff.imwrite(save_path, recon_img_np)
        
        # 计算和保存差异图像
        difference = torch.abs(images - recon_images)
        for i, diff_img in enumerate(difference):
            diff_img_np = diff_img.cpu().numpy()
            save_path = os.path.join(args.results_path, f'difference_image_{i}.tif')
            tiff.imwrite(save_path, diff_img_np)
        
        # 打印统计信息
        mean_difference = difference.mean().item()
        print(f'原始图像和重建图像的平均差异: {mean_difference:.4f}')
        """

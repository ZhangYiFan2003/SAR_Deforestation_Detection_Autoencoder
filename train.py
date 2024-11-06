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
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                    help='how many batches to wait before logging training status')#16, 32, 64, ...
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='FOREST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay for the optimizer')
parser.add_argument('--step_size', type=int, default=3, 
                    help='Step size for learning rate scheduler')
parser.add_argument('--gamma', type=float, default=0.3, 
                    help='Gamma for learning rate scheduler')
parser.add_argument('--patience', type=int, default=10, 
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
            should_stop = autoenc.test(epoch)  # 测试并检查EarlyStopping
            
            # 检查EarlyStopping条件
            if should_stop:
                print("Early stopping triggered. Training terminated.")
                break  # 提前结束训练
            
            # 保存模型权重
            save_path = os.path.join(args.results_path, f'{args.model}_epoch_{epoch}.pth')
            torch.save(autoenc.model.state_dict(), save_path)
            print(f'Model weights saved at {save_path}')
        
        # 训练结束后，计算误差阈值
        threshold = autoenc.calculate_threshold()
        print(f"Calculated Threshold for Anomaly Detection: {threshold:.4f}")
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")
    
    with torch.no_grad():
        
        # Training结束后导出模型
        print("Exporting model after training completion...")
        
        """
        # 导出为 ONNX 格式
        onnx_save_path = os.path.join(args.results_path, f'{args.model}_final.onnx')
        dummy_input = torch.randn(1, 2, 256, 256).to(autoenc.device)  # 根据模型的输入维度调整
        torch.onnx.export(autoenc.model, dummy_input, onnx_save_path, verbose=True)
        print(f'Model saved in ONNX format at {onnx_save_path}')
        """
        # 导出为 TorchScript 格式
        dummy_input = torch.randn(1, 2, 256, 256).to(autoenc.device)  # 根据模型的输入维度调整
        script_save_path = os.path.join(args.results_path, f'{args.model}_final.pt')
        traced_script_module = torch.jit.trace(autoenc.model, dummy_input)
        traced_script_module.save(script_save_path)
        print(f'Model saved in TorchScript format at {script_save_path}')

        # 从测试集中获取一个图像
        single_image = next(iter(autoenc.test_loader))[0]  # 直接获取第一个图像数据
        if isinstance(single_image, tuple):  # 如果返回的是tuple，只取第一个元素（图像数据）
            single_image = single_image[0]
        single_image = single_image.unsqueeze(0).to(autoenc.device)  # 添加batch维度以适应模型输入

        # 对于VAE模型，需要特别处理
        if args.model == 'VAE':
            recon_image, _, _ = autoenc.model(single_image)  # VAE返回重建图像、均值和方差
        else:
            recon_image = autoenc.model(single_image)  # AE只返回重建图像

        # 确保结果保存目录存在
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

        # 保存原始图像
        img_np = single_image.cpu().squeeze().numpy()
        save_path = os.path.join(args.results_path, 'original_image.tif')
        tiff.imwrite(save_path, img_np)

        # 保存重建图像
        recon_img_np = recon_image.cpu().squeeze().numpy()
        save_path = os.path.join(args.results_path, 'reconstructed_image.tif')
        tiff.imwrite(save_path, recon_img_np)

        # 计算和保存差异图像
        difference = torch.abs(single_image - recon_image)
        diff_img_np = difference.cpu().squeeze().numpy()
        save_path = os.path.join(args.results_path, 'difference_image.tif')
        tiff.imwrite(save_path, diff_img_np)

        # 打印统计信息
        mean_difference = difference.mean().item()
        print(f'原始图像和重建图像的平均差异: {mean_difference:.4f}')

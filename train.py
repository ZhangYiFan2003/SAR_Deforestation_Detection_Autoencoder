import argparse, os, sys
import numpy as np
import imageio
import tifffile as tiff
import torch
import random
import matplotlib.pyplot as plt
from scipy import ndimage
from torchvision.utils import save_image
from models.VAE import VAE
from models.AE import AE
from utils import get_interpolations
from datasets import ProcessedForestDataLoader  # 假设你的数据集代码保存为 dataset.py
from scipy.signal import find_peaks
from loss_distribution.loss_distribution_analyse import LossDistributionAnalysis  # 假设你的损失分布分析代码保存为 loss_analysis.py

parser = argparse.ArgumentParser(
    description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                    help='embedding size for latent space') #16, 32, 64, ...
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
                    help='Step size for learning rate scheduler StepLR')
parser.add_argument('--gamma', type=float, default=0.3, 
                    help='Gamma for learning rate scheduler StepLR')
parser.add_argument('--eta_min', type=float, default=1e-5,
                    help='Minimum learning rate in scheduler CosineAnnealingLR')
parser.add_argument('--patience', type=int, default=10, 
                    help='Patience for early stopping')
parser.add_argument('--delta', type=float, default=0.01, 
                    help='Minimum change to qualify as improvement for early stopping')
parser.add_argument('--train', action='store_true', default=True,
                    help='Choose whether to train the model (default: True)')
parser.add_argument('--test', action='store_true', default=True,
                    help='Choose whether to test the model with the latest saved weights (default: False)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

vae = VAE(args)
ae = AE(args)
architectures = {'AE': ae,
                 'VAE': vae}

# Load DataLoader
data_loader = ProcessedForestDataLoader(args)

if __name__ == "__main__":

    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('---------------------------------------------------------')
        sys.exit()

    # 实例化 LossDistributionAnalysis
    loss_analysis = LossDistributionAnalysis(autoenc.model, data_loader.train_loader,
                                             data_loader.validation_loader, autoenc.device, args)

    if args.train:
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

        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        # 在训练结束后，计算逐像素误差分布并保存对比图
        loss_analysis.calculate_pixelwise_threshold()

    if args.test:
        # 加载 "best_model.pth" 模型权重
        weight_path = os.path.join(args.results_path, "best_model.pth")
        if not os.path.exists(weight_path):
            print("No weight file named 'best_model.pth' found for testing.")
            sys.exit()

        # 加载模型时注意模型类型
        if args.model == 'VAE':
            autoenc = architectures['VAE']
        elif args.model == 'AE':
            autoenc = architectures['AE']
        else:
            print("Unsupported model type.")
            sys.exit()

        # 使用 `torch.load()` 并显式设置 `map_location`
        state_dict = torch.load(weight_path, weights_only=True, map_location=autoenc.device)
        autoenc.model.load_state_dict(state_dict)

        autoenc.model.eval()

        print(f'Loaded weights from {weight_path}')

        # 使用测试集进行评估
        if data_loader.test_loader is None:
            print("Test loader is not initialized. Please use --add-deforestation-test to add test dataset.")
            sys.exit()

        mse_losses = []

        with torch.no_grad():
            print("Evaluating model on deforestation test dataset...")
            for batch_idx, data in enumerate(data_loader.test_loader):
                data = data.to(autoenc.device)

                # AE or VAE processing
                if args.model == 'VAE':
                    recon_batch, _, _ = autoenc.model(data)
                else:
                    recon_batch = autoenc.model(data)

                # Loss computation (assuming MSE loss for this autoencoder)
                loss_fn = torch.nn.MSELoss(reduction='none')
                batch_loss = loss_fn(recon_batch, data)
                
                # 计算每个图像的整体MSE
                mse_batch = batch_loss.mean(dim=(1, 2, 3))  # 假设图像是 (B, C, H, W)，求每个图像的均值
                mse_losses.extend(mse_batch.cpu().numpy())

                if batch_idx % args.log_interval == 0:
                    print(f'Test Batch {batch_idx}: Average Loss = {mse_batch.mean().item():.4f}')

            # 转换为 NumPy 数组
            mse_losses = np.array(mse_losses)

            # 寻找第一个波峰及其后的最低点
            peaks, _ = find_peaks(mse_losses)
            if len(peaks) > 0:
                first_peak_index = peaks[0]
                # 在第一个波峰之后寻找最低点
                subsequent_values = mse_losses[first_peak_index:]
                min_index = np.argmin(subsequent_values)
                threshold_index = first_peak_index + min_index
                threshold_value = mse_losses[threshold_index]

                # 分割数据集
                forest_indices = np.where(mse_losses <= threshold_value)[0]
                deforestation_indices = np.where(mse_losses > threshold_value)[0]

                print(f'Found threshold value: {threshold_value:.4f}')
                print(f'Number of forest samples: {len(forest_indices)}')
                print(f'Number of deforestation samples: {len(deforestation_indices)}')
            else:
                print("No significant peaks found in the MSE distribution.")
            
            plt.figure(figsize=(10, 6))
            
            # 绘制MSE损失的直方图，增加bins数量以增加细节
            plt.hist(mse_losses, bins=500, color='blue', alpha=0.7)
            
            # 设置x轴和y轴的标签
            plt.xlabel('每像素的 MSE 损失', fontsize=12)
            plt.ylabel('频率', fontsize=12)
            plt.title('每像素 MSE 损失的频率分布', fontsize=14)
            
            # 设置x轴和y轴的范围
            plt.xlim(0, 0.002)  # 这里你可以调整 x 轴范围，比如从 0 到 0.002
            
            # 保存图像
            plot_path = os.path.join(args.results_path, 'mse_loss_pixelwise_distribution_zoomed.png')
            plt.savefig(plot_path)
            
            print(f'每像素 MSE 损失的放大分布图已保存至 {plot_path}')
            """
            # 随机选择 10 张图像用于重构和差异分析
            print("Randomly selecting 10 images for reconstruction...")
            all_test_images = list(data_loader.test_loader)
            selected_batches = random.sample(all_test_images, 10)

            for idx, batch in enumerate(selected_batches):
                batch = batch.to(autoenc.device)
                
                # AE or VAE processing for reconstruction
                if args.model == 'VAE':
                    recon_batch, _, _ = autoenc.model(batch)
                else:
                    recon_batch = autoenc.model(batch)

                # Save original, reconstructed, and difference images
                for img_idx in range(batch.size(0)):
                    original_img = batch[img_idx].cpu().numpy()
                    recon_img = recon_batch[img_idx].cpu().numpy()
                    diff_img = np.abs(original_img - recon_img)

                    # 保存原始图像
                    orig_save_path = os.path.join(args.results_path, f'original_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(orig_save_path, original_img)
                    # 保存重建图像
                    recon_save_path = os.path.join(args.results_path, f'reconstructed_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(recon_save_path, recon_img)
                    # 保存差异图像
                    diff_save_path = os.path.join(args.results_path, f'difference_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(diff_save_path, diff_img)

                    print(f"Image {img_idx} in batch {idx} saved: Original, Reconstructed, and Difference images.")

                # 计算每张图像的平均差异
                mean_diff = diff_img.mean()
                print(f'Batch {idx}, Image {img_idx}: Mean difference = {mean_diff:.4f}')
                """
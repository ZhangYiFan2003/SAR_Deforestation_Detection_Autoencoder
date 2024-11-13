import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tifffile as tiff

#####################################################################################################################################################

class LossDistributionAnalysis:
    def __init__(self, model, train_loader, validation_loader, test_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

#####################################################################################################################################################

    def _calculate_pixel_losses(self, loader, dataset_type, num_batches=20):
        """计算并返回给定数据加载器中的每个像素的 MSE 损失"""
        pixel_losses = []
        
        with torch.no_grad():
            print("Evaluating model on dataset...")
            
            # 随机抽样 20 个批次
            sampled_batches = random.sample(list(loader), min(num_batches, len(loader)))
            
            # 遍历采样到的批次
            for i, data in enumerate(sampled_batches):
                data = data.to(self.device)
                
                # AE 或 VAE 重建处理
                recon_batch = self.model(data)
                
                if isinstance(recon_batch, tuple):
                    recon_batch = recon_batch[0]  # 只取重建的图像部分
                
                # 计算逐像素的 MSE 误差 (形状为 B x C x H x W)
                loss_fn = torch.nn.MSELoss(reduction='none')
                batch_loss = loss_fn(recon_batch, data)
                
                # 将所有像素的误差添加到 pixel_losses 列表中
                pixel_losses.extend(batch_loss.view(-1).cpu().numpy())  # 展平成一维数组，收集所有像素的 MSE 误差
        
        # 转换为 NumPy 数组
        pixel_losses = np.array(pixel_losses)
        
        # 记录误差的直方图
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=0)
        
        return pixel_losses

#####################################################################################################################################################

    def _plot_histogram(self, data, title, xlabel, ylabel, save_path, color='blue', alpha=0.7, bins=1000, xlim=(0, 0.05)):
        """通用的绘制直方图函数"""
        plt.figure(figsize=(10, 6))
        """
        hist, bin_edges = np.histogram(data, bins=500)
        print("Histogram bin counts:", hist)
        print("Bin edges:", bin_edges)
        """
        plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black', density=True)
        plt.yscale('log')  # 将 y 轴设置为对数刻度
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        #plt.xlim(xlim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path)
        plt.close()
        print(f"{title} saved at {save_path}")

#####################################################################################################################################################

    def train_and_validation_loss_distribution(self):
        """计算训练集和测试集的逐像素误差分布，并生成直方图"""
        self.model.eval()
        
        # 计算训练集和测试集的逐像素误差
        train_pixel_losses = self._calculate_pixel_losses(self.train_loader, 'Train')
        test_pixel_losses = self._calculate_pixel_losses(self.validation_loader, 'Validation')
        
        # 绘制训练集和测试集的逐像素误差分布直方图
        train_plot_path = os.path.join(self.args.results_path, 'train_pixelwise_mse_distribution.png')
        self._plot_histogram(train_pixel_losses, 'Train Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             train_plot_path, color='blue')
        
        test_plot_path = os.path.join(self.args.results_path, 'validation_pixelwise_mse_distribution.png')
        self._plot_histogram(test_pixel_losses, 'Validation Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             test_plot_path, color='red')


#####################################################################################################################################################

    def test_loss_distribution(self):
        """用于在测试过程中计算和绘制逐像素误差分布的封装方法"""
        self.model.eval()
        
        # 计算测试集的逐像素误差
        mse_losses = self._calculate_pixel_losses(self.test_loader, 'Test')
        
        # 绘制并保存MSE损失的频率分布
        plot_path = os.path.join(self.args.results_path, 'test_pixelwise_mse_distribution.png')
        self._plot_histogram(mse_losses, 'Test Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             plot_path, color='green')
        
        # 调用重构和差异分析方法
        #self._reconstruct_and_analyze_images()

#####################################################################################################################################################

    def calculate_pixelwise_loss_distribution(self, loader, dataset_type, epoch):
        """计算给定数据加载器中逐像素的损失分布，并在 TensorBoard 中记录"""
        pixel_losses = []
        
        with torch.no_grad():
            for i, data in enumerate(random.sample(list(loader), 20)):
                data = data.to(self.device)
                recon_batch = self.model(data)
                
                pixel_loss = F.mse_loss(recon_batch, data, reduction='none')
                pixel_loss = pixel_loss.view(-1, 2, 256, 256)
                pixel_losses.append(pixel_loss.cpu().numpy())
                
        pixel_losses = np.concatenate(pixel_losses, axis=0).flatten()
        
        # 在 TensorBoard 中记录逐像素误差分布
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=epoch)

#####################################################################################################################################################

    def _reconstruct_and_analyze_images(self):
            """随机选择 10 张图像用于重构和差异分析"""
            print("Randomly selecting 10 images for reconstruction...")
            all_test_images = list(self.test_loader)
            selected_batches = random.sample(all_test_images, 10)

            for idx, batch in enumerate(selected_batches):
                batch = batch.to(self.device)
                
                # AE or VAE processing for reconstruction
                if isinstance(self.model, torch.nn.Module):  # 根据模型类型进行处理
                    recon_batch = self.model(batch)
                else:
                    recon_batch, _, _ = self.model(batch)

                # Save original, reconstructed, and difference images
                for img_idx in range(batch.size(0)):
                    original_img = batch[img_idx].cpu().numpy()
                    recon_img = recon_batch[img_idx].detach().cpu().numpy()
                    diff_img = np.abs(original_img - recon_img)

                    # 保存原始图像
                    orig_save_path = os.path.join(self.args.results_path, f'original_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(orig_save_path, original_img)
                    # 保存重建图像
                    recon_save_path = os.path.join(self.args.results_path, f'reconstructed_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(recon_save_path, recon_img)
                    # 保存差异图像
                    diff_save_path = os.path.join(self.args.results_path, f'difference_image_{idx}_{img_idx}.tif')
                    tiff.imwrite(diff_save_path, diff_img)

                    print(f"Image {img_idx} in batch {idx} saved: Original, Reconstructed, and Difference images.")

                # 计算每张图像的平均差异
                mean_diff = diff_img.mean()
                print(f'Batch {idx}, Image {img_idx}: Mean difference = {mean_diff:.4f}')

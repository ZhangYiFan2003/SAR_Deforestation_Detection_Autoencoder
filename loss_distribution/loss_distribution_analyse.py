import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class LossDistributionAnalysis:
    def __init__(self, model, train_loader, test_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')
    
    
    
    def _calculate_pixel_losses(self, loader, dataset_type):
        """计算并返回给定数据加载器中的逐像素误差"""
        pixel_losses = []
        
        with torch.no_grad():
            # 从数据加载器中随机选取20个批次
            for i, data in enumerate(random.sample(list(loader), 20)):
                data = data.to(self.device)
                recon_batch = self.model(data)
                
                # 计算逐像素的MSE误差
                pixel_loss = F.mse_loss(recon_batch, data, reduction='none')
                pixel_loss = pixel_loss.view(-1, 2, 256, 256)
                pixel_losses.append(pixel_loss.cpu().numpy())
        
        # 展平成一维数组
        pixel_losses = np.concatenate(pixel_losses, axis=0).flatten()
        
        # 记录误差的直方图
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=0)
        
        return pixel_losses
    
    
    
    def _plot_pixel_loss_distribution(self, train_pixel_losses, test_pixel_losses):
        """绘制训练集和测试集的逐像素误差对比直方图，分为两个子图保存到同一张图片中"""
        
        # 创建图形并设置子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        
        # 绘制训练集的误差直方图
        ax1.hist(train_pixel_losses, bins=100, color='blue', edgecolor='black', alpha=0.5, range=(0, 0.01))
        ax1.set_title('Train Pixel-wise MSE Loss Distribution')
        ax1.set_xlabel('MSE Loss per Pixel')
        ax1.set_ylabel('Frequency')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制测试集的误差直方图
        ax2.hist(test_pixel_losses, bins=100, color='red', edgecolor='black', alpha=0.5, range=(0, 0.01))
        ax2.set_title('Test Pixel-wise MSE Loss Distribution')
        ax2.set_xlabel('MSE Loss per Pixel')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 设置图例
        fig.suptitle('Comparison of Pixel-wise MSE Loss Distribution (Train vs Test)', fontsize=16)
        
        # 保存直方图
        hist_path = os.path.join(self.args.results_path, 'train_test_pixelwise_mse_loss_distribution.jpg')
        plt.savefig(hist_path)
        plt.close()
        print(f"Pixel-wise MSE Loss Distribution comparison saved at {hist_path}")
    
    
    
    def plot_pixel_loss_distribution_kde(self, train_pixel_losses, test_pixel_losses):
        """使用 TensorBoard 记录训练集和测试集的逐像素误差近似 KDE 分布"""
        
        # 将训练集和测试集的逐像素误差上传到 TensorBoard 的直方图
        self.writer.add_histogram('Train_Pixelwise_MSE_Loss_Distribution', train_pixel_losses, bins='auto')
        self.writer.add_histogram('Test_Pixelwise_MSE_Loss_Distribution', test_pixel_losses, bins='auto')
        
        print("Pixel-wise MSE Loss Distribution for Train and Test added to TensorBoard")
    
    
    
    def calculate_pixelwise_threshold(self):
        """计算训练集和测试集的逐像素误差分布，并生成对比直方图和 KDE 图"""
        self.model.eval()
        
        # 计算训练集和测试集的逐像素误差
        train_pixel_losses = self._calculate_pixel_losses(self.train_loader, 'Train')
        test_pixel_losses = self._calculate_pixel_losses(self.test_loader, 'Test')
        
        # 绘制训练集和测试集的逐像素误差分布直方图
        self._plot_pixel_loss_distribution(train_pixel_losses, test_pixel_losses)
        
        # 绘制逐像素误差分布的 KDE 图
        #self.plot_pixel_loss_distribution_kde(train_pixel_losses, test_pixel_losses)
    
    
    
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

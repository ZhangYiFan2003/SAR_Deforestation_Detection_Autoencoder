import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tifffile as tiff
from scipy.stats import norm

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
        self.model.eval()
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
                mse_batch = batch_loss.mean(dim=(1, 2, 3))
                pixel_losses.extend(mse_batch.cpu().numpy())
                #pixel_losses.extend(batch_loss.view(-1).cpu().numpy())  # 展平成一维数组，收集所有像素的 MSE 误差
        
        # 转换为 NumPy 数组
        pixel_losses = np.array(pixel_losses)
        
        # 记录误差的直方图
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=0)
        
        return pixel_losses

#####################################################################################################################################################

    def _plot_histogram(self, data, title, xlabel, ylabel, save_path, hyperparameters, color='blue', alpha=0.7, bins=1000, xlim=(0, 0.0015)):
        """绘制直方图并在图中包含超参数信息"""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black', density=True)
        plt.yscale('log')  # 将 y 轴设置为对数刻度
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlim(xlim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在图中添加超参数信息
        hyperparameters_text = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])
        plt.gcf().text(0.95, 0.5, hyperparameters_text, fontsize=10, ha='right', va='center', transform=plt.gcf().transFigure)
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"{title} saved at {save_path}")

#####################################################################################################################################################

    def train_and_validation_and_test_loss_distribution(self):
        """用于在测试过程中计算和绘制逐像素误差分布的封装方法"""
        self.model.eval()
        
        # 计算训练集、验证集和测试集的逐像素误差
        train_pixel_losses = self._calculate_pixel_losses(self.train_loader, 'Train')
        validation_pixel_losses = self._calculate_pixel_losses(self.validation_loader, 'Validation')
        test_pixel_losses = self._calculate_pixel_losses(self.test_loader, 'Test')
        
        # 获取超参数
        hyperparameters = {
            'batch_size': self.args.batch_size,
            'epochs': self.args.epochs,
            'embedding_size': self.args.embedding_size,
            'learning_rate': self.args.lr,
            'weight_decay': self.args.weight_decay,
            'model': self.args.model,
            'step_size': self.args.step_size,
            'gamma': self.args.gamma
            # 可以根据需要添加更多超参数
        }
        
        # 绘制训练集和测试集的逐像素误差分布直方图
        train_plot_path = os.path.join(self.args.results_path, 'train_pixelwise_mse_distribution.png')
        self._plot_histogram(train_pixel_losses, 'Train Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             train_plot_path, hyperparameters, color='blue')
        
        validation_plot_path = os.path.join(self.args.results_path, 'validation_pixelwise_mse_distribution.png')
        self._plot_histogram(validation_pixel_losses, 'Validation Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             validation_plot_path, hyperparameters, color='red')
        
        # 绘制并保存MSE损失的频率分布
        plot_path = os.path.join(self.args.results_path, 'test_pixelwise_mse_distribution.png')
        self._plot_histogram(test_pixel_losses, 'Test Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             plot_path, hyperparameters, color='green')
        """
        # 计算训练、验证和测试之间的误差差异并绘制直方图
        # 差异计算
        train_val_diff = train_pixel_losses - validation_pixel_losses[:len(train_pixel_losses)]
        train_test_diff = train_pixel_losses - test_pixel_losses[:len(train_pixel_losses)]
        val_test_diff = validation_pixel_losses - test_pixel_losses[:len(validation_pixel_losses)]
        
        # 绘制差异的直方图
        train_val_diff_path = os.path.join(self.args.results_path, 'train_validation_mse_diff_distribution.png')
        self._plot_histogram(train_val_diff, 'Train vs Validation MSE Difference Distribution', 'MSE Difference per Pixel', 'Frequency',
                             train_val_diff_path, color='purple')
        
        train_test_diff_path = os.path.join(self.args.results_path, 'train_test_mse_diff_distribution.png')
        self._plot_histogram(train_test_diff, 'Train vs Test MSE Difference Distribution', 'MSE Difference per Pixel', 'Frequency',
                             train_test_diff_path, color='orange')
        
        val_test_diff_path = os.path.join(self.args.results_path, 'validation_test_mse_diff_distribution.png')
        self._plot_histogram(val_test_diff, 'Validation vs Test MSE Difference Distribution', 'MSE Difference per Pixel', 'Frequency',
                             val_test_diff_path, color='cyan')
        
        print("All difference histograms saved.")
        """
        # 计算训练集误差的统计特征（均值和标准差）
        train_mean = np.mean(train_pixel_losses)
        train_std = np.std(train_pixel_losses)
        print(f"Train Pixel-wise MSE - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
        """
        # 使用Z-score方法确定异常阈值（例如，设定阈值为均值加上3倍标准差）
        anomaly_threshold = train_mean + 100 * train_std
        print(f"Anomaly detection threshold (Mean + 3*Std): {anomaly_threshold:.6f}")
        
        # 使用正态分布计算指定置信区间的异常阈值（例如99%置信区间）
        confidence_level = 0.99
        z_score = norm.ppf(confidence_level)
        anomaly_threshold = train_mean + z_score * train_std
        print(f"Anomaly detection threshold (99% confidence interval): {anomaly_threshold:.6f}")
        """
        anomaly_threshold = np.quantile(train_pixel_losses, 0.999)
        
        image_index = 1566 #1566-1592
        
        # 调用重构和差异分析方法
        self._reconstruct_and_analyze_images(anomaly_threshold, image_index=image_index)

#####################################################################################################################################################

    def _reconstruct_and_analyze_images(self, anomaly_threshold, image_index=None):
        """根据指定索引选择一张测试集图像进行重构和异常检测，如果未指定索引则随机选择"""
        if image_index is not None:
            print(f"Selecting image at index {image_index} from the test dataset for anomaly detection...")
        else:
            print("Randomly selecting one image in the test dataset for anomaly detection...")

        self.model.eval()
        with torch.no_grad():
            # 如果提供了图像索引
            if image_index is not None:
                dataset = self.test_loader.dataset
                if image_index < 0 or image_index >= len(dataset):
                    print(f"Image index {image_index} is out of bounds. Valid range: 0 to {len(dataset) - 1}.")
                    return
                
                # 获取指定索引的图像数据
                data = dataset[image_index]
                # 如果返回的是（图像，标签）元组，只取图像部分
                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]
                data = data.unsqueeze(0).to(self.device)  # 增加批次维度
            else:
                # 随机选择一个批次
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                # 从选定的批次中随机选择一张图像
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch with index {rand_image_index}.")
            
            # AE 或 VAE 重建处理
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]  # 只取重建的图像部分

            # 计算逐像素的 MSE 误差 (形状为 1 x C x H x W)
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data)  # 保持形状为 (1, C, H, W)
            pixel_loss = pixel_loss.squeeze(0).cpu().numpy()  # 移除批次维度，转换为 NumPy 数组

            # 计算异常检测图（像素级阈值判断）
            anomaly_map = np.ones_like(pixel_loss[0])  # 假设使用第一个通道
            anomaly_map[pixel_loss[0] > anomaly_threshold] = 0  # 异常像素置为1，其余为0

            # 保存原始图像、重建图像、差异图像和异常检测结果
            original_img = data.squeeze(0).cpu().numpy()
            recon_img = recon_data.squeeze(0).cpu().numpy()
            diff_img = np.abs(original_img - recon_img)
            """
            # 保存原始图像
            orig_save_path = os.path.join(self.args.results_path, 'original_image.tif')
            tiff.imwrite(orig_save_path, original_img)
            # 保存重建图像
            recon_save_path = os.path.join(self.args.results_path, 'reconstructed_image.tif')
            tiff.imwrite(recon_save_path, recon_img)
            # 保存差异图像
            diff_save_path = os.path.join(self.args.results_path, 'difference_image.tif')
            tiff.imwrite(diff_save_path, diff_img)
            # 保存异常检测结果（异常像素图）
            anomaly_save_path = os.path.join(self.args.results_path, 'anomaly_map.tif')
            tiff.imwrite(anomaly_save_path, anomaly_map)
            print(f"Anomaly detection result saved at {anomaly_save_path}")
            """
            # 可视化异常检测结果（可选）
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img[0], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(anomaly_map, cmap='hot')
            plt.title('Anomaly Map')
            plt.axis('off')

            plt.tight_layout()
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result.png')
            plt.savefig(vis_save_path)
            plt.close()
            print(f"Anomaly detection visualization saved at {vis_save_path}")

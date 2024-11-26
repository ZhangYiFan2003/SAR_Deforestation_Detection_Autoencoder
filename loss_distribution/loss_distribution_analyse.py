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
        }
        """
        # 绘制训练集逐像素误差分布直方图
        train_plot_path = os.path.join(self.args.results_path, 'train_pixelwise_mse_distribution.png')
        self._plot_histogram(train_pixel_losses, 'Train Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             train_plot_path, hyperparameters, color='blue')
        
        # 绘制验证集的逐像素误差分布直方图
        validation_plot_path = os.path.join(self.args.results_path, 'validation_pixelwise_mse_distribution.png')
        self._plot_histogram(validation_pixel_losses, 'Validation Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             validation_plot_path, hyperparameters, color='red')
        
        # 绘制测试集的逐像素误差分布直方图
        plot_path = os.path.join(self.args.results_path, 'test_pixelwise_mse_distribution.png')
        self._plot_histogram(test_pixel_losses, 'Test Pixel-wise MSE Distribution', 'MSE per Pixel', 'Frequency', 
                             plot_path, hyperparameters, color='green')
        """
        # 计算训练集误差的统计特征（均值和标准差）
        train_mean = np.mean(train_pixel_losses)
        train_std = np.std(train_pixel_losses)
        print(f"Train Pixel-wise MSE - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
        
        anomaly_threshold = np.quantile(train_pixel_losses, 0.99)
        
        val_image_index = None
        test_image_index = 1566 #1566-1592
        
        # 调用重构和差异分析方法
        self._reconstruct_and_analyze_images(anomaly_threshold, image_index=test_image_index)
        #self._compare_pixel_mse_histograms(val_image_index=val_image_index, test_image_index=test_image_index)

#####################################################################################################################################################

    def _reconstruct_and_analyze_images(self, anomaly_threshold, image_index=None):
        """根据指定索引选择一张测试集图像进行重构和异常检测，并整合热力图到结果中"""
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
                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]
                data = data.unsqueeze(0).to(self.device)
            else:
                # 随机选择一个批次
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch with index {rand_image_index}.")

            # AE 或 VAE 重建处理
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]

            # 计算逐像素的 MSE 误差 (形状为 1 x C x H x W)
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data)  # 保持形状为 (1, C, H, W)
            pixel_loss = pixel_loss.squeeze(0).cpu().numpy()  # 移除批次维度，转换为 NumPy 数组

            # 计算异常检测图（像素级阈值判断）
            anomaly_map = np.ones_like(pixel_loss[0])  # 假设使用第一个通道
            anomaly_map[pixel_loss[0] > anomaly_threshold] = 0  # 异常像素置为1，其余为0

            # 保存原始图像、重建图像、差异图像和异常检测结果
            original_img = data.squeeze(0).cpu().numpy()
            
            # Sum over channels to get per-pixel loss
            pixel_loss_sum = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)

            # Apply logarithm to enhance differences
            pixel_loss_sum = np.log(pixel_loss_sum + 1e-8)

            # Normalize the pixel loss to [0, 1]
            min_loss = pixel_loss_sum.min()
            max_loss = pixel_loss_sum.max()
            norm_pixel_loss = (pixel_loss_sum - min_loss) / (max_loss - min_loss + 1e-8)

            # 可视化异常检测结果（包括热力图）
            plt.figure(figsize=(12, 6))

            # 原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(original_img[0], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # 热力图
            plt.subplot(1, 2, 2)
            plt.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            plt.colorbar(label='Normalized MSE Loss')
            plt.title('Anomaly Heat Map')
            plt.axis('off')
            """
            # 异常检测图
            plt.subplot(1, 3, 3)
            plt.imshow(anomaly_map, cmap='hot')
            plt.title('Anomaly Map')
            plt.axis('off')
            """
            # 保存结果
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result_with_heatmap.png')
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"Anomaly detection visualization with heat map saved at {vis_save_path}")

#####################################################################################################################################################

    def _compare_pixel_mse_histograms(self, val_image_index=None, test_image_index=None):
        """
        从验证集和测试集中选择图像，绘制像素级 MSE 分布直方图，并将两者绘制在同一张图中。
        
        Args:
            val_image_index (int): 验证集中的图像索引。如果为 None，则随机选择。
            test_image_index (int): 测试集中的图像索引。如果为 None，则随机选择。
        """
        def _get_image_and_loss(dataset, image_index):
            """获取图像和像素级 MSE 损失"""
            if image_index is not None:
                if image_index < 0 or image_index >= len(dataset):
                    raise ValueError(f"Image index {image_index} is out of bounds. Valid range: 0 to {len(dataset) - 1}.")
                data = dataset[image_index]
            else:
                image_index = random.randint(0, len(dataset) - 1)
                data = dataset[image_index]

            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]  # 如果是 (图像, 标签) 元组，只取图像部分
            data = data.unsqueeze(0).to(self.device)

            # AE 或 VAE 重建
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]

            # 逐像素计算 MSE 损失
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()  # 合并通道，转为 NumPy 数组
            return image_index, pixel_loss

        self.model.eval()
        with torch.no_grad():
            # 从验证集获取图像和损失
            val_index, val_pixel_loss = _get_image_and_loss(self.validation_loader.dataset, val_image_index)
            print(f"Selected validation image index: {val_index}")

            # 从测试集获取图像和损失
            test_index, test_pixel_loss = _get_image_and_loss(self.test_loader.dataset, test_image_index)
            print(f"Selected test image index: {test_index}")

            # 绘制直方图
            plt.figure(figsize=(10, 6))
            plt.hist(val_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Validation Image {val_index}', color='blue', density=True)
            plt.hist(test_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Test Image {test_index}', color='orange', density=True)
            plt.yscale('log')  # 对数刻度
            plt.xlabel('MSE per Pixel', fontsize=12)
            plt.ylabel('Frequency (Log Scale)', fontsize=12)
            plt.title('Pixel MSE Distribution: Validation vs Test', fontsize=14)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # 保存图像
            histogram_save_path = os.path.join(self.args.results_path, f'pixel_mse_comparison_val_{val_index}_test_{test_index}.png')
            plt.savefig(histogram_save_path, bbox_inches='tight')
            plt.close()
            print(f"Pixel MSE histogram comparison saved at {histogram_save_path}")
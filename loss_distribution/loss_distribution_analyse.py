import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label

import os
import glob
import re
from datetime import datetime
from torch.nn import MSELoss
from PIL import Image
import torchvision.transforms as transforms
import tifffile as tiff

#####################################################################################################################################################

class LossDistributionAnalysis:
    def __init__(self, model, train_loader, validation_loader, test_loader, device, args):
        """
        Initializes the LossDistributionAnalysis class for analyzing the pixel-level loss distributions.
        
        Args:
        - model: The trained model (Autoencoder or Variational Autoencoder).
        - train_loader: DataLoader for the training set.
        - validation_loader: DataLoader for the validation set.
        - test_loader: DataLoader for the test set.
        - device: The device to run the analysis (CPU or GPU).
        - args: Additional arguments including hyperparameters and result paths.
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

#####################################################################################################################################################

    def _calculate_pixel_losses(self, loader, dataset_type, num_batches=20):
        """
        Calculates the Mean Squared Error (MSE) loss for each pixel in the dataset.
        
        Args:
        - loader: DataLoader for the dataset.
        - dataset_type: Type of dataset ('Train', 'Validation', or 'Test').
        - num_batches: Number of random batches to sample.
        
        Returns:
        - pixel_losses: Array of pixel-wise MSE losses.
        """
        pixel_losses = []
        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            print("Evaluating model on dataset...")
            
            # Randomly sample batches
            sampled_batches = random.sample(list(loader), min(num_batches, len(loader)))
            
            # Iterate over the sampled batches
            for i, data in enumerate(sampled_batches):
                data = data.to(self.device)
                
                # Forward pass to reconstruct the image
                recon_batch = self.model(data)
                
                if isinstance(recon_batch, tuple):
                    # Extract the reconstructed image if tuple is returned
                    recon_batch = recon_batch[0]  
                
                # Calculate pixel-wise MSE loss (shape: Batch x Channel x Height x Width)
                loss_fn = torch.nn.MSELoss(reduction='none')
                batch_loss = loss_fn(recon_batch, data)
                
                # Collect batch-wise mean losses
                mse_batch = batch_loss.mean(dim=(1, 2, 3))
                pixel_losses.extend(mse_batch.cpu().numpy())
                #pixel_losses.extend(batch_loss.view(-1).cpu().numpy())  # 展平成一维数组，收集所有像素的 MSE 误差
        
        # Convert losses to NumPy array
        pixel_losses = np.array(pixel_losses)
        
        # Log pixel loss distribution as a histogram in TensorBoard
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=0)
        
        return pixel_losses

#####################################################################################################################################################

    def _plot_histogram(self, data, title, xlabel, ylabel, save_path, hyperparameters, color='blue', alpha=0.7, bins=1000, xlim=(0, 0.0015)):
        """
        Plots a histogram of the data with additional hyperparameter information.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black', density=True)
        # Set y-axis to logarithmic scale
        plt.yscale('log')  
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlim(xlim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate the plot with hyperparameters
        hyperparameters_text = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])
        plt.gcf().text(0.95, 0.5, hyperparameters_text, fontsize=10, ha='right', va='center', transform=plt.gcf().transFigure)
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"{title} saved at {save_path}")

#####################################################################################################################################################

    def train_and_validation_and_test_loss_distribution(self):
        """
        Wrapper function to compute and plot pixel-wise loss distributions for training, validation, and test datasets.
        """
        self.model.eval()
        
        # Compute pixel-wise MSE losses for each dataset
        train_pixel_losses = self._calculate_pixel_losses(self.train_loader, 'Train')
        
        # Collect hyperparameters for annotation in plots
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
        
        # Log and print statistics for training losses
        train_mean = np.mean(train_pixel_losses)
        train_std = np.std(train_pixel_losses)
        print(f"Train Pixel-wise MSE - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
        
        #(Not verified method) Set an anomaly threshold based on the 99th percentile of training losses
        anomaly_threshold = np.quantile(train_pixel_losses, 0.99)
        
        val_image_index = None
        test_image_index = 1566 #2022.9.19: with index from 1566 to 1592
        
        # Analyze images for anomaly detection based on this threshold
        self._reconstruct_and_analyze_images(anomaly_threshold, image_index=test_image_index)
        self._reconstruct_and_analyze_images_over_time(anomaly_threshold=0.5)
        #self._compare_pixel_mse_histograms(val_image_index=val_image_index, test_image_index=test_image_index)

#####################################################################################################################################################

    def _reconstruct_and_analyze_images(self, anomaly_threshold, image_index=None):
        """
        Reconstructs and analyzes a specific image for anomaly detection.
        
        Args:
        - anomaly_threshold: Pixel-level anomaly detection threshold.
        - image_index: Index of the image to analyze from the test dataset.
        """
        if image_index is not None:
            print(f"Selecting image at index {image_index} from the test dataset for anomaly detection...")
        else:
            print("Randomly selecting one image in the test dataset for anomaly detection...")
        
        self.model.eval()
        with torch.no_grad():
            # Fetch the specified image from the dataset
            if image_index is not None:
                dataset = self.test_loader.dataset
                if image_index < 0 or image_index >= len(dataset):
                    print(f"Image index {image_index} is out of bounds. Valid range: 0 to {len(dataset) - 1}.")
                    return
                
                # Retrieve image at the specified index
                data = dataset[image_index]
                if isinstance(data, tuple) or isinstance(data, list):
                    data = data[0]
                data = data.unsqueeze(0).to(self.device)
            else:
                # Randomly select a batch from the test loader
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch with index {rand_image_index}.")
            
            # Forward pass to reconstruct the image
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
            
            # Calculate pixel-wise MSE losses
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data)  
            # Convert to NumPy array
            pixel_loss = pixel_loss.squeeze(0).cpu().numpy()  
            
            # Save original images, reconstructed images, difference images, and anomaly detection results
            original_img = data.squeeze(0).cpu().numpy()
            
            # Generate anomaly heatmap
            pixel_loss_sum = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
            # Apply logarithm to enhance differences
            pixel_loss_sum = np.log(pixel_loss_sum + 1e-8)
            # Normalize the pixel loss to [0, 1]
            min_loss = np.percentile(pixel_loss_sum, 1)
            max_loss = np.percentile(pixel_loss_sum, 99)
            clipped_loss = np.clip(pixel_loss_sum, min_loss, max_loss)
            norm_pixel_loss = (clipped_loss - min_loss) / (max_loss - min_loss + 1e-8)
            
            #Kmeans 2 classes clustering
            # Apply KMeans clustering to classify normal and anomalous pixels
            flattened_loss = norm_pixel_loss.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_loss)
            anomaly_labels = kmeans.labels_.reshape(norm_pixel_loss.shape)
            
            # Post-process to ensure spatial continuity using connected component analysis
            # Assume cluster with higher mean loss is anomaly
            cluster_mean_loss = [norm_pixel_loss[anomaly_labels == i].mean() for i in range(2)]
            anomaly_cluster = cluster_mean_loss.index(max(cluster_mean_loss))
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
            
            # Label connected components in the binary anomaly map
            labeled_map, num_features = label(binary_anomaly_map)
            
            # Filter out small connected components (e.g., less than 50 pixels)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 30:  # Threshold for spatial continuity
                    filtered_anomaly_map[component] = 1
            
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_img[0], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            plt.colorbar(label='Normalized MSE Loss')
            plt.title('Anomaly Heat Map')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            plt.title('Semantic Anomaly Map')
            plt.axis('off')
            
            # Save the visualization
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result.png')
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"Anomaly detection visualization with refined deforestation clustering saved at {vis_save_path}")

#####################################################################################################################################################

    def _reconstruct_and_analyze_images_over_time(self, anomaly_threshold, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif", num_images=30):
        """
        重建并分析同一地理位置的多张图像，生成热力图和异常语义图，并绘制随时间变化的可视化结果。

        参数:
        - anomaly_threshold: 像素级异常检测阈值。
        - base_filename_part: 文件名的共同部分。
        - suffix: 文件名的后缀部分。
        - num_images: 要加载和分析的图像数量（默认15张）。
        """
        # 定义图像目录
        image_dir = "/home/yifan/Documents/data/forest/test/processed"

        # 构建文件匹配的正则表达式
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)

        if len(image_paths) == 0:
            print("未找到匹配的图像文件。请检查文件路径和命名格式。")
            return

        # 提取日期并排序
        def extract_date(path):
            match = re.search(r'D_(\d{8})T', path)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d")
            else:
                return datetime.min  # 如果无法提取日期，放在最前面

        image_paths.sort(key=extract_date)

        # 选择前 num_images 张图像
        selected_image_paths = image_paths[:num_images]

        if len(selected_image_paths) < num_images:
            print(f"找到的图像数量少于 {num_images} 张：{len(selected_image_paths)} 张。")
            return

        print(f"已选择 {num_images} 张图像进行分析，按照时间顺序排序。")

        # 定义图像转换，与 ProcessedForestDataset 中的转换保持一致
        transform = transforms.Compose([
            # 在这里添加任何必要的转换，例如归一化
            # 目前没有应用任何转换，与数据集加载器保持一致
        ])
        
        # 准备绘图，num_images 行 2 列（原始图像和热力图）
        fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))
        
        min_val = None
        max_val = None
        
        for idx, img_path in enumerate(selected_image_paths):
            # 加载图像
            combined_image = tiff.imread(img_path)  # 读取多通道 TIFF 图像
            #print(f"加载图像 {img_path} 的形状为: {combined_image.shape}")
            
            # 处理图像维度，确保为 (C, H, W) 并且有2个通道
            if combined_image.ndim == 2:
                # 单通道 (H, W) 转为 (1, H, W)
                combined_image = combined_image[np.newaxis, ...]
            elif combined_image.ndim == 3:
                if combined_image.shape[0] == 2:
                    # (C, H, W) 已经是正确的格式
                    pass
                elif combined_image.shape[-1] == 2:
                    # (H, W, C) 转为 (C, H, W)
                    combined_image = np.transpose(combined_image, (2, 0, 1))
                else:
                    raise ValueError(f"期望的通道数为2，但在文件 {img_path} 中找到了 {combined_image.shape[-1]} 个通道。")
            else:
                raise ValueError(f"图像维度不正确：{combined_image.ndim}，文件路径：{img_path}")
            
            if combined_image.shape[0] != 2:
                raise ValueError(f"期望的通道数为2，但在文件 {img_path} 中找到了 {combined_image.shape[0]} 个通道。")
            
            # 将图像转换为 Tensor
            img_tensor = torch.from_numpy(combined_image).float()
            
            # 归一化（如果需要）
            if min_val is not None and max_val is not None:
                img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)  # 将值限制在 [0, 1] 之间

            # 应用转换（如果有）
            if transform:
                img_tensor = transform(img_tensor)

            # 添加 batch 维度并移动到设备
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # 模型推理
            self.model.eval()
            with torch.no_grad():
                recon = self.model(img_tensor)
                if isinstance(recon, tuple):
                    recon = recon[0]

            # 计算像素级 MSE 损失
            loss_fn = MSELoss(reduction='none')
            pixel_loss = loss_fn(recon, img_tensor)
            # 对所有通道求和，得到 (H, W) 的损失图
            pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()

            # 生成热力图
            pixel_loss_sum = np.log(pixel_loss_sum + 1e-8)  # 应用对数以增强差异
            min_loss = np.percentile(pixel_loss_sum, 1)
            max_loss = np.percentile(pixel_loss_sum, 99)
            clipped_loss = np.clip(pixel_loss_sum, min_loss, max_loss)
            norm_pixel_loss = (clipped_loss - min_loss) / (max_loss - min_loss + 1e-8)

            # KMeans 聚类，将像素分为正常和异常两类
            flattened_loss = norm_pixel_loss.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_loss)
            anomaly_labels = kmeans.labels_.reshape(norm_pixel_loss.shape)

            # 确定异常类别（均值较大的类别）
            cluster_mean_loss = [norm_pixel_loss[anomaly_labels == i].mean() for i in range(2)]
            anomaly_cluster = cluster_mean_loss.index(max(cluster_mean_loss))
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)

            # 连通组件分析，过滤小的异常区域
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 30:  # 空间连续性阈值
                    filtered_anomaly_map[component] = 1

            # **两通道按像素相加，生成单通道图像**
            summed_image = combined_image.sum(axis=0)  # 形状从 (2, H, W) -> (H, W)

            # 将相加后的图像进行归一化
            if min_val is not None and max_val is not None:
                summed_image = (summed_image - min_val) / (max_val - min_val + 1e-8)
                summed_image = np.clip(summed_image, 0.0, 1.0)  # 将值限制在 [0, 1] 之间

            # **绘制相加后的原始图像**
            ax_orig = axes[idx, 0] if num_images > 1 else axes[0]
            ax_orig.imshow(summed_image, cmap='gray')
            ax_orig.set_title(f'original image {idx+1}')
            ax_orig.axis('off')

            # **绘制热力图**
            ax_heat = axes[idx, 1] if num_images > 1 else axes[1]
            heatmap = ax_heat.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            ax_heat.set_title(f'anomaly heat map {idx+1}')
            ax_heat.axis('off')
            # 为每个热力图添加单独的颜色条
            plt.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04, label=' MSE loss')

        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_over_time.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"随时间变化的异常检测结果已保存到 {vis_save_path}")

#####################################################################################################################################################

    def _compare_pixel_mse_histograms(self, val_image_index=None, test_image_index=None):
        """
        从验证集和测试集中选择图像，绘制像素级 MSE 分布直方图，并将两者绘制在同一张图中。
        
        Args:
            val_image_index (int): 验证集中的图像索引。如果为 None，则随机选择。
            test_image_index (int): 测试集中的图像索引。如果为 None，则随机选择。
        """
        def _get_image_and_loss(dataset, image_index):
            if image_index is not None:
                if image_index < 0 or image_index >= len(dataset):
                    raise ValueError(f"Image index {image_index} is out of bounds. Valid range: 0 to {len(dataset) - 1}.")
                data = dataset[image_index]
            else:
                image_index = random.randint(0, len(dataset) - 1)
                data = dataset[image_index]
            
            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]  
            data = data.unsqueeze(0).to(self.device)
            
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
            
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()  # 合并通道，转为 NumPy 数组
            return image_index, pixel_loss
        
        self.model.eval()
        with torch.no_grad():
            val_index, val_pixel_loss = _get_image_and_loss(self.validation_loader.dataset, val_image_index)
            print(f"Selected validation image index: {val_index}")
            
            test_index, test_pixel_loss = _get_image_and_loss(self.test_loader.dataset, test_image_index)
            print(f"Selected test image index: {test_index}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(val_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Validation Image {val_index}', color='blue', density=True)
            plt.hist(test_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Test Image {test_index}', color='orange', density=True)
            plt.yscale('log')  
            plt.xlabel('MSE per Pixel', fontsize=12)
            plt.ylabel('Frequency (Log Scale)', fontsize=12)
            plt.title('Pixel MSE Distribution: Validation vs Test', fontsize=14)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            histogram_save_path = os.path.join(self.args.results_path, f'pixel_mse_comparison_val_{val_index}_test_{test_index}.png')
            plt.savefig(histogram_save_path, bbox_inches='tight')
            plt.close()
            print(f"Pixel MSE histogram comparison saved at {histogram_save_path}")
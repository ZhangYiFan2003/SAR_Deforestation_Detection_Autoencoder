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
        #self._reconstruct_and_analyze_images(anomaly_threshold, image_index=test_image_index)
        self._reconstruct_and_analyze_images_over_time(target_date="20220721")
        #self.plot_pixel_error_histogram()
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

    def _reconstruct_and_analyze_images_over_time(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_256_512_fused.tif"):
        """
        重建并分析特定日期及其前后五天的图像，生成热力图和异常语义图，计算语义异常图之间的差异，并绘制随时间变化的可视化结果。
        
        参数:
        - target_date: 目标日期，格式为 "YYYYMMDD"。
        - base_filename_part: 文件名的共同部分。
        - suffix: 文件名的后缀部分。
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
        
        # 转换目标日期为 datetime 对象
        try:
            target_datetime = datetime.strptime(target_date, "%Y%m%d")
        except ValueError:
            print("目标日期格式不正确。请使用 'YYYYMMDD' 格式。")
            return
        
        # 查找目标日期的索引
        dates = [extract_date(path) for path in image_paths]
        if target_datetime not in dates:
            print(f"未找到目标日期 {target_date} 的图像。")
            return
        
        target_index = dates.index(target_datetime)
        
        # 选择目标图像及其前后五张图像
        start_index = max(target_index - 5, 0)
        end_index = min(target_index + 5 + 1, len(image_paths))  # +1 因为切片不包括 end_index
        selected_image_paths = image_paths[start_index:end_index]
        
        # 检查是否选中11张图像
        if len(selected_image_paths) < 11:
            print(f"选择的图像数量少于11张：{len(selected_image_paths)} 张。")
            return
        
        print(f"已选择日期 {target_date} 及其前后5天的11张图像进行分析，按照时间顺序排序。")
        
        # 定义图像转换，与 ProcessedForestDataset 中的转换保持一致
        transform = transforms.Compose([
            # 在这里添加任何必要的转换，例如归一化
            # 目前没有应用任何转换，与数据集加载器保持一致
        ])
        
        num_images = len(selected_image_paths)
        
        # 准备绘图，4 行 num_images 列（原始图像、热力图、语义异常图、差异图）
        fig, axes = plt.subplots(4, num_images, figsize=(num_images * 4, 16))
        
        min_val = None
        max_val = None
        
        mse_min = 0
        mse_max = 1000
        
        # 收集所有图像的像素级误差
        all_pixel_errors = []
        pixel_loss_sums = []  # 存储每张图像的 pixel_loss_sum
        image_dates = []      # 存储每张图像的日期
        summed_images = []    # 存储每张图像的 summed_image
        semantic_anomaly_maps = []  # 存储每张图像的语义异常图
        
        print("开始计算选择的11张图像的像素级误差...")
        
        for idx, img_path in enumerate(selected_image_paths):
            try:
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
                
                # 收集所有像素误差
                all_pixel_errors.extend(pixel_loss_sum.flatten())
                pixel_loss_sums.append(pixel_loss_sum)
                
                # 记录图像日期
                current_date = extract_date(img_path).strftime("%Y-%m-%d")
                image_dates.append(current_date)
                
                # 生成并存储 summed_image
                summed_image = combined_image.sum(axis=0)  # 形状从 (2, H, W) -> (H, W)
                if min_val is not None and max_val is not None:
                    summed_image = (summed_image - min_val) / (max_val - min_val + 1e-8)
                    summed_image = np.clip(summed_image, 0.0, 1.0)  # 将值限制在 [0, 1] 之间
                summed_images.append(summed_image)
                
                if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_image_paths):
                    print(f"已处理 {idx + 1} / {len(selected_image_paths)} 张图像。")
            
            except Exception as e:
                print(f"处理图像 {img_path} 时出错：{e}")
                continue
        
        # 转换为 NumPy 数组
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        
        # 训练全局GMM
        print("开始训练全局GMM...")
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_errors)
        
        # 确定异常类别（均值较大的组件）
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        print(f"异常类别为GMM的组件 {anomaly_cluster}，均值为 {component_means[anomaly_cluster]:.4f}")
        
        # 开始绘制每张图像的热力图和语义异常图
        for idx in range(num_images):
            pixel_loss_sum = pixel_loss_sums[idx]
            current_date = image_dates[idx]
            summed_image = summed_images[idx]
            
            # 使用GMM预测标签
            pixel_losses = pixel_loss_sum.flatten().reshape(-1, 1)
            predicted_labels = gmm.predict(pixel_losses)
            anomaly_labels = predicted_labels.reshape(pixel_loss_sum.shape)
            
            # 生成二值化的异常图
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
            semantic_anomaly_maps.append(binary_anomaly_map)
            
            # 连通组件分析，过滤小的异常区域
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 50:  # 空间连续性阈值
                    filtered_anomaly_map[component] = 1
            
            # 存储过滤后的语义异常图
            semantic_anomaly_maps[-1] = filtered_anomaly_map
            
            # 生成热力图，使用全局的边界值
            # 这里假设您已经通过直方图确定了 min_loss 和 max_loss
            # 您可以根据实际情况调整这些值
            clipped_loss = np.clip(pixel_loss_sum, mse_min, mse_max)
            norm_pixel_loss = (clipped_loss - mse_min) / (mse_max - mse_min + 1e-8)
            
            # 绘制原始图像
            ax_orig = axes[0, idx]
            ax_orig.imshow(summed_image, cmap='gray')
            ax_orig.set_title(f'原始图像 {current_date}')
            ax_orig.axis('off')
            
            # 绘制热力图
            ax_heat = axes[1, idx]
            heatmap = ax_heat.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            ax_heat.set_title(f'热力图 {current_date}')
            ax_heat.axis('off')
            # 为每个热力图添加单独的颜色条
            plt.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04, label='MSE loss')
            
            # 绘制语义异常图
            ax_cluster = axes[2, idx]
            cluster_map = semantic_anomaly_maps[-1]  # 二值图，1表示异常，0表示正常
            ax_cluster.imshow(cluster_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            ax_cluster.set_title(f'语义异常图 {current_date}')
            ax_cluster.axis('off')
        
        # 计算并绘制差异图（正常 -> 异常）
        print("开始计算语义异常图之间的差异...")
        difference_maps = []
        for i in range(1, len(semantic_anomaly_maps)):
            previous_map = semantic_anomaly_maps[i - 1]
            current_map = semantic_anomaly_maps[i]
            # 计算差异：前一张正常（0）且当前异常（1）
            difference_map = np.logical_and(previous_map == 0, current_map == 1).astype(int)
            difference_maps.append(difference_map)
        
        # 为差异图准备绘图区域
        # 如果有差异图，添加第四行
        if difference_maps:
            # 扩展子图到4行
            fig, axes = plt.subplots(4, num_images, figsize=(num_images * 4, 20))
            # 重新绘制前三行
            for idx in range(num_images):
                # 原始图像
                ax_orig = axes[0, idx]
                ax_orig.imshow(summed_images[idx], cmap='gray')
                ax_orig.set_title(f'original image {image_dates[idx]}')
                ax_orig.axis('off')
                
                # 热力图
                ax_heat = axes[1, idx]
                clipped_loss = np.clip(pixel_loss_sums[idx], mse_min, mse_max)
                norm_pixel_loss = (clipped_loss - mse_min) / (mse_max - mse_min + 1e-8)
                heatmap = ax_heat.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
                ax_heat.set_title(f'heat map {image_dates[idx]}')
                ax_heat.axis('off')
                plt.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04, label='MSE loss')
                
                # 语义异常图
                ax_cluster = axes[2, idx]
                cluster_map = semantic_anomaly_maps[idx]
                ax_cluster.imshow(cluster_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
                ax_cluster.set_title(f'semantic anomaly map {image_dates[idx]}')
                ax_cluster.axis('off')
            
            # 绘制差异图
            for idx, diff_map in enumerate(difference_maps):
                current_date = image_dates[idx + 1]  # 差异对应的是后一个日期
                ax_diff = axes[3, idx]
                ax_diff.imshow(diff_map, cmap='bone', vmin=0, vmax=1)
                ax_diff.set_title(f'chanegment {current_date}')
                ax_diff.axis('off')
            
            # 对最后一张没有差异图的图像进行空白处理
            if num_images > len(difference_maps):
                ax_diff = axes[3, -1]
                ax_diff.axis('off')
            
            plt.tight_layout()
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_over_time_with_differences.png')
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"带有差异图的随时间变化的异常检测结果已保存到 {vis_save_path}")

#####################################################################################################################################################

    def plot_pixel_error_histogram(self, image_dir="/home/yifan/Documents/data/forest/test/processed", num_bins=1000):
        """
        计算测试数据集中所有图像的像素级MSE，绘制误差直方图，并计算最小值和最大值。
        
        参数:
        - image_dir: 测试图像存储目录。
        - num_bins: 直方图的柱子数量（默认100）。
        """
        
        # 获取所有TIFF图像文件
        pattern = os.path.join(image_dir, "*.tif")
        image_paths = glob.glob(pattern)
        
        if len(image_paths) == 0:
            print("未找到测试数据集中的TIFF图像文件。请检查文件路径和文件格式。")
            return
        
        # 定义图像转换，与 ProcessedForestDataset 中的转换保持一致
        transform = transforms.Compose([
            # 在这里添加任何必要的转换，例如归一化
            # 目前没有应用任何转换，与数据集加载器保持一致
        ])
        
        loss_fn = MSELoss(reduction='none')
        all_pixel_errors = []
        
        print("开始计算所有测试图像的像素级误差...")
        
        for idx, img_path in enumerate(image_paths):
            try:
                # 加载图像
                combined_image = tiff.imread(img_path)  # 读取多通道 TIFF 图像
                
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
                # 根据您的需求，这里暂时不进行归一化操作
                # 如果需要，可以根据实际情况添加
                
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
                pixel_loss = loss_fn(recon, img_tensor)
                # 对所有通道求和，得到 (H, W) 的损失图
                pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
                
                # 收集所有像素误差
                all_pixel_errors.extend(pixel_loss_sum.flatten())
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
                    print(f"{idx + 1} / {len(image_paths)} images traité。")
                    
            except Exception as e:
                print(f"处理图像 {img_path} 时出错：{e}")
                continue
            
        # 转换为 NumPy 数组
        all_pixel_errors = np.array(all_pixel_errors)
        
        # 计算最小值和最大值
        min_mse = all_pixel_errors.min()
        max_mse = all_pixel_errors.max()
        print(f"MSE minimum au niveau des pixels de toutes les images de test: {min_mse}")
        print(f"MSE maximum au niveau des pixels de toutes les images de test: {max_mse}")
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(all_pixel_errors, bins=num_bins, color='skyblue', edgecolor='black')
        plt.title('Histogramme d erreur au niveau des pixels pour toutes les images de test')
        plt.xlabel('Erreur MSE au niveau du pixel')
        plt.ylabel('Nombre de pixels')
        plt.grid(True)
        hist_save_path = os.path.join(self.args.results_path, 'pixel_error_histogram.png')
        plt.savefig(hist_save_path, bbox_inches='tight')
        plt.close()
        print(f"像素级误差直方图已保存到 {hist_save_path}")

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
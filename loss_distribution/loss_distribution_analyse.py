import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from scipy.ndimage import label

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
        validation_pixel_losses = self._calculate_pixel_losses(self.validation_loader, 'Validation')
        test_pixel_losses = self._calculate_pixel_losses(self.test_loader, 'Test')
        
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
            
            # 保存原始图像、重建图像、差异图像和异常检测结果
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
            
            # Apply KMeans clustering to classify pixels into three categories
            flattened_loss = norm_pixel_loss.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=0).fit(flattened_loss)
            anomaly_labels = kmeans.labels_.reshape(norm_pixel_loss.shape)
            
            # Post-process to assign each cluster a semantic meaning
            cluster_mean_loss = [norm_pixel_loss[anomaly_labels == i].mean() for i in range(3)]
            sorted_clusters = np.argsort(cluster_mean_loss)  # Sort clusters by their mean MSE loss
            forest_label = sorted_clusters[0]  # Lowest mean loss (forest)
            deforestation_label = sorted_clusters[1]  # Middle mean loss (new deforestation)
            no_forest_label = sorted_clusters[2]  # Highest mean loss (originally no forest)
            
            # Generate semantic anomaly map
            semantic_anomaly_map = np.zeros_like(anomaly_labels)
            semantic_anomaly_map[anomaly_labels == forest_label] = 0
            semantic_anomaly_map[anomaly_labels == deforestation_label] = 1
            semantic_anomaly_map[anomaly_labels == no_forest_label] = 2
            
            # 可视化异常检测结果（包括热力图）
            plt.figure(figsize=(18, 6))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(original_img[0], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # 热力图
            plt.subplot(1, 3, 2)
            plt.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            plt.colorbar(label='Normalized MSE Loss')
            plt.title('Anomaly Heat Map')
            plt.axis('off')
            
            # 语义分类图
            plt.subplot(1, 3, 3)
            plt.imshow(semantic_anomaly_map, cmap='tab10', alpha=0.8)
            plt.title('Semantic Anomaly Map (3 Categories)')
            plt.axis('off')
            
            # Save the visualization
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result_with_heatmap.png')
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"Anomaly detection visualization with semantic map saved at {vis_save_path}")

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
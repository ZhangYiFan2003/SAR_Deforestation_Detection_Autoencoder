import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label
import tifffile as tiff
import glob
import re
from datetime import datetime
import torchvision.transforms as transforms

#####################################################################################################################################################

class AnomalyDetection:
    def reconstruct_and_analyze_images(self, anomaly_threshold, image_index=None):
        if image_index is not None:
            print(f"Selecting image at index {image_index} from the test dataset...")
        else:
            print("Randomly selecting one image...")
        self.model.eval()
        with torch.no_grad():
            if image_index is not None:
                dataset = self.test_loader.dataset
                if image_index < 0 or image_index >= len(dataset):
                    print(f"Image index {image_index} out of bounds.")
                    return
                data = dataset[image_index]
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.unsqueeze(0).to(self.device)
            else:
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch index {rand_image_index}.")

            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
            loss_fn = MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data)  
            pixel_loss = pixel_loss.squeeze(0).cpu().numpy()
            original_img = data.squeeze(0).cpu().numpy()
            pixel_loss_sum = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()
            pixel_loss_sum = np.log(pixel_loss_sum + 1e-8)
            min_loss = np.percentile(pixel_loss_sum, 1)
            max_loss = np.percentile(pixel_loss_sum, 99)
            clipped_loss = np.clip(pixel_loss_sum, min_loss, max_loss)
            norm_pixel_loss = (clipped_loss - min_loss) / (max_loss - min_loss + 1e-8)
            flattened_loss = norm_pixel_loss.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_loss)
            anomaly_labels = kmeans.labels_.reshape(norm_pixel_loss.shape)
            cluster_mean_loss = [norm_pixel_loss[anomaly_labels == i].mean() for i in range(2)]
            anomaly_cluster = cluster_mean_loss.index(max(cluster_mean_loss))
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 30:
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
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result.png')
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"Anomaly detection visualization saved at {vis_save_path}")

#####################################################################################################################################################

    def reconstruct_and_analyze_images_by_time_sequence(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("未找到匹配的图像文件。")
            return
        def extract_date(path):
            match = re.search(r'D_(\d{8})T', path)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d")
            else:
                return datetime.min
        image_paths.sort(key=extract_date)
        try:
            target_datetime = datetime.strptime(target_date, "%Y%m%d")
        except ValueError:
            print("目标日期格式不正确。")
            return
        dates = [extract_date(p) for p in image_paths]
        if target_datetime not in dates:
            print(f"未找到目标日期 {target_date} 的图像。")
            return
        target_index = dates.index(target_datetime)
        start_index = max(target_index - 5, 0)
        end_index = min(target_index + 6, len(image_paths))
        selected_image_paths = image_paths[start_index:end_index]
        if len(selected_image_paths) < 11:
            print(f"选择的图像数量少于11张：{len(selected_image_paths)} 张。")
            return
        print(f"已选择日期 {target_date} 及其前后5天的11张图像。")
        transform = transforms.Compose([])
        num_images = len(selected_image_paths)
        fig, axes = plt.subplots(5, num_images, figsize=(num_images * 4, 25))
        mse_min = 0
        mse_max = 1000
        all_pixel_errors = []
        pixel_loss_sums = []
        image_dates = []
        summed_images = []
        semantic_anomaly_maps = []
        print("开始计算选择的11张图像的像素级误差...")
        for idx, img_path in enumerate(selected_image_paths):
            try:
                combined_image = tiff.imread(img_path)
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] != 2 and combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))
                if combined_image.shape[0] != 2:
                    raise ValueError("通道数错误")
                img_tensor = torch.from_numpy(combined_image).float()
                if transform:
                    img_tensor = transform(img_tensor)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(img_tensor)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                loss_fn = MSELoss(reduction='none')
                pixel_loss = loss_fn(recon, img_tensor)
                pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
                all_pixel_errors.extend(pixel_loss_sum.flatten())
                pixel_loss_sums.append(pixel_loss_sum)
                current_date = extract_date(img_path).strftime("%Y-%m-%d")
                image_dates.append(current_date)
                summed_image = combined_image.sum(axis=0)
                if transform:
                    summed_image = (summed_image - summed_image.min()) / (summed_image.max() - summed_image.min() + 1e-8)
                    summed_image = np.clip(summed_image, 0.0, 1.0)
                summed_images.append(summed_image)
                if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_image_paths):
                    print(f"已处理 {idx + 1} / {len(selected_image_paths)} 张图像。")
            except Exception as e:
                print(f"处理图像 {img_path} 时出错：{e}")
                continue
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("开始训练全局GMM...")
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_errors)
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        print(f"异常类别为GMM的组件 {anomaly_cluster}，均值为 {component_means[anomaly_cluster]:.4f}")
        H, W = pixel_loss_sums[0].shape
        previous_anomalies = np.zeros((H, W), dtype=bool)
        for idx in range(num_images):
            pixel_loss_sum = pixel_loss_sums[idx]
            current_date = image_dates[idx]
            summed_image = summed_images[idx]
            pixel_losses = pixel_loss_sum.flatten().reshape(-1, 1)
            predicted_labels = gmm.predict(pixel_losses)
            anomaly_labels = predicted_labels.reshape(pixel_loss_sum.shape)
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 50:
                    filtered_anomaly_map[component] = 1
            semantic_anomaly_maps.append(filtered_anomaly_map)
            current_deforestation_map = (filtered_anomaly_map == 1) & (~previous_anomalies)
            ancient_deforestation_map = (filtered_anomaly_map == 1) & (previous_anomalies)
            previous_anomalies = previous_anomalies | (filtered_anomaly_map == 1)
            clipped_loss = np.clip(pixel_loss_sum, mse_min, mse_max)
            norm_pixel_loss = (clipped_loss - mse_min) / (mse_max - mse_min + 1e-8)
            ax_orig = axes[0, idx]
            ax_orig.imshow(summed_image, cmap='gray')
            ax_orig.set_title(f'original image {current_date}')
            ax_orig.axis('off')
            ax_heat = axes[1, idx]
            heatmap = ax_heat.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            ax_heat.set_title(f'heat map {current_date}')
            ax_heat.axis('off')
            plt.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04, label='MSE loss')
            ax_cluster = axes[2, idx]
            cluster_map = semantic_anomaly_maps[-1]
            ax_cluster.imshow(cluster_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            ax_cluster.set_title(f'semantic anomaly map {current_date}')
            ax_cluster.axis('off')
            ax_current = axes[3, idx]
            ax_current.imshow(current_deforestation_map, cmap='Reds', vmin=0, vmax=1)
            ax_current.set_title(f'current deforestation {current_date}')
            ax_current.axis('off')
            ax_ancient = axes[4, idx]
            ax_ancient.imshow(ancient_deforestation_map, cmap='Blues', vmin=0, vmax=1)
            ax_ancient.set_title(f'ancient deforestation {current_date}')
            ax_ancient.axis('off')
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_over_time_with_classification.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"带有异常分类的结果已保存到 {vis_save_path}")

#####################################################################################################################################################

    def reconstruct_and_analyze_images_by_clustering(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_256_512_fused.tif"):
        
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
        mse_max = 1050
        
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
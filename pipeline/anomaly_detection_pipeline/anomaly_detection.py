import os, re, glob
import numpy as np
import torch
import random
import tifffile as tiff
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.nn import MSELoss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Polygon

#####################################################################################################################################################

class AnomalyDetection:
    def reconstruct_and_analyze_images(self, anomaly_threshold, image_index=None):
        """
        重建并分析单张图像以检测异常。
        
        Args:
            anomaly_threshold (float): 异常检测的阈值。
            image_index (int, optional): 指定要分析的图像索引。如果为 None，则随机选择一张图像。
        """
        # 选择图像
        if image_index is not None:
            print(f"Selecting image at index {image_index} from the test dataset...")
        else:
            print("Randomly selecting one image...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        with torch.no_grad():
            if image_index is not None:
                dataset = self.test_loader.dataset
                if image_index < 0 or image_index >= len(dataset):
                    print(f"Image index {image_index} out of bounds.")
                    return
                
                # 获取指定索引的图像数据
                data = dataset[image_index]
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = data.unsqueeze(0).to(self.device)
            else:
                # 随机选择一个批次中的随机图像
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch index {rand_image_index}.")
        
            # 模型前向传播，获取重建图像
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
            
            # 计算像素级MSE损失
            loss_fn = MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data)  
            pixel_loss = pixel_loss.squeeze(0).cpu().numpy()
            
            # 获取原始图像
            original_img = data.squeeze(0).cpu().numpy()
            
            # 计算每个像素的损失总和，并进行对数变换
            pixel_loss_sum = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()
            pixel_loss_sum = np.log(pixel_loss_sum + 1e-8)
            
            # 计算损失的百分位数用于裁剪
            min_loss = np.percentile(pixel_loss_sum, 1)
            max_loss = np.percentile(pixel_loss_sum, 99)
            clipped_loss = np.clip(pixel_loss_sum, min_loss, max_loss)
            
            # 归一化损失
            norm_pixel_loss = (clipped_loss - min_loss) / (max_loss - min_loss + 1e-8)
            
            # 扁平化损失并进行KMeans聚类
            flattened_loss = norm_pixel_loss.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_loss)
            anomaly_labels = kmeans.labels_.reshape(norm_pixel_loss.shape)
            
            # 计算每个簇的平均损失，确定异常簇
            cluster_mean_loss = [norm_pixel_loss[anomaly_labels == i].mean() for i in range(2)]
            anomaly_cluster = cluster_mean_loss.index(max(cluster_mean_loss))
            
            # 生成二值异常图
            binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
            
            # 连通组件标记
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            
            # 过滤掉小的异常区域
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 30:
                    filtered_anomaly_map[component] = 1
            
            # 可视化结果
            plt.figure(figsize=(18, 6))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(original_img[0], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # 异常热力图
            plt.subplot(1, 3, 2)
            plt.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            plt.colorbar(label='Normalized MSE Loss')
            plt.title('Anomaly Heat Map')
            plt.axis('off')
            
            # 语义异常图
            plt.subplot(1, 3, 3)
            plt.imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            plt.title('Semantic Anomaly Map')
            plt.axis('off')
            
            # 保存可视化结果
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result.png')
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"Anomaly detection visualization saved at {vis_save_path}")
    
    #####################################################################################################################################################
    
    def reconstruct_and_analyze_images_by_time_sequence(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        """
        按时间序列重建并分析图像，生成热力图和异常分类图。
        
        Args:
            target_date (str): 目标日期，格式为 "YYYYMMDD"。
            base_filename_part (str, optional): 文件名的共同部分。
            suffix (str, optional): 文件名的后缀部分。
        """
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        
        if len(image_paths) == 0:
            print("未找到匹配的图像文件。")
            return
        
        # 提取并排序图像日期
        def extract_date(path):
            match = re.search(r'D_(\d{8})T', path)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d")
            else:
                return datetime.min
        
        image_paths.sort(key=extract_date)
        
        # 转换目标日期为 datetime 对象
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
        
        # 选择目标图像及其前后五张图像
        start_index = max(target_index - 5, 0)
        end_index = min(target_index + 6, len(image_paths))
        selected_image_paths = image_paths[start_index:end_index]
        
        if len(selected_image_paths) < 11:
            print(f"选择的图像数量少于11张：{len(selected_image_paths)} 张。")
            return
        
        print(f"已选择日期 {target_date} 及其前后5天的11张图像。")
        
        # 定义图像转换
        transform = transforms.Compose([])
        
        num_images = len(selected_image_paths)
        
        # 准备绘图，5行每行对应不同的可视化内容
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
                # 加载图像
                combined_image = tiff.imread(img_path)
                
                # 处理图像维度，确保为 (C, H, W) 且有2个通道
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] != 2 and combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))
                
                if combined_image.shape[0] != 2:
                    raise ValueError("通道数错误")
                
                # 转换为Tensor
                img_tensor = torch.from_numpy(combined_image).float()
                
                if transform:
                    img_tensor = transform(img_tensor)
                
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # 模型前向传播
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(img_tensor)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                
                # 计算像素级MSE损失
                loss_fn = MSELoss(reduction='none')
                pixel_loss = loss_fn(recon, img_tensor)
                pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
                
                # 收集所有像素误差
                all_pixel_errors.extend(pixel_loss_sum.flatten())
                pixel_loss_sums.append(pixel_loss_sum)
                
                # 记录图像日期
                current_date = extract_date(img_path).strftime("%Y-%m-%d")
                image_dates.append(current_date)
                
                # 生成并存储summed_image
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
        
        # 转换为NumPy数组并训练GMM
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("开始训练全局GMM...")
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_errors)
        
        # 确定异常类别（均值较大的组件）
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        print(f"异常类别为GMM的组件 {anomaly_cluster}，均值为 {component_means[anomaly_cluster]:.4f}")
        
        H, W = pixel_loss_sums[0].shape
        previous_anomalies = np.zeros((H, W), dtype=bool)
        
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
            
            # 连通组件标记
            labeled_map, num_features = label(binary_anomaly_map)
            filtered_anomaly_map = np.zeros_like(binary_anomaly_map)
            
            # 过滤掉小的异常区域
            for i in range(1, num_features + 1):
                component = (labeled_map == i)
                if component.sum() >= 50:
                    filtered_anomaly_map[component] = 1
            
            semantic_anomaly_maps.append(filtered_anomaly_map)
            
            # 分类异常类型
            current_deforestation_map = (filtered_anomaly_map == 1) & (~previous_anomalies)
            ancient_deforestation_map = (filtered_anomaly_map == 1) & (previous_anomalies)
            
            # 更新之前的异常状态
            previous_anomalies = previous_anomalies | (filtered_anomaly_map == 1)
            
            # 归一化损失
            clipped_loss = np.clip(pixel_loss_sum, mse_min, mse_max)
            norm_pixel_loss = (clipped_loss - mse_min) / (mse_max - mse_min + 1e-8)
            
            # 绘制原始图像
            ax_orig = axes[0, idx]
            ax_orig.imshow(summed_image, cmap='gray')
            ax_orig.set_title(f'original image {current_date}')
            ax_orig.axis('off')
            
            # 绘制热力图
            ax_heat = axes[1, idx]
            heatmap = ax_heat.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            ax_heat.set_title(f'heat map {current_date}')
            ax_heat.axis('off')
            plt.colorbar(heatmap, ax=ax_heat, fraction=0.046, pad=0.04, label='MSE loss')
            
            # 绘制语义异常图
            ax_cluster = axes[2, idx]
            cluster_map = semantic_anomaly_maps[-1]
            ax_cluster.imshow(cluster_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            ax_cluster.set_title(f'semantic anomaly map {current_date}')
            ax_cluster.axis('off')
            
            # 绘制当前砍伐图
            ax_current = axes[3, idx]
            ax_current.imshow(current_deforestation_map, cmap='Reds', vmin=0, vmax=1)
            ax_current.set_title(f'current deforestation {current_date}')
            ax_current.axis('off')
            
            # 绘制古老砍伐图
            ax_ancient = axes[4, idx]
            ax_ancient.imshow(ancient_deforestation_map, cmap='Blues', vmin=0, vmax=1)
            ax_ancient.set_title(f'ancient deforestation {current_date}')
            ax_ancient.axis('off')
        
        # 调整布局并保存可视化结果
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_time_sequence.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"带有异常分类的结果已保存到 {vis_save_path}")

#####################################################################################################################################################

    def reconstruct_and_analyze_images_by_clustering(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        
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
        
        print(f"已选择日期 {target_date} 及其前后5天的11张图像")
        
        # 定义图像转换，与 ProcessedForestDataset 中的转换保持一致
        transform = transforms.Compose([
            # 在这里添加任何必要的转换，例如归一化
            # 目前没有应用任何转换，与数据集加载器保持一致
        ])
        
        num_images = len(selected_image_paths)
        
        # 准备绘图，4 行 num_images 列（原始图像、热力图、语义异常图、差异图）
        fig, axes = plt.subplots(4, num_images, figsize=(num_images * 4, 25))
        
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
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_clustering.png')
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close()
            print(f"带有异常分类的结果已保存到 {vis_save_path}")

#####################################################################################################################################################

    def generate_large_change_map(self, target_date, base_filename_part="622_975_S1A__IW___D_", 
                              suffix_template="_VV_gamma0-rtc_db_{row}_{col}_fused.tif", 
                              image_dir="/home/yifan/Documents/data/forest/test/processed",
                              max_row=768, max_col=2304, tile_size=256):
        """
        1. 找出目标日期 target_date 所有符合条件的图像及其前一日期（previous_date）所有符合条件的图像。
        2. 对两个日期的所有图像一起计算像素级 MSE 损失，并对全部像素进行全局GMM聚类，以区分异常与正常类。
        3. 利用训练好的GMM对两个日期的所有图像生成对应的异常二值图。
        4. 将两日期的所有小块异常图拼接成两个大图（一个是 target_date 的大图，一个是 previous_date 的大图）。
        5. 对拼接好的两个大图做差分，得到大尺寸的变化检测图并保存。
        """

        # 将目标日期字符串转换为 datetime 对象
        target_datetime = datetime.strptime(target_date, "%Y%m%d")

        # 从已存在的图像文件中提取日期
        all_images = glob.glob(os.path.join(image_dir, f"{base_filename_part}*"))
        if len(all_images) == 0:
            print("未找到任何匹配的图像文件。")
            return None

        # 提取所有图像日期
        date_to_paths = {}
        for path in all_images:
            match = re.search(r'D_(\d{8})T', path)
            if match:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                if date_obj not in date_to_paths:
                    date_to_paths[date_obj] = []
                date_to_paths[date_obj].append(path)
        
        # 找到 target_date 之前的最近日期
        all_dates = sorted(date_to_paths.keys())
        prev_date = None
        for d in all_dates:
            if d < target_datetime:
                prev_date = d
            elif d == target_datetime:
                continue
            else:
                break

        if prev_date is None:
            print(f"未能找到 {target_date} 的前一日期图像。")
            return None

        # 过滤出 target_date 和 prev_date 对应的所有图像
        target_images_all = date_to_paths.get(target_datetime, [])
        prev_images_all = date_to_paths.get(prev_date, [])

        if len(target_images_all) == 0:
            print(f"未找到目标日期 {target_date} 的图像。")
            return None
        if len(prev_images_all) == 0:
            print(f"未找到前一日期 {prev_date.strftime('%Y%m%d')} 的图像。")
            return None

        # 从文件名中提取行列信息，并筛选出符合 suffix_template 模式的文件
        # suffix_template 中有 {row} 和 {col}，例如 "_VV_gamma0-rtc_db_{row}_{col}_fused.tif"
        # 我们需要匹配 *_VV_gamma0-rtc_db_数字_数字_fused.tif
        # 尝试用正则表达式从文件名中提取 row, col
        pattern_suffix = r'_VV_gamma0-rtc_db_(\d+)_(\d+)_fused\.tif$'

        def extract_row_col(path):
            m = re.search(pattern_suffix, path)
            if m:
                row = int(m.group(1))
                col = int(m.group(2))
                return row, col
            return None, None

        # 将两个日期的图像用字典储存：{(row, col): path}
        target_map = {}
        prev_map = {}
        for p in target_images_all:
            row, col = extract_row_col(p)
            if row is not None and col is not None:
                target_map[(row, col)] = p
        for p in prev_images_all:
            row, col = extract_row_col(p)
            if row is not None and col is not None:
                prev_map[(row, col)] = p

        # 获取所有有效的 (row, col) 坐标集合(两者都存在的 tile)
        common_tiles = set(target_map.keys()).intersection(set(prev_map.keys()))
        if len(common_tiles) == 0:
            print("未找到目标日期和前一日期共有的图像块。")
            return None

        # 准备对所有图像块进行像素级 MSE 计算
        def load_and_compute_pixel_loss(image_path):
            try:
                combined_image = tiff.imread(image_path)
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] == 2:
                        pass
                    elif combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))
                    else:
                        raise ValueError(f"期望2通道，但在 {image_path} 中找到 {combined_image.shape[-1]} 通道。")
                else:
                    raise ValueError(f"图像维度不正确：{combined_image.ndim}, 文件: {image_path}")

                if combined_image.shape[0] != 2:
                    raise ValueError(f"期望2通道，但在 {image_path} 中找到 {combined_image.shape[0]} 通道。")

                img_tensor = torch.from_numpy(combined_image).float().unsqueeze(0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(img_tensor)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                mse_loss = MSELoss(reduction='none')
                pixel_loss = mse_loss(recon, img_tensor)
                pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
                return pixel_loss_sum
            except Exception as e:
                print(f"处理图像 {image_path} 时出错：{e}")
                return None

        # 分别加载所有公共 tile 的两张图像，计算像素级误差，并存储
        pixel_loss_target_dict = {}
        pixel_loss_prev_dict = {}

        all_pixel_losses = []
        # 我们要做全局聚类，所以要将两个日期所有块的像素误差全部收集
        for (row, col) in common_tiles:
            pl_target = load_and_compute_pixel_loss(target_map[(row, col)])
            pl_prev = load_and_compute_pixel_loss(prev_map[(row, col)])
            if pl_target is None or pl_prev is None:
                # 如果有任意一张无法处理，则跳过该 tile
                print(f"跳过块(row={row}, col={col})，因为其中一张图像处理失败。")
                continue
            pixel_loss_target_dict[(row, col)] = pl_target
            pixel_loss_prev_dict[(row, col)] = pl_prev
            all_pixel_losses.append(pl_target.flatten())
            all_pixel_losses.append(pl_prev.flatten())

        if len(all_pixel_losses) == 0:
            print("没有可用于聚类的图像块数据。")
            return None

        all_pixel_losses = np.concatenate(all_pixel_losses, axis=0).reshape(-1, 1)

        # 全局 GMM 聚类
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_losses)
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)

        # 使用全局GMM对每个 tile 的两幅图像计算异常图
        # 然后分别将所有 tile 的异常图拼接为两张大图
        large_map_target = np.zeros((max_row, max_col), dtype=np.uint8)
        large_map_prev = np.zeros((max_row, max_col), dtype=np.uint8)

        for (row, col), pl_target in pixel_loss_target_dict.items():
            pl_prev = pixel_loss_prev_dict[(row, col)]
            # 对目标日期图像像素预测
            pred_target = gmm.predict(pl_target.flatten().reshape(-1, 1))
            anomaly_target = (pred_target.reshape(pl_target.shape) == anomaly_cluster).astype(np.uint8)
            
            # 对前一日期图像像素预测
            pred_prev = gmm.predict(pl_prev.flatten().reshape(-1, 1))
            anomaly_prev = (pred_prev.reshape(pl_prev.shape) == anomaly_cluster).astype(np.uint8)
            
            end_row = min(row + tile_size, max_row)
            end_col = min(col + tile_size, max_col)
            # 填入大图
            large_map_target[row:end_row, col:end_col] = anomaly_target[:end_row - row, :end_col - col]
            large_map_prev[row:end_row, col:end_col] = anomaly_prev[:end_row - row, :end_col - col]

        # 计算两个大图的差异图(变化检测图)
        # 差异 = target_date_异常图 - prev_date_异常图
        difference_map = large_map_target.astype(int) - large_map_prev.astype(int)
        difference_map = np.where(difference_map != 0, 1, 0).astype(np.uint8)

        # 保存结果
        # 保存目标日期、前一日期和差异图
        os.makedirs(self.args.results_path, exist_ok=True)
        save_target_path = os.path.join(self.args.results_path, f'anomaly_map_target_{target_date}.tif')
        save_prev_path = os.path.join(self.args.results_path, f'anomaly_map_prev_{prev_date.strftime("%Y%m%d")}.tif')
        save_diff_path = os.path.join(self.args.results_path, f'anomaly_difference_{target_date}.tif')

        tiff.imwrite(save_target_path, (large_map_target * 255).astype(np.uint8))
        tiff.imwrite(save_prev_path, (large_map_prev * 255).astype(np.uint8))
        tiff.imwrite(save_diff_path, (difference_map * 255).astype(np.uint8))

        print(f"已保存目标日期大图: {save_target_path}")
        print(f"已保存前一日期大图: {save_prev_path}")
        print(f"已保存变化检测图: {save_diff_path}")

        return difference_map
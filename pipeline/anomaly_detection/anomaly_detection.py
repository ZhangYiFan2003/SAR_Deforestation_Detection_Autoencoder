import os, re, glob
import math
import torch
import random
import rasterio
import numpy as np
import tifffile as tiff
import geopandas as gpd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from scipy.ndimage import label
from shapely.geometry import shape
from skimage.transform import rescale
from skimage.morphology import disk, opening, closing
from datetime import datetime
from rasterio import features
from rasterio.transform import from_origin
from torch.nn import MSELoss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pyproj import Transformer
from affine import Affine

#####################################################################################################################################################

class AnomalyDetection:

    def _extract_date_from_path(self, path, pattern=r'D_(\d{8})T'):
        """
        从路径中提取日期，如果失败返回 datetime.min。
        """
        match = re.search(pattern, path)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d")
        else:
            return datetime.min

    def _load_and_preprocess_image(self, image_path, device, transform=None, min_val=None, max_val=None):
        """
        加载并预处理单张影像，返回对应的Tensor和原始np数组。
        """
        combined_image = tiff.imread(image_path)
        if combined_image.ndim == 2:
            combined_image = combined_image[np.newaxis, ...]
        elif combined_image.ndim == 3:
            if combined_image.shape[0] == 2:
                pass
            elif combined_image.shape[-1] == 2:
                combined_image = np.transpose(combined_image, (2, 0, 1))
            else:
                raise ValueError(f"期望的通道数为2，但在文件 {image_path} 中找到了 {combined_image.shape[-1]} 个通道。")
        else:
            raise ValueError(f"图像维度不正确：{combined_image.ndim}，文件路径：{image_path}")
        
        if combined_image.shape[0] != 2:
            raise ValueError(f"期望的通道数为2，但在文件 {image_path} 中找到了 {combined_image.shape[0]} 个通道。")
        
        img_tensor = torch.from_numpy(combined_image).float()
        
        # 归一化
        if min_val is not None and max_val is not None:
            img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
        # 转换
        if transform:
            img_tensor = transform(img_tensor)
            
        img_tensor = img_tensor.unsqueeze(0).to(device)
        return img_tensor, combined_image

    def _model_inference(self, model, img_tensor):
        """
        对图像进行模型推理，并返回重建结果。
        """
        model.eval()
        with torch.no_grad():
            recon = model(img_tensor)
            if isinstance(recon, tuple):
                recon = recon[0]
        return recon

    def _compute_pixel_loss_map(self, recon, img_tensor):
        """
        计算像素级别的MSE损失图（对通道求和后为单通道图）。
        """
        loss_fn = MSELoss(reduction='none')
        pixel_loss = loss_fn(recon, img_tensor)
        pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
        return pixel_loss_sum

    def _filter_small_components(self, binary_map, min_size=50):
        """
        对二值化异常图进行区域过滤与形态学优化:
        1. 连通域过滤：移除小于 min_size 的连通组件。
        2. 形态学开运算 (Opening)：去除小噪声点，平滑区域边界。
        3. 形态学闭运算 (Closing)：填补内部孔洞，使异常区域更紧实连贯。
        
        Args:
            binary_map (np.ndarray): 二值化的异常图 (H, W)，0或1。
            min_size (int): 最小区域大小阈值，小于此值的连通区域将被移除。
            
        Returns:
            np.ndarray: 经过形态学优化和最小区域过滤后的二值图。
        """
        # Step 1: 连通组件过滤
        labeled_map, num_features = label(binary_map)
        filtered_map = np.zeros_like(binary_map, dtype=np.uint8)
        for i in range(1, num_features + 1):
            component = (labeled_map == i)
            if component.sum() >= min_size:
                filtered_map[component] = 1
                
        # 引入形态学操作
        # 选择合适的结构元素（如3x3的圆盘结构元素）
        #selem = disk(3)  # 可以根据需要调整大小，如disk(5)
        
        # Step 2: 开运算 (Opening)
        # Opening = Erosion + Dilation，用于去除小噪点与分离细小的突出部分
        #filtered_map = opening(filtered_map, selem)
        
        # Step 3: 闭运算 (Closing)
        # Closing = Dilation + Erosion，用于填补区域内的小孔洞，使区域更连贯
        #filtered_map = closing(filtered_map, selem)
        
        return filtered_map

    def _fit_global_gmm(self, all_pixel_errors):
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_errors)
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        return gmm, anomaly_cluster, component_means[anomaly_cluster]

    def _predict_anomaly_map(self, pixel_loss_sum, gmm, anomaly_cluster, mse_min=0, mse_max=1050, min_size=50):
        pixel_losses = pixel_loss_sum.flatten().reshape(-1, 1)
        predicted_labels = gmm.predict(pixel_losses)
        anomaly_labels = predicted_labels.reshape(pixel_loss_sum.shape)
        
        binary_anomaly_map = (anomaly_labels == anomaly_cluster).astype(int)
        filtered_anomaly_map = self._filter_small_components(binary_anomaly_map, min_size=min_size)
        
        clipped_loss = np.clip(pixel_loss_sum, mse_min, mse_max)
        norm_pixel_loss = (clipped_loss - mse_min) / (mse_max - mse_min + 1e-8)
        return filtered_anomaly_map, norm_pixel_loss

    def _select_images_by_date(self, image_paths, target_date, backward=5, forward=5):
        # 提取日期并排序
        image_paths.sort(key=lambda p: self._extract_date_from_path(p))
        try:
            target_datetime = datetime.strptime(target_date, "%Y%m%d")
        except ValueError:
            print("目标日期格式不正确。")
            return None
        
        dates = [self._extract_date_from_path(p) for p in image_paths]
        if target_datetime not in dates:
            print(f"未找到目标日期 {target_date} 的图像。")
            return None
        
        target_index = dates.index(target_datetime)
        start_index = max(target_index - backward, 0)
        end_index = min(target_index + forward + 1, len(image_paths))
        selected_image_paths = image_paths[start_index:end_index]
        return selected_image_paths

    def _compute_all_pixel_losses(self, selected_image_paths, transform, device, min_val=None, max_val=None):
        all_pixel_errors = []
        pixel_loss_sums = []
        image_dates = []
        summed_images = []
        
        for idx, img_path in enumerate(selected_image_paths):
            try:
                img_tensor, combined_image = self._load_and_preprocess_image(img_path, device, transform, min_val, max_val)
                recon = self._model_inference(self.model, img_tensor)
                pixel_loss_sum = self._compute_pixel_loss_map(recon, img_tensor)
                
                all_pixel_errors.extend(pixel_loss_sum.flatten())
                pixel_loss_sums.append(pixel_loss_sum)
                
                current_date = self._extract_date_from_path(img_path).strftime("%Y-%m-%d")
                image_dates.append(current_date)
                
                summed_image = combined_image.sum(axis=0)
                if min_val is not None and max_val is not None:
                    summed_image = (summed_image - min_val) / (max_val - min_val + 1e-8)
                    summed_image = np.clip(summed_image, 0.0, 1.0)
                else:
                    # 如果不做归一化，但有transform则尝试归一化
                    if transform:
                        summed_image = (summed_image - summed_image.min()) / (summed_image.max() - summed_image.min() + 1e-8)
                        summed_image = np.clip(summed_image, 0.0, 1.0)
                        
                summed_images.append(summed_image)
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错：{e}")
                # 跳过出错的图像，但继续处理其他的
                continue
            
        return all_pixel_errors, pixel_loss_sums, image_dates, summed_images

    def _draw_single_image_result(self, original_img, norm_pixel_loss, filtered_anomaly_map, vis_save_path):
        """
        绘制单张图像的原始图像、异常热力图和语义异常图的可视化结果。
        """
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
        
        plt.tight_layout()
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()

    #####################################################################################################################################################
    # 原有函数：reconstruct_and_analyze_images 不变，只是调用了辅助函数
    #####################################################################################################################################################

    def reconstruct_and_analyze_images(self, image_index=None):
        # 选择图像
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
                # 随机选择一个批次中的随机图像
                all_test_images = list(self.test_loader)
                selected_batch = random.choice(all_test_images)
                rand_image_index = random.randint(0, selected_batch.size(0) - 1)
                data = selected_batch[rand_image_index].unsqueeze(0).to(self.device)
                print(f"Randomly selected image from batch index {rand_image_index}.")
                
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
                
            loss_fn = MSELoss(reduction='none')
            
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
            filtered_anomaly_map = self._filter_small_components(binary_anomaly_map, min_size=50)
            
            vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_result.png')
            self._draw_single_image_result(original_img, norm_pixel_loss, filtered_anomaly_map, vis_save_path)
            print(f"Anomaly detection visualization saved at {vis_save_path}")

    #####################################################################################################################################################
    # 原有函数：reconstruct_and_analyze_images_by_time_sequence 不变，只是调用了辅助函数
    #####################################################################################################################################################

    def reconstruct_and_analyze_images_by_time_sequence(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("未找到匹配的图像文件。")
            return
        
        selected_image_paths = self._select_images_by_date(image_paths, target_date, backward=5, forward=5)
        if not selected_image_paths or len(selected_image_paths) < 11:
            print(f"选择的图像数量少于11张。")
            return
        
        print(f"已选择日期 {target_date} 及其前后5天的11张图像。")
        
        transform = transforms.Compose([])
        num_images = len(selected_image_paths)
        fig, axes = plt.subplots(5, num_images, figsize=(num_images * 4, 25))
        
        mse_min = 0
        mse_max = 1050
        
        print("开始计算选择的11张图像的像素级误差...")
        all_pixel_errors, pixel_loss_sums, image_dates, summed_images = self._compute_all_pixel_losses(
            selected_image_paths, transform, self.device
        )
        if len(pixel_loss_sums) < 11:
            print("有效图像不足，无法继续处理。")
            return
        
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("开始训练全局GMM...")
        gmm, anomaly_cluster, anomaly_mean = self._fit_global_gmm(all_pixel_errors)
        print(f"异常类别为GMM的组件 {anomaly_cluster}，均值为 {anomaly_mean:.4f}")
        
        H, W = pixel_loss_sums[0].shape
        previous_anomalies = np.zeros((H, W), dtype=bool)
        semantic_anomaly_maps = []
        
        for idx in range(num_images):
            pixel_loss_sum = pixel_loss_sums[idx]
            current_date = image_dates[idx]
            summed_image = summed_images[idx]
            
            filtered_anomaly_map, norm_pixel_loss = self._predict_anomaly_map(pixel_loss_sum, gmm, anomaly_cluster, mse_min, mse_max, min_size=50)
            semantic_anomaly_maps.append(filtered_anomaly_map)
            
            current_deforestation_map = (filtered_anomaly_map == 1) & (~previous_anomalies)
            ancient_deforestation_map = (filtered_anomaly_map == 1) & (previous_anomalies)
            previous_anomalies = previous_anomalies | (filtered_anomaly_map == 1)
            
            # 原始图像
            axes[0, idx].imshow(summed_image, cmap='gray')
            axes[0, idx].set_title(f'original image {current_date}')
            axes[0, idx].axis('off')
            
            # 热力图
            heatmap = axes[1, idx].imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            axes[1, idx].set_title(f'heat map {current_date}')
            axes[1, idx].axis('off')
            plt.colorbar(heatmap, ax=axes[1, idx], fraction=0.046, pad=0.04, label='MSE loss')
            
            # 语义异常图
            axes[2, idx].imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            axes[2, idx].set_title(f'semantic anomaly map {current_date}')
            axes[2, idx].axis('off')
            
            # 当前砍伐
            axes[3, idx].imshow(current_deforestation_map, cmap='Reds', vmin=0, vmax=1)
            axes[3, idx].set_title(f'current deforestation {current_date}')
            axes[3, idx].axis('off')
            
            # 古老砍伐
            axes[4, idx].imshow(ancient_deforestation_map, cmap='Blues', vmin=0, vmax=1)
            axes[4, idx].set_title(f'ancient deforestation {current_date}')
            axes[4, idx].axis('off')
            
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_time_sequence.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"带有异常分类的结果已保存到 {vis_save_path}")

    #####################################################################################################################################################
    # 原有函数：reconstruct_and_analyze_images_by_clustering 不变，只是调用了辅助函数简化部分代码
    #####################################################################################################################################################

    def reconstruct_and_analyze_images_by_clustering(self, target_date, base_filename_part="622_975_S1A__IW___D_", suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("未找到匹配的图像文件。请检查文件路径和命名格式。")
            return
        
        selected_image_paths = self._select_images_by_date(image_paths, target_date, backward=5, forward=5)
        if not selected_image_paths or len(selected_image_paths) < 11:
            print("选择的图像数量少于11张。")
            return
        
        print(f"已选择日期 {target_date} 及其前后5天的11张图像")
        
        transform = transforms.Compose([])
        num_images = len(selected_image_paths)
        
        mse_min = 0
        mse_max = 1050
        
        print("开始计算选择的11张图像的像素级误差...")
        all_pixel_errors, pixel_loss_sums, image_dates, summed_images = self._compute_all_pixel_losses(
            selected_image_paths, transform, self.device
        )
        if len(pixel_loss_sums) < 11:
            print("有效图像不足，无法继续处理。")
            return
        
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("开始训练全局GMM...")
        gmm, anomaly_cluster, anomaly_mean = self._fit_global_gmm(all_pixel_errors)
        print(f"异常类别为GMM的组件 {anomaly_cluster}，均值为 {anomaly_mean:.4f}")
        
        # 计算语义异常图和差异图
        semantic_anomaly_maps = []
        for idx in range(num_images):
            pixel_loss_sum = pixel_loss_sums[idx]
            filtered_anomaly_map, norm_pixel_loss = self._predict_anomaly_map(pixel_loss_sum, gmm, anomaly_cluster, mse_min, mse_max, min_size=50)
            semantic_anomaly_maps.append((filtered_anomaly_map, norm_pixel_loss))
            
        difference_maps = []
        for i in range(1, len(semantic_anomaly_maps)):
            previous_map = semantic_anomaly_maps[i - 1][0]
            current_map = semantic_anomaly_maps[i][0]
            difference_map = np.logical_and(previous_map == 0, current_map == 1).astype(int)
            difference_maps.append(difference_map)
            
        # 准备绘制最终结果：4 行(num_images 列)
        # 第1行：原始图像
        # 第2行：热力图
        # 第3行：语义异常图
        # 第4行：差异图（如果有）
        total_rows = 4 if difference_maps else 3
        fig, axes = plt.subplots(total_rows, num_images, figsize=(num_images * 4, 20 if difference_maps else 15))
        
        for idx in range(num_images):
            current_date = image_dates[idx]
            summed_image = summed_images[idx]
            filtered_anomaly_map, norm_pixel_loss = semantic_anomaly_maps[idx]
            
            # 原始图像
            axes[0, idx].imshow(summed_image, cmap='gray')
            axes[0, idx].set_title(f'original image {current_date}')
            axes[0, idx].axis('off')
            
            # 热力图
            axes[1, idx].imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            axes[1, idx].set_title(f'heat map {current_date}')
            axes[1, idx].axis('off')
            plt.colorbar(axes[1, idx].images[0], ax=axes[1, idx], fraction=0.046, pad=0.04, label='MSE loss')
            
            # 语义异常图
            axes[2, idx].imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            axes[2, idx].set_title(f'semantic anomaly map {current_date}')
            axes[2, idx].axis('off')
            
        # 如果有差异图，绘制差异图行
        if difference_maps:
            for idx, diff_map in enumerate(difference_maps):
                current_date = image_dates[idx + 1]
                axes[3, idx].imshow(diff_map, cmap='bone', vmin=0, vmax=1)
                axes[3, idx].set_title(f'change {current_date}')
                axes[3, idx].axis('off')
                
            # 如果最后一列没有差异图（因为差异图少一个），将其隐藏
            if num_images > len(difference_maps):
                axes[3, -1].axis('off')
                
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_clustering.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"带有异常分类的结果已保存到 {vis_save_path}")

    #####################################################################################################################################################
    # 原有函数：generate_large_change_map 不变，只是引入辅助函数简化重复代码
    #####################################################################################################################################################

    def generate_large_change_map(
        self,
        target_date,
        base_filename_part="622_975_S1A__IW___D_",
        suffix_template="_VV_gamma0-rtc_db_{row}_{col}_fused.tif",
        image_dir="/home/yifan/Documents/data/forest/test/processed",
        tile_size=256,
        min_size=100,
        pixel_loss_threshold=1.0
    ):
        """
        将原先在内存中拼成大图后投影的做法，改为：
          1) 一次性获取所有 tile 的 pixel loss，用 GMM 训练
          2) 逐块应用 GMM，输出每个瓦片的 anomaly_target / anomaly_prev / difference
          3) 读取瓦片原始的 transform/crs，并写出地理对齐的结果 GeoTIFF
          4) 最后将所有差异瓦片矢量化，合并为一个 Shapefile
        """
        from rasterio.features import shapes as rasterio_shapes
        
        # ========== 1. 根据 target_date 找到对应的前一日期，以及瓦片数据 ==========
        target_datetime = datetime.strptime(target_date, "%Y%m%d")
        all_images = glob.glob(os.path.join(image_dir, f"{base_filename_part}*"))
        if len(all_images) == 0:
            print("未找到任何匹配的图像文件。")
            return None
        
        date_to_paths = {}
        for path in all_images:
            date_obj = self._extract_date_from_path(path)
            if date_obj not in date_to_paths:
                date_to_paths[date_obj] = []
            date_to_paths[date_obj].append(path)
        
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
        
        target_images_all = date_to_paths.get(target_datetime, [])
        prev_images_all = date_to_paths.get(prev_date, [])
        
        if len(target_images_all) == 0:
            print(f"未找到目标日期 {target_date} 的图像。")
            return None
        if len(prev_images_all) == 0:
            print(f"未找到前一日期 {prev_date.strftime('%Y%m%d')} 的图像。")
            return None
        
        # ========== 2. 匹配并收集公共瓦片 ==========
        pattern_suffix = r'_VV_gamma0-rtc_db_(\d+)_(\d+)_fused\.tif$'
        
        def extract_row_col(path):
            m = re.search(pattern_suffix, path)
            if m:
                row = int(m.group(1))
                col = int(m.group(2))
                return row, col
            return None, None
        
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
        
        common_tiles = set(target_map.keys()).intersection(set(prev_map.keys()))
        if len(common_tiles) == 0:
            print("未找到目标日期和前一日期共有的图像块。")
            return None
        
        # ========== 3. 函数：读取瓦片并计算 pixel_loss ==========
        def load_and_compute_pixel_loss(image_path):
            """
            读取2波段影像, 做model推断, 得到 pixel_loss_sum。
            """
            try:
                # 这里继续用 tifffile 或者用 rasterio 读取都可以。
                # 若要拿 transform, crs 等, 后面再用 rasterio.open(...) 读取
                combined_image = tiff.imread(image_path)
                
                # 校验波段维度
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]  # shape=(1,H,W)
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] == 2:
                        pass
                    elif combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))  # (H,W,2)->(2,H,W)
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
                pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()  # shape=(H,W)
                return pixel_loss_sum
            except Exception as e:
                print(f"处理图像 {image_path} 时出错：{e}")
                return None
        
        # ========== 4. 收集并训练 GMM ==========
        pixel_loss_target_dict = {}
        pixel_loss_prev_dict = {}
        all_pixel_losses = []
        
        for (row, col) in common_tiles:
            pl_target = load_and_compute_pixel_loss(target_map[(row, col)])
            pl_prev = load_and_compute_pixel_loss(prev_map[(row, col)])
            if pl_target is None or pl_prev is None:
                print(f"跳过块(row={row}, col={col})，因为其中一张图像处理失败。")
                continue
            pixel_loss_target_dict[(row, col)] = pl_target
            pixel_loss_prev_dict[(row, col)] = pl_prev
            
            # 收集像素损失到 GMM 训练集
            all_pixel_losses.append(pl_target.flatten())
            all_pixel_losses.append(pl_prev.flatten())
        
        if len(all_pixel_losses) == 0:
            print("没有可用于聚类的图像块数据。")
            return None
        
        all_pixel_losses = np.concatenate(all_pixel_losses, axis=0).reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_losses)
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        
        # ========== 5. 对每个瓦片应用分类, 并分别写出结果 ==========
        # 这里我们准备收集所有切片的差异多边形, 最后合并到一个Shapefile
        polygons_all = []
        
        # 创建输出文件夹
        os.makedirs(self.args.results_path, exist_ok=True)
        
        for (row, col), pl_target in pixel_loss_target_dict.items():
            pl_prev = pixel_loss_prev_dict[(row, col)]
            
            # 应用 GMM
            pred_target = gmm.predict(pl_target.flatten().reshape(-1, 1))
            anomaly_target = (pred_target.reshape(pl_target.shape) == anomaly_cluster).astype(np.uint8)
            
            pred_prev = gmm.predict(pl_prev.flatten().reshape(-1, 1))
            anomaly_prev = (pred_prev.reshape(pl_prev.shape) == anomaly_cluster).astype(np.uint8)
            
            # ========== (A) 强制 pixel_loss 小于阈值时不可判为“not forest” ========== 
            # 如果该像素损失低于阈值，则认定它是“forest”(0)
            anomaly_target[pl_target < pixel_loss_threshold] = 0
            anomaly_prev[pl_prev < pixel_loss_threshold] = 0
            
            # 可选：对 anomaly_target 和 anomaly_prev 做小连通域过滤
            anomaly_target = self._filter_small_components(anomaly_target, min_size=min_size)
            anomaly_prev = self._filter_small_components(anomaly_prev, min_size=min_size)
            
            # ========== (B) 修正 difference_tile 的定义，仅标记“forest→not forest” ========== 
            # forest=0, not forest=1，因此仅在 anomaly_prev=0 且 anomaly_target=1 时记为1
            difference_tile = np.where((anomaly_prev == 0) & (anomaly_target == 1),1, 0).astype(np.uint8)
            difference_tile = self._filter_small_components(difference_tile, min_size=min_size)
            
            # ==== 在这里读取瓦片原始的 transform / crs，并写出地理对齐的结果 ====
            tile_target_path = os.path.join(self.args.results_path, 
                f"anomaly_target_{target_date}_r{row}_c{col}.tif")
            tile_prev_path = os.path.join(self.args.results_path, 
                f"anomaly_prev_{prev_date.strftime('%Y%m%d')}_r{row}_c{col}.tif")
            tile_diff_path = os.path.join(self.args.results_path, 
                f"anomaly_diff_{target_date}_r{row}_c{col}.tif")
            
            # 以目标瓦片为例, 读取其 transform/crs 来写 anomaly_target
            # 你也可以选择先读取 target_map 再读取 prev_map, 确保二者一致
            with rasterio.open(target_map[(row, col)]) as src_tile:
                tile_crs = src_tile.crs
                tile_transform = src_tile.transform
                height = src_tile.height
                width = src_tile.width
                
                # 写 anomaly_target
                with rasterio.open(
                    tile_target_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=anomaly_target.dtype,
                    crs=tile_crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(anomaly_target, 1)
            
            # 写 anomaly_prev
            # 我们假设 prev_map 与 target_map 在 transform / crs 方面是一致的
            # 否则需要再单独打开 prev_map[(row,col)] 取 transform。
            with rasterio.open(prev_map[(row, col)]) as src_tile:
                tile_crs = src_tile.crs
                tile_transform = src_tile.transform
                height = src_tile.height
                width = src_tile.width
                
                with rasterio.open(
                    tile_prev_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=anomaly_prev.dtype,
                    crs=tile_crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(anomaly_prev, 1)
            
            # 写 difference
            # 这里也可以复用 target_map 的 transform, 不过为安全起见，也跟前面一样单独打开一次
            # 只要确实确认target与prev地理信息是一致的，也可以只打开一次
            with rasterio.open(target_map[(row, col)]) as src_tile:
                tile_crs = src_tile.crs
                tile_transform = src_tile.transform
                height = src_tile.height
                width = src_tile.width
                
                with rasterio.open(
                    tile_diff_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=difference_tile.dtype,
                    crs=tile_crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(difference_tile, 1)
            
            print(f"已输出瓦片 row={row}, col={col} 的 anomaly_target、anomaly_prev、difference。")
            
            # ========== 6. 矢量化该瓦片的 difference ==========
            # 如果想要把所有瓦片的多边形合并到一个 Shapefile，就把这里得到的 polygons 累加起来
            shapes_gen = rasterio_shapes(difference_tile, transform=tile_transform)
            for geom, val in shapes_gen:
                if val == 1:
                    polygons_all.append(shape(geom))
        
        # ========== 7. 合并所有瓦片的多边形，保存为一个Shapefile ==========
        if len(polygons_all) > 0:
            # 以最后一个瓦片的 CRS 或者任意一个瓦片的 CRS 为准
            # 这里假设所有瓦片 crs 一致，所以选用 tile_crs (最后读到的一个)
            gdf = gpd.GeoDataFrame(geometry=polygons_all, crs=tile_crs)
            # 也可考虑 dissolve 或 unary_union 合并重叠多边形
            shp_path = os.path.join(self.args.results_path, f'anomaly_difference_{target_date}.shp')
            gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            print(f"已保存合并差异区域 Shapefile: {shp_path}")
        else:
            print("差异图中未找到任何异常区域对应的多边形。")
        
        return True  # 或返回你需要的其他信息
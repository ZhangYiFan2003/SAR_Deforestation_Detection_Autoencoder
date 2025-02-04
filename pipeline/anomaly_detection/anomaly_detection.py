import os, re, glob
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
from datetime import datetime
from torch.nn import MSELoss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#####################################################################################################################################################

class AnomalyDetection:

    def _extract_date_from_path(self, path, pattern=r'D_(\d{8})T'):
        match = re.search(pattern, path)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d")
        else:
            return datetime.min

    def _load_and_preprocess_image(self, image_path, device, transform=None, min_val=None, max_val=None):
        # Loads and preprocesses a single image, returning the corresponding Tensor and the original numpy array.
        combined_image = tiff.imread(image_path)
        if combined_image.ndim == 2:
            combined_image = combined_image[np.newaxis, ...]
        elif combined_image.ndim == 3:
            if combined_image.shape[0] == 2:
                pass
            elif combined_image.shape[-1] == 2:
                combined_image = np.transpose(combined_image, (2, 0, 1))
            else:
                raise ValueError(f"Expected 2 channels, but found {combined_image.shape[-1]} channels in file {image_path}.")
        else:
            raise ValueError(f"Incorrect image dimensions: {combined_image.ndim}, file path: {image_path}")
        
        if combined_image.shape[0] != 2:
            raise ValueError(f"Expected 2 channels, but found {combined_image.shape[0]} channels in file {image_path}.")
        
        img_tensor = torch.from_numpy(combined_image).float()
        
        # Normalize the image
        if min_val is not None and max_val is not None:
            img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
        # Apply transform if provided
        if transform:
            img_tensor = transform(img_tensor)
            
        img_tensor = img_tensor.unsqueeze(0).to(device)
        return img_tensor, combined_image

    def _model_inference(self, model, img_tensor):
        # Performs model inference on the image and returns the reconstruction result.
        model.eval()
        with torch.no_grad():
            recon = model(img_tensor)
            if isinstance(recon, tuple):
                recon = recon[0]
        return recon

    def _compute_pixel_loss_map(self, recon, img_tensor):
        # Computes the pixel-level MSE loss map (summing over channels to produce a single-channel map).
        loss_fn = MSELoss(reduction='none')
        pixel_loss = loss_fn(recon, img_tensor)
        pixel_loss_sum = pixel_loss.sum(dim=1).squeeze(0).cpu().numpy()
        return pixel_loss_sum

    def _filter_small_components(self, binary_map, min_size=50):
        """
        Applies region filtering and morphological optimization to the binary anomaly map:
        1. Connected component filtering: remove connected components smaller than min_size.
        2. Morphological opening: remove small noise points and smooth region boundaries.
        3. Morphological closing: fill internal holes to make the anomaly regions more coherent.
        
        Args:
            binary_map (np.ndarray): Binary anomaly map (H, W) with values 0 or 1.
            min_size (int): Minimum area threshold; connected components smaller than this will be removed.
            
        Returns:
            np.ndarray: The binary map after morphological optimization and filtering.
        """
        # Step 1: Connected component filtering
        labeled_map, num_features = label(binary_map)
        filtered_map = np.zeros_like(binary_map, dtype=np.uint8)
        for i in range(1, num_features + 1):
            component = (labeled_map == i)
            if component.sum() >= min_size:
                filtered_map[component] = 1
                
        # Introduce morphological operations
        # Choose an appropriate structuring element (e.g., a 3x3 disk-shaped element)
        # selem = disk(3)  # Adjust size as needed, e.g., disk(5)
        
        # Step 2: Morphological Opening (Erosion followed by Dilation) to remove small noise and separate fine protrusions
        # filtered_map = opening(filtered_map, selem)
        
        # Step 3: Morphological Closing (Dilation followed by Erosion) to fill small holes and smooth regions
        # filtered_map = closing(filtered_map, selem)
        
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
        # Extract dates and sort the image paths
        image_paths.sort(key=lambda p: self._extract_date_from_path(p))
        try:
            target_datetime = datetime.strptime(target_date, "%Y%m%d")
        except ValueError:
            print("The target date format is incorrect.")
            return None
        
        dates = [self._extract_date_from_path(p) for p in image_paths]
        if target_datetime not in dates:
            print(f"No image found for the target date {target_date}.")
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
                    # If not normalizing, but a transform is provided, attempt to normalize
                    if transform:
                        summed_image = (summed_image - summed_image.min()) / (summed_image.max() - summed_image.min() + 1e-8)
                        summed_image = np.clip(summed_image, 0.0, 1.0)
                        
                summed_images.append(summed_image)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
            
        return all_pixel_errors, pixel_loss_sums, image_dates, summed_images

    def _draw_single_image_result(self, original_img, norm_pixel_loss, filtered_anomaly_map, vis_save_path):
        """
        Visualizes the results for a single image: the original image, the anomaly heat map, and the semantic anomaly map.
        """
        plt.figure(figsize=(18, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img[0], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Anomaly heat map
        plt.subplot(1, 3, 2)
        plt.imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
        plt.colorbar(label='Normalized MSE Loss')
        plt.title('Anomaly Heat Map')
        plt.axis('off')
        
        # Semantic anomaly map
        plt.subplot(1, 3, 3)
        plt.imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
        plt.title('Semantic Anomaly Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()

#####################################################################################################################################################

    def reconstruct_and_analyze_images(self, image_index=None):
        # Select an image
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
                # Randomly select an image from a random batch
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

    def reconstruct_and_analyze_images_by_time_sequence(self, 
                                                        target_date, 
                                                        base_filename_part="622_975_S1A__IW___D_", 
                                                        suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("No matching image files found.")
            return
        
        selected_image_paths = self._select_images_by_date(image_paths, target_date, backward=2, forward=2)
        if not selected_image_paths or len(selected_image_paths) < 5:
            print(f"The number of selected images is less than 5.")
            return
        
        print(f"Selected images for date {target_date} and 2 images before and after.")
        
        transform = transforms.Compose([])
        num_images = len(selected_image_paths)
        fig, axes = plt.subplots(5, num_images, figsize=(num_images * 4, 25))
        
        mse_min = 0
        mse_max = 1050
        
        print("Starting to compute pixel-level errors for the selected 5 images...")
        all_pixel_errors, pixel_loss_sums, image_dates, summed_images = self._compute_all_pixel_losses(
            selected_image_paths, transform, self.device
        )
        if len(pixel_loss_sums) < 5:
            print("Insufficient valid images to continue processing.")
            return
        
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("Training GMM...")
        gmm, anomaly_cluster, anomaly_mean = self._fit_global_gmm(all_pixel_errors)
        print(f"Anomaly cluster in GMM is component {anomaly_cluster}, with mean {anomaly_mean:.4f}")
        
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
            
            # Original image
            axes[0, idx].imshow(summed_image, cmap='gray')
            axes[0, idx].set_title(f'original image {current_date}')
            axes[0, idx].axis('off')
            
            # Heat map
            heatmap = axes[1, idx].imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            axes[1, idx].set_title(f'heat map {current_date}')
            axes[1, idx].axis('off')
            plt.colorbar(heatmap, ax=axes[1, idx], fraction=0.046, pad=0.04, label='MSE loss')
            
            # Semantic anomaly map
            axes[2, idx].imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            axes[2, idx].set_title(f'semantic anomaly map {current_date}')
            axes[2, idx].axis('off')
            
            # Current deforestation
            axes[3, idx].imshow(current_deforestation_map, cmap='Reds', vmin=0, vmax=1)
            axes[3, idx].set_title(f'current deforestation {current_date}')
            axes[3, idx].axis('off')
            
            # Ancient deforestation
            axes[4, idx].imshow(ancient_deforestation_map, cmap='Blues', vmin=0, vmax=1)
            axes[4, idx].set_title(f'ancient deforestation {current_date}')
            axes[4, idx].axis('off')
            
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_time_sequence.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"Results with anomaly classification saved at {vis_save_path}")

#####################################################################################################################################################

    def reconstruct_and_analyze_images_by_clustering(self, 
                                                     target_date, 
                                                     base_filename_part="622_975_S1A__IW___D_", 
                                                     suffix="_VV_gamma0-rtc_db_0_0_fused.tif"):
        image_dir = "/home/yifan/Documents/data/forest/test/processed"
        pattern = os.path.join(image_dir, f"{base_filename_part}*{suffix}")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("No matching image files found. Please check the file paths and naming format.")
            return
        
        selected_image_paths = self._select_images_by_date(image_paths, target_date, backward=2, forward=2)
        if not selected_image_paths or len(selected_image_paths) < 5:
            print("The number of selected images is less than 5.")
            return
        
        print(f"Selected images for date {target_date} and 2 images before and after")
        
        transform = transforms.Compose([])
        num_images = len(selected_image_paths)
        
        mse_min = 0
        mse_max = 1050
        
        print("Starting to compute pixel-level errors for the selected 5 images...")
        all_pixel_errors, pixel_loss_sums, image_dates, summed_images = self._compute_all_pixel_losses(
            selected_image_paths, transform, self.device
        )
        if len(pixel_loss_sums) < 5:
            print("Insufficient valid images to continue processing.")
            return
        
        all_pixel_errors = np.array(all_pixel_errors).reshape(-1, 1)
        print("Training GMM...")
        gmm, anomaly_cluster, anomaly_mean = self._fit_global_gmm(all_pixel_errors)
        print(f"Anomaly cluster in GMM is component {anomaly_cluster}, with mean {anomaly_mean:.4f}")
        
        # Compute semantic anomaly maps and difference maps
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
            
        # Prepare to plot final results: 4 rows (num_images columns)
        # Row 1: Original image
        # Row 2: Heat map
        # Row 3: Semantic anomaly map
        # Row 4: Difference map
        total_rows = 4 if difference_maps else 3
        fig, axes = plt.subplots(total_rows, num_images, figsize=(num_images * 4, 20 if difference_maps else 15))
        
        for idx in range(num_images):
            current_date = image_dates[idx]
            summed_image = summed_images[idx]
            filtered_anomaly_map, norm_pixel_loss = semantic_anomaly_maps[idx]
            
            # Original image
            axes[0, idx].imshow(summed_image, cmap='gray')
            axes[0, idx].set_title(f'original image {current_date}')
            axes[0, idx].axis('off')
            
            # Heat map
            axes[1, idx].imshow(norm_pixel_loss, cmap='magma', vmin=0, vmax=1.0)
            axes[1, idx].set_title(f'heat map {current_date}')
            axes[1, idx].axis('off')
            plt.colorbar(axes[1, idx].images[0], ax=axes[1, idx], fraction=0.046, pad=0.04, label='MSE loss')
            
            # Semantic anomaly map
            axes[2, idx].imshow(filtered_anomaly_map, cmap='bone', vmin=0, vmax=1, alpha=0.8)
            axes[2, idx].set_title(f'semantic anomaly map {current_date}')
            axes[2, idx].axis('off')
            
        # If difference maps exist, plot the difference map row
        if difference_maps:
            for idx, diff_map in enumerate(difference_maps):
                current_date = image_dates[idx + 1]
                axes[3, idx].imshow(diff_map, cmap='bone', vmin=0, vmax=1)
                axes[3, idx].set_title(f'change {current_date}')
                axes[3, idx].axis('off')
                
            
            if num_images > len(difference_maps):
                axes[3, -1].axis('off')
                
        plt.tight_layout()
        vis_save_path = os.path.join(self.args.results_path, 'anomaly_detection_analysed_by_clustering.png')
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close()
        print(f"Results with anomaly classification saved at {vis_save_path}")

#####################################################################################################################################################

    def generate_large_change_map(
        self,
        target_date,
        prev_date,
        base_filename_part="622_975_S1A__IW___D_",
        suffix_template="_VV_gamma0-rtc_db_{row}_{col}_fused.tif",
        image_dir="/home/yifan/Documents/data/forest/test/processed",
        tile_size=256,
        min_size=100,
        pixel_loss_threshold=1.0
    ):
        
        from rasterio.features import shapes as rasterio_shapes
        
        # 1. Parse the target date and manually specified previous date
        target_datetime = datetime.strptime(target_date, "%Y%m%d")
        prev_datetime = datetime.strptime(prev_date, "%Y%m%d")
        
        all_images = glob.glob(os.path.join(image_dir, f"{base_filename_part}*"))
        if len(all_images) == 0:
            print("No matching image files found.")
            return None
        
        date_to_paths = {}
        for path in all_images:
            date_obj = self._extract_date_from_path(path)
            if date_obj not in date_to_paths:
                date_to_paths[date_obj] = []
            date_to_paths[date_obj].append(path)
        
        target_images_all = date_to_paths.get(target_datetime, [])
        prev_images_all = date_to_paths.get(prev_datetime, [])
        
        if len(target_images_all) == 0:
            print(f"No images found for target date {target_date}.")
            return None
        if len(prev_images_all) == 0:
            print(f"No images found for previous date {prev_date}.")
            return None
        
        # 2. Match and collect common tiles
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
            print("No common image tiles found for the target and previous dates.")
            return None
        
        # 3. Function: Read tiles and compute pixel loss
        def load_and_compute_pixel_loss(image_path):
            # Reads a 2-channel image, performs model inference, and obtains the pixel_loss_sum.
            try:
                combined_image = tiff.imread(image_path)
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]  # shape=(1,H,W)
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] == 2:
                        pass
                    elif combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))  # (H,W,2)->(2,H,W)
                    else:
                        raise ValueError(f"Expected 2 channels, but found {combined_image.shape[-1]} channels in {image_path}.")
                else:
                    raise ValueError(f"Incorrect image dimensions: {combined_image.ndim}, file: {image_path}")
                
                if combined_image.shape[0] != 2:
                    raise ValueError(f"Expected 2 channels, but found {combined_image.shape[0]} channels in {image_path}.")
                
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
                print(f"Error processing image {image_path}: {e}")
                return None
        
        # ========== 4. 收集并训练 GMM ==========
        pixel_loss_target_dict = {}
        pixel_loss_prev_dict = {}
        all_pixel_losses = []
        
        for (row, col) in common_tiles:
            pl_target = load_and_compute_pixel_loss(target_map[(row, col)])
            pl_prev = load_and_compute_pixel_loss(prev_map[(row, col)])
            if pl_target is None or pl_prev is None:
                print(f"Skipping tile (row={row}, col={col}) because one of the images failed to process.")
                continue
            pixel_loss_target_dict[(row, col)] = pl_target
            pixel_loss_prev_dict[(row, col)] = pl_prev
            
            all_pixel_losses.append(pl_target.flatten())
            all_pixel_losses.append(pl_prev.flatten())
        
        if len(all_pixel_losses) == 0:
            print("No image tile data available for clustering.")
            return None
        
        all_pixel_losses = np.concatenate(all_pixel_losses, axis=0).reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(all_pixel_losses)
        component_means = gmm.means_.flatten()
        anomaly_cluster = np.argmax(component_means)
        
        # 5. Apply classification on each tile and write out the results
        polygons_all = []
        os.makedirs(self.args.results_path, exist_ok=True)
        
        for (row, col), pl_target in pixel_loss_target_dict.items():
            pl_prev = pixel_loss_prev_dict[(row, col)]
            
            pred_target = gmm.predict(pl_target.flatten().reshape(-1, 1))
            anomaly_target = (pred_target.reshape(pl_target.shape) == anomaly_cluster).astype(np.uint8)
            
            pred_prev = gmm.predict(pl_prev.flatten().reshape(-1, 1))
            anomaly_prev = (pred_prev.reshape(pl_prev.shape) == anomaly_cluster).astype(np.uint8)
            
            # Set regions below the threshold to forest=0
            anomaly_target[pl_target < pixel_loss_threshold] = 0
            anomaly_prev[pl_prev < pixel_loss_threshold] = 0
            
            anomaly_target = self._filter_small_components(anomaly_target, min_size=min_size)
            anomaly_prev = self._filter_small_components(anomaly_prev, min_size=min_size)
            
            difference_tile = np.where((anomaly_prev == 0) & (anomaly_target == 1), 1, 0).astype(np.uint8)
            difference_tile = self._filter_small_components(difference_tile, min_size=min_size)
            
            with rasterio.open(target_map[(row, col)]) as src_tile:
                tile_crs = src_tile.crs
                tile_transform = src_tile.transform
                
            tile_polygons = []
            shapes_gen = rasterio_shapes(difference_tile, transform=tile_transform)
            for geom, val in shapes_gen:
                if val == 1:
                    tile_polygons.append(shape(geom))
                    
            if len(tile_polygons) > 0:
                tile_gdf = gpd.GeoDataFrame(geometry=tile_polygons, crs=tile_crs)
                subfolder = os.path.join(self.args.results_path, f"{row}_{col}")
                os.makedirs(subfolder, exist_ok=True)
                tile_shp_path = os.path.join(subfolder, f"difference_{row}_{col}.shp")
                tile_gdf.to_file(tile_shp_path, driver='ESRI Shapefile', encoding='utf-8')
                polygons_all.extend(tile_polygons)
                
            print(f"Completed vectorization of differences for tile row={row}, col={col}.")
        
        # 6. Merge all tile polygons and save as a single Shapefile
        if len(polygons_all) > 0:
            gdf = gpd.GeoDataFrame(geometry=polygons_all, crs=tile_crs)
            shp_path = os.path.join(self.args.results_path, f'anomaly_difference_{target_date}.shp')
            gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            print(f"Merged difference area Shapefile saved: {shp_path}")
        else:
            print("No polygons corresponding to anomaly areas were found in the difference map.")
        
        return True

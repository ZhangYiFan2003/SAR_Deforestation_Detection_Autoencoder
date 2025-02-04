import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import torch
import tifffile as tiff
import random

#####################################################################################################################################################

class Plot:
    def _plot_histogram(self, data, title, xlabel, ylabel, save_path, hyperparameters, color='blue', alpha=0.7, bins=1000, xlim=(0, 0.0015)):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black', density=True)
        plt.yscale('log')  
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlim(xlim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        hyperparameters_text = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])
        plt.gcf().text(0.95, 0.5, hyperparameters_text, fontsize=10, ha='right', va='center', transform=plt.gcf().transFigure)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"{title} saved at {save_path}")

#####################################################################################################################################################

    def plot_pixel_error_histogram(self, image_dir="/home/yifan/Documents/data/forest/test/processed", num_bins=1000):
        pattern = os.path.join(image_dir, "*.tif")
        image_paths = glob.glob(pattern)
        if len(image_paths) == 0:
            print("No TIFF image files found in the test dataset.")
            return
        
        transform = None  
        from torch.nn import MSELoss
        loss_fn = MSELoss(reduction='none')
        all_pixel_errors = []
        print("Starting to compute pixel-level errors for all test images...")
        
        for idx, img_path in enumerate(image_paths):
            try:
                combined_image = tiff.imread(img_path)
                if combined_image.ndim == 2:
                    combined_image = combined_image[np.newaxis, ...]
                elif combined_image.ndim == 3:
                    if combined_image.shape[0] != 2 and combined_image.shape[-1] == 2:
                        combined_image = np.transpose(combined_image, (2, 0, 1))
                img_tensor = torch.from_numpy(combined_image).float()
                if transform:
                    img_tensor = transform(img_tensor)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(img_tensor)
                    if isinstance(recon, tuple):
                        recon = recon[0]
                pixel_loss = loss_fn(recon, img_tensor).sum(dim=1).squeeze(0).cpu().numpy()
                all_pixel_errors.extend(pixel_loss.flatten())
                if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
                    print(f"{idx + 1} / {len(image_paths)} images traité。")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
            
        all_pixel_errors = np.array(all_pixel_errors)
        min_mse = all_pixel_errors.min()
        max_mse = all_pixel_errors.max()
        print(f"MSE minimum: {min_mse}")
        print(f"MSE maximum: {max_mse}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_pixel_errors, bins=num_bins, color='skyblue', edgecolor='black')
        plt.title('Histogramme erreur pixel')
        plt.xlabel('Erreur MSE')
        plt.ylabel('Nombre de pixels')
        plt.grid(True)
        hist_save_path = os.path.join(self.args.results_path, 'pixel_error_histogram.png')
        plt.savefig(hist_save_path, bbox_inches='tight')
        plt.close()
        print(f"Pixel error histogram saved to {hist_save_path}")

#####################################################################################################################################################

    def _compare_pixel_mse_histograms(self, val_image_index=None, test_image_index=None):
        def _get_image_and_loss(dataset, image_index):
            if image_index is not None:
                if image_index < 0 or image_index >= len(dataset):
                    raise ValueError(f"Image index {image_index} out of bounds.")
                data = dataset[image_index]
            else:
                image_index = random.randint(0, len(dataset) - 1)
                data = dataset[image_index]
            if isinstance(data, (tuple, list)):
                data = data[0]  
            data = data.unsqueeze(0).to(self.device)
            recon_data = self.model(data)
            if isinstance(recon_data, tuple):
                recon_data = recon_data[0]
            loss_fn = torch.nn.MSELoss(reduction='none')
            pixel_loss = loss_fn(recon_data, data).sum(dim=1).squeeze(0).cpu().numpy()
            return image_index, pixel_loss
        
        self.model.eval()
        with torch.no_grad():
            val_index, val_pixel_loss = _get_image_and_loss(self.validation_loader.dataset, val_image_index)
            print(f"Selected validation image index: {val_index}")
            test_index, test_pixel_loss = _get_image_and_loss(self.test_loader.dataset, test_image_index)
            print(f"Selected test image index: {test_index}")
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(val_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Validation Image {val_index}', color='blue', density=True)
            plt.hist(test_pixel_loss.flatten(), bins=100, alpha=0.7, label=f'Test Image {test_index}', color='orange', density=True)
            plt.yscale('log')  
            plt.xlabel('MSE per Pixel')
            plt.ylabel('Frequency (Log Scale)')
            plt.title('Pixel MSE Distribution: Validation vs Test')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            histogram_save_path = os.path.join(self.args.results_path, f'pixel_mse_comparison_val_{val_index}_test_{test_index}.png')
            plt.savefig(histogram_save_path, bbox_inches='tight')
            plt.close()
            print(f"Pixel MSE histogram comparison saved at {histogram_save_path}")

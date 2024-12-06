import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#####################################################################################################################################################

class ProcessedForestDataset(Dataset):
    """
    Custom Dataset for loading preprocessed 2-channel forest images in .tif format.
    """
    def __init__(self, root_dir, min_val=None, max_val=None, transform=None):
        """
        Args:
            root_dir (string): Root directory containing preprocessed 2-channel TIFF images.
            min_val (float): Global minimum value for normalization (calculated from train dataset).
            max_val (float): Global maximum value for normalization (calculated from train dataset).
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.root_dir = root_dir
        self.min_val = min_val
        self.max_val = max_val
        self.transform = transform
        
        # List all files in the directory ending with '.tif'
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])
    
    def __len__(self):
        # Return the total number of image files
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load and return the image at the specified index.
        
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            torch.Tensor: The preprocessed 2-channel image tensor.
        """
        # Construct the full path to the image file
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Read the multi-channel TIFF image using tifffile
        # Image shape could be (H, W, C) or (C, H, W)
        combined_image = tiff.imread(img_path)  
        
        # Handle cases where the image has unexpected dimensions
        if combined_image.ndim == 2:
            # If the image is single-channel (H, W), add a channel dimension to make it (1, H, W)
            combined_image = combined_image[np.newaxis, ...]  
        
        elif combined_image.ndim == 3:
            if combined_image.shape[-1] == 2:
                # If the image is in (H, W, C) format, transpose to (C, H, W)
                combined_image = np.transpose(combined_image, (2, 0, 1))  
        
        # Ensure the image is in (C, H, W) format with 2 channels
        if combined_image.shape[0] != 2:
            raise ValueError(f"Expected 2 channels, but got {combined_image.shape[0]}")
        
        # Normalize the image using global min and max values
        #if self.min_val is not None and self.max_val is not None:
            #combined_image = (combined_image - self.min_val) / (self.max_val - self.min_val)
        
        # Convert the image to a PyTorch tensor
        combined_image = torch.from_numpy(combined_image).float()
        
        # Apply optional transformations
        if self.transform:
            combined_image = self.transform(combined_image)
        
        return combined_image

#####################################################################################################################################################

class ProcessedForestDataLoader(object):
    """
    Wrapper class for creating DataLoaders for train, validation, and test datasets.
    """
    def __init__(self, args):
        """
        Args:
            args: Command-line arguments containing batch size and CUDA information.
        """
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        
        self.min_train = -15
        self.max_train = -3
        #root_dir = '/home/yifan/Documents/data/forest/train/processed'
        #self.min_train, self.max_train = compute_percentile_min_max(root_dir=root_dir, lower_percentile=1, upper_percentile=99,batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define image transformations (currently none are applied)
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),                     # Random horizontal flip
            #transforms.RandomVerticalFlip(),                       # Random vertical flip
            #transforms.RandomRotation(90),                         # Random rotation by multiples of 90 degrees
        ])
        
        # Create DataLoader for the training dataset
        self.train_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/train/processed', 
                                    min_val=self.min_train, max_val=self.max_train, transform=transform),
                                    batch_size=args.batch_size, shuffle=True, **kwargs)
        
        # Create DataLoader for the validation dataset
        self.validation_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/validation/processed',
                                    min_val=self.min_train, max_val=self.max_train, transform=transform),
                                    batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # Create DataLoader for the test dataset
        self.test_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/test/processed',
                                    min_val=self.min_train, max_val=self.max_train, transform=transform),
                                    batch_size=args.batch_size, shuffle=False, **kwargs)

#####################################################################################################################################################

def compute_percentile_min_max(root_dir, lower_percentile=1, upper_percentile=99, batch_size=100, device='cuda'):
    """
    Efficiently compute global percentile minimum and maximum values for normalization using GPU.

    Args:
        root_dir (string): Root directory containing preprocessed 2-channel TIFF images.
        lower_percentile (float): Lower percentile (e.g., 1).
        upper_percentile (float): Upper percentile (e.g., 99).
        batch_size (int): Number of images to process in each batch.
        device (string): Device to use ('cuda' or 'cpu').

    Returns:
        tuple: Estimated global percentile minimum and maximum values across all images in the directory.
    """
    image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])
    num_files = len(image_files)
    
    print(f"[INFO] Found {num_files} images in {root_dir}. Processing on {device.upper()}...")
    
    batch_min_values = []
    batch_max_values = []
    
    for i in range(0, num_files, batch_size):
        # Process a batch of images
        batch_files = image_files[i:i + batch_size]
        batch_pixels = []
        print(f"[INFO] Processing batch {i // batch_size + 1}/{(num_files + batch_size - 1) // batch_size}...")
        
        for img_file in batch_files:
            img_path = os.path.join(root_dir, img_file)
            combined_image = tiff.imread(img_path)  # Load the image
            
            # Flatten the image and add to batch
            batch_pixels.extend(combined_image.flatten())
        
        # Convert batch pixels to GPU tensor
        batch_pixels_tensor = torch.tensor(batch_pixels, device=device, dtype=torch.float32)
        
        # Compute percentiles for the current batch
        batch_min = torch.quantile(batch_pixels_tensor, lower_percentile / 100.0).item()
        batch_max = torch.quantile(batch_pixels_tensor, upper_percentile / 100.0).item()
        
        # Append results to lists and release memory
        batch_min_values.append(batch_min)
        batch_max_values.append(batch_max)
        del batch_pixels_tensor  # Release GPU memory
        
        print(f"[INFO] Batch {i // batch_size + 1} processed. Min: {batch_min}, Max: {batch_max}.")
    
    print(f"[INFO] Calculating global min and max as the mean of batch results...")
    
    # Calculate global min and max as the mean of batch results
    global_min = np.mean(batch_min_values)
    global_max = np.mean(batch_max_values)
    
    print(f"[INFO] Calculation complete. Global Min: {global_min}, Global Max: {global_max}")
    return global_min, global_max

#####################################################################################################################################################
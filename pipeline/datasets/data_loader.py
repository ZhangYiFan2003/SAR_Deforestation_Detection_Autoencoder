import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from pipeline.transforms import SARTransform, SARTransformConfig

#####################################################################################################################################################

class ProcessedForestDataset(Dataset):
    """
    Custom Dataset for loading preprocessed 2-channel forest images in .tif format.
    """
    def __init__(self, root_dir, min_val=None, max_val=None, transform=None, sar_transform=None):
        """
        Args:
            root_dir (string): Root directory containing preprocessed 2-channel TIFF images.
            min_val (float): Global minimum value for normalization (calculated from train dataset).
            max_val (float): Global maximum value for normalization (calculated from train dataset).
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.root_dir = root_dir
        if sar_transform is not None and (min_val is not None or max_val is not None):
            raise ValueError("Pass sar_transform or min_val/max_val, not both")
        if sar_transform is None:
            config = SARTransformConfig(
                min_value=-15.0 if min_val is None else min_val,
                max_value=-3.0 if max_val is None else max_val,
            )
            sar_transform = SARTransform(config)
        self.sar_transform = sar_transform
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
        
        combined_image = self.sar_transform.read(img_path)
        
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
        num_workers = getattr(args, 'num_workers', None)
        if num_workers is None:
            num_workers = 1 if args.cuda else 0
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        pin_memory = getattr(args, 'pin_memory', None)
        if pin_memory is None:
            pin_memory = args.cuda
        if getattr(args, 'non_blocking', False) and not pin_memory:
            raise ValueError("non_blocking transfers require pin_memory=True")
        if num_workers == 0 and getattr(args, 'persistent_workers', False):
            raise ValueError("persistent_workers requires num_workers > 0")
        if num_workers > 0 and getattr(args, 'prefetch_factor', 2) <= 0:
            raise ValueError("prefetch_factor must be > 0")
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        if num_workers > 0:
            kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 2)
            kwargs['persistent_workers'] = getattr(args, 'persistent_workers', False)

        self.transform_config = SARTransformConfig(
            min_value=getattr(args, 'min_value', -15.0),
            max_value=getattr(args, 'max_value', -3.0),
            clamp=getattr(args, 'clamp_input', False),
            expected_channels=getattr(args, 'expected_channels', 2),
        )
        self.sar_transform = SARTransform(self.transform_config)
        #self.min_train, self.max_train = compute_percentile_min_max(root_dir=root_dir, lower_percentile=1, upper_percentile=99,batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define image transformations (currently none are applied)
        train_dir = getattr(args, 'train_dir', None)
        validation_dir = getattr(args, 'validation_dir', None)
        test_dir = getattr(args, 'test_dir', None)
        if not all((train_dir, validation_dir, test_dir)):
            raise ValueError("train_dir, validation_dir, and test_dir must all be configured")

        # Create DataLoader for the training dataset
        self.train_loader = DataLoader(
            ProcessedForestDataset(root_dir=train_dir, sar_transform=self.sar_transform),
                                    batch_size=args.batch_size, shuffle=True, **kwargs)
        
        # Create DataLoader for the validation dataset
        self.validation_loader = DataLoader(
            ProcessedForestDataset(root_dir=validation_dir, sar_transform=self.sar_transform),
                                    batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # Create DataLoader for the test dataset
        self.test_loader = DataLoader(
            ProcessedForestDataset(root_dir=test_dir, sar_transform=self.sar_transform),
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

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
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory containing preprocessed 2-channel TIFF images.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.root_dir = root_dir
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
        
        # Normalize the image to the range [0, 1]
        combined_image = (combined_image - combined_image.min()) / (combined_image.max() - combined_image.min())
        
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
        
        # Define image transformations (currently none are applied)
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),                     # Random horizontal flip
            #transforms.RandomVerticalFlip(),                       # Random vertical flip
            #transforms.RandomRotation(90),                         # Random rotation by multiples of 90 degrees
        ])
        
        # Create DataLoader for the training dataset
        self.train_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/train/processed', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        # Create DataLoader for the validation dataset
        self.validation_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/validation/processed', transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # Create DataLoader for the test dataset
        self.test_loader = DataLoader(
            ProcessedForestDataset(root_dir='/home/yifan/Documents/data/forest/test/processed', transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)

#####################################################################################################################################################

"""
def main():
    parser = argparse.ArgumentParser(description='Test ProcessedForestDataset and DataLoader')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    args = parser.parse_args()
    
    # Set CUDA usage based on availability and user preference
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Instantiate the ProcessedForestDataLoader
    data_loader = ProcessedForestDataLoader(args)
    
    # Test the train DataLoader
    print("Testing train DataLoader...")
    for batch_idx, data in enumerate(data_loader.train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image block size: {data.size()}")
        print(f"  Image min value: {data.min().item()}, max value: {data.max().item()}")
        
        if batch_idx == 1:  # Only test the first two batches
            break
        
    # Test the test DataLoader
    print("Testing test DataLoader...")
    for batch_idx, data in enumerate(data_loader.test_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image block size: {data.size()}")
        print(f"  Image min value: {data.min().item()}, max value: {data.max().item()}")
        
        if batch_idx == 1:  # Only test the first two batches
            break

if __name__ == "__main__":
    main()
"""
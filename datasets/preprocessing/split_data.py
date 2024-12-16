"""
import os
import numpy as np
from PIL import Image
import tifffile as tiff  

def split_and_fuse_images(vv_dir, vh_dir, output_dir, tile_size=256):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    vv_files = sorted([f for f in os.listdir(vv_dir) if f.endswith('.tif')])
    vh_files = sorted([f for f in os.listdir(vh_dir) if f.endswith('.tif')])
    
    assert len(vv_files) == len(vh_files), "Mismatch in the number of VV and VH files"
    
    for idx in range(0, len(vv_files) - 1):  # Use -1 to ensure two time points can be retrieved
        # Current time point files
        vv_file_1 = vv_files[idx]
        vh_file_1 = vh_files[idx]
        # Next time point files
        vv_file_2 = vv_files[idx + 1]
        vh_file_2 = vh_files[idx + 1]
        
        # Load four images
        vv_image_1 = np.array(Image.open(os.path.join(vv_dir, vv_file_1)))
        vh_image_1 = np.array(Image.open(os.path.join(vh_dir, vh_file_1)))
        vv_image_2 = np.array(Image.open(os.path.join(vv_dir, vv_file_2)))
        vh_image_2 = np.array(Image.open(os.path.join(vh_dir, vh_file_2)))
        
        # Check if image sizes match; skip if they don't
        if vv_image_1.shape != vh_image_1.shape or vv_image_1.shape != vv_image_2.shape or vv_image_1.shape != vh_image_2.shape:
            print(f"Skipping images due to size mismatch: {vv_file_1}, {vh_file_1}, {vv_file_2}, {vh_file_2}")
            continue
        
        # Calculate divisible dimensions
        h, w = vv_image_1.shape
        new_h = (h // tile_size) * tile_size
        new_w = (w // tile_size) * tile_size
        
        vv_image_1 = vv_image_1[:new_h, :new_w]
        vh_image_1 = vh_image_1[:new_h, :new_w]
        vv_image_2 = vv_image_2[:new_h, :new_w]
        vh_image_2 = vh_image_2[:new_h, :new_w]
        
        # Extract timestamps from filenames
        date_1 = vv_file_1.split('_')[6]  # Assume timestamp is the seventh part of the filename
        date_2 = vv_file_2.split('_')[6]  # Assume timestamp is the seventh part of the filename
        
        # Split and save image tiles
        for i in range(0, new_h, tile_size):
            for j in range(0, new_w, tile_size):
                vv_tile_1 = vv_image_1[i:i+tile_size, j:j+tile_size]
                vh_tile_1 = vh_image_1[i:i+tile_size, j:j+tile_size]
                vv_tile_2 = vv_image_2[i:i+tile_size, j:j+tile_size]
                vh_tile_2 = vh_image_2[i:i+tile_size, j:j+tile_size]
                
                # Combine four images into a 4-channel image
                fused_image = np.stack([vv_tile_1, vh_tile_1, vv_tile_2, vh_tile_2], axis=0)  # Stack into (C, H, W)
                
                # Name each channel
                metadata = {
                    'axes': 'CXY',
                    'channel_names': [
                        f"{date_1}_VV",
                        f"{date_1}_VH",
                        f"{date_2}_VV",
                        f"{date_2}_VH"
                    ]
                }
                
                # Save as a 4-channel TIFF file
                fused_filename = os.path.join(output_dir, f"{vv_file_1[:-4]}_{date_1}_{date_2}_{i}_{j}_fused_4channels.tif")
                tiff.imwrite(fused_filename, fused_image, photometric='minisblack', metadata=metadata)
        
        print(f"Processed and fused time points {vv_file_1} and {vv_file_2}")

if __name__ == "__main__":
    vv_dir = '/home/yifan/Documents/data/forest/train/VV'
    vh_dir = '/home/yifan/Documents/data/forest/train/VH'
    output_dir = '/home/yifan/Documents/data/forest/train/processed'
    split_and_fuse_images(vv_dir, vh_dir, output_dir)
"""

import os
import numpy as np
from PIL import Image
import tifffile as tiff  

def split_and_fuse_images(vv_dir, vh_dir, output_dir, tile_size=256):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    vv_files = sorted([f for f in os.listdir(vv_dir) if f.endswith('.tif')])
    vh_files = sorted([f for f in os.listdir(vh_dir) if f.endswith('.tif')])
    
    assert len(vv_files) == len(vh_files), "The number of VV and VH files does not match"
    
    for idx, (vv_file, vh_file) in enumerate(zip(vv_files, vh_files)):
        vv_image = Image.open(os.path.join(vv_dir, vv_file))
        vh_image = Image.open(os.path.join(vh_dir, vh_file))
        
        vv_image = np.array(vv_image)
        vh_image = np.array(vh_image)
        
        # Ensure VV and VH images have the same size
        assert vv_image.shape == vh_image.shape, f"The dimensions of {vv_file} and {vh_file} do not match"
        
        # Calculate dimensions that are divisible by the tile size
        h, w = vv_image.shape
        new_h = (h // tile_size) * tile_size
        new_w = (w // tile_size) * tile_size
        
        vv_image = vv_image[:new_h, :new_w]
        vh_image = vh_image[:new_h, :new_w]
        
        # Split and save image tiles
        for i in range(0, new_h, tile_size):
            for j in range(0, new_w, tile_size):
                vv_tile = vv_image[i:i+tile_size, j:j+tile_size]
                vh_tile = vh_image[i:i+tile_size, j:j+tile_size]
                
                # Combine VV and VH into a 2-channel image
                fused_image = np.stack([vv_tile, vh_tile], axis=0)  # Stack into (C, H, W)
                
                # Save as a 2-channel TIFF file
                fused_filename = os.path.join(output_dir, f"{vv_file[:-4]}_{i}_{j}_fused.tif")
                tiff.imwrite(fused_filename, fused_image, photometric='minisblack')
        
        print(f"Processed and fused {vv_file} and {vh_file}")

if __name__ == "__main__":
    vv_dir = '/home/yifan/Documents/data/forest/test/VV'
    vh_dir = '/home/yifan/Documents/data/forest/test/VH'
    output_dir = '/home/yifan/Documents/data/forest/test/processed'
    split_and_fuse_images(vv_dir, vh_dir, output_dir)
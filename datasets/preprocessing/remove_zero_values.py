import os
import numpy as np
import tifffile as tiff

def remove_invalid_tif_files(tiff_dir):
    """
    Read all TIFF files in the directory, check their maximum and minimum values, 
    and delete the file if both are 0.
    
    Args:
        tiff_dir (string): Directory containing TIFF files.
    """
    # Get all TIFF files
    tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tif')]
    
    removed_files = 0
    
    for tiff_file in tiff_files:
        tiff_path = os.path.join(tiff_dir, tiff_file)
        
        try:
            # Read the TIFF file
            img = tiff.imread(tiff_path)
            
            # Check minimum and maximum values
            img_min = np.min(img)
            img_max = np.max(img)
            
            if img_min == 0 and img_max == 0:
                print(f"The maximum and minimum values of file {tiff_file} are both 0. Deleting the file.")
                os.remove(tiff_path)  # Delete the file
                removed_files += 1
            else:
                print(f"File {tiff_file} - Maximum value: {img_max}, Minimum value: {img_min}")
        except Exception as e:
            print(f"An error occurred while reading file {tiff_file}: {e}")
    
    print(f"Deleted a total of {removed_files} invalid TIFF files.")

if __name__ == "__main__":
    # Directory to be processed
    tiff_dir = '/home/yifan/Documents/data/forest/test/processed'  # Modify this to your file path
    remove_invalid_tif_files(tiff_dir)

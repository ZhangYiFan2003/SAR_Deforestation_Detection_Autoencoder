import os
import numpy as np
import tifffile as tiff

def check_and_remove_nan_images(directory):
    """
    Traverse all TIFF files in the directory, check if they contain NaN values, and delete those files if they do.
    
    Args:
        directory (string): Directory containing .tif images.
    """
    # Get all .tif files in the directory
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    
    for file in tif_files:
        file_path = os.path.join(directory, file)
        
        try:
            # Read the TIFF file
            img = tiff.imread(file_path)
            
            # Check if it contains NaN values
            if np.isnan(img).any():
                print(f"File {file} contains NaN values. Deleting...")
                os.remove(file_path)  # Delete the file
            else:
                print(f"File {file} does not contain NaN values. Keeping it.")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
if __name__ == "__main__":
    # Update the path to the folder where processed .tif files are stored
    processed_directory = '/home/yifan/Documents/data/forest/test/processed'
    
    # Process the 'processed' folder
    print("Processing the 'processed' folder...")
    check_and_remove_nan_images(processed_directory)